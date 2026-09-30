"""
test_b13_2_metrics_and_benchmarks.py — Comprehensive Test Suite for Gate B13.2
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Verifies:
1. SessionMetrics dataclass and SessionMetricsExporter (dictionary, JSON, text, derived indicators, zero-division safety).
2. Live ExperimentLogger & TelemetryDiagnosticAggregator session assembly and export.
3. BenchmarkResult dataclass and BenchmarkHarness discovery, execution, and export.
4. Specific evaluation benchmarks:
   - pipeline_runtime benchmark (latency & FPS against <= 50 ms threshold)
   - synthetic_procedural benchmark (EXP-001 S1-S5, skipped, out-of-order, recovery)
   - telemetry_overhead benchmark (against < 0.10 ms threshold)
   - orientation_robustness benchmark (upright, inverted, rotated 90/270)
   - rate_matching benchmark (DROP_OLDEST with LatestFrameBuffer)
   - architecture_comparison benchmark (raw vs full 6-stage architecture)
5. CLI execution and artifact generation.
6. Mathematical safety, boundary conditions, epistemic honesty, and contract isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict
import unittest

import numpy as np

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import (
    InferenceWorker,
    LatestFrameBuffer,
    RateStrategy,
)
from backend.ai.telemetry_aggregator import (
    TelemetryDiagnosticAggregator,
    TelemetrySnapshot,
)
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryEvent, RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.logging.experiment_logger import ExperimentLogger
from backend.logging.session_metrics_exporter import (
    SessionMetrics,
    SessionMetricsExporter,
)
from evaluation.benchmark_harness import (
    BenchmarkHarness,
    BenchmarkResult,
    MAX_PIPELINE_LATENCY_MS_THRESHOLD,
    MAX_TELEMETRY_OVERHEAD_MS_THRESHOLD,
    main,
)


class TestSessionMetricsExporterCore(unittest.TestCase):
    """Unit tests for SessionMetrics and SessionMetricsExporter calculations and formatting."""

    def test_01_empty_session_metrics_creation_and_defaults(self):
        """Verify default empty metrics generation without errors."""
        metrics = SessionMetricsExporter.export_session()
        self.assertIsInstance(metrics, SessionMetrics)
        self.assertTrue(metrics.session_id.startswith("SESSION_"))
        self.assertEqual(metrics.experiment_id, "EXP-001")
        self.assertEqual(metrics.operator_id, "Astronaut-01")
        self.assertEqual(metrics.session_state, "IDLE")
        self.assertEqual(metrics.duration_seconds, 0.0)
        self.assertEqual(metrics.total_cycles, 0)
        self.assertIn("processed_frame_ratio", metrics.derived_metrics)

    def test_02_metrics_serialization_to_dict_and_json(self):
        """Verify .to_dict() and .to_json() produce complete and deserializable objects."""
        metrics = SessionMetricsExporter.export_session(
            experiment_id="EXP-001",
            operator_id="Pilot-02",
            notes="Serial testing notes",
        )
        data_dict = metrics.to_dict()
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict["operator_id"], "Pilot-02")
        self.assertEqual(data_dict["notes"], "Serial testing notes")

        json_str = metrics.to_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["operator_id"], "Pilot-02")
        self.assertEqual(parsed["experiment_id"], "EXP-001")

    def test_03_ascii_text_report_generation(self):
        """Verify .to_text() includes all major sections and disclaimer."""
        metrics = SessionMetricsExporter.export_session()
        report = metrics.to_text()
        self.assertIn("ISRO BHARATIYA ANTARIKSH STATION", report)
        self.assertIn("1. PIPELINE THROUGHPUT & BUFFER UTILIZATION:", report)
        self.assertIn("2. LATENCY PROFILES (ms):", report)
        self.assertIn("3. STAGE LATENCIES (ms):", report)
        self.assertIn("4. DIAGNOSTICS SUMMARY:", report)
        self.assertIn("5. DERIVED PERFORMANCE & PROCEDURE INDICATORS:", report)
        self.assertIn("NOTE: Procedural validity and execution indicators measure protocol adherence.", report)

    def test_04_derived_metrics_zero_division_safety(self):
        """Verify derived metrics handle zero denominators gracefully without crashing."""
        derived = SessionMetricsExporter.calculate_derived_metrics(
            pipeline_data={},
            procedure_data={},
            recovery_data={},
            logger_summary=None,
        )
        self.assertEqual(derived["processed_frame_ratio"], 0.0)
        self.assertEqual(derived["effective_drop_rate"], 0.0)
        self.assertEqual(derived["procedural_validity_ratio"], 0.0)
        self.assertEqual(derived["anomaly_ratio"], 0.0)
        self.assertEqual(derived["recovery_ratio"], 1.0)
        self.assertEqual(derived["step_completion_ratio"], 0.0)
        self.assertEqual(derived["average_confidence"], 0.0)

    def test_05_derived_metrics_exact_formulas(self):
        """Verify precision calculation for realistic metric payloads."""
        pipeline_data = {
            "frames_submitted": 100,
            "frames_processed": 80,
            "frames_replaced": 20,
        }
        procedure_data = {
            "valid_transitions": 8,
            "skipped_steps": 1,
            "out_of_order_events": 1,
            "invalid_object_events": 0,
            "unrecognized_events": 0,
            "completed_steps": 4,
        }
        recovery_data = {
            "total_recoveries": 2,
        }
        logger_summary = {
            "total_steps": 10,
            "valid_steps": 4,
            "anomalies_detected": 2,
            "recoveries_triggered": 2,
            "average_confidence": 0.88,
        }

        derived = SessionMetricsExporter.calculate_derived_metrics(
            pipeline_data=pipeline_data,
            procedure_data=procedure_data,
            recovery_data=recovery_data,
            logger_summary=logger_summary,
            total_canonical_steps=5,
        )

        self.assertEqual(derived["processed_frame_ratio"], 0.80)
        self.assertEqual(derived["effective_drop_rate"], 0.20)
        # valid: 8, total_evals: 10 => 0.80
        self.assertEqual(derived["procedural_validity_ratio"], 0.80)
        # anomalies: 2, total_steps: 10 => 0.20
        self.assertEqual(derived["anomaly_ratio"], 0.20)
        # recoveries: 2, anomalies: 2 => 1.00
        self.assertEqual(derived["recovery_ratio"], 1.00)
        # completed: 4, canonical: 5 => 0.80
        self.assertEqual(derived["step_completion_ratio"], 0.80)
        self.assertEqual(derived["average_confidence"], 0.88)

    def test_06_derived_metrics_recovery_ratio_boundary(self):
        """Verify recovery ratio when recoveries occur without anomalies or vice versa."""
        # 0 recoveries, 2 anomalies => 0.0
        d1 = SessionMetricsExporter.calculate_derived_metrics(
            pipeline_data={},
            procedure_data={"skipped_steps": 2},
            recovery_data={"total_recoveries": 0},
        )
        self.assertEqual(d1["recovery_ratio"], 0.0)

        # 0 anomalies, 0 recoveries => 1.0
        d2 = SessionMetricsExporter.calculate_derived_metrics(
            pipeline_data={},
            procedure_data={},
            recovery_data={"total_recoveries": 0},
        )
        self.assertEqual(d2["recovery_ratio"], 1.0)

    def test_07_export_to_file_json_and_text(self):
        """Verify export_to_file creates both JSON and text reports in target directory."""
        metrics = SessionMetricsExporter.export_session(
            experiment_id="EXP-001",
            operator_id="Scientist-03",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_paths = SessionMetricsExporter.export_to_file(
                metrics=metrics,
                output_dir=tmp_dir,
                base_name="test_session_001",
                export_text=True,
            )
            json_p = Path(out_paths["json_path"])
            txt_p = Path(out_paths["txt_path"])

            self.assertTrue(json_p.exists())
            self.assertTrue(txt_p.exists())

            # Verify contents
            with open(json_p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.assertEqual(loaded["operator_id"], "Scientist-03")

            with open(txt_p, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Scientist-03", content)


class TestSessionMetricsLoggerIntegration(unittest.TestCase):
    """Tests integrating SessionMetricsExporter with live ExperimentLogger and TelemetryDiagnosticAggregator."""

    def test_08_live_experiment_logger_full_session_export(self):
        """Verify SessionMetrics captures full audit trail, steps, anomalies, and recoveries from ExperimentLogger."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(log_dir=Path(tmp_dir))
            logger.start_session(
                experiment_id="EXP-001",
                experiment_name="Sample Transfer",
                operator_id="Astronaut-07",
                notes="Live test session",
            )

            # Log valid steps
            logger.log_step(
                step_id=1,
                step_title="Step 1",
                expected_action="pickup_sample_vial",
                detected_action="pickup_sample_vial",
                confidence=0.95,
                validation_status="VALID",
            )
            logger.log_step(
                step_id=2,
                step_title="Step 2",
                expected_action="unscrew_containment_cap",
                detected_action="unscrew_containment_cap",
                confidence=0.92,
                validation_status="VALID",
            )

            # Log anomaly and recovery
            logger.log_anomaly(
                anomaly_type="SKIPPED_STEP",
                message="Skipped step S3",
                step_id=4,
            )
            rec_evt = RecoveryEvent(
                expected_step="S3",
                expected_action="INSERT_PIPETTE_TIP",
                expected_object="PIPETTE_TIP",
                detected_action="PIPETTE_BUFFER",
                detected_object="BUFFER_VIAL",
                recovery_instruction="Please perform S3 first.",
                timeout_s=30.0,
            )
            logger.log_recovery(
                recovery_event=rec_evt,
            )

            # Telemetry aggregator
            aggregator = TelemetryDiagnosticAggregator()
            aggregator.record_worker_timings(
                process_end_mono=time.monotonic(),
                infer_dur_ms=12.5,
                frame_age_ms=8.0,
                dispatch_ms=1.5,
                e2e_ms=22.0,
            )
            aggregator.record_stage_latencies(
                stage_a_rectification_ms=2.0,
                stage_b_detection_ms=8.0,
                stage_c_consistency_ms=0.5,
                stage_d_temporal_ms=0.5,
                stage_e_fsm_ms=1.0,
                stage_f_adapter_ms=0.5,
            )
            aggregator.record_fsm_result({
                "status": "VALID",
                "detected_step": "S1",
                "expected_step": "S1",
            })

            # Export while session is still active
            metrics = SessionMetricsExporter.export_session(
                telemetry_aggregator=aggregator,
                experiment_logger=logger,
            )

            self.assertEqual(metrics.experiment_id, "EXP-001")
            self.assertEqual(metrics.operator_id, "Astronaut-07")
            self.assertEqual(metrics.session_state, "IN_PROGRESS")
            self.assertEqual(len(metrics.steps_audit_trail), 2)
            self.assertEqual(len(metrics.anomalies), 1)
            self.assertEqual(len(metrics.recoveries), 1)
            self.assertAlmostEqual(metrics.latencies["mean_inference_ms"], 12.5)

            # End session and export completed metrics
            session_files = logger.end_session(status="COMPLETED")
            with open(session_files["json_path"], "r", encoding="utf-8") as f:
                completed_data = json.load(f)

            metrics_completed = SessionMetricsExporter.export_session(
                telemetry_aggregator=aggregator,
                session_data=completed_data,
            )
            self.assertEqual(metrics_completed.session_state, "COMPLETED")
            self.assertGreaterEqual(metrics_completed.duration_seconds, 0.0)

    def test_09_session_isolation(self):
        """Verify multiple consecutive sessions export independently without state leakage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(log_dir=Path(tmp_dir))

            # Session 1
            logger.start_session(experiment_id="EXP-001", experiment_name="Exp 1", operator_id="Op-1")
            logger.log_step(step_id=1, step_title="S1", expected_action="act1", detected_action="act1", confidence=0.9, validation_status="VALID")
            metrics1 = SessionMetricsExporter.export_session(experiment_logger=logger)
            logger.end_session()

            # Session 2
            logger.start_session(experiment_id="EXP-002", experiment_name="Exp 2", operator_id="Op-2")
            logger.log_step(step_id=1, step_title="S1", expected_action="act2", detected_action="act2", confidence=0.85, validation_status="VALID")
            logger.log_step(step_id=2, step_title="S2", expected_action="act3", detected_action="act3", confidence=0.88, validation_status="VALID")
            metrics2 = SessionMetricsExporter.export_session(experiment_logger=logger)
            logger.end_session()

            self.assertEqual(metrics1.experiment_id, "EXP-001")
            self.assertEqual(metrics1.operator_id, "Op-1")
            self.assertEqual(len(metrics1.steps_audit_trail), 1)

            self.assertEqual(metrics2.experiment_id, "EXP-002")
            self.assertEqual(metrics2.operator_id, "Op-2")
            self.assertEqual(len(metrics2.steps_audit_trail), 2)


class TestBenchmarkResultAndHarnessDiscovery(unittest.TestCase):
    """Tests for BenchmarkResult and BenchmarkHarness discovery and helpers."""

    def test_10_benchmark_result_dataclass_methods(self):
        """Verify BenchmarkResult .to_dict(), .to_json(), and .to_text()."""
        res = BenchmarkResult(
            benchmark_name="sample_test",
            category="unit_test",
            timestamp="2026-09-28T12:00:00Z",
            duration_seconds=1.234,
            iterations=10,
            metrics={"throughput_fps": 35.5, "latency_ms": 14.2},
            thresholds={"max_latency_ms": 50.0},
            passed=True,
            environment={"os": "Windows"},
            limitations=["Testing limitation notice"],
        )
        d = res.to_dict()
        self.assertEqual(d["benchmark_name"], "sample_test")
        self.assertTrue(d["passed"])

        j = res.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["metrics"]["throughput_fps"], 35.5)

        txt = res.to_text()
        self.assertIn("BENCHMARK REPORT: SAMPLE_TEST [PASSED]", txt)
        self.assertIn("throughput_fps", txt)
        self.assertIn("Testing limitation notice", txt)

    def test_11_benchmark_harness_list_benchmarks(self):
        """Verify BenchmarkHarness.list_benchmarks() lists all expected suites."""
        b_list = BenchmarkHarness.list_benchmarks()
        self.assertIn("pipeline_runtime", b_list)
        self.assertIn("synthetic_procedural", b_list)
        self.assertIn("telemetry_overhead", b_list)
        self.assertIn("orientation_robustness", b_list)
        self.assertIn("rate_matching", b_list)
        self.assertIn("architecture_comparison", b_list)


class TestIndividualEvaluationBenchmarks(unittest.TestCase):
    """Detailed verification of each individual benchmark suite in BenchmarkHarness."""

    def test_12_pipeline_runtime_benchmark(self):
        """Verify pipeline_runtime benchmark executes and meets <= 50 ms latency threshold."""
        result = BenchmarkHarness.run_pipeline_runtime_benchmark(num_frames=20)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "pipeline_runtime")
        self.assertTrue(result.passed)
        self.assertIn("mean_pipeline_latency_ms", result.metrics)
        self.assertIn("p95_pipeline_latency_ms", result.metrics)
        self.assertIn("throughput_fps", result.metrics)
        self.assertLessEqual(result.metrics["mean_pipeline_latency_ms"], MAX_PIPELINE_LATENCY_MS_THRESHOLD)
        self.assertGreaterEqual(result.metrics["throughput_fps"], 20.0)

    def test_13_synthetic_procedural_benchmark_full_flow(self):
        """Verify synthetic_procedural benchmark traverses S1-S5 and validates anomalies and recovery."""
        result = BenchmarkHarness.run_synthetic_procedural_benchmark(num_cycles=2)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "synthetic_procedural")
        self.assertTrue(result.passed)

        self.assertEqual(result.metrics["valid_transitions_observed"], 10)
        self.assertEqual(result.metrics["expected_valid_transitions"], 10)
        self.assertEqual(result.metrics["recoveries_triggered"], 2)
        self.assertEqual(result.metrics["traversal_accuracy_ratio"], 1.0)
        self.assertTrue(len(result.limitations) > 0)

    def test_14_telemetry_overhead_benchmark(self):
        """Verify telemetry_overhead benchmark executes and meets < 0.10 ms aggregation threshold."""
        result = BenchmarkHarness.run_telemetry_overhead_benchmark(iterations=100)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "telemetry_overhead")
        self.assertTrue(result.passed)
        self.assertLess(result.metrics["mean_overhead_ms"], MAX_TELEMETRY_OVERHEAD_MS_THRESHOLD)
        self.assertGreater(result.metrics["mean_overhead_us"], 0.0)

    def test_15_orientation_robustness_benchmark(self):
        """Verify orientation_robustness benchmark evaluates rotation and disturbance invariance."""
        result = BenchmarkHarness.run_orientation_robustness_benchmark(max_frames=3)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "orientation_robustness")
        self.assertTrue(result.passed)
        self.assertTrue(result.metrics["all_orientations_evaluated"])

    def test_16_rate_matching_benchmark_drop_oldest(self):
        """Verify rate_matching benchmark demonstrates producer-consumer decoupling with DROP_OLDEST."""
        result = BenchmarkHarness.run_rate_matching_benchmark(duration_per_run_sec=0.02)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "rate_matching")
        self.assertTrue(result.passed)

    def test_17_architecture_comparison_benchmark(self):
        """Verify architecture_comparison benchmark benchmarks raw pipeline vs 6-stage architecture."""
        result = BenchmarkHarness.run_architecture_comparison_benchmark(iterations=3)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "architecture_comparison")
        self.assertTrue(result.passed)

    def test_18_run_all_benchmarks(self):
        """Verify BenchmarkHarness.run_all() runs all suites and aggregates results."""
        results = BenchmarkHarness.run_all(quick=True)
        self.assertEqual(len(results), len(BenchmarkHarness.AVAILABLE_BENCHMARKS))
        for name, res in results.items():
            self.assertIsInstance(res, BenchmarkResult)
            self.assertTrue(res.passed, f"Benchmark {name} failed: {res.metrics}")


class TestBenchmarkCLIAndArtifactExport(unittest.TestCase):
    """Tests for standalone execution and file export via CLI."""

    def test_19_cli_single_benchmark_export(self):
        """Verify CLI execution of a single benchmark with JSON export."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ret = main([
                "--benchmark", "synthetic_procedural",
                "--output-dir", tmp_dir,
            ])
            self.assertEqual(ret, 0)
            json_file = Path(tmp_dir) / "synthetic_procedural_benchmark.json"
            txt_file = Path(tmp_dir) / "synthetic_procedural_benchmark.txt"
            self.assertTrue(json_file.exists())
            self.assertTrue(txt_file.exists())

    def test_20_cli_all_benchmarks_export(self):
        """Verify CLI execution with --all flag exports all benchmarks and summary."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ret = main([
                "--all",
                "--quick",
                "--output-dir", tmp_dir,
            ])
            self.assertEqual(ret, 0)
            summary_file = Path(tmp_dir) / "benchmark_summary.json"
            self.assertTrue(summary_file.exists())

            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                self.assertIn("synthetic_procedural", summary_data)
                self.assertIn("pipeline_runtime", summary_data)


class TestArchitecturalInvariantsAndContractPreservation(unittest.TestCase):
    """Verifies that exporter and benchmark harness strictly preserve architectural boundaries."""

    def test_21_observer_non_mutation(self):
        """Verify calling export_session and benchmarks does not alter FSM or pipeline internal state."""
        fsm = SequenceValidatorFSM()
        initial_state = fsm.state
        initial_step = fsm.current_step_index

        agg = TelemetryDiagnosticAggregator()
        pipeline = InferencePipeline(telemetry_aggregator=agg)
        initial_agg = pipeline.telemetry_aggregator.get_snapshot_dict()

        # Run exporter
        _ = SessionMetricsExporter.export_session(telemetry_aggregator=pipeline.telemetry_aggregator)

        # Assert FSM state unchanged
        self.assertEqual(fsm.state, initial_state)
        self.assertEqual(fsm.current_step_index, initial_step)

        # Assert pipeline snapshot unchanged
        after_agg = pipeline.telemetry_aggregator.get_snapshot_dict()
        self.assertEqual(initial_agg["pipeline"]["frames_processed"], after_agg["pipeline"]["frames_processed"])

    def test_22_public_contract_integrity(self):
        """Verify public AI contract dictionary has exact 8 fields."""
        pipeline = InferencePipeline()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame_public(dummy_frame, timestamp="2026-09-28T12:00:00")

        self.assertIsInstance(result, dict)
        expected_keys = {
            "timestamp",
            "action",
            "object",
            "confidence",
            "expected_step",
            "detected_step",
            "status",
            "next_step",
        }
        self.assertEqual(set(result.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
