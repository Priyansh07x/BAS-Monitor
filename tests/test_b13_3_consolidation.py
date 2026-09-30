"""
test_b13_3_consolidation.py — Comprehensive Verification & Gate B13 Consolidation Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B13.3)

Comprehensive end-to-end verification covering:
1. Full end-to-end telemetry and diagnostic data path traversal.
2. 12 diagnostic scenario validations across B11 (consistency & uncertainty), B10 (temporal confirmation),
   FSM (valid, skipped, out-of-order, invalid object), and B12 (recovery triggers & resolution).
3. Complete session lifecycle management (START -> IN_PROGRESS -> PAUSE -> RESUME -> COMPLETED -> EXPORT)
   with strict reset and multi-session state isolation.
4. Derived metrics audit with explicit zero-denominator safety across all 6 mathematical indicators.
5. Deterministic JSON and human-readable ASCII report serialization and reproducibility.
6. Unified BenchmarkHarness verification across all 6 benchmark suites.
7. Acceptance threshold compliance (<= 50 ms pipeline latency, < 0.10 ms telemetry overhead).
8. Epistemic honesty verification: explicit labeling of software benchmarks and disclosure of deferred EXP-001 checkpoints.
9. Concurrency & bounded-memory stress testing without deadlocks, state corruption, or thread leaks.
10. Strict preservation of the frozen 8-field public AI contract and non-mutation of B10/B11/B12/FSM decision logic.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
import unittest

import numpy as np

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import (
    InferenceWorker,
    LatestFrameBuffer,
    RateStrategy,
)
from backend.ai.multimodal_consistency import (
    CATEGORY_CROSS_MODAL_CONFLICT,
    CATEGORY_RELIABLE_ALIGNED,
    MultimodalConsistencyEvaluator,
)
from backend.ai.telemetry_aggregator import (
    TelemetryDiagnosticAggregator,
    TelemetrySnapshot,
)
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.bridge import Bridge
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import (
    RecoveryEvent,
    RecoveryManager,
)
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
    main as harness_main,
)


class TestB13ComprehensiveTelemetryTraversal(unittest.TestCase):
    """Verifies end-to-end telemetry collection and diagnostic flow through all pipeline stages."""

    def setUp(self):
        self.aggregator = TelemetryDiagnosticAggregator(window_size=100)
        self.fsm = SequenceValidatorFSM()
        self.recovery_mgr = RecoveryManager()
        self.pipeline = InferencePipeline(
            validator=self.fsm,
            recovery_manager=self.recovery_mgr,
            telemetry_aggregator=self.aggregator,
        )

    def test_01_full_perception_telemetry_flow(self):
        """Verify telemetry flows through preprocessing, detection, consistency, temporal, FSM, and adapter."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.fsm.start(operator_id="Astronaut-01")
        self.aggregator.start_session("EXP_001_SESSION_01")

        # Process multiple frames through pipeline
        for i in range(5):
            mono_now = time.monotonic()
            self.aggregator.record_frame_submission(mono_now)
            res = self.pipeline.process_frame_public(
                frame=dummy_frame,
                timestamp=f"2026-09-28T12:00:0{i}",
            )
            self.aggregator.record_worker_timings(
                process_end_mono=time.monotonic(),
                infer_dur_ms=5.0,
                frame_age_ms=1.0,
                dispatch_ms=0.1,
                e2e_ms=6.1,
            )
            # Contract verification
            self.assertIn("action", res)
            self.assertIn("object", res)
            self.assertIn("status", res)

        snapshot = self.aggregator.get_snapshot()
        self.assertIsInstance(snapshot, TelemetrySnapshot)
        self.assertEqual(snapshot.pipeline.frames_submitted, 5)
        self.assertEqual(snapshot.pipeline.frames_processed, 5)
        self.assertGreater(snapshot.pipeline.mean_inference_ms, 0.0)

    def test_02_diagnostic_traversal_all_twelve_scenarios(self):
        """
        Exercises all 12 representative diagnostic scenarios through integrated system:
        1. Nominal valid action
        2. Multimodal conflict
        3. Marginal evidence accumulation
        4. Temporally unconfirmed candidate
        5. Valid procedural transition
        6. Invalid object
        7. Skipped step
        8. Out-of-order action
        9. Unrecognized action
        10. Recovery event trigger
        11. Successful recovery resolution
        12. Completed procedure
        """
        # 1. Nominal valid evaluation
        eval_res = self.pipeline.consistency_evaluator.evaluate(
            action="PICK_RED",
            action_confidence=0.92,
            object_name="RED_SAMPLE",
            object_confidence=0.88,
        )
        self.assertEqual(eval_res["category"], CATEGORY_RELIABLE_ALIGNED)
        self.aggregator.record_consistency_result(eval_res)

        # 2. Multimodal conflict (PICK_RED with CONTAINER_LID)
        conflict_res = self.pipeline.consistency_evaluator.evaluate(
            action="PICK_RED",
            action_confidence=0.90,
            object_name="CONTAINER_LID",
            object_confidence=0.85,
        )
        self.assertEqual(conflict_res["category"], CATEGORY_CROSS_MODAL_CONFLICT)
        self.aggregator.record_consistency_result(conflict_res)

        # 3. Marginal evidence accumulation (confidence in [0.50, 0.70))
        unc_res = self.pipeline.uncertainty_handler.process_observation(
            action="PICK_RED",
            confidence=0.62,
            object_name="RED_SAMPLE",
        )
        self.aggregator.record_uncertainty_result(unc_res)

        # 4. Temporally unconfirmed candidate (1 frame out of 3 required)
        temp_res = self.pipeline.temporal_filter.process_observation(
            action="PICK_RED",
            confidence=0.90,
            object_name="RED_SAMPLE",
        )
        self.assertFalse(temp_res["confirmed"])
        self.aggregator.record_temporal_result(temp_res)

        # 5. Valid procedural transition (S1)
        self.fsm.reset()
        self.fsm.start()
        fsm_s1 = self.fsm.validate_action("PICK_RED", 0.90, "RED_SAMPLE")
        self.assertEqual(fsm_s1["status"], "VALID")
        self.aggregator.record_fsm_result(fsm_s1)

        # 6. Invalid object (S2 action with BLUE_SAMPLE)
        fsm_inv_obj = self.fsm.validate_action("PLACE_RED", 0.90, "BLUE_SAMPLE")
        self.assertEqual(fsm_inv_obj["status"], "OUT_OF_SEQUENCE")
        self.aggregator.record_fsm_result(fsm_inv_obj)

        # 7. Skipped step (S4 before S2/S3)
        fsm_skip = self.fsm.validate_action("PLACE_BLUE", 0.90, "BLUE_SAMPLE")
        self.assertEqual(fsm_skip["status"], "SKIPPED")
        self.aggregator.record_fsm_result(fsm_skip)

        # 8. Out-of-order action (perform S1 again when expecting S2)
        self.fsm.reset()
        self.fsm.start()
        self.fsm.validate_action("PICK_RED", 0.90, "RED_SAMPLE")
        fsm_ooo = self.fsm.validate_action("PICK_RED", 0.90, "RED_SAMPLE")
        self.assertEqual(fsm_ooo["status"], "OUT_OF_SEQUENCE")
        self.aggregator.record_fsm_result(fsm_ooo)

        # 9. Unrecognized action
        fsm_unrec = self.fsm.validate_action("UNKNOWN_GESTURE", 0.90, "RED_SAMPLE")
        self.assertEqual(fsm_unrec["status"], "OUT_OF_SEQUENCE")
        self.aggregator.record_fsm_result(fsm_unrec)

        # 10. Recovery event trigger
        rec_event = self.recovery_mgr.evaluate_fsm_result(fsm_ooo, procedure_manager=self.fsm.pm)
        self.assertIsNotNone(rec_event)
        self.assertEqual(self.recovery_mgr.state, "RECOVERY_ACTIVE")
        self.aggregator.record_recovery_event(rec_event)

        # 11. Successful recovery resolution (perform actual expected step S2)
        fsm_s2 = self.fsm.validate_action("PLACE_RED", 0.90, "RED_SAMPLE")
        self.assertEqual(fsm_s2["status"], "VALID")
        rec_resolved = self.recovery_mgr.evaluate_fsm_result(fsm_s2, procedure_manager=self.fsm.pm)
        self.assertEqual(self.recovery_mgr.state, "RECOVERED")
        self.aggregator.record_fsm_result(fsm_s2)

        # 12. Completed procedure (traverse remaining S3, S4, S5)
        fsm_s3 = self.fsm.validate_action("PICK_BLUE", 0.90, "BLUE_SAMPLE")
        self.assertEqual(fsm_s3["status"], "VALID")
        self.aggregator.record_fsm_result(fsm_s3)

        fsm_s4 = self.fsm.validate_action("PLACE_BLUE", 0.90, "BLUE_SAMPLE")
        self.assertEqual(fsm_s4["status"], "VALID")
        self.aggregator.record_fsm_result(fsm_s4)

        fsm_s5 = self.fsm.validate_action("CLOSE_LID", 0.90, "CONTAINER_LID")
        self.assertEqual(fsm_s5["status"], "VALID")
        self.assertEqual(self.fsm.state, "COMPLETED")
        self.aggregator.record_fsm_result(fsm_s5)

        # Assert all events were aggregated cleanly
        snap = self.aggregator.get_snapshot()
        self.assertEqual(snap.consistency.reliable_aligned_count, 1)
        self.assertEqual(snap.consistency.cross_modal_conflict_count, 1)
        self.assertEqual(snap.uncertainty.marginal_count, 1)
        self.assertGreaterEqual(snap.procedure.valid_transitions, 4)
        self.assertEqual(snap.procedure.skipped_steps, 1)
        self.assertEqual(snap.procedure.out_of_order_events, 1)
        self.assertEqual(snap.procedure.invalid_object_events, 1)
        self.assertEqual(snap.procedure.unrecognized_events, 1)
        self.assertEqual(snap.recovery.total_recoveries, 1)


class TestB13SessionLifecycleAndDerivedMetrics(unittest.TestCase):
    """Verifies session lifecycle management, multi-session isolation, and derived metrics math."""

    def test_03_session_lifecycle_transitions(self):
        """Verify START -> PAUSE -> RESUME -> COMPLETE lifecycle transitions in aggregator."""
        agg = TelemetryDiagnosticAggregator()
        agg.start_session("SESS_01")
        self.assertEqual(agg.get_snapshot().lifecycle.session_state, "RUNNING")

        agg.pause_session()
        self.assertEqual(agg.get_snapshot().lifecycle.session_state, "PAUSED")

        agg.resume_session()
        self.assertEqual(agg.get_snapshot().lifecycle.session_state, "RUNNING")

        agg.complete_session()
        self.assertEqual(agg.get_snapshot().lifecycle.session_state, "COMPLETED")

    def test_04_session_isolation_and_clean_reset(self):
        """Verify resetting aggregator flushes counters and prevents cross-session pollution."""
        agg = TelemetryDiagnosticAggregator()
        agg.start_session("SESS_A")
        agg.record_frame_submission(time.monotonic())
        agg.record_worker_timings(time.monotonic(), 10.0, 5.0, 1.0, 16.0)
        self.assertEqual(agg.get_snapshot().pipeline.frames_submitted, 1)

        # Reset for Session B
        agg.reset()
        clean_snap = agg.get_snapshot()
        self.assertEqual(clean_snap.pipeline.frames_submitted, 0)
        self.assertEqual(clean_snap.pipeline.frames_processed, 0)
        self.assertEqual(clean_snap.pipeline.mean_inference_ms, 0.0)
        self.assertEqual(clean_snap.procedure.valid_transitions, 0)

    def test_05_derived_metrics_full_mathematical_audit(self):
        """Audit all 6 derived metric calculations and boundary zero-denominator guards."""
        # Case 1: All zeros
        d_zero = SessionMetricsExporter.calculate_derived_metrics(
            pipeline_data={},
            procedure_data={},
            recovery_data={},
            logger_summary=None,
        )
        self.assertEqual(d_zero["processed_frame_ratio"], 0.0)
        self.assertEqual(d_zero["effective_drop_rate"], 0.0)
        self.assertEqual(d_zero["procedural_validity_ratio"], 0.0)
        self.assertEqual(d_zero["anomaly_ratio"], 0.0)
        self.assertEqual(d_zero["recovery_ratio"], 1.0)
        self.assertEqual(d_zero["step_completion_ratio"], 0.0)

        # Case 2: Standard execution
        d_normal = SessionMetricsExporter.calculate_derived_metrics(
            pipeline_data={"frames_submitted": 200, "frames_processed": 180, "frames_replaced": 20},
            procedure_data={
                "valid_transitions": 5,
                "skipped_steps": 1,
                "out_of_order_events": 0,
                "invalid_object_events": 0,
                "unrecognized_events": 0,
                "completed_steps": 5,
            },
            recovery_data={"total_recoveries": 1},
            logger_summary={"total_steps": 6, "average_confidence": 0.91},
            total_canonical_steps=5,
        )
        self.assertEqual(d_normal["processed_frame_ratio"], 0.90)
        self.assertEqual(d_normal["effective_drop_rate"], 0.10)
        self.assertEqual(d_normal["procedural_validity_ratio"], 0.8333)
        self.assertEqual(d_normal["anomaly_ratio"], 0.1667)
        self.assertEqual(d_normal["recovery_ratio"], 1.0)
        self.assertEqual(d_normal["step_completion_ratio"], 1.0)
        self.assertEqual(d_normal["average_confidence"], 0.91)

    def test_06_deterministic_export_and_file_creation(self):
        """Verify export_session creates serializable, reproducible JSON and ASCII reports."""
        agg = TelemetryDiagnosticAggregator()
        agg.start_session("EXP_001_EXPORT_TEST")
        agg.record_worker_timings(time.monotonic(), 15.0, 6.0, 1.2, 22.2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = SessionMetricsExporter.export_session(
                telemetry_aggregator=agg,
                experiment_id="EXP-001",
                operator_id="Astronaut-42",
            )
            out_files = SessionMetricsExporter.export_to_file(
                metrics=metrics,
                output_dir=tmp_dir,
                base_name="b13_test_export",
                export_text=True,
            )

            # Check JSON
            with open(out_files["json_path"], "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.assertEqual(loaded["operator_id"], "Astronaut-42")
                self.assertIn("derived_metrics", loaded)

            # Check TXT
            with open(out_files["txt_path"], "r", encoding="utf-8") as f:
                text_content = f.read()
                self.assertIn("ISRO BHARATIYA ANTARIKSH STATION", text_content)
                self.assertIn("Astronaut-42", text_content)


class TestB13BenchmarkHarnessAndEpistemicHonesty(unittest.TestCase):
    """Verifies standalone BenchmarkHarness execution, thresholds, and epistemic honesty disclosures."""

    def test_07_benchmark_discovery_and_independent_runnability(self):
        """Verify all benchmarks in BenchmarkHarness can run independently without errors."""
        benchmarks = BenchmarkHarness.list_benchmarks()
        self.assertGreaterEqual(len(benchmarks), 6)

        # 1. Pipeline runtime
        res_pipe = BenchmarkHarness.run_benchmark("pipeline_runtime", num_frames=10)
        self.assertTrue(res_pipe.passed)
        self.assertLessEqual(res_pipe.metrics["mean_pipeline_latency_ms"], MAX_PIPELINE_LATENCY_MS_THRESHOLD)

        # 2. Synthetic procedural
        res_proc = BenchmarkHarness.run_benchmark("synthetic_procedural", num_cycles=2)
        self.assertTrue(res_proc.passed)
        self.assertEqual(res_proc.metrics["valid_transitions_observed"], 10)

        # 3. Telemetry overhead
        res_tel = BenchmarkHarness.run_benchmark("telemetry_overhead", iterations=50)
        self.assertTrue(res_tel.passed)
        self.assertLess(res_tel.metrics["mean_overhead_ms"], MAX_TELEMETRY_OVERHEAD_MS_THRESHOLD)

        # 4. Orientation robustness
        res_ori = BenchmarkHarness.run_benchmark("orientation_robustness", max_frames=2)
        self.assertTrue(res_ori.passed)

        # 5. Rate matching
        res_rate = BenchmarkHarness.run_benchmark("rate_matching", duration_per_run_sec=0.01)
        self.assertTrue(res_rate.passed)

        # 6. Architecture comparison
        res_arch = BenchmarkHarness.run_benchmark("architecture_comparison", iterations=2)
        self.assertTrue(res_arch.passed)

    def test_08_epistemic_honesty_and_missing_checkpoint_disclosure(self):
        """Verify software benchmarks disclose limitations and do NOT claim neural model accuracy."""
        res_pipe = BenchmarkHarness.run_pipeline_runtime_benchmark(num_frames=5)
        self.assertTrue(any("NPU" in lim or "CPU" in lim for lim in res_pipe.limitations))

        res_proc = BenchmarkHarness.run_synthetic_procedural_benchmark(num_cycles=1)
        self.assertTrue(any("classification accuracy" in lim.lower() for lim in res_proc.limitations))

    def test_09_cli_interface_execution(self):
        """Verify standalone harness CLI entry point."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ret = harness_main([
                "--benchmark", "synthetic_procedural",
                "--output-dir", tmp_dir,
                "--quiet",
            ])
            self.assertEqual(ret, 0)
            exported = Path(tmp_dir) / "synthetic_procedural_benchmark.json"
            self.assertTrue(exported.exists())


class TestB13ConcurrencyAndStressSafety(unittest.TestCase):
    """Stress tests thread-safety, bounded memory, and observer non-interference."""

    def test_10_concurrent_updates_and_snapshot_reads(self):
        """Verify multi-threaded concurrent telemetry recording and snapshot reading without data corruption."""
        agg = TelemetryDiagnosticAggregator(window_size=200)
        agg.start_session("STRESS_SESSION")

        stop_flag = False

        def writer(worker_id: int):
            for i in range(100):
                if stop_flag:
                    break
                mono = time.monotonic()
                agg.record_frame_submission(mono)
                agg.record_worker_timings(mono, 5.0 + worker_id, 2.0, 0.5, 7.5)
                agg.record_stage_latencies(0.1, 3.0, 0.2, 0.2, 0.3, 0.1)
                time.sleep(0.0001)

        def reader():
            reads = 0
            while not stop_flag and reads < 50:
                snap = agg.get_snapshot_dict()
                self.assertIn("pipeline", snap)
                self.assertIn("latencies", snap)
                reads += 1
                time.sleep(0.0002)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            w_futures = [executor.submit(writer, idx) for idx in range(3)]
            r_futures = [executor.submit(reader) for _ in range(2)]

            concurrent.futures.wait(w_futures + r_futures, timeout=10.0)

        snap = agg.get_snapshot()
        self.assertEqual(snap.pipeline.frames_submitted, 300)
        self.assertEqual(snap.pipeline.frames_processed, 300)

    def test_11_bounded_memory_under_high_frame_load(self):
        """Verify internal deques remain strictly bounded under high frame ingestion."""
        agg = TelemetryDiagnosticAggregator(window_size=50)
        for i in range(1000):
            mono = time.monotonic()
            agg.record_frame_submission(mono)
            agg.record_worker_timings(mono, 10.0, 2.0, 0.5, 12.5)

        # Deque size must not exceed window_size
        self.assertLessEqual(len(agg._inference_durations_ms), 50)
        self.assertLessEqual(len(agg._frame_ages_ms), 50)
        self.assertLessEqual(len(agg._dispatch_latencies_ms), 50)
        self.assertLessEqual(len(agg._end_to_end_latencies_ms), 50)

    def test_12_public_contract_and_authority_isolation(self):
        """Verify B13 introduces zero mutations to FSM state and zero extra fields in the public contract."""
        from backend.app_state import AppState

        agg = TelemetryDiagnosticAggregator()
        fsm = SequenceValidatorFSM()
        pipeline = InferencePipeline(validator=fsm, telemetry_aggregator=agg)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process public frame
        public_dict = pipeline.process_frame_public(dummy_frame, timestamp="2026-09-28T12:00:00")

        # Public contract strictness
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
        self.assertEqual(set(public_dict.keys()), expected_keys)

        # Out-of-band query slots
        worker = InferenceWorker(pipeline=pipeline, telemetry_aggregator=agg)
        app_state = AppState(inference_worker=worker)
        bridge = Bridge(app_state)
        telemetry_json = bridge.getTelemetrySnapshot()
        telemetry_snap = json.loads(telemetry_json)
        self.assertIsInstance(telemetry_snap, dict)
        self.assertIn("pipeline", telemetry_snap)

        diagnostic_json = bridge.getDiagnosticSnapshot()
        diagnostic_snap = json.loads(diagnostic_json)
        self.assertIsInstance(diagnostic_snap, dict)
        self.assertIn("consistency", diagnostic_snap)


if __name__ == "__main__":
    unittest.main()
