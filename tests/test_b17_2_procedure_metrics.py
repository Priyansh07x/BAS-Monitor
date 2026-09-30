"""
Workstream B — Gate B17.2 Test Suite
Procedure-Level Protocol Adherence & Sequence Metric Suite

Verifies:
1. Benchmark discovery and execution of 'procedure_protocol_metrics' and aliases ('procedure_metrics', 'b17_2', 'protocol_adherence').
2. Complete-Sequence Success Rate (CSSR = 1.0) under nominal EXP-001 execution.
3. Skipped-Step Detection Rate (SSDR = 1.0) across single-step and multi-step forward jumps.
4. Out-of-Order Detection Rate (OODR = 1.0) across backward transitions and repeated past steps.
5. False Alarm Rate (FAR = 0.0) across nominal executions and perception noise streams.
6. Missed-Violation Rate (MVR = 0.0) across all injected procedural anomalies.
7. Ground-truth scenario mapping and detailed anomaly category breakdown.
8. Multi-cycle stress execution ensuring deterministic FSM behavior and zero state leakage.
9. Strict preservation of the frozen 8-field public AI contract (docs/architecture.md Section 2).
10. Serialization and CLI invocation of B17.2 procedure metrics.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import AIResultAdapter, PUBLIC_CONTRACT_KEYS
from backend.ai.telemetry_aggregator import TelemetryDiagnosticAggregator
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM
from evaluation.benchmark_harness import BenchmarkHarness, BenchmarkResult, main as benchmark_main


def test_procedure_metrics_benchmark_execution():
    """Verify B17.2 procedure metrics benchmark executes cleanly and returns passed BenchmarkResult."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=3, num_anomalous_cycles=3)
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_name == "procedure_protocol_metrics"
    assert result.category == "PROCEDURE_PROTOCOL_METRICS"
    assert result.passed is True
    assert result.duration_seconds > 0.0
    assert result.iterations > 0


def test_procedure_metrics_benchmark_aliases():
    """Verify aliases 'procedure_metrics', 'protocol_adherence', and 'b17_2' route to B17.2."""
    for alias in ["procedure_metrics", "protocol_adherence", "b17_2", "sequence_metrics"]:
        res = BenchmarkHarness.run_benchmark(alias, num_nominal_cycles=2, num_anomalous_cycles=2)
        assert res.benchmark_name == "procedure_protocol_metrics"
        assert res.passed is True


def test_complete_sequence_success_rate_nominal():
    """Verify CSSR reaches 100% (1.0) on nominal EXP-001 step traversals."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=5, num_anomalous_cycles=2)
    cssr = result.metrics["complete_sequence_success_rate"]
    assert cssr == 1.0, f"Expected CSSR 1.0, got {cssr}"

    nom_eval = result.metrics["nominal_evaluation"]
    assert nom_eval["nominal_sequences_total"] == 5
    assert nom_eval["nominal_sequences_completed"] == 5
    assert nom_eval["nominal_steps_total"] == 25
    assert nom_eval["nominal_steps_passed"] == 25
    assert nom_eval["nominal_false_alarms"] == 0


def test_skipped_step_detection_rate():
    """Verify SSDR reaches 100% (1.0) across single-step and multi-step forward skip injections."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=2, num_anomalous_cycles=5)
    ssdr = result.metrics["skipped_step_detection_rate"]
    assert ssdr == 1.0, f"Expected SSDR 1.0, got {ssdr}"

    v_breakdown = result.metrics["violation_detection_breakdown"]
    assert v_breakdown["injected_skips_total"] == 10  # 2 per anomalous cycle * 5 cycles
    assert v_breakdown["detected_skips_true_positive"] == 10


def test_out_of_order_detection_rate():
    """Verify OODR reaches 100% (1.0) on repeated past steps and out-of-order actions."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=2, num_anomalous_cycles=5)
    oodr = result.metrics["out_of_order_detection_rate"]
    assert oodr == 1.0, f"Expected OODR 1.0, got {oodr}"

    v_breakdown = result.metrics["violation_detection_breakdown"]
    assert v_breakdown["injected_out_of_order_total"] == 5
    assert v_breakdown["detected_out_of_order_true_positive"] == 5


def test_false_alarm_rate_zero():
    """Verify FAR remains 0.0 under nominal execution and transient perception noise."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=4, num_anomalous_cycles=4)
    far = result.metrics["false_alarm_rate"]
    assert far == 0.0, f"Expected FAR 0.0, got {far}"


def test_missed_violation_rate_zero():
    """Verify MVR remains 0.0 across all injected skips, out-of-order, invalid objects, and unrecognized actions."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=3, num_anomalous_cycles=4)
    mvr = result.metrics["missed_violation_rate"]
    assert mvr == 0.0, f"Expected MVR 0.0, got {mvr}"

    v_breakdown = result.metrics["violation_detection_breakdown"]
    assert v_breakdown["total_missed_violations"] == 0
    assert v_breakdown["total_injected_violations"] == v_breakdown["total_detected_violations"]


def test_ground_truth_scenario_matrix_breakdown():
    """Verify ground truth scenarios and anomaly categories are explicitly reported."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=3, num_anomalous_cycles=3)
    scenarios = result.metrics["scenarios"]
    assert len(scenarios) >= 4

    categories = [s.get("category") for s in scenarios]
    assert "NOMINAL_EXECUTION" in categories
    assert "VIOLATION_INJECTION" in categories
    assert "NOISE_REJECTION" in categories


def test_stress_multi_cycle_traversal():
    """Verify multi-cycle stress execution (10 cycles) executes deterministically without state leakage."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=10, num_anomalous_cycles=10)
    assert result.passed is True
    assert result.metrics["complete_sequence_success_rate"] == 1.0
    assert result.metrics["skipped_step_detection_rate"] == 1.0
    assert result.metrics["out_of_order_detection_rate"] == 1.0
    assert result.metrics["false_alarm_rate"] == 0.0
    assert result.metrics["missed_violation_rate"] == 0.0
    assert result.metrics["procedural_accuracy"] == 1.0


def test_frozen_public_ai_contract_preservation():
    """Verify public AI contract remains exactly 8 fields during procedure evaluation."""
    pipeline = InferencePipeline()
    fsm = SequenceValidatorFSM()
    fsm.start()
    pipeline.validator = fsm

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.process_frame_public(frame=dummy_frame, timestamp="2026-09-30T12:00:00")

    assert set(res.keys()) == set(PUBLIC_CONTRACT_KEYS)
    assert len(res) == 8


def test_procedure_metrics_serialization_and_cli():
    """Verify JSON/text export and CLI runner operation for B17.2."""
    result = BenchmarkHarness.run_benchmark("procedure_protocol_metrics", num_nominal_cycles=2, num_anomalous_cycles=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = BenchmarkHarness.export_results(result, tmpdir, base_name="test_b17_2_export")
        assert os.path.exists(paths["json_path"])
        assert os.path.exists(paths["txt_path"])

        # CLI invocation
        cli_code = benchmark_main(["--benchmark", "procedure_protocol_metrics", "--output-dir", tmpdir, "--quiet"])
        assert cli_code == 0
