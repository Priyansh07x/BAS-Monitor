"""
test_b17_3_classification_and_consolidation.py
Workstream B — Gate B17.3: Action-Level Classification Metrics & Gate B17 Consolidation

Tests:
1. compute_action_classification_metrics mathematical correctness on perfect nominal vector.
2. compute_action_classification_metrics on hand-calculated mixed substitution vector.
3. compute_action_classification_metrics zero support and zero predictions safety.
4. compute_action_classification_metrics empty input safety.
5. compute_action_classification_metrics unmapped labels handling.
6. Confusion matrix shape, class ordering, and total sample conservation.
7. BenchmarkHarness discovery and execution of action classification benchmark.
8. BenchmarkHarness consolidated B17 evaluation aggregation across Dimensions A, B, C, D.
9. Epistemic honesty disclosures and deferred dataset / neural model annotations.
10. Frozen 8-field public AI contract invariance.
11. Export of JSON / text reports and CLI compatibility.
"""

import json
import pytest
import numpy as np
from pathlib import Path
import tempfile

from evaluation.benchmark_harness import (
    EXP001_ACTION_CLASSES,
    BenchmarkHarness,
    BenchmarkResult,
    compute_action_classification_metrics,
)
from backend.ai.result_adapter import AIResultAdapter


class TestB173ActionClassificationMetrics:
    """Mathematical verification of the action-level classification metric engine."""

    def test_compute_action_classification_metrics_perfect_vector(self):
        """Vector 1: Perfect nominal sequence S1->S5 (N=5)."""
        y_true = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"]
        y_pred = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"]

        res = compute_action_classification_metrics(y_true, y_pred, classes=EXP001_ACTION_CLASSES)

        assert res["total_samples"] == 5
        assert res["accuracy"] == 1.0
        assert res["macro_precision"] == 1.0
        assert res["macro_recall"] == 1.0
        assert res["macro_f1"] == 1.0
        assert res["unmapped_samples"] == 0

        # Confusion matrix must be identity 5x5
        cm = res["confusion_matrix"]
        assert len(cm) == 5
        for i in range(5):
            assert len(cm[i]) == 5
            for j in range(5):
                expected = 1 if i == j else 0
                assert cm[i][j] == expected

        # Per-class checks
        for c in EXP001_ACTION_CLASSES:
            pc = res["per_class"][c]
            assert pc["tp"] == 1
            assert pc["fp"] == 0
            assert pc["fn"] == 0
            assert pc["tn"] == 4
            assert pc["support"] == 1
            assert pc["precision"] == 1.0
            assert pc["recall"] == 1.0
            assert pc["f1_score"] == 1.0

    def test_compute_action_classification_metrics_hand_calculated_mixed_vector(self):
        """
        Vector 2: Hand-calculated mixed vector (N=6):
        y_true = [PICK_RED, PICK_RED, PLACE_RED, PICK_BLUE, PLACE_BLUE, CLOSE_LID]
        y_pred = [PICK_RED, PICK_BLUE, PLACE_RED, PICK_BLUE, PLACE_BLUE, CLOSE_LID]
        
        Hand calculations:
        - PICK_RED:   TP=1, FP=0, FN=1, TN=4, Support=2 -> P=1.0/1.0=1.0, R=1.0/2.0=0.5, F1=2*(1.0*0.5)/1.5 = 2/3 ≈ 0.6667
        - PLACE_RED:  TP=1, FP=0, FN=0, TN=5, Support=1 -> P=1.0, R=1.0, F1=1.0
        - PICK_BLUE:  TP=1, FP=1, FN=0, TN=4, Support=1 -> P=1.0/2.0=0.5, R=1.0/1.0=1.0, F1=2*(0.5*1.0)/1.5 = 2/3 ≈ 0.6667
        - PLACE_BLUE: TP=1, FP=0, FN=0, TN=5, Support=1 -> P=1.0, R=1.0, F1=1.0
        - CLOSE_LID:  TP=1, FP=0, FN=0, TN=5, Support=1 -> P=1.0, R=1.0, F1=1.0
        
        Averages (5 classes):
        - Accuracy = 5/6 ≈ 0.8333
        - Macro P = (1.0 + 1.0 + 0.5 + 1.0 + 1.0) / 5 = 4.5 / 5 = 0.9000
        - Macro R = (0.5 + 1.0 + 1.0 + 1.0 + 1.0) / 5 = 4.5 / 5 = 0.9000
        - Macro F1 = (0.6667 + 1.0 + 0.6667 + 1.0 + 1.0) / 5 = 4.3333 / 5 = 0.8667
        """
        y_true = ["PICK_RED", "PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"]
        y_pred = ["PICK_RED", "PICK_BLUE", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"]

        res = compute_action_classification_metrics(y_true, y_pred, classes=EXP001_ACTION_CLASSES)

        assert res["total_samples"] == 6
        assert abs(res["accuracy"] - 0.8333) < 0.001
        assert abs(res["macro_precision"] - 0.9000) < 0.001
        assert abs(res["macro_recall"] - 0.9000) < 0.001
        assert abs(res["macro_f1"] - 0.8667) < 0.001

        # Check per-class values
        pr = res["per_class"]["PICK_RED"]
        assert pr["tp"] == 1 and pr["fp"] == 0 and pr["fn"] == 1 and pr["tn"] == 4 and pr["support"] == 2
        assert abs(pr["precision"] - 1.0) < 0.001
        assert abs(pr["recall"] - 0.5) < 0.001
        assert abs(pr["f1_score"] - 0.6667) < 0.001

        pb = res["per_class"]["PICK_BLUE"]
        assert pb["tp"] == 1 and pb["fp"] == 1 and pb["fn"] == 0 and pb["tn"] == 4 and pb["support"] == 1
        assert abs(pb["precision"] - 0.5) < 0.001
        assert abs(pb["recall"] - 1.0) < 0.001
        assert abs(pb["f1_score"] - 0.6667) < 0.001

        # Check confusion matrix exact counts
        # Classes: 0: PICK_RED, 1: PLACE_RED, 2: PICK_BLUE, 3: PLACE_BLUE, 4: CLOSE_LID
        cm = res["confusion_matrix"]
        assert cm[0] == [1, 0, 1, 0, 0]  # PICK_RED: 1 correctly predicted as PICK_RED, 1 as PICK_BLUE
        assert cm[1] == [0, 1, 0, 0, 0]  # PLACE_RED
        assert cm[2] == [0, 0, 1, 0, 0]  # PICK_BLUE
        assert cm[3] == [0, 0, 0, 1, 0]  # PLACE_BLUE
        assert cm[4] == [0, 0, 0, 0, 1]  # CLOSE_LID

    def test_compute_action_classification_metrics_zero_support_and_predictions(self):
        """Zero support for a class and zero predictions for a class must return 0.0 without errors."""
        # CLOSE_LID never appears in y_true (support=0)
        # PLACE_BLUE never appears in y_pred (predictions=0)
        y_true = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PICK_BLUE"]
        y_pred = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PICK_RED"]

        res = compute_action_classification_metrics(y_true, y_pred, classes=EXP001_ACTION_CLASSES)

        assert res["total_samples"] == 4
        # CLOSE_LID has support=0 -> recall=0.0, precision=0.0, f1=0.0
        cl = res["per_class"]["CLOSE_LID"]
        assert cl["support"] == 0
        assert cl["precision"] == 0.0
        assert cl["recall"] == 0.0
        assert cl["f1_score"] == 0.0

        # PLACE_BLUE has tp=0, fp=0, fn=0, support=0 -> precision=0.0, recall=0.0, f1=0.0
        plb = res["per_class"]["PLACE_BLUE"]
        assert plb["tp"] == 0
        assert plb["fp"] == 0
        assert plb["precision"] == 0.0

        # No crashes, valid float values returned
        assert isinstance(res["macro_precision"], float)
        assert isinstance(res["macro_recall"], float)
        assert isinstance(res["macro_f1"], float)

    def test_compute_action_classification_metrics_empty_input(self):
        """Empty lists y_true=[] and y_pred=[] return zero metrics cleanly."""
        res = compute_action_classification_metrics([], [], classes=EXP001_ACTION_CLASSES)

        assert res["total_samples"] == 0
        assert res["accuracy"] == 0.0
        assert res["macro_precision"] == 0.0
        assert res["macro_recall"] == 0.0
        assert res["macro_f1"] == 0.0
        assert res["unmapped_samples"] == 0
        assert res["confusion_matrix"] == [[0] * 5 for _ in range(5)]

    def test_compute_action_classification_metrics_mismatched_length(self):
        """Mismatched y_true and y_pred lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have identical length"):
            compute_action_classification_metrics(["PICK_RED"], ["PICK_RED", "PLACE_RED"])

    def test_compute_action_classification_metrics_unmapped_labels(self):
        """Unknown or unmapped action labels are tracked in unmapped_samples."""
        y_true = ["PICK_RED", "UNKNOWN_ACTION", "PLACE_RED"]
        y_pred = ["PICK_RED", "PLACE_RED", "CUSTOM_LABEL"]

        res = compute_action_classification_metrics(y_true, y_pred, classes=EXP001_ACTION_CLASSES)
        assert res["total_samples"] == 3
        assert res["unmapped_samples"] == 2
        # Only index 0 was valid
        assert res["per_class"]["PICK_RED"]["tp"] == 1

    def test_confusion_matrix_dimensions_ordering_and_conservation(self):
        """Confusion matrix is exactly 5x5, follows canonical order, and conserves mapped sample counts."""
        y_true = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID", "PICK_RED"]
        y_pred = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID", "PLACE_RED"]

        res = compute_action_classification_metrics(y_true, y_pred, classes=EXP001_ACTION_CLASSES)
        cm = res["confusion_matrix"]

        assert len(cm) == 5
        for row in cm:
            assert len(row) == 5

        # Sum of all elements in cm == mapped samples
        total_cm_count = sum(sum(row) for row in cm)
        assert total_cm_count == res["total_samples"] - res["unmapped_samples"]
        assert total_cm_count == 6


class TestB173BenchmarkHarnessIntegration:
    """BenchmarkHarness execution and registration tests for Gate B17.3."""

    def test_action_classification_benchmark_discovery_and_execution(self):
        """Action classification benchmark is discoverable and executable via primary and alias names."""
        benchmarks = BenchmarkHarness.list_benchmarks()
        assert "action_classification_metrics" in benchmarks
        assert "consolidated_b17_evaluation" in benchmarks

        for alias in ["action_classification_metrics", "action_metrics", "classification_metrics", "b17_3"]:
            res = BenchmarkHarness.run_benchmark(alias)
            assert isinstance(res, BenchmarkResult)
            assert res.benchmark_name == "action_classification_metrics"
            assert res.passed is True
            assert res.metrics["vectors_evaluated"] >= 3
            assert "combined_confusion_matrix" in res.metrics

    def test_consolidated_b17_benchmark_execution_and_aggregation(self):
        """Consolidated Gate B17 benchmark runs and aggregates Dimensions A, B, C, D."""
        for alias in ["consolidated_b17_evaluation", "consolidated_b17", "b17", "b17_all"]:
            res = BenchmarkHarness.run_benchmark(
                alias,
                runtime_frames=20,
                procedure_nominal_cycles=2,
                procedure_anomalous_cycles=2,
            )
            assert isinstance(res, BenchmarkResult)
            assert res.benchmark_name == "consolidated_b17_evaluation"
            assert res.passed is True
            assert res.metrics["overall_verdict"] == "PASSED"

            # Dimension A: Runtime Profiling
            dim_a = res.metrics["dimension_a_runtime_profiling"]
            assert dim_a["passed"] is True
            assert dim_a["pipeline_fps"] > 0
            assert dim_a["mean_latency_ms"] > 0
            assert dim_a["rss_ram_mb"] > 0

            # Dimension B: Procedure Protocol
            dim_b = res.metrics["dimension_b_procedure_protocol"]
            assert dim_b["passed"] is True
            assert dim_b["complete_sequence_success_rate"] == 1.0
            assert dim_b["skipped_step_detection_rate"] == 1.0
            assert dim_b["out_of_order_detection_rate"] == 1.0
            assert dim_b["false_alarm_rate"] == 0.0
            assert dim_b["missed_violation_rate"] == 0.0

            # Dimension C: Action Classification
            dim_c = res.metrics["dimension_c_action_classification"]
            assert dim_c["passed"] is True
            assert dim_c["accuracy"] > 0
            assert dim_c["macro_f1"] > 0
            assert len(dim_c["confusion_matrix"]) == 5

            # Dimension D: Epistemic / Status Disclosures
            dim_d = res.metrics["dimension_d_model_dataset_status"]
            assert dim_d["exp001_video_dataset"] == "UNAVAILABLE / DEFERRED"
            assert dim_d["part1_neural_checkpoint"] == "UNAVAILABLE / DEFERRED"

    def test_epistemic_honesty_and_blocked_model_disclosure(self):
        """Limitations and status disclosures transparently state synthetic validation and deferred weights."""
        res = BenchmarkHarness.run_action_classification_benchmark()
        lims_text = " ".join(res.limitations)
        assert "synthetic" in lims_text.lower()
        assert "deferred" in lims_text.lower()
        assert "zero synthetic neural accuracy is fabricated" in lims_text.lower()

        res_cons = BenchmarkHarness.run_consolidated_b17_benchmark(runtime_frames=10, procedure_nominal_cycles=1, procedure_anomalous_cycles=1)
        cons_lims = " ".join(res_cons.limitations)
        assert "deferred" in cons_lims.lower()

    def test_frozen_8_field_public_ai_contract_invariance(self):
        """Public AI result schema retains exactly 8 frozen fields."""
        res_raw = {
            "action": "PICK_RED",
            "confidence": 0.95,
            "object": "RED_SAMPLE",
            "detected_step": "S1",
            "timestamp": "2026-09-30T12:00:00",
        }
        res_public = AIResultAdapter.adapt(internal_result=res_raw)
        assert len(res_public) == 8
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
        assert set(res_public.keys()) == expected_keys

    def test_b17_3_export_and_cli_execution(self):
        """Benchmark results export cleanly to JSON and text summary files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = BenchmarkHarness.run_action_classification_benchmark()
            paths = BenchmarkHarness.export_results(res, output_dir=tmp_dir, base_name="b17_3_test")

            assert paths["json_path"].exists()
            assert paths["txt_path"].exists()

            with open(paths["json_path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["benchmark_name"] == "action_classification_metrics"
            assert data["passed"] is True
            assert "combined_confusion_matrix" in data["metrics"]

            txt = paths["txt_path"].read_text(encoding="utf-8")
            assert "ACTION_CLASSIFICATION_METRICS" in txt
            assert "PASSED" in txt
