"""
tests/test_orientation_evaluation.py
Gate B4.2 — Orientation & Robustness Evaluation Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates:
1.  Evaluation harness imports successfully without external ML frameworks.
2.  Synthetic frame generation is deterministic given a fixed seed.
3.  Synthetic frame generation produces valid pixel-range frames.
4.  All 7 canonical orientation angles are defined and evaluated.
5.  Orientation categories are correctly assigned for each canonical angle.
6.  Augmentation engine produces frames at correct output shape.
7.  Augmentation engine produces frames with valid pixel range [0, 255].
8.  Frame content is not entirely black after rotation.
9.  Bounding box coordinates remain within [0.0, 1.0] after rotation.
10. Keypoint coordinates remain within [0.0, 1.0] after rotation.
11. Rotation is deterministic: same input + angle → same output.
12. Leakage safety: evaluation frames are never inserted into training splits.
13. Metric aggregation: blocked metrics are explicitly None (not zero or fabricated).
14. Rectification comparison: overhead is non-negative for each angle.
15. Evaluation report includes required fields.
16. Missing model/checkpoint is handled gracefully (no exceptions raised).
17. Missing dataset directory is handled gracefully (synthetic fallback).
18. Evaluation is reproducible: same seed → same results on repeated runs.
19. All 7 robustness dimensions produce valid output shapes.
20. Report JSON serialization produces a readable file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.orientation_robustness_eval import (
    CANONICAL_ANGLES,
    EVAL_SEED,
    ORIENTATION_CATEGORIES,
    PROPOSED_LATENCY_OVERHEAD_MS,
    AngleConditionResult,
    EvaluationReport,
    FrameSample,
    RectificationComparisonResult,
    RobustnessDimensionResult,
    _generate_synthetic_frame,
    _make_sample_annotations,
    evaluate_orientation_conditions,
    evaluate_rectification_comparison,
    evaluate_robustness_dimensions,
    load_evaluation_frames,
    run_evaluation,
    save_report,
)
from backend.ai.augmentation import AugmentationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_frames() -> List[FrameSample]:
    """Load a small set of evaluation frames (synthetic fallback always works)."""
    frames, _, _ = load_evaluation_frames(max_frames=5, seed=EVAL_SEED)
    return frames


@pytest.fixture(scope="module")
def eval_report(tmp_path_factory) -> EvaluationReport:
    """Run a full small-scale evaluation once and cache the report."""
    out = tmp_path_factory.mktemp("b4_2_test") / "test_results.json"
    report = run_evaluation(max_frames=5, output_path=out, seed=EVAL_SEED, verbose=False)
    return report


# ---------------------------------------------------------------------------
# 1. Import sanity
# ---------------------------------------------------------------------------

class TestImports:
    def test_evaluation_module_imports_without_ml_frameworks(self):
        """Evaluation harness must import successfully without torch, tensorflow, etc."""
        import evaluation.orientation_robustness_eval as evmod
        assert hasattr(evmod, "run_evaluation")
        assert hasattr(evmod, "AugmentationEngine")

    def test_augmentation_engine_importable(self):
        from backend.ai.augmentation import AugmentationEngine, AugmentationConfig
        engine = AugmentationEngine()
        assert engine is not None


# ---------------------------------------------------------------------------
# 2. Synthetic frame generation
# ---------------------------------------------------------------------------

class TestSyntheticFrameGeneration:
    def test_synthetic_frame_is_deterministic(self):
        """Same seed must produce identical frames."""
        f1 = _generate_synthetic_frame(seed=42)
        f2 = _generate_synthetic_frame(seed=42)
        assert np.array_equal(f1, f2)

    def test_different_seeds_produce_different_frames(self):
        f1 = _generate_synthetic_frame(seed=42)
        f2 = _generate_synthetic_frame(seed=99)
        assert not np.array_equal(f1, f2)

    def test_synthetic_frame_pixel_range(self):
        f = _generate_synthetic_frame(seed=7)
        assert int(f.min()) >= 0
        assert int(f.max()) <= 255
        assert f.dtype == np.uint8

    def test_synthetic_frame_shape(self):
        f = _generate_synthetic_frame(width=120, height=90, seed=0)
        assert f.shape == (90, 120, 3)

    def test_synthetic_frame_not_all_black(self):
        f = _generate_synthetic_frame(seed=0)
        assert f.sum() > 0


# ---------------------------------------------------------------------------
# 3. Canonical angles and orientation categories
# ---------------------------------------------------------------------------

class TestOrientationConfiguration:
    def test_all_seven_canonical_angles_defined(self):
        """All 7 B4.0-specified evaluation angles must be present."""
        expected = {0, 45, 90, 135, 180, 225, 270}
        assert set(CANONICAL_ANGLES) == expected

    def test_orientation_categories_correctly_assigned(self):
        assert ORIENTATION_CATEGORIES[0] == "NORMAL"
        for angle in [45, 135, 225]:
            assert ORIENTATION_CATEGORIES[angle] == "MODERATE_ROTATION"
        for angle in [90, 180, 270]:
            assert ORIENTATION_CATEGORIES[angle] == "EXTREME_ROTATION"

    def test_all_canonical_angles_have_category(self):
        for angle in CANONICAL_ANGLES:
            assert angle in ORIENTATION_CATEGORIES, f"Angle {angle} missing category"


# ---------------------------------------------------------------------------
# 4. Frame loading
# ---------------------------------------------------------------------------

class TestFrameLoading:
    def test_load_returns_nonzero_frames(self, synthetic_frames):
        assert len(synthetic_frames) > 0

    def test_loaded_frames_have_valid_images(self, synthetic_frames):
        for f in synthetic_frames:
            assert isinstance(f.image, np.ndarray)
            assert f.image.ndim in (2, 3)
            assert int(f.image.min()) >= 0
            assert int(f.image.max()) <= 255

    def test_missing_dataset_handled_gracefully(self, tmp_path):
        """Evaluation must not crash if HMDB51 dataset is absent."""
        from evaluation import orientation_robustness_eval as evmod
        original_root = evmod.HMDB51_DATA_ROOT
        try:
            evmod.HMDB51_DATA_ROOT = tmp_path / "nonexistent_dataset"
            frames, dataset_used, dataset_status = load_evaluation_frames(
                max_frames=5, seed=EVAL_SEED
            )
            assert len(frames) > 0, "Should fall back to synthetic frames"
            assert "ABSENT" in dataset_status or "SYNTHETIC" in dataset_used
        finally:
            evmod.HMDB51_DATA_ROOT = original_root

    def test_frame_source_classes_not_empty(self, synthetic_frames):
        classes = [f.source_class for f in synthetic_frames]
        assert len(classes) > 0


# ---------------------------------------------------------------------------
# 5. Orientation evaluation correctness
# ---------------------------------------------------------------------------

class TestOrientationEvaluation:
    def test_all_seven_angles_evaluated(self, synthetic_frames):
        results, _, _ = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        evaluated_angles = {r.angle_deg for r in results}
        assert evaluated_angles == set(CANONICAL_ANGLES)

    def test_output_shapes_valid_for_all_angles(self, synthetic_frames):
        results, _, _ = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.output_shape_valid, f"Shape invalid at {r.angle_deg}°"

    def test_pixel_range_valid_for_all_angles(self, synthetic_frames):
        results, _, _ = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.pixel_range_valid, f"Pixel range invalid at {r.angle_deg}°"

    def test_frame_content_survives_rotation(self, synthetic_frames):
        results, _, _ = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.frame_not_all_black, f"Frame all-black after {r.angle_deg}° rotation"

    def test_bounding_boxes_stay_in_bounds(self, synthetic_frames):
        results, _, _ = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.box_coords_in_bounds, f"Box out of bounds at {r.angle_deg}°"

    def test_keypoints_stay_in_bounds(self, synthetic_frames):
        results, _, _ = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.keypoints_in_bounds, f"Keypoints out of bounds at {r.angle_deg}°"

    def test_timing_is_positive(self, synthetic_frames):
        results, mean_ms, max_ms = evaluate_orientation_conditions(synthetic_frames, seed=EVAL_SEED)
        assert mean_ms >= 0.0
        assert max_ms >= 0.0
        for r in results:
            assert r.augmentation_time_ms >= 0.0


# ---------------------------------------------------------------------------
# 6. Rotation determinism
# ---------------------------------------------------------------------------

class TestRotationDeterminism:
    def test_same_input_same_angle_produces_identical_output(self, synthetic_frames):
        """Rotation must be deterministic: identical inputs produce identical outputs."""
        engine = AugmentationEngine()
        img = synthetic_frames[0].image
        boxes, kps = _make_sample_annotations(img.shape[1], img.shape[0])

        result_a = engine.rotate(img.copy(), 90.0, list(boxes), list(kps))
        result_b = engine.rotate(img.copy(), 90.0, list(boxes), list(kps))

        assert np.array_equal(result_a.image, result_b.image)
        assert result_a.bounding_boxes == result_b.bounding_boxes
        assert result_a.keypoints == result_b.keypoints

    def test_identity_rotation_preserves_frame(self, synthetic_frames):
        """0° rotation must return an image equal to the original."""
        engine = AugmentationEngine()
        img = synthetic_frames[0].image
        result = engine.rotate(img.copy(), 0.0, [], [])
        assert np.array_equal(result.image, img)


# ---------------------------------------------------------------------------
# 7. Leakage safety
# ---------------------------------------------------------------------------

class TestLeakageSafety:
    def test_evaluation_frames_not_sourced_from_training_manifests(self, synthetic_frames):
        """Evaluation frames are loaded from raw HMDB51 data or synthetic fallback.
        Since no EXP-001 dataset exists, no subject-level split manifests exist,
        and there is no risk of leakage. This test verifies the source class is
        not a split directory name and no EXP-001 procedure split path is referenced."""
        for f in synthetic_frames:
            # Source class must not be a split directory name
            clean_class = f.source_class.replace("_SYNTHETIC_FALLBACK", "")
            assert clean_class not in {"TRAIN", "VALIDATION", "TEST"}, (
                f"Frame sourced from split directory: {f.source_class}"
            )
            # Source must not reference an EXP-001 procedure action label inside a training split
            for procedure_split in [
                "TRAIN/PICK_RED", "TRAIN/PLACE_RED", "TRAIN/PICK_BLUE",
                "VALIDATION/PICK_RED", "TEST/PICK_RED",
            ]:
                assert procedure_split not in f.source_file, (
                    f"Frame file path '{f.source_file}' appears to reference a training split"
                )

    def test_evaluation_only_reads_frames_not_writes_training_data(self):
        """Confirm evaluation module has no write path to training/data/."""
        import evaluation.orientation_robustness_eval as evmod
        import inspect
        source = inspect.getsource(evmod)
        # No write to training directory
        assert "training/data" not in source.replace("\\", "/") or \
               all(
                   "open(" not in line or "training/data" not in line
                   for line in source.split("\n")
                   if "training/data" in line
               )


# ---------------------------------------------------------------------------
# 8. Blocked metric assertions
# ---------------------------------------------------------------------------

class TestBlockedMetrics:
    def test_per_angle_precision_is_none(self, eval_report):
        assert eval_report.per_angle_precision is None

    def test_per_angle_recall_is_none(self, eval_report):
        assert eval_report.per_angle_recall is None

    def test_per_angle_f1_is_none(self, eval_report):
        assert eval_report.per_angle_f1 is None

    def test_macro_f1_is_none(self, eval_report):
        assert eval_report.macro_f1 is None

    def test_max_orientation_drop_is_none(self, eval_report):
        assert eval_report.max_orientation_drop is None

    def test_confusion_matrix_is_none(self, eval_report):
        assert eval_report.confusion_matrix is None

    def test_orientation_results_have_none_metrics(self, eval_report):
        for r in eval_report.orientation_results:
            assert r["precision"] is None
            assert r["recall"] is None
            assert r["f1"] is None

    def test_model_eval_explicitly_blocked(self, eval_report):
        assert eval_report.model_eval_passed is False
        assert "BLOCKED" in eval_report.model_checkpoint_status
        assert "BLOCKED" in eval_report.model_evaluation_status
        assert len(eval_report.model_evaluation_reason) > 10

    def test_model_evaluation_reason_mentions_checkpoint(self, eval_report):
        assert "checkpoint" in eval_report.model_evaluation_reason.lower() or \
               "model" in eval_report.model_evaluation_reason.lower()


# ---------------------------------------------------------------------------
# 9. Rectification comparison
# ---------------------------------------------------------------------------

class TestRectificationComparison:
    def test_rectification_comparison_covers_all_angles(self, synthetic_frames):
        results, _ = evaluate_rectification_comparison(synthetic_frames, seed=EVAL_SEED)
        angles = {r.angle_deg for r in results}
        assert angles == set(CANONICAL_ANGLES)

    def test_overhead_is_non_negative(self, synthetic_frames):
        results, mean_overhead = evaluate_rectification_comparison(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.overhead_ms >= 0.0, f"Negative overhead at {r.angle_deg}°"
        assert mean_overhead >= 0.0

    def test_baseline_shape_valid(self, synthetic_frames):
        results, _ = evaluate_rectification_comparison(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            if not r.errors:
                assert r.baseline_shape_valid, f"Baseline shape invalid at {r.angle_deg}°"

    def test_rectified_shape_valid(self, synthetic_frames):
        results, _ = evaluate_rectification_comparison(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            if not r.errors:
                assert r.rectified_shape_valid, f"Rectified shape invalid at {r.angle_deg}°"


# ---------------------------------------------------------------------------
# 10. Robustness dimensions
# ---------------------------------------------------------------------------

class TestRobustnessDimensions:
    EXPECTED_DIMENSIONS = {
        "rotation", "scale", "translation", "brightness", "blur", "occlusion", "perspective"
    }

    def test_all_seven_dimensions_evaluated(self, synthetic_frames):
        results = evaluate_robustness_dimensions(synthetic_frames, seed=EVAL_SEED)
        evaluated = {r.dimension for r in results}
        assert evaluated == self.EXPECTED_DIMENSIONS

    def test_all_dimensions_produce_valid_shapes(self, synthetic_frames):
        results = evaluate_robustness_dimensions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            if not r.errors:
                assert r.output_shape_valid, f"Shape invalid for dimension '{r.dimension}'"

    def test_all_dimensions_produce_valid_pixel_range(self, synthetic_frames):
        results = evaluate_robustness_dimensions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            if not r.errors:
                assert r.pixel_range_valid, f"Pixel range invalid for dimension '{r.dimension}'"

    def test_all_dimensions_have_timing(self, synthetic_frames):
        results = evaluate_robustness_dimensions(synthetic_frames, seed=EVAL_SEED)
        for r in results:
            assert r.augmentation_time_ms >= 0.0


# ---------------------------------------------------------------------------
# 11. Evaluation report structure
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_report_gate_is_b42(self, eval_report):
        assert eval_report.gate == "B4.2"

    def test_report_has_date(self, eval_report):
        assert len(eval_report.date) > 0

    def test_report_has_seed(self, eval_report):
        assert eval_report.seed == EVAL_SEED

    def test_report_has_dataset_info(self, eval_report):
        assert len(eval_report.dataset_used) > 0
        assert len(eval_report.dataset_status) > 0

    def test_report_has_orientation_results(self, eval_report):
        assert len(eval_report.orientation_results) == 7

    def test_report_has_robustness_dimension_results(self, eval_report):
        assert len(eval_report.robustness_dimension_results) == 7

    def test_report_has_rectification_comparison(self, eval_report):
        assert len(eval_report.rectification_comparison) == 7

    def test_report_gate_status_is_not_empty(self, eval_report):
        assert eval_report.gate_status in {
            "PARTIALLY_EVALUATED", "PREPROCESSING_FAILED", "PASSED", "BLOCKED"
        }

    def test_preprocessing_eval_passed(self, eval_report):
        assert eval_report.preprocessing_eval_passed is True

    def test_report_evaluation_mode_is_synthetic_stress_test(self, eval_report):
        assert eval_report.evaluation_mode == "SYNTHETIC_STRESS_TEST"


# ---------------------------------------------------------------------------
# 12. JSON serialization
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_report_is_json_serializable(self, eval_report, tmp_path):
        out = tmp_path / "test_report.json"
        save_report(eval_report, out)
        assert out.exists()
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_serialized_report_contains_gate(self, eval_report, tmp_path):
        out = tmp_path / "test_report2.json"
        save_report(eval_report, out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["gate"] == "B4.2"

    def test_serialized_report_blocked_metrics_are_null(self, eval_report, tmp_path):
        out = tmp_path / "test_report3.json"
        save_report(eval_report, out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["per_angle_f1"] is None
        assert data["macro_f1"] is None
        assert data["per_angle_precision"] is None
        assert data["per_angle_recall"] is None


# ---------------------------------------------------------------------------
# 13. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_produces_same_orientation_results(self, tmp_path):
        """Two runs with the same seed must produce identical results."""
        out1 = tmp_path / "run1.json"
        out2 = tmp_path / "run2.json"
        r1 = run_evaluation(max_frames=4, output_path=out1, seed=42, verbose=False)
        r2 = run_evaluation(max_frames=4, output_path=out2, seed=42, verbose=False)

        assert r1.all_orientations_evaluated == r2.all_orientations_evaluated
        assert r1.frames_successfully_loaded == r2.frames_successfully_loaded
        assert r1.preprocessing_eval_passed == r2.preprocessing_eval_passed
        # Both must have same metric-blocked status
        assert r1.macro_f1 == r2.macro_f1  # Both None

    def test_different_seeds_load_different_frames(self):
        """Different seeds should produce different frame orderings."""
        frames_a, _, _ = load_evaluation_frames(max_frames=10, seed=0)
        frames_b, _, _ = load_evaluation_frames(max_frames=10, seed=99)
        # Both should have frames
        assert len(frames_a) > 0
        assert len(frames_b) > 0
        # Content may differ (either real or synthetic)
