"""
tests/test_orientation_robustness.py

Gate B4.0 — Orientation & Robustness Strategy Specification
Validation tests for docs/workstream-b-orientation-robustness.md and config/orientation_robustness_spec.json.

IMPORTANT:
- These tests do NOT require any model training or video processing to exist.
- Validates schema correctness, 7 canonical angles, 3 orientation categories,
  7 robustness dimensions, evaluation subsets, metrics, and leakage policy invariants.
"""

import json
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(REPO_ROOT, "config", "orientation_robustness_spec.json")
DOC_PATH = os.path.join(REPO_ROOT, "docs", "workstream-b-orientation-robustness.md")
DATASET_SPEC_PATH = os.path.join(REPO_ROOT, "config", "dataset_spec.json")
EXPERIMENT_JSON_PATH = os.path.join(REPO_ROOT, "config", "experiment.json")

EXPECTED_ANGLES = {0, 45, 90, 135, 180, 225, 270}
EXPECTED_CATEGORIES = {"NORMAL", "MODERATE_ROTATION", "EXTREME_ROTATION"}
EXPECTED_DIMENSIONS = {"rotation", "scale", "translation", "brightness", "blur", "occlusion", "perspective"}
EXPECTED_SUBSETS = {
    "EVAL_NORMAL", "EVAL_ORIENTATION", "EVAL_OCCLUSION",
    "EVAL_LIGHTING", "EVAL_VIEWPOINT", "EVAL_FAILURE", "EVAL_INCOMPLETE"
}
EXPECTED_ACTIONS = {"PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"}


@pytest.fixture(scope="module")
def robustness_spec():
    """Load the orientation robustness spec JSON."""
    assert os.path.exists(SPEC_PATH), f"File not found: {SPEC_PATH}"
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def robustness_doc():
    """Load the orientation robustness specification document."""
    assert os.path.exists(DOC_PATH), f"File not found: {DOC_PATH}"
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestOrientationSpecFileIntegrity:
    """Validate JSON file existence and basic top-level schema."""

    def test_spec_file_exists(self):
        assert os.path.exists(SPEC_PATH), "config/orientation_robustness_spec.json must exist"

    def test_doc_file_exists(self):
        assert os.path.exists(DOC_PATH), "docs/workstream-b-orientation-robustness.md must exist"

    def test_spec_is_valid_json(self, robustness_spec):
        assert isinstance(robustness_spec, dict), "Spec must be a JSON dictionary"

    def test_experiment_id_matches_exp001(self, robustness_spec):
        assert robustness_spec.get("experiment_id") == "EXP-001", "experiment_id must be 'EXP-001'"
        assert robustness_spec.get("experiment_version") == "1.0.0", "experiment_version must be '1.0.0'"


class TestOrientationAnglesAndCategories:
    """Validate the 7 required evaluation angles and 3 categories."""

    def test_all_seven_required_angles_present(self, robustness_spec):
        angles = set(robustness_spec.get("required_orientation_degrees", []))
        assert angles == EXPECTED_ANGLES, (
            f"Expected angles {EXPECTED_ANGLES}, got {angles}"
        )

    def test_three_orientation_categories_defined(self, robustness_spec):
        cats = {c["category"] for c in robustness_spec.get("orientation_categories", [])}
        assert cats == EXPECTED_CATEGORIES, (
            f"Expected categories {EXPECTED_CATEGORIES}, got {cats}"
        )

    def test_category_degrees_cover_all_seven_angles(self, robustness_spec):
        all_degrees = set()
        for c in robustness_spec.get("orientation_categories", []):
            all_degrees.update(c.get("degrees", []))
        assert all_degrees == EXPECTED_ANGLES, (
            f"Category degrees union must exactly equal {EXPECTED_ANGLES}, got {all_degrees}"
        )

    def test_alignment_with_dataset_spec(self, robustness_spec):
        assert os.path.exists(DATASET_SPEC_PATH)
        with open(DATASET_SPEC_PATH, "r", encoding="utf-8") as f:
            ds_spec = json.load(f)

        ds_degrees = set()
        for cat in ds_spec.get("orientation_categories", []):
            ds_degrees.update(cat.get("degrees", []))

        spec_degrees = set(robustness_spec.get("required_orientation_degrees", []))
        assert spec_degrees == ds_degrees, (
            f"Orientation angles in orientation_robustness_spec ({spec_degrees}) must match dataset_spec ({ds_degrees})"
        )


class TestRobustnessDimensions:
    """Validate coverage of the 7 core environmental disturbance dimensions."""

    def test_all_seven_robustness_dimensions_present(self, robustness_spec):
        dims = {d["dimension"] for d in robustness_spec.get("augmentation_dimensions", [])}
        assert dims == EXPECTED_DIMENSIONS, (
            f"Expected dimensions {EXPECTED_DIMENSIONS}, got {dims}"
        )

    def test_each_dimension_has_purpose_and_status(self, robustness_spec):
        for dim in robustness_spec.get("augmentation_dimensions", []):
            assert "purpose" in dim and len(dim["purpose"]) > 5, f"Missing purpose in dimension {dim.get('dimension')}"
            assert "training_augmentation" in dim, f"Missing training_augmentation in dimension {dim.get('dimension')}"
            assert "evaluation_presence" in dim, f"Missing evaluation_presence in dimension {dim.get('dimension')}"


class TestEvaluationSubsetsAndMetrics:
    """Validate evaluation subsets and metric definitions."""

    def test_all_evaluation_subsets_represented(self, robustness_spec):
        subsets = {s["subset_id"] for s in robustness_spec.get("evaluation_subsets", [])}
        assert subsets == EXPECTED_SUBSETS, (
            f"Expected subsets {EXPECTED_SUBSETS}, got {subsets}"
        )

    def test_required_metrics_defined(self, robustness_spec):
        metric_names = {m["metric_name"] for m in robustness_spec.get("required_metrics", [])}
        required = {
            "per_angle_precision", "per_angle_recall", "per_angle_f1",
            "macro_average_f1", "relative_angular_degradation", "maximum_orientation_drop"
        }
        assert required.issubset(metric_names), (
            f"Missing required metrics from spec: {required - metric_names}"
        )


class TestLeakagePolicyAndNoFabrication:
    """Ensure subject-level isolation and no fabricated numerical benchmark scores."""

    def test_leakage_policy_is_subject_level(self, robustness_spec):
        policy = robustness_spec.get("leakage_policy", {})
        assert policy.get("strategy") == "subject_level_split_isolation", (
            f"Leakage policy strategy must be 'subject_level_split_isolation', got: {policy.get('strategy')}"
        )
        assert len(policy.get("rules", [])) >= 3, "Leakage policy must contain explicit rules"

    def test_evaluation_status_is_not_evaluated(self, robustness_spec):
        status = robustness_spec.get("evaluation_status")
        assert status == "NOT_EVALUATED", (
            f"evaluation_status must be 'NOT_EVALUATED' (no data collected yet), got: {status}"
        )

    def test_canonical_actions_are_exp001_not_binary_classes(self, robustness_spec):
        actions = set(robustness_spec.get("canonical_procedure_actions", []))
        assert actions == EXPECTED_ACTIONS, (
            f"Canonical actions must be {EXPECTED_ACTIONS}, got: {actions}"
        )
        assert "catch" not in actions, "'catch' must not appear as a canonical action"
        assert "not-catch" not in actions, "'not-catch' must not appear as a canonical action"


class TestRobustnessDocumentContent:
    """Ensure markdown document has all required sections and key concepts."""

    REQUIRED_SECTIONS = [
        "## 1. Scope",
        "## 2. Current Repository Reality",
        "## 3. Orientation Categories",
        "## 4. Required Evaluation Angles",
        "## 5. Training-Time Augmentation Strategy",
        "## 6. Real-Data Evaluation Strategy",
        "## 7. Other Robustness Dimensions",
        "## 8. Evaluation Subsets",
        "## 9. Leakage Prevention",
        "## 10. Metrics",
        "## 11. Microgravity vs Camera Orientation",
        "## 12. Current Implementation Gaps",
        "## 13. B4.1 Implementation Requirements",
        "## 14. B4.2 Evaluation Requirements",
        "## 15. Open Parameters",
        "## 16. Acceptance Criteria",
    ]

    def test_document_has_all_required_sections(self, robustness_doc):
        for sec in self.REQUIRED_SECTIONS:
            assert sec in robustness_doc, f"Missing required section heading: '{sec}'"

    def test_document_explicitly_states_no_evaluation_performed(self, robustness_doc):
        doc_lower = robustness_doc.lower()
        markers = [
            "no robustness evaluation has been performed",
            "not yet collected",
            "no numerical benchmark",
            "not yet available",
            "pending",
        ]
        assert any(m in doc_lower for m in markers), (
            "Document must state that robustness evaluation is pending dataset/model availability"
        )
