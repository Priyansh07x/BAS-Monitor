"""
tests/test_dataset_protocol.py

Gate B3 — Dataset Planning / Collection Protocol
Validation tests for docs/workstream-b-dataset-protocol.md and config/dataset_spec.json.

IMPORTANT:
- These tests do NOT require any actual video dataset to exist.
- No clips, raw recordings, or HMDB51 data is read.
- Tests validate schema correctness, documentation consistency,
  and critical protocol invariants only.
"""

import json
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SPEC_PATH = os.path.join(REPO_ROOT, "config", "dataset_spec.json")
PROTOCOL_DOC_PATH = os.path.join(REPO_ROOT, "docs", "workstream-b-dataset-protocol.md")
EXPERIMENT_JSON_PATH = os.path.join(REPO_ROOT, "config", "experiment.json")


@pytest.fixture(scope="module")
def dataset_spec():
    """Load the canonical dataset specification JSON."""
    assert os.path.exists(DATASET_SPEC_PATH), (
        f"config/dataset_spec.json does not exist at: {DATASET_SPEC_PATH}"
    )
    with open(DATASET_SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def experiment_json():
    """Load the canonical experiment definition JSON."""
    assert os.path.exists(EXPERIMENT_JSON_PATH), (
        f"config/experiment.json does not exist at: {EXPERIMENT_JSON_PATH}"
    )
    with open(EXPERIMENT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def protocol_doc():
    """Load the dataset protocol markdown document."""
    assert os.path.exists(PROTOCOL_DOC_PATH), (
        f"docs/workstream-b-dataset-protocol.md does not exist at: {PROTOCOL_DOC_PATH}"
    )
    with open(PROTOCOL_DOC_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Test 1: dataset_spec.json file exists and is valid JSON
# ---------------------------------------------------------------------------

class TestDatasetSpecFileExists:
    """Ensure dataset_spec.json exists and is parseable."""

    def test_dataset_spec_file_exists(self):
        assert os.path.exists(DATASET_SPEC_PATH), \
            "config/dataset_spec.json must exist"

    def test_dataset_spec_is_valid_json(self, dataset_spec):
        assert isinstance(dataset_spec, dict), \
            "dataset_spec.json must parse to a dict"

    def test_protocol_doc_file_exists(self):
        assert os.path.exists(PROTOCOL_DOC_PATH), \
            "docs/workstream-b-dataset-protocol.md must exist"


# ---------------------------------------------------------------------------
# Test 2: Experiment ID and version matches canonical experiment.json
# ---------------------------------------------------------------------------

class TestDatasetSpecExperimentAlignment:
    """dataset_spec.json must reference the correct experiment ID and version."""

    def test_experiment_id_is_EXP_001(self, dataset_spec):
        assert dataset_spec["experiment_id"] == "EXP-001", (
            f"experiment_id must be 'EXP-001', got: {dataset_spec.get('experiment_id')}"
        )

    def test_experiment_version_is_1_0_0(self, dataset_spec):
        assert dataset_spec["experiment_version"] == "1.0.0", (
            f"experiment_version must be '1.0.0', got: {dataset_spec.get('experiment_version')}"
        )

    def test_experiment_id_matches_experiment_json(self, dataset_spec, experiment_json):
        exp_id = experiment_json.get("experiment_id") or experiment_json.get("id")
        spec_id = dataset_spec["experiment_id"]
        assert exp_id == spec_id, (
            f"experiment_id mismatch: experiment.json={exp_id}, dataset_spec.json={spec_id}"
        )


# ---------------------------------------------------------------------------
# Test 3: All 5 canonical actions are defined in dataset_spec.json
# ---------------------------------------------------------------------------

EXPECTED_ACTIONS = {"PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"}

class TestCanonicalActionsInSpec:
    """All 5 procedure actions must appear in dataset_spec.json."""

    def test_canonical_actions_field_exists(self, dataset_spec):
        assert "canonical_actions" in dataset_spec, \
            "dataset_spec.json must have a 'canonical_actions' field"

    def test_canonical_actions_count_is_5(self, dataset_spec):
        actions = dataset_spec["canonical_actions"]
        assert len(actions) == 5, \
            f"Expected 5 canonical actions, got {len(actions)}"

    def test_all_5_action_ids_present(self, dataset_spec):
        action_ids = {a["action_id"] for a in dataset_spec["canonical_actions"]}
        assert action_ids == EXPECTED_ACTIONS, (
            f"Missing or unexpected actions.\n"
            f"Expected: {EXPECTED_ACTIONS}\n"
            f"Got: {action_ids}"
        )

    def test_actions_have_required_fields(self, dataset_spec):
        required_fields = {"action_id", "step_id", "step_number", "target_object",
                           "description", "onset_evidence", "completion_evidence", "timeout_s"}
        for action in dataset_spec["canonical_actions"]:
            missing = required_fields - set(action.keys())
            assert not missing, \
                f"Action '{action.get('action_id')}' is missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 4: All 4 canonical objects are defined in dataset_spec.json
# ---------------------------------------------------------------------------

EXPECTED_OBJECTS = {"RED_SAMPLE", "BLUE_SAMPLE", "SAMPLE_CONTAINER", "CONTAINER_LID"}

class TestCanonicalObjectsInSpec:
    """All 4 procedure objects must appear in dataset_spec.json."""

    def test_canonical_objects_field_exists(self, dataset_spec):
        assert "canonical_objects" in dataset_spec, \
            "dataset_spec.json must have a 'canonical_objects' field"

    def test_canonical_objects_count_is_4(self, dataset_spec):
        objects = dataset_spec["canonical_objects"]
        assert len(objects) == 4, \
            f"Expected 4 canonical objects, got {len(objects)}"

    def test_all_4_object_ids_present(self, dataset_spec):
        object_ids = {o["object_id"] for o in dataset_spec["canonical_objects"]}
        assert object_ids == EXPECTED_OBJECTS, (
            f"Missing or unexpected objects.\n"
            f"Expected: {EXPECTED_OBJECTS}\n"
            f"Got: {object_ids}"
        )


# ---------------------------------------------------------------------------
# Test 5: Split strategy is subject-level (not random frame-level)
# ---------------------------------------------------------------------------

class TestSplitStrategySubjectLevel:
    """Split strategy must be 'subject_level'."""

    def test_split_definitions_exist(self, dataset_spec):
        assert "split_definitions" in dataset_spec, \
            "dataset_spec.json must have a 'split_definitions' field"

    def test_split_strategy_is_subject_level(self, dataset_spec):
        strategy = dataset_spec["split_definitions"].get("strategy")
        assert strategy == "subject_level", (
            f"split strategy must be 'subject_level' (never 'random_frame' or 'random_clip'), "
            f"got: {strategy}"
        )

    def test_all_three_splits_defined(self, dataset_spec):
        splits = dataset_spec["split_definitions"].get("splits", [])
        split_ids = {s["split_id"] for s in splits}
        expected = {"TRAIN", "VALIDATION", "TEST"}
        assert split_ids == expected, \
            f"Expected splits {expected}, got {split_ids}"

    def test_split_fractions_sum_to_1(self, dataset_spec):
        splits = dataset_spec["split_definitions"].get("splits", [])
        total = sum(s["subject_fraction"] for s in splits)
        assert abs(total - 1.0) < 1e-9, \
            f"Split fractions must sum to 1.0, got {total}"


# ---------------------------------------------------------------------------
# Test 6: All 7 orientation degrees are defined
# ---------------------------------------------------------------------------

EXPECTED_ORIENTATION_DEGREES = {0, 45, 90, 135, 180, 225, 270}

class TestOrientationMetadata:
    """All 7 orientation degrees must be covered in dataset_spec.json."""

    def test_orientation_categories_exist(self, dataset_spec):
        assert "orientation_categories" in dataset_spec, \
            "dataset_spec.json must have an 'orientation_categories' field"

    def test_all_7_orientation_degrees_covered(self, dataset_spec):
        all_degrees = set()
        for cat in dataset_spec["orientation_categories"]:
            all_degrees.update(cat.get("degrees", []))
        assert all_degrees == EXPECTED_ORIENTATION_DEGREES, (
            f"Expected orientation degrees {EXPECTED_ORIENTATION_DEGREES}, "
            f"got {all_degrees}"
        )

    def test_three_orientation_categories_defined(self, dataset_spec):
        cats = {c["orientation_category"] for c in dataset_spec["orientation_categories"]}
        expected = {"NORMAL", "MODERATE_ROTATION", "EXTREME_ROTATION"}
        assert cats == expected, \
            f"Expected orientation categories {expected}, got {cats}"


# ---------------------------------------------------------------------------
# Test 7: All required metadata fields are documented
# ---------------------------------------------------------------------------

REQUIRED_METADATA_FIELDS = {
    "sample_id", "subject_id", "session_id", "take_id", "source_video",
    "clip_start", "clip_end", "procedure_id", "procedure_version",
    "step_id", "action", "object", "hand", "speed_category", "viewpoint",
    "orientation_deg", "orientation_category", "lighting", "occlusion",
    "completion_state", "procedure_correctness", "failure_type", "split",
}

class TestMetadataFields:
    """All required metadata fields must be in dataset_spec.json."""

    def test_metadata_fields_section_exists(self, dataset_spec):
        assert "metadata_fields" in dataset_spec, \
            "dataset_spec.json must have a 'metadata_fields' section"

    def test_all_required_fields_documented(self, dataset_spec):
        documented = {f["field"] for f in dataset_spec["metadata_fields"]}
        missing = REQUIRED_METADATA_FIELDS - documented
        assert not missing, \
            f"The following required metadata fields are missing from dataset_spec.json: {missing}"

    def test_action_field_has_allowed_values(self, dataset_spec):
        action_field = next(
            (f for f in dataset_spec["metadata_fields"] if f["field"] == "action"), None
        )
        assert action_field is not None, "'action' metadata field must exist"
        allowed = set(action_field.get("allowed_values", []))
        required_actions = {"PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"}
        assert required_actions.issubset(allowed), (
            f"'action' field allowed_values must include {required_actions}. "
            f"Currently: {allowed}"
        )

    def test_split_field_has_allowed_values(self, dataset_spec):
        split_field = next(
            (f for f in dataset_spec["metadata_fields"] if f["field"] == "split"), None
        )
        assert split_field is not None, "'split' metadata field must exist"
        allowed = set(split_field.get("allowed_values", []))
        assert {"TRAIN", "VALIDATION", "TEST"}.issubset(allowed), \
            f"'split' allowed_values must include TRAIN/VALIDATION/TEST. Got: {allowed}"


# ---------------------------------------------------------------------------
# Test 8: Action-object relationships are exclusive and correct
# ---------------------------------------------------------------------------

EXPECTED_ACTION_OBJECT_MAP = {
    "PICK_RED": "RED_SAMPLE",
    "PLACE_RED": "RED_SAMPLE",
    "PICK_BLUE": "BLUE_SAMPLE",
    "PLACE_BLUE": "BLUE_SAMPLE",
    "CLOSE_LID": "CONTAINER_LID",
}

class TestActionObjectRelationships:
    """Each action must map to exactly one canonical object."""

    def test_each_action_has_one_target_object(self, dataset_spec):
        for action in dataset_spec["canonical_actions"]:
            action_id = action["action_id"]
            target = action["target_object"]
            expected = EXPECTED_ACTION_OBJECT_MAP.get(action_id)
            assert target == expected, (
                f"Action '{action_id}' must map to '{expected}', "
                f"but maps to '{target}' in dataset_spec.json"
            )

    def test_catch_not_catch_are_not_procedure_actions(self, dataset_spec):
        action_ids = {a["action_id"] for a in dataset_spec["canonical_actions"]}
        assert "catch" not in action_ids, \
            "'catch' (Part-1 ML class) must NOT appear as a canonical procedure action"
        assert "not-catch" not in action_ids, \
            "'not-catch' (Part-1 ML class) must NOT appear as a canonical procedure action"


# ---------------------------------------------------------------------------
# Test 9: Failure categories include both procedure-level and AI-level failures
# ---------------------------------------------------------------------------

class TestFailureCategoryTaxonomy:
    """Both procedure-level and AI-recognition-level failure categories must exist."""

    def test_failure_categories_exist(self, dataset_spec):
        assert "failure_categories" in dataset_spec, \
            "dataset_spec.json must have 'failure_categories'"

    def test_procedure_level_failures_present(self, dataset_spec):
        proc_failures = [
            f for f in dataset_spec["failure_categories"] if f["level"] == "PROCEDURE"
        ]
        assert len(proc_failures) >= 5, (
            f"Expected at least 5 PROCEDURE-level failure categories, "
            f"found {len(proc_failures)}"
        )

    def test_ai_recognition_level_failures_present(self, dataset_spec):
        ai_failures = [
            f for f in dataset_spec["failure_categories"] if f["level"] == "AI_RECOGNITION"
        ]
        assert len(ai_failures) >= 5, (
            f"Expected at least 5 AI_RECOGNITION-level failure categories, "
            f"found {len(ai_failures)}"
        )

    def test_mandatory_failure_tags_present(self, dataset_spec):
        tags = {f["category_tag"] for f in dataset_spec["failure_categories"]}
        required_tags = {
            "FAILURE_WRONG_ACTION",
            "FAILURE_SKIPPED_STEP",
            "FAILURE_OUT_OF_ORDER",
            "FAILURE_PREMATURE",
            "FAILURE_TIMEOUT",
            "FAILURE_WRONG_OBJECT",
            "INCOMPLETE_APPROACH",
        }
        missing = required_tags - tags
        assert not missing, \
            f"Missing required failure category tags: {missing}"


# ---------------------------------------------------------------------------
# Test 10: Current model is documented as NOT integrated
# ---------------------------------------------------------------------------

class TestCurrentModelStatus:
    """The existing catch/not-catch Part-1 model must be documented as not integrated."""

    def test_current_model_section_exists(self, dataset_spec):
        assert "current_part1_model" in dataset_spec, \
            "dataset_spec.json must have a 'current_part1_model' section"

    def test_current_model_is_not_integrated(self, dataset_spec):
        status = dataset_spec["current_part1_model"].get("integration_status")
        assert status == "NOT_INTEGRATED", (
            f"current_part1_model.integration_status must be 'NOT_INTEGRATED', got: {status}"
        )

    def test_current_model_classes_are_binary(self, dataset_spec):
        classes = dataset_spec["current_part1_model"].get("classes", [])
        assert set(classes) == {"catch", "not-catch"}, (
            f"current_part1_model.classes must be ['catch', 'not-catch'], got: {classes}"
        )

    def test_no_synthetic_mapping_catch_to_procedure_actions(self, dataset_spec):
        # There must be no field that creates a synthetic mapping from catch/not-catch
        # to the 5 procedure action labels. Verify by checking none of the canonical
        # action names equals a binary class.
        action_ids = {a["action_id"] for a in dataset_spec["canonical_actions"]}
        assert "catch" not in action_ids, \
            "No synthetic mapping from 'catch' to procedure actions should exist"
        assert "not-catch" not in action_ids, \
            "No synthetic mapping from 'not-catch' to procedure actions should exist"


# ---------------------------------------------------------------------------
# Test 11: Dataset Sources Alignment (HMDB51 is Part-1 baseline IN_USE vs EXP-001 NOT_COLLECTED)
# ---------------------------------------------------------------------------

class TestDatasetSourcesAlignment:
    """dataset_spec.json must distinguish current baseline (HMDB51) from future EXP-001 dataset."""

    def test_dataset_sources_section_exists(self, dataset_spec):
        assert "dataset_sources" in dataset_spec, \
            "dataset_spec.json must have a 'dataset_sources' section"

    def test_current_baseline_dataset_is_hmdb51(self, dataset_spec):
        baseline = dataset_spec["dataset_sources"].get("current_baseline_dataset", {})
        assert baseline.get("name") == "HMDB51", \
            f"current_baseline_dataset name must be 'HMDB51', got: {baseline.get('name')}"
        assert baseline.get("status") == "IN_USE_IN_PART_1", \
            f"current_baseline_dataset status must be 'IN_USE_IN_PART_1', got: {baseline.get('status')}"
        assert baseline.get("current_model_classes") == ["catch", "not-catch"], \
            f"current_model_classes must be ['catch', 'not-catch'], got: {baseline.get('current_model_classes')}"

    def test_procedure_specific_dataset_is_not_collected(self, dataset_spec):
        proc_data = dataset_spec["dataset_sources"].get("procedure_specific_dataset", {})
        assert proc_data.get("status") == "NOT_COLLECTED", (
            f"procedure_specific_dataset status must be 'NOT_COLLECTED', got: {proc_data.get('status')}"
        )
        labels = proc_data.get("future_procedure_specific_labels", [])
        expected_labels = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"]
        assert labels == expected_labels, (
            f"future_procedure_specific_labels mismatch. Expected: {expected_labels}, got: {labels}"
        )

    def test_existing_training_data_documents_part1_in_use(self, dataset_spec):
        hmdb51 = dataset_spec.get("existing_training_data", {}).get("hmdb51", {})
        assert hmdb51.get("part1_baseline_status") == "IN_USE_IN_PART_1", (
            f"hmdb51 part1_baseline_status must be 'IN_USE_IN_PART_1', got: {hmdb51.get('part1_baseline_status')}"
        )


# ---------------------------------------------------------------------------
# Test 12: Collection status confirms no actual data collected yet
# ---------------------------------------------------------------------------

class TestCollectionStatus:
    """No actual dataset should be marked as collected in this gate."""

    def test_collection_targets_section_exists(self, dataset_spec):
        assert "collection_targets" in dataset_spec, \
            "dataset_spec.json must have a 'collection_targets' section"

    def test_collection_status_is_not_collected(self, dataset_spec):
        status = dataset_spec["collection_targets"].get("status")
        assert status == "NOT_COLLECTED", (
            f"collection_targets.status must be 'NOT_COLLECTED'. Got: {status}. "
            "Gate B3/B3.1 only defines the protocol; no actual data is collected."
        )

    def test_no_actual_dataset_directory_required(self):
        # data/dataset/ should not be required to exist in B3/B3.1
        dataset_dir = os.path.join(REPO_ROOT, "data", "dataset")
        # This test passes whether or not the directory exists —
        # it simply documents that its absence is expected in B3/B3.1.
        # We just verify that no assertion forces its presence.
        assert True, "data/dataset/ is not required to exist in Gate B3/B3.1"


# ---------------------------------------------------------------------------
# Test 13: Protocol document covers all 5 canonical actions and 4 objects
# ---------------------------------------------------------------------------

class TestProtocolDocumentContent:
    """The protocol markdown document must mention all canonical actions and objects."""

    @pytest.mark.parametrize("action", sorted(EXPECTED_ACTIONS))
    def test_protocol_mentions_action(self, protocol_doc, action):
        assert action in protocol_doc, (
            f"docs/workstream-b-dataset-protocol.md must mention action '{action}'"
        )

    @pytest.mark.parametrize("obj", sorted(EXPECTED_OBJECTS))
    def test_protocol_mentions_object(self, protocol_doc, obj):
        assert obj in protocol_doc, (
            f"docs/workstream-b-dataset-protocol.md must mention object '{obj}'"
        )

    def test_protocol_explicitly_states_no_data_collected(self, protocol_doc):
        # One of: "not collected", "NOT COLLECTED", "no actual dataset", etc.
        doc_lower = protocol_doc.lower()
        markers = [
            "not been collected",
            "not collected",
            "no actual dataset",
            "no dataset has been collected",
            "no dataset files are created",
            "no experiment-specific dataset",
        ]
        assert any(m in doc_lower for m in markers), (
            "Protocol doc must explicitly state that no actual dataset has been collected in this gate"
        )

    def test_protocol_documents_hmdb51_as_baseline_source(self, protocol_doc):
        assert "hmdb51" in protocol_doc.lower(), \
            "Protocol doc must mention HMDB51 dataset"
        assert "current part-1 baseline" in protocol_doc.lower() or \
               "part-1 baseline dataset source" in protocol_doc.lower(), \
            "Protocol doc must explicitly describe HMDB51 as the current Part-1 baseline dataset source"
        assert "not equivalent to" in protocol_doc.lower() or \
               "not map directly" in protocol_doc.lower(), \
            "Protocol doc must state HMDB51 generic classes are not equivalent to EXP-001 procedure actions"
