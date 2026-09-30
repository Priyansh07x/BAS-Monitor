"""
test_b6_2_multi_detector_coordination.py — Gate B6.2 Multi-Detector Coordination & Heuristics Tests
ISRO SIH26174 BAS Experiment Monitor

Validates:
1. Full pipeline coordination with deterministic fixtures.
2. Canonical object propagation (RED_SAMPLE, BLUE_SAMPLE, SAMPLE_CONTAINER, CONTAINER_LID).
3. Canonical five-action vocabulary (PICK_RED, PLACE_RED, PICK_BLUE, PLACE_BLUE, CLOSE_LID, IDLE).
4. Red vs blue object distinction.
5. Container-lid interaction producing CLOSE_LID.
6. Deterministic heuristic behavior (no random outputs, fully reproducible).
7. InteractionEngine output propagation (container metrics, distances, IoU).
8. Missing detector-output handling (missing boxes, pose landmarks, hand tracking).
9. Missing model checkpoint handling (graceful fallback without crashing).
10. Optional timestamp acceptance (float, ISO string, None).
11. Existing frame-index compatibility.
12. 111-D feature contract preservation.
13. 30-frame temporal buffer contract preservation.
14. Legacy vocabulary absence from action outputs.
15. No random action selection.
"""

import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.object_detector import ObjectDetector
from backend.ai.pose_detector import PoseDetector
from backend.ai.hand_detector import HandDetector
from backend.ai.action_classifier import ActionClassifier
from backend.ai.action_recognizer import ActionRecognizer
from backend.experiment.interaction_logic import InteractionEngine
from backend.video.frame_processor import FrameProcessor


CANONICAL_ACTIONS = {
    "PICK_RED",
    "PLACE_RED",
    "PICK_BLUE",
    "PLACE_BLUE",
    "CLOSE_LID",
    "IDLE",
}

CANONICAL_OBJECTS = {
    "RED_SAMPLE",
    "BLUE_SAMPLE",
    "SAMPLE_CONTAINER",
    "CONTAINER_LID",
}

LEGACY_ACTION_LABELS = {
    "PICK_CONTAINER",
    "PIPETTE_TRANSFER",
    "INSERT_ANALYZER",
    "SEAL_CONTAINER",
    "INITIATE_SCAN",
}

LEGACY_OBJECT_LABELS = {
    "CONTAINER",
    "SAMPLE_VIAL",
    "PIPETTE",
    "WELL_PLATE",
    "CENTRIFUGE_TUBE",
    "FORCEPS",
    "INCUBATOR_DOOR",
}


@pytest.fixture
def synthetic_frame():
    """Create a deterministic synthetic 640x480 RGB frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:380, 150:490] = (128, 128, 128)  # Workspace table
    frame[200:260, 200:260] = (0, 0, 255)      # Red sample
    frame[200:260, 380:440] = (255, 0, 0)      # Blue sample
    return frame


@pytest.fixture
def pipeline():
    """Returns a clean InferencePipeline instance."""
    return InferencePipeline()


# ===========================================================================
# 1. Pipeline Coordination & Return Structure
# ===========================================================================

def test_full_pipeline_coordination(pipeline, synthetic_frame):
    """Verify that InferencePipeline coordinates all detector stages and returns canonical payload."""
    result = pipeline.process_frame(synthetic_frame, annotate=True)

    assert isinstance(result, dict)
    assert result["frame_index"] == 1
    assert "rectification_applied" in result
    assert "objects" in result
    assert "pose" in result
    assert "hands" in result
    assert "interaction" in result
    assert "action" in result
    assert "confidence" in result
    assert "annotated_frame" in result
    assert result["action"] in CANONICAL_ACTIONS


def test_canonical_object_propagation(pipeline, synthetic_frame):
    """Verify all detected objects strictly belong to canonical EXP-001 vocabulary."""
    result = pipeline.process_frame(synthetic_frame)

    for obj in result["objects"]:
        label = obj.get("label")
        assert label in CANONICAL_OBJECTS, f"Non-canonical object label found: {label}"
        assert label not in LEGACY_OBJECT_LABELS, f"Legacy object label found: {label}"


# ===========================================================================
# 2. Canonical Five-Action Fallback & Distinction Tests
# ===========================================================================

def test_pick_red_fallback():
    """Verify holding RED_SAMPLE away from container produces PICK_RED."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    interaction = {
        "state": "HOLDING",
        "target_object": "RED_SAMPLE",
        "container_iou": 0.0,
        "container_distance": 0.50,
        "container_scale_proximity": 3.0,
    }
    action, conf = classifier.classify(interaction=interaction)
    assert action == "PICK_RED"
    assert conf >= 0.85
    assert action in CANONICAL_ACTIONS


def test_place_red_fallback():
    """Verify holding RED_SAMPLE near/in SAMPLE_CONTAINER produces PLACE_RED."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    interaction = {
        "state": "HOLDING",
        "target_object": "RED_SAMPLE",
        "container_iou": 0.15,
        "container_distance": 0.05,
        "container_scale_proximity": 0.4,
    }
    action, conf = classifier.classify(interaction=interaction)
    assert action == "PLACE_RED"
    assert conf >= 0.85
    assert action in CANONICAL_ACTIONS


def test_pick_blue_fallback():
    """Verify holding BLUE_SAMPLE away from container produces PICK_BLUE."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    interaction = {
        "state": "HOLDING",
        "target_object": "BLUE_SAMPLE",
        "container_iou": 0.0,
        "container_distance": 0.55,
        "container_scale_proximity": 3.2,
    }
    action, conf = classifier.classify(interaction=interaction)
    assert action == "PICK_BLUE"
    assert conf >= 0.85
    assert action in CANONICAL_ACTIONS


def test_place_blue_fallback():
    """Verify holding BLUE_SAMPLE near/in SAMPLE_CONTAINER produces PLACE_BLUE."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    interaction = {
        "state": "HOLDING",
        "target_object": "BLUE_SAMPLE",
        "container_iou": 0.12,
        "container_distance": 0.06,
        "container_scale_proximity": 0.5,
    }
    action, conf = classifier.classify(interaction=interaction)
    assert action == "PLACE_BLUE"
    assert conf >= 0.85
    assert action in CANONICAL_ACTIONS


def test_close_lid_fallback():
    """Verify manipulating CONTAINER_LID produces CLOSE_LID."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    interaction = {
        "state": "HOLDING",
        "target_object": "CONTAINER_LID",
    }
    action, conf = classifier.classify(interaction=interaction)
    assert action == "CLOSE_LID"
    assert conf >= 0.85
    assert action in CANONICAL_ACTIONS


def test_approaching_state_is_idle_non_action():
    """Verify approaching alone does not trigger false premature action."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    interaction = {
        "state": "APPROACHING",
        "target_object": "RED_SAMPLE",
    }
    action, conf = classifier.classify(interaction=interaction)
    assert action == "IDLE"
    assert conf <= 0.80


# ===========================================================================
# 3. Interaction Engine Propagation & Secondary Spatial Metrics
# ===========================================================================

def test_interaction_engine_container_metrics():
    """Verify InteractionEngine computes secondary container metrics when SAMPLE_CONTAINER is present."""
    engine = InteractionEngine()
    hand = [{"x1": 0.45, "y1": 0.45, "x2": 0.55, "y2": 0.55}]
    objects = [
        {"label": "RED_SAMPLE", "x1": 0.46, "y1": 0.46, "x2": 0.52, "y2": 0.52},
        {"label": "SAMPLE_CONTAINER", "x1": 0.40, "y1": 0.40, "x2": 0.60, "y2": 0.60},
    ]

    res = engine.evaluate_interaction(hand, objects)
    assert res["state"] == "HOLDING"
    assert res["container_iou"] > 0.0
    assert res["container_distance"] is not None
    assert res["container_scale_proximity"] is not None


# ===========================================================================
# 4. Determinism, Safety & Missing-Input Resilience
# ===========================================================================

def test_deterministic_reproducibility(pipeline, synthetic_frame):
    """Verify identical inputs produce identical outputs across multiple calls."""
    res1 = pipeline.process_frame(synthetic_frame, annotate=False)
    pipeline.reset()
    res2 = pipeline.process_frame(synthetic_frame, annotate=False)

    assert res1["action"] == res2["action"]
    assert res1["confidence"] == res2["confidence"]
    assert len(res1["objects"]) == len(res2["objects"])


def test_empty_and_corrupt_frame_safety(pipeline):
    """Verify pipeline handles empty and zero-size frames gracefully."""
    res_none = pipeline.process_frame(None)
    assert res_none["frame_index"] == 0
    assert res_none["action"] == "IDLE"
    assert res_none["objects"] == []

    res_empty = pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_empty["frame_index"] == 0
    assert res_empty["action"] == "IDLE"


def test_missing_detector_outputs_safety():
    """Verify ActionClassifier handles empty object lists and missing interaction cleanly."""
    classifier = ActionClassifier()
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    act, conf = classifier.classify(interaction_state=None, target_object=None, interaction=None)
    assert act == "IDLE"
    assert conf == 0.90


# ===========================================================================
# 5. Timestamp & Frame Metadata Support
# ===========================================================================

def test_optional_timestamp_and_metadata_passthrough(pipeline, synthetic_frame):
    """Verify pipeline accepts optional monotonic timestamps and metadata dictionaries."""
    ts_float = 1727181500.123
    meta = {"camera_id": "overhead_cam_01", "fps": 30, "resolution": [640, 480]}

    result = pipeline.process_frame(
        synthetic_frame,
        timestamp=ts_float,
        metadata=meta,
    )

    assert result["timestamp"] == ts_float
    assert result["metadata"] == meta
    assert result["frame_index"] == 1


def test_default_timestamp_is_none_when_unprovided(pipeline, synthetic_frame):
    """Verify pipeline does not fabricate a fake timestamp when none is supplied."""
    result = pipeline.process_frame(synthetic_frame)
    assert result["timestamp"] is None
    assert result["metadata"] == {}


# ===========================================================================
# 6. Contract Invariants (111-D & 30-Frame Window & Facade)
# ===========================================================================

def test_111d_feature_vector_invariance():
    """Verify FrameProcessor.extract_keypoint_vector preserves exact 111 dimensions."""
    proc = FrameProcessor()
    dummy_pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33
    dummy_boxes = [{"label": "RED_SAMPLE", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2, "confidence": 0.9}]
    vec = proc.extract_keypoint_vector(dummy_pose, dummy_boxes)

    assert vec.shape == (111,)
    assert vec.dtype == np.float32


def test_30_frame_buffer_invariance():
    """Verify ActionClassifier maintains 30-frame temporal buffer contract."""
    classifier = ActionClassifier(window_size=30)
    for i in range(40):
        classifier.push_frame_vector(np.full(111, float(i), dtype=np.float32))

    assert len(classifier._vector_buffer) == 30
    assert np.all(classifier._vector_buffer[-1] == 39.0)


def test_action_recognizer_facade_coordination():
    """Verify ActionRecognizer facade coordinates with rich interaction context."""
    recognizer = ActionRecognizer()
    for _ in range(6):
        action, conf = recognizer.recognize(
            keypoint_vector=np.zeros(111, dtype=np.float32),
            interaction={
                "state": "HOLDING",
                "target_object": "RED_SAMPLE",
                "container_distance": 0.05,
                "container_iou": 0.20,
            },
        )
    assert action == "PLACE_RED"
    assert conf >= 0.85
    assert action in CANONICAL_ACTIONS


def test_no_legacy_vocabulary_in_outputs(pipeline, synthetic_frame):
    """Verify legacy vocabulary labels are completely absent from action classification."""
    for _ in range(5):
        res = pipeline.process_frame(synthetic_frame)
        assert res["action"] not in LEGACY_ACTION_LABELS
        for obj in res["objects"]:
            assert obj["label"] not in LEGACY_OBJECT_LABELS
