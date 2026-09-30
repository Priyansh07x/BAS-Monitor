"""
test_b6_1_vocabulary_rectification.py — Gate B6.1 Verification Test Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates:
1. Canonical Object Vocabulary: ObjectDetector KNOWN_CLASSES matches EXP-001 canonical vocabulary.
2. Canonical Action Vocabulary: ActionClassifier DEFAULT_ACTIONS matches EXP-001 canonical actions.
3. Absence of Legacy Vocabulary: Obsolete action/object labels do not exist in runtime definitions or fallback outputs.
4. Fallback Behavior: ObjectDetector and ActionClassifier produce canonical outputs when models are absent.
5. Rectification Integration: InferencePipeline seamlessly integrates the CameraRectifier hook.
6. Rectification Default Disabled: Default execution bypasses rectification with rectification_applied=False.
7. Rectification Enabled: When configured/enabled, rectification_applied=True and frame is rotated/rectified cleanly.
8. Error Handling & Safety: Invalid rectification configs degrade safely to unrectified pass-through without crashing.
9. Contract Preservation: 111-D feature vector and 30-frame temporal window contracts remain exact and unchanged.
10. Reset & Lifecycle: InferencePipeline.reset() and release() operate cleanly without errors.
"""

import numpy as np
import pytest

from backend.ai.object_detector import ObjectDetector
from backend.ai.action_classifier import ActionClassifier
from backend.ai.action_recognizer import ActionRecognizer
from backend.ai.inference_pipeline import InferencePipeline
from backend.video.frame_processor import FrameProcessor
from backend.video.camera_rectification import CameraRectifier, RectificationConfig


CANONICAL_OBJECTS = {"RED_SAMPLE", "BLUE_SAMPLE", "SAMPLE_CONTAINER", "CONTAINER_LID"}
CANONICAL_ACTIONS = {"PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID", "IDLE"}

LEGACY_OBJECT_LABELS = {"CONTAINER", "SAMPLE_VIAL", "PIPETTE", "ANALYZER_CHAMBER", "REAGENT_BOTTLE", "FORCEPS"}
LEGACY_ACTION_LABELS = {"PICK_CONTAINER", "PIPETTE_TRANSFER", "INSERT_ANALYZER", "SEAL_CONTAINER", "INITIATE_SCAN"}


# ===========================================================================
# 1. Object Vocabulary Tests
# ===========================================================================

def test_object_detector_known_classes_canonical():
    """Verify ObjectDetector.KNOWN_CLASSES contains exactly the 4 canonical EXP-001 objects."""
    detector = ObjectDetector()
    assert set(detector.KNOWN_CLASSES) == CANONICAL_OBJECTS, (
        f"ObjectDetector.KNOWN_CLASSES mismatch: expected {CANONICAL_OBJECTS}, got {detector.KNOWN_CLASSES}"
    )


def test_object_detector_no_legacy_classes():
    """Verify no legacy object classes exist in ObjectDetector.KNOWN_CLASSES."""
    detector = ObjectDetector()
    for legacy in LEGACY_OBJECT_LABELS:
        assert legacy not in detector.KNOWN_CLASSES, f"Legacy object class '{legacy}' found in KNOWN_CLASSES"


def test_object_detector_fallback_uses_canonical_labels():
    """Verify fallback detection outputs use canonical EXP-001 object labels."""
    detector = ObjectDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert len(detections) > 0, "Fallback detections should not be empty"
    for det in detections:
        assert det["label"] in CANONICAL_OBJECTS, (
            f"Non-canonical object label '{det['label']}' produced in fallback"
        )
        assert 0.0 <= det["x1"] <= det["x2"] <= 1.0
        assert 0.0 <= det["y1"] <= det["y2"] <= 1.0


# ===========================================================================
# 2. Action Vocabulary Tests
# ===========================================================================

def test_action_classifier_default_actions_canonical():
    """Verify ActionClassifier.DEFAULT_ACTIONS contains only canonical EXP-001 actions + IDLE."""
    classifier = ActionClassifier()
    assert set(classifier.DEFAULT_ACTIONS) == CANONICAL_ACTIONS, (
        f"ActionClassifier.DEFAULT_ACTIONS mismatch: expected {CANONICAL_ACTIONS}, got {classifier.DEFAULT_ACTIONS}"
    )


def test_action_classifier_no_legacy_actions():
    """Verify no legacy action classes exist in ActionClassifier.DEFAULT_ACTIONS."""
    classifier = ActionClassifier()
    for legacy in LEGACY_ACTION_LABELS:
        assert legacy not in classifier.DEFAULT_ACTIONS, f"Legacy action class '{legacy}' found in DEFAULT_ACTIONS"


def test_action_classifier_fallback_uses_canonical_labels():
    """Verify heuristic classification fallback returns canonical EXP-001 actions."""
    classifier = ActionClassifier()
    # Feed dummy vectors to fill minimal buffer
    for _ in range(6):
        classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

    act_idle, _ = classifier.classify(interaction_state="IDLE")
    assert act_idle in CANONICAL_ACTIONS

    act_hold_red, conf = classifier.classify(interaction_state="HOLDING", target_object="RED_SAMPLE")
    assert act_hold_red == "PICK_RED"
    assert conf > 0.80

    act_hold_blue, conf = classifier.classify(interaction_state="HOLDING", target_object="BLUE_SAMPLE")
    assert act_hold_blue == "PICK_BLUE"

    act_close_lid, conf = classifier.classify(interaction_state="HOLDING", target_object="CONTAINER_LID")
    assert act_close_lid == "CLOSE_LID"


def test_action_recognizer_facade_canonical():
    """Verify ActionRecognizer facade supports canonical actions and passes target_object."""
    recognizer = ActionRecognizer()
    for _ in range(6):
        action, conf = recognizer.recognize(
            keypoint_vector=np.zeros(111, dtype=np.float32),
            interaction_state="HOLDING",
            target_object="BLUE_SAMPLE",
        )
    assert action == "PICK_BLUE"
    assert action in CANONICAL_ACTIONS


# ===========================================================================
# 3. Rectification Hook Integration in InferencePipeline
# ===========================================================================

def test_inference_pipeline_default_rectification_disabled():
    """Verify default InferencePipeline execution leaves rectification disabled (rectification_applied=False)."""
    pipeline = InferencePipeline()
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = pipeline.process_frame(frame, annotate=False)

    assert result["rectification_applied"] is False
    assert result["frame_index"] == 1
    assert "objects" in result
    assert "pose" in result
    assert "hands" in result
    assert "interaction" in result
    assert "action" in result
    assert result["action"] in CANONICAL_ACTIONS


def test_inference_pipeline_with_active_rectifier():
    """Verify InferencePipeline applies camera rectification when an active CameraRectifier is provided."""
    rect_cfg = RectificationConfig(enabled=True, rotation_deg=90.0)
    rectifier = CameraRectifier(rect_cfg)

    pipeline = InferencePipeline(rectifier=rectifier)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[10:50, 10:50] = 255

    result = pipeline.process_frame(frame, annotate=True)

    assert result["rectification_applied"] is True
    assert result["frame_index"] == 1
    if result["annotated_frame"] is not None:
        assert result["annotated_frame"].shape[:2] == (480, 640)


def test_inference_pipeline_per_frame_rectifier_override():
    """Verify process_frame allows per-call rectifier override."""
    pipeline = InferencePipeline()  # Default disabled

    # Override on call
    rectifier_90 = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=90.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, annotate=False, rectifier=rectifier_90)

    assert result["rectification_applied"] is True


def test_inference_pipeline_invalid_rectifier_safety():
    """Verify InferencePipeline handles invalid rectifier configs safely without crashing."""
    pipeline = InferencePipeline(rectifier={"invalid": "config", "enabled": True})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, annotate=False)

    # Should safely degrade to unrectified pass-through
    assert result["rectification_applied"] is False
    assert result["action"] in CANONICAL_ACTIONS


# ===========================================================================
# 4. Feature Vector & Temporal Contract Preservation
# ===========================================================================

def test_feature_vector_111d_contract_preserved():
    """Verify FrameProcessor.extract_keypoint_vector maintains the exact 111-D float32 shape."""
    processor = FrameProcessor()
    dummy_pose = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(33)]
    dummy_objects = [
        {"label": "RED_SAMPLE", "confidence": 0.9, "x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6},
        {"label": "SAMPLE_CONTAINER", "confidence": 0.95, "x1": 0.3, "y1": 0.3, "x2": 0.7, "y2": 0.7},
    ]

    vec = processor.extract_keypoint_vector(dummy_pose, dummy_objects)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (111,)
    assert vec.dtype == np.float32


def test_temporal_buffer_window_contract_preserved():
    """Verify ActionClassifier maintains the exact 30-frame window size contract."""
    classifier = ActionClassifier(window_size=30)
    assert classifier.window_size == 30

    for i in range(40):
        vec = np.full(111, float(i), dtype=np.float32)
        classifier.push_frame_vector(vec)

    assert len(classifier._vector_buffer) == 30
    # First item should be vector 10 (FIFO eviction of vectors 0-9)
    assert classifier._vector_buffer[0][0] == 10.0
    assert classifier._vector_buffer[-1][0] == 39.0


def test_inference_pipeline_empty_frame_safety():
    """Verify InferencePipeline gracefully handles None and empty frames."""
    pipeline = InferencePipeline()
    res_none = pipeline.process_frame(None)
    assert res_none["action"] == "IDLE"
    assert res_none["objects"] == []
    assert res_none["rectification_applied"] is False

    res_empty = pipeline.process_frame(np.empty((0, 0, 3), dtype=np.uint8))
    assert res_empty["action"] == "IDLE"
    assert res_empty["objects"] == []


def test_inference_pipeline_reset_and_release():
    """Verify reset() and release() methods execute cleanly."""
    pipeline = InferencePipeline()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.process_frame(frame)
    assert pipeline._frame_index == 1

    pipeline.reset()
    assert pipeline._frame_index == 0
    assert len(pipeline.action_classifier._vector_buffer) == 0

    pipeline.release()
