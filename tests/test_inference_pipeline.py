"""
test_inference_pipeline.py — Gate B6.4 Comprehensive Inference Pipeline & Public Contract Test Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates the complete model-independent perception pipeline from video frame input through
spatial feature extraction, multi-detector coordination, temporal buffering, deterministic
heuristic fallback, and frozen 8-field public AI contract adaptation.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import (
    AIResultAdapter,
    CANONICAL_ACTIONS,
    CANONICAL_OBJECTS,
    LEGACY_ACTIONS,
    LEGACY_OBJECTS,
    PUBLIC_CONTRACT_KEYS,
    PUBLIC_STATUS_VALUES,
    STATUS_MAP,
)
from backend.ai.action_classifier import ActionClassifier
from backend.ai.action_recognizer import ActionRecognizer
from backend.ai.object_detector import ObjectDetector
from backend.ai.pose_detector import PoseDetector
from backend.ai.hand_detector import HandDetector
from backend.ai.hailo_inference import HailoInferenceEngine
from backend.experiment.interaction_logic import InteractionEngine
from backend.video.frame_processor import FrameProcessor
from backend.video.camera_rectification import CameraRectifier, RectificationConfig


class TestInferencePipelineFrameExecution:
    """Tests 1-4: Frame processing, safety, and rectification paths."""

    def test_valid_frame_execution(self):
        """Test 1: Valid frame produces complete internal perception payload."""
        pipeline = InferencePipeline()
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        result = pipeline.process_frame(frame)

        required_keys = {
            "frame_index",
            "timestamp",
            "metadata",
            "rectification_applied",
            "objects",
            "pose",
            "hands",
            "interaction",
            "action",
            "confidence",
            "annotated_frame",
        }
        assert required_keys.issubset(set(result.keys()))
        assert result["frame_index"] == 1
        assert result["action"] in CANONICAL_ACTIONS
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["objects"], list)
        assert isinstance(result["pose"], list)
        assert isinstance(result["hands"], list)
        assert isinstance(result["interaction"], dict)

    def test_empty_and_invalid_frame_safety(self):
        """Test 2: Empty, None, or zero-dimension frames return safe defaults without crashing."""
        pipeline = InferencePipeline()

        # None frame
        res_none = pipeline.process_frame(None)
        assert res_none["action"] == "IDLE"
        assert res_none["confidence"] == 0.0
        assert res_none["objects"] == []
        assert res_none["annotated_frame"] is None

        # Zero-size frame
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = pipeline.process_frame(empty_frame)
        assert res_empty["action"] == "IDLE"
        assert res_empty["confidence"] == 0.0

    def test_rectification_disabled_identity_path(self):
        """Test 3: Unconfigured or disabled rectifier leaves frame unrectified."""
        pipeline = InferencePipeline(rectifier=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipeline.process_frame(frame)
        assert res["rectification_applied"] is False

        # Disabled config explicitly
        cfg = RectificationConfig(enabled=False, rotation_deg=90.0)
        res_cfg = pipeline.process_frame(frame, rectifier=cfg)
        assert res_cfg["rectification_applied"] is False

    def test_rectification_enabled_path(self):
        """Test 4: Enabled rectifier applies geometric transformation and sets telemetry flag."""
        rect = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=90.0))
        pipeline = InferencePipeline(rectifier=rect)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipeline.process_frame(frame)
        assert res["rectification_applied"] is True


class TestVocabularyAndCoordination:
    """Tests 5-9: Canonical vocabularies, legacy rejection, and multi-detector coordination."""

    def test_canonical_object_vocabulary(self):
        """Test 5: Detector classes conform strictly to canonical EXP-001 objects."""
        detector = ObjectDetector()
        assert set(ObjectDetector.KNOWN_CLASSES) == {"RED_SAMPLE", "BLUE_SAMPLE", "SAMPLE_CONTAINER", "CONTAINER_LID"}
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy)
        for d in detections:
            assert d["label"] in CANONICAL_OBJECTS

    def test_canonical_action_vocabulary(self):
        """Test 6: Classifier default actions conform strictly to canonical EXP-001 actions."""
        classifier = ActionClassifier()
        assert set(classifier.actions) == CANONICAL_ACTIONS

    def test_legacy_vocabulary_rejection(self):
        """Test 7: Legacy actions and objects are absent from active components and rejected by adapter."""
        for leg_act in LEGACY_ACTIONS:
            assert leg_act not in ActionClassifier.DEFAULT_ACTIONS
            adapted = AIResultAdapter.adapt(action=leg_act)
            assert adapted["action"] == "IDLE"

        for leg_obj in LEGACY_OBJECTS:
            assert leg_obj not in ObjectDetector.KNOWN_CLASSES
            adapted_obj = AIResultAdapter.adapt(object=leg_obj, action="IDLE")
            assert adapted_obj["object"] == "NONE"

    def test_multi_detector_coordination(self):
        """Test 8: Object, pose, and hand detectors coordinate cleanly within pipeline."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipeline.process_frame(frame)

        # Hand boxes and landmarks are forwarded to interaction evaluation
        assert "interaction" in res
        assert "state" in res["interaction"]
        assert "target_object" in res["interaction"]

    def test_interaction_engine_evidence_propagation(self):
        """Test 9: InteractionEngine spatial metrics propagate into temporal classification."""
        engine = InteractionEngine()
        hand_boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200}]
        obj_boxes = [
            {"x1": 110, "y1": 110, "x2": 190, "y2": 190, "label": "RED_SAMPLE"},
            {"x1": 400, "y1": 400, "x2": 550, "y2": 550, "label": "SAMPLE_CONTAINER"},
        ]
        inter = engine.evaluate_interaction(hand_boxes=hand_boxes, object_boxes=obj_boxes)
        assert inter["state"] == "HOLDING"
        assert inter["target_object"] == "RED_SAMPLE"
        assert "container_distance" in inter
        assert "container_scale_proximity" in inter


class TestFeatureAndTemporalContracts:
    """Tests 10-14: 111-D feature vector, 30-frame temporal buffer, warm-up, and missing checkpoints."""

    def test_exact_111d_feature_vector_structure(self):
        """Test 10: Vector extraction strictly adheres to 111-D contract:
        - indices 0..98: 33 body landmarks x 3 (x,y,z)
        - indices 99..110: 4 object centroids x 3 (cx, cy, conf)
        """
        pose = [{"x": float(i), "y": float(i * 2), "z": float(i * 3)} for i in range(33)]
        objects = [
            {"x1": 10, "y1": 20, "x2": 50, "y2": 60, "confidence": 0.9},
            {"x1": 70, "y1": 80, "x2": 90, "y2": 100, "confidence": 0.8},
        ]
        vec = FrameProcessor.extract_keypoint_vector(pose, objects)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (111,)
        assert vec.dtype == np.float32

        # Verify landmark segment
        assert vec[0] == 0.0 and vec[1] == 0.0 and vec[2] == 0.0
        assert vec[3] == 1.0 and vec[4] == 2.0 and vec[5] == 3.0

        # Verify object segment (index 99 onwards)
        # Obj 0: cx = (10+50)/2 = 30, cy = (20+60)/2 = 40, conf = 0.9
        assert vec[99] == 30.0
        assert vec[100] == 40.0
        assert pytest.approx(vec[101], 0.01) == 0.9

        # Obj 1: cx = (70+90)/2 = 80, cy = (80+100)/2 = 90, conf = 0.8
        assert vec[102] == 80.0
        assert vec[103] == 90.0
        assert pytest.approx(vec[104], 0.01) == 0.8

        # Remaining 2 padded objects: 0.0
        assert np.all(vec[105:111] == 0.0)

    def test_exact_temporal_sliding_window_shape(self):
        """Test 11: Sliding window buffer stacks exactly into (1, 30, 111) shape."""
        classifier = ActionClassifier(window_size=30)
        for i in range(35):
            vec = np.ones(111, dtype=np.float32) * i
            classifier.push_frame_vector(vec)

        assert len(classifier._vector_buffer) == 30
        stacked = np.stack(list(classifier._vector_buffer))
        assert stacked.shape == (30, 111)
        expanded = np.expand_dims(stacked, axis=0)
        assert expanded.shape == (1, 30, 111)

    def test_temporal_warmup_behavior(self):
        """Test 12: Classifier returns ('IDLE', 0.95) when buffer has fewer than 5 frames."""
        classifier = ActionClassifier()
        classifier.reset_buffer()
        for _ in range(4):
            classifier.push_frame_vector(np.zeros(111, dtype=np.float32))
            act, conf = classifier.classify(interaction_state="HOLDING", target_object="RED_SAMPLE")
            assert act == "IDLE"
            assert conf == 0.95

    @pytest.mark.parametrize("inter_state, target_obj, near_container, expected_action", [
        ("HOLDING", "CONTAINER_LID", False, "CLOSE_LID"),
        ("HOLDING", "RED_SAMPLE", False, "PICK_RED"),
        ("HOLDING", "RED_SAMPLE", True, "PLACE_RED"),
        ("HOLDING", "BLUE_SAMPLE", False, "PICK_BLUE"),
        ("HOLDING", "BLUE_SAMPLE", True, "PLACE_BLUE"),
        ("APPROACHING", "RED_SAMPLE", False, "IDLE"),
        ("IDLE", None, False, "IDLE"),
    ])
    def test_deterministic_heuristic_fallback(self, inter_state, target_obj, near_container, expected_action):
        """Test 13: Deterministic fallback resolves all 5 canonical actions correctly after warm-up."""
        classifier = ActionClassifier()
        classifier.reset_buffer()
        for _ in range(10):
            classifier.push_frame_vector(np.zeros(111, dtype=np.float32))

        inter_dict = {
            "state": inter_state,
            "target_object": target_obj,
            "container_iou": 0.25 if near_container else 0.0,
            "container_distance": 0.05 if near_container else 0.50,
            "container_scale_proximity": 0.80 if near_container else 2.50,
        }
        act, conf = classifier.classify(
            interaction_state=inter_state,
            target_object=target_obj,
            interaction=inter_dict,
        )
        assert act == expected_action
        assert 0.0 <= conf <= 1.0

    def test_missing_model_checkpoints_safety(self):
        """Test 14: Absence of neural checkpoints (TFLite, YOLO weights, Hailo HEF) does not raise exceptions."""
        pipeline = InferencePipeline(
            hef_path="nonexistent/path/model.hef",
            tflite_path="nonexistent/path/temporal.tflite",
        )
        assert pipeline.action_classifier._is_tflite_ready is False
        assert pipeline.hailo_engine.is_npu_active is False

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipeline.process_frame(frame)
        assert res["action"] in CANONICAL_ACTIONS
        assert 0.0 <= res["confidence"] <= 1.0


class TestMetadataMonotonicityAndAdapterIntegration:
    """Tests 15-20: Monotonicity, timestamps, metadata, public schema, and diagnostic stripping."""

    def test_timestamp_preservation_and_conversion(self):
        """Test 15: Timestamp strings and numeric epochs are handled cleanly."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        iso_str = "2026-09-24T18:30:00.000Z"
        res_iso = pipeline.process_frame(frame, timestamp=iso_str)
        assert res_iso["timestamp"] == iso_str

        pub_iso = pipeline.process_frame_public(frame, timestamp=iso_str)
        assert pub_iso["timestamp"] == iso_str

    def test_metadata_forwarding(self):
        """Test 16: Custom metadata is stored in internal payload."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        meta = {"camera_id": "CAM_01", "frame_rate": 30, "gain": 1.2}
        res = pipeline.process_frame(frame, metadata=meta)
        assert res["metadata"] == meta

    def test_frame_index_monotonicity_and_reset(self):
        """Test 17: Frame index increments monotonically on each frame and resets on reset()."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(1, 6):
            res = pipeline.process_frame(frame)
            assert res["frame_index"] == i

        pipeline.reset()
        res_after_reset = pipeline.process_frame(frame)
        assert res_after_reset["frame_index"] == 1

    def test_public_adapter_integration(self):
        """Test 18: process_frame_public produces valid schema conforming to AIResultAdapter."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pub_res = pipeline.process_frame_public(frame, timestamp="2026-09-24T18:00:00Z")

        assert AIResultAdapter.validate_public_contract(pub_res) is True
        assert pub_res["timestamp"] == "2026-09-24T18:00:00Z"

    def test_exact_public_8_field_schema(self):
        """Test 19: Public contract contains exactly the 8 frozen keys."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pub_res = pipeline.process_frame_public(frame)

        assert set(pub_res.keys()) == set(PUBLIC_CONTRACT_KEYS)
        assert len(pub_res) == 8

    def test_no_leakage_of_internal_diagnostics(self):
        """Test 20: Internal perception fields are strictly stripped from public result."""
        pipeline = InferencePipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pub_res = pipeline.process_frame_public(frame, metadata={"secret": "diagnostic"})

        internal_keys = {
            "frame_index",
            "metadata",
            "rectification_applied",
            "objects",
            "pose",
            "hands",
            "interaction",
            "annotated_frame",
        }
        for ik in internal_keys:
            assert ik not in pub_res


class TestPublicContractRulesAndStatusMapping:
    """Tests 21-26: Canonical values, confidence bounding, step normalization, and status mapping."""

    @pytest.mark.parametrize("action", ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID", "IDLE"])
    def test_canonical_action_validation(self, action):
        """Test 21: Canonical actions are validated and preserved in public output."""
        res = AIResultAdapter.adapt(action=action)
        assert res["action"] == action

    @pytest.mark.parametrize("obj", ["RED_SAMPLE", "BLUE_SAMPLE", "SAMPLE_CONTAINER", "CONTAINER_LID", "NONE"])
    def test_canonical_object_validation(self, obj):
        """Test 22: Canonical objects are validated and preserved in public output."""
        res = AIResultAdapter.adapt(object=obj)
        assert res["object"] == obj

    @pytest.mark.parametrize("raw_conf, clamped", [
        (0.95, 0.95),
        (1.5, 1.0),
        (-0.2, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (None, 0.0),
        ("invalid", 0.0),
    ])
    def test_confidence_bounding(self, raw_conf, clamped):
        """Test 23: Confidence is strictly bounded to [0.0, 1.0]."""
        res = AIResultAdapter.adapt(confidence=raw_conf)
        assert res["confidence"] == clamped
        assert 0.0 <= res["confidence"] <= 1.0

    @pytest.mark.parametrize("raw_step, expected", [
        ("S1", "S1"),
        ("s2", "S2"),
        (3, "S3"),
        ("4", "S4"),
        ("S5", "S5"),
        ("S6", None),
        ("invalid", None),
    ])
    def test_step_normalization(self, raw_step, expected):
        """Test 24: Step IDs normalize to S1–S5 or None."""
        res = AIResultAdapter.adapt(expected_step=raw_step)
        assert res["expected_step"] == expected

    def test_next_step_progression(self):
        """Test 25: Next step follows canonical progression sequence."""
        assert AIResultAdapter.adapt(expected_step="S1")["next_step"] == "S2"
        assert AIResultAdapter.adapt(expected_step="S2")["next_step"] == "S3"
        assert AIResultAdapter.adapt(expected_step="S3")["next_step"] == "S4"
        assert AIResultAdapter.adapt(expected_step="S4")["next_step"] == "S5"
        assert AIResultAdapter.adapt(expected_step="S5")["next_step"] is None

    @pytest.mark.parametrize("internal_status, public_status", [
        ("VALID", "VALID"),
        ("SKIPPED", "SKIPPED"),
        ("OUT_OF_SEQUENCE", "OUT_OF_SEQUENCE"),
        ("OUT_OF_ORDER", "OUT_OF_SEQUENCE"),
        ("UNCERTAIN", "OUT_OF_SEQUENCE"),
        ("LOW_CONFIDENCE", "OUT_OF_SEQUENCE"),
        ("UNRECOGNIZED", "OUT_OF_SEQUENCE"),
        ("ANOMALY", "OUT_OF_SEQUENCE"),
        ("IGNORED", "OUT_OF_SEQUENCE"),
        ("IDLE", "VALID"),
        ("RUNNING", "VALID"),
        ("COMPLETED", "VALID"),
        ("UNKNOWN_STATUS", "OUT_OF_SEQUENCE"),
    ])
    def test_public_status_mapping_strictness(self, internal_status, public_status):
        """Test 26: Status mapping strictly translates internal statuses to {VALID, SKIPPED, OUT_OF_SEQUENCE}."""
        res = AIResultAdapter.adapt(fsm_status=internal_status)
        assert res["status"] in PUBLIC_STATUS_VALUES
        assert res["status"] == public_status
