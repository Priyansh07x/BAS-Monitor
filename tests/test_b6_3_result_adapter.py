"""
test_b6_3_result_adapter.py — Gate B6.3 Public AI Result Adapter & Contract Compliance Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B
"""

import pytest
import numpy as np
from datetime import datetime

from backend.ai.result_adapter import (
    AIResultAdapter,
    CANONICAL_ACTIONS,
    CANONICAL_OBJECTS,
    LEGACY_ACTIONS,
    LEGACY_OBJECTS,
    PUBLIC_CONTRACT_KEYS,
    PUBLIC_STATUS_VALUES,
    normalize_action,
    normalize_object,
    normalize_confidence,
    normalize_timestamp,
    normalize_step_id,
    normalize_status,
)
from backend.ai.inference_pipeline import InferencePipeline


class TestPublicContractStructure:
    """Tests 1-3: Frozen 8-field schema integrity and validation."""

    def test_exact_8_fields_present(self):
        """Test 1: Adapter returns exact 8 frozen keys."""
        res = AIResultAdapter.adapt()
        assert set(res.keys()) == set(PUBLIC_CONTRACT_KEYS)
        assert len(res) == 8

    def test_no_extra_keys(self):
        """Test 2: Internal fields (e.g. pose, objects, interaction) do not leak into public contract."""
        internal = {
            "action": "PICK_RED",
            "confidence": 0.88,
            "objects": [{"label": "RED_SAMPLE", "confidence": 0.9}],
            "pose": [{"x": 0.5, "y": 0.5}],
            "hands": [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}],
            "interaction": {"state": "GRASPING"},
            "rectification_applied": True,
            "extra_field": "leak_test",
        }
        res = AIResultAdapter.adapt(internal)
        assert "objects" not in res
        assert "pose" not in res
        assert "hands" not in res
        assert "interaction" not in res
        assert "rectification_applied" not in res
        assert "extra_field" not in res
        assert set(res.keys()) == set(PUBLIC_CONTRACT_KEYS)

    def test_no_missing_keys_on_empty_input(self):
        """Test 3: Empty or None input yields all 8 keys with valid defaults."""
        res_none = AIResultAdapter.adapt(None)
        res_empty = AIResultAdapter.adapt({})
        for r in [res_none, res_empty]:
            for k in PUBLIC_CONTRACT_KEYS:
                assert k in r


class TestVocabularyValidation:
    """Tests 4-7: Action and object vocabulary compliance."""

    @pytest.mark.parametrize("action", ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID", "IDLE"])
    def test_canonical_actions_accepted(self, action):
        """Test 4: All canonical EXP-001 actions are accepted without modification."""
        assert normalize_action(action) == action
        res = AIResultAdapter.adapt(action=action)
        assert res["action"] == action

    @pytest.mark.parametrize("legacy", ["PICK_CONTAINER", "PIPETTE_TRANSFER", "INSERT_ANALYZER", "SEAL_CONTAINER", "INITIATE_SCAN", "UNKNOWN_ACTION"])
    def test_legacy_actions_rejected(self, legacy):
        """Test 5: Legacy actions are safely remapped to IDLE."""
        assert normalize_action(legacy) == "IDLE"
        res = AIResultAdapter.adapt(action=legacy)
        assert res["action"] == "IDLE"

    @pytest.mark.parametrize("obj", ["RED_SAMPLE", "BLUE_SAMPLE", "SAMPLE_CONTAINER", "CONTAINER_LID", "NONE"])
    def test_canonical_objects_accepted(self, obj):
        """Test 6: All canonical EXP-001 objects are accepted."""
        assert normalize_object(obj, action="IDLE") == obj
        res = AIResultAdapter.adapt(object=obj)
        assert res["object"] == obj

    @pytest.mark.parametrize("legacy_obj", ["SAMPLE_VIAL", "PIPETTE", "WELL_PLATE", "CENTRIFUGE_TUBE", "FORCEPS", "INCUBATOR_DOOR", "INVALID_OBJ"])
    def test_legacy_objects_rejected(self, legacy_obj):
        """Test 7: Legacy objects are safely remapped to canonical default or NONE."""
        norm = normalize_object(legacy_obj, action="IDLE")
        assert norm == "NONE"
        assert norm not in LEGACY_OBJECTS

        norm_pick_red = normalize_object(legacy_obj, action="PICK_RED")
        assert norm_pick_red == "RED_SAMPLE"


class TestConfidenceAndTimestampNormalization:
    """Tests 8-10: Confidence clamping and timestamp normalization."""

    @pytest.mark.parametrize("val, expected", [
        (0.85, 0.85),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.5, 1.0),
        (-0.3, 0.0),
        ("0.75", 0.75),
        ("invalid", 0.0),
        (None, 0.0),
    ])
    def test_confidence_normalization(self, val, expected):
        """Tests 8-9: Confidence values are clamped to [0.0, 1.0] and gracefully handle invalid types."""
        assert normalize_confidence(val) == expected

    def test_timestamp_preservation_and_conversion(self):
        """Test 10: Timestamp strings are preserved; epoch timestamps are converted to ISO-8601."""
        iso_str = "2026-09-24T18:00:00.000Z"
        res_iso = AIResultAdapter.adapt(timestamp=iso_str)
        assert res_iso["timestamp"] == iso_str

        # Epoch float (seconds)
        epoch_ts = 1774432800.0  # arbitrary valid epoch
        res_epoch = AIResultAdapter.adapt(timestamp=epoch_ts)
        assert isinstance(res_epoch["timestamp"], str)
        assert "T" in res_epoch["timestamp"]


class TestStepAndStatusMapping:
    """Tests 11-17: Step normalization, next step progression, and FSM status mapping."""

    @pytest.mark.parametrize("raw_step, expected", [
        ("S1", "S1"),
        ("s2", "S2"),
        (3, "S3"),
        ("4", "S4"),
        ("S5", "S5"),
        ("S6", None),
        ("invalid", None),
        (None, None),
    ])
    def test_step_id_normalization(self, raw_step, expected):
        """Test 11: Normalizes step IDs across strings, integers, and case variations."""
        assert normalize_step_id(raw_step) == expected

    def test_step_inference_from_action(self):
        """Tests 12-13: Action correctly infers detected_step, expected_step, and next_step."""
        res_s1 = AIResultAdapter.adapt(action="PICK_RED")
        assert res_s1["detected_step"] == "S1"
        assert res_s1["expected_step"] == "S1"
        assert res_s1["next_step"] == "S2"

        res_s5 = AIResultAdapter.adapt(action="CLOSE_LID")
        assert res_s5["detected_step"] == "S5"
        assert res_s5["next_step"] is None

    @pytest.mark.parametrize("status_in, status_out", [
        ("VALID", "VALID"),
        ("valid", "VALID"),
        ("SKIPPED", "SKIPPED"),
        ("skipped", "SKIPPED"),
        ("OUT_OF_SEQUENCE", "OUT_OF_SEQUENCE"),
        ("OUT_OF_ORDER", "OUT_OF_SEQUENCE"),
        ("UNCERTAIN", "OUT_OF_SEQUENCE"),
        ("LOW_CONFIDENCE", "OUT_OF_SEQUENCE"),
        ("UNRECOGNIZED", "OUT_OF_SEQUENCE"),
        ("ANOMALY", "OUT_OF_SEQUENCE"),
        ("IGNORED", "OUT_OF_SEQUENCE"),
        ("UNKNOWN_STATUS", "OUT_OF_SEQUENCE"),
    ])
    def test_status_mapping_strictness(self, status_in, status_out):
        """Tests 15-17: Status values map strictly to {VALID, SKIPPED, OUT_OF_SEQUENCE}."""
        assert normalize_status(status_in) == status_out
        res = AIResultAdapter.adapt(fsm_status=status_in)
        assert res["status"] in PUBLIC_STATUS_VALUES
        assert res["status"] == status_out


class TestResilienceAndPipelineIntegration:
    """Tests 18-20: Missing input resilience, pipeline method, and contract validator."""

    def test_missing_input_graceful_defaults(self):
        """Test 18: Handles completely empty / corrupted input dict gracefully."""
        res = AIResultAdapter.adapt({})
        assert AIResultAdapter.validate_public_contract(res) is True
        assert res["action"] == "IDLE"
        assert res["object"] == "NONE"
        assert res["confidence"] == 0.0
        assert res["status"] == "VALID"

    def test_inference_pipeline_process_frame_public(self):
        """Test 19: InferencePipeline.process_frame_public produces valid contract."""
        pipeline = InferencePipeline()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pub_res = pipeline.process_frame_public(dummy_frame, timestamp="2026-09-24T18:00:00Z")

        assert AIResultAdapter.validate_public_contract(pub_res) is True
        assert pub_res["timestamp"] == "2026-09-24T18:00:00Z"
        assert pub_res["action"] in CANONICAL_ACTIONS
        assert pub_res["object"] in CANONICAL_OBJECTS
        assert 0.0 <= pub_res["confidence"] <= 1.0
        assert pub_res["status"] in PUBLIC_STATUS_VALUES

    def test_contract_validator_accuracy(self):
        """Test 20: validate_public_contract distinguishes valid vs malformed payloads."""
        valid_payload = {
            "timestamp": "2026-09-24T18:00:00Z",
            "action": "PICK_RED",
            "object": "RED_SAMPLE",
            "confidence": 0.95,
            "expected_step": "S1",
            "detected_step": "S1",
            "status": "VALID",
            "next_step": "S2",
        }
        assert AIResultAdapter.validate_public_contract(valid_payload) is True

        # Missing key
        invalid_missing = dict(valid_payload)
        del invalid_missing["next_step"]
        assert AIResultAdapter.validate_public_contract(invalid_missing) is False

        # Extra key
        invalid_extra = dict(valid_payload, extra_key="leak")
        assert AIResultAdapter.validate_public_contract(invalid_extra) is False

        # Legacy action
        invalid_action = dict(valid_payload, action="PIPETTE_TRANSFER")
        assert AIResultAdapter.validate_public_contract(invalid_action) is False

        # Legacy object
        invalid_obj = dict(valid_payload, object="PIPETTE")
        assert AIResultAdapter.validate_public_contract(invalid_obj) is False

        # Out-of-bounds confidence
        invalid_conf = dict(valid_payload, confidence=1.5)
        assert AIResultAdapter.validate_public_contract(invalid_conf) is False

        # Invalid status
        invalid_status = dict(valid_payload, status="UNCERTAIN")
        assert AIResultAdapter.validate_public_contract(invalid_status) is False
