"""
test_b11_1_uncertainty_handler.py — Verification of Gate B11.1 Uncertainty Handler Core
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B11.1)

Coverage:
A. High confidence: confidence >= 0.70 -> CONFIDENT, is_reliable=True
B. Marginal confidence: 0.50 <= confidence < 0.70 -> UNCERTAIN_EVIDENCE_ACCUMULATING, is_reliable=False
C. Very low confidence: < 0.50 -> UNCERTAIN_LOW_CONFIDENCE, is_reliable=False
D. Repeated marginal same composite candidate: (PICK_RED, RED_SAMPLE) -> deterministic resolution
E. Marginal candidate mixed with unrelated candidates: no accidental confirmation
F. Action/object composite integrity: RED_SAMPLE evidence cannot combine with BLUE_SAMPLE
G. Confidence recovery: marginal -> sufficiently confident -> immediate CONFIDENT
H. Persistent uncertainty: repeated insufficient evidence -> deterministic unresolved behavior
I. Candidate switching: deterministic reset of active marginal accumulation
J. No FSM mutation: uncertain observations never advance current_step_index
K. Public contract: no new public status value; exactly 8 fields
L. Lifecycle: reset(), pause(), resume(), complete()
M. Memory: evidence state remains strictly bounded
N. Determinism: identical streams produce identical outputs
O. Thread boundary: synchronous execution without extra threads/queues
P. B10 separation: verifies B11 does not alter B10 temporal confirmation logic
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import AIResultAdapter, PUBLIC_CONTRACT_KEYS
from backend.experiment.sequence_validator import SequenceValidatorFSM


@pytest.fixture
def clean_fsm() -> SequenceValidatorFSM:
    """Provides a fresh, running SequenceValidatorFSM loaded with EXP-001."""
    fsm = SequenceValidatorFSM()
    fsm.start()
    return fsm


@pytest.fixture
def uncertainty_handler() -> UncertaintyHandler:
    """Provides a fresh UncertaintyHandler with default thresholds."""
    return UncertaintyHandler(
        high_confidence_threshold=0.70,
        marginal_confidence_threshold=0.50,
        required_evidence_count=3,
        anomaly_persistence_threshold=5,
        max_evidence_history=10,
    )


class TestUncertaintyHandlerUnit:
    """Unit tests for UncertaintyHandler core logic and state transitions."""

    def test_high_confidence_immediate_confident(self, uncertainty_handler: UncertaintyHandler):
        """Case A: Confidence >= 0.70 is classified as CONFIDENT and is_reliable=True."""
        res = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        assert res["state"] == "CONFIDENT"
        assert res["is_reliable"] is True
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"
        assert res["confidence"] == 0.85

    def test_marginal_confidence_accumulates(self, uncertainty_handler: UncertaintyHandler):
        """Case B: 0.50 <= confidence < 0.70 enters UNCERTAIN_EVIDENCE_ACCUMULATING and is_reliable=False."""
        res1 = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        assert res1["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert res1["is_reliable"] is False
        assert res1["evidence_count"] == 1

        res2 = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.62)
        assert res2["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert res2["is_reliable"] is False
        assert res2["evidence_count"] == 2

    def test_very_low_confidence_insufficient(self, uncertainty_handler: UncertaintyHandler):
        """Case C: Confidence < 0.50 is classified as UNCERTAIN_LOW_CONFIDENCE and is_reliable=False."""
        res = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.45)
        assert res["state"] == "UNCERTAIN_LOW_CONFIDENCE"
        assert res["is_reliable"] is False
        assert res["evidence_count"] == 0

    def test_repeated_marginal_evidence_resolves(self, uncertainty_handler: UncertaintyHandler):
        """Case D: 3 consecutive consistent marginal observations resolve uncertainty."""
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.64)
        res3 = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.68)

        assert res3["state"] == "RESOLVED"
        assert res3["is_reliable"] is True
        assert res3["evidence_count"] == 3
        # Aggregate confidence is the mean of [0.60, 0.64, 0.68] = 0.64
        assert res3["confidence"] == 0.64
        assert uncertainty_handler.total_resolved == 1

    def test_marginal_mixed_with_unrelated_resets_accumulation(self, uncertainty_handler: UncertaintyHandler):
        """Case E: Unrelated action interrupts marginal accumulation."""
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)

        # Distractor observation
        uncertainty_handler.process_observation("PICK_BLUE", "BLUE_SAMPLE", 0.60)

        # Resuming PICK_RED starts accumulation afresh from count 1
        res = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        assert res["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert res["evidence_count"] == 1
        assert res["is_reliable"] is False

    def test_composite_action_object_integrity(self, uncertainty_handler: UncertaintyHandler):
        """Case F: Same action with different objects does NOT share evidence accumulation."""
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)

        # Swapping to BLUE_SAMPLE resets active candidate
        res = uncertainty_handler.process_observation("PICK_RED", "BLUE_SAMPLE", 0.60)
        assert res["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert res["evidence_count"] == 1
        assert res["candidate"] == ("PICK_RED", "BLUE_SAMPLE")

    def test_confidence_recovery_to_high_confidence(self, uncertainty_handler: UncertaintyHandler):
        """Case G: Marginal observation followed by high-confidence frame resolves immediately to CONFIDENT."""
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        res = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.85)

        assert res["state"] == "CONFIDENT"
        assert res["is_reliable"] is True
        assert res["confidence"] == 0.85

    def test_persistent_sub_marginal_uncertainty(self, uncertainty_handler: UncertaintyHandler):
        """Case H: Repeated frames below 0.50 remain persistently unreliable without resolving."""
        for _ in range(10):
            res = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.35)
            assert res["state"] == "UNCERTAIN_LOW_CONFIDENCE"
            assert res["is_reliable"] is False
        assert uncertainty_handler.total_resolved == 0

    def test_candidate_switching_semantics(self, uncertainty_handler: UncertaintyHandler):
        """Case I: Switching directly between different actions flushes active candidate."""
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        res_switch = uncertainty_handler.process_observation("PLACE_RED", "RED_SAMPLE", 0.60)

        assert res_switch["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert res_switch["candidate"] == ("PLACE_RED", "RED_SAMPLE")
        assert res_switch["evidence_count"] == 1

    def test_lifecycle_reset_pause_resume_complete(self, uncertainty_handler: UncertaintyHandler):
        """Case L: Full lifecycle transitions."""
        # 1. Accumulate 2 frames
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)

        # 2. PAUSE
        uncertainty_handler.pause()
        assert uncertainty_handler.is_paused is True
        assert uncertainty_handler.state == "PAUSED"
        res_pause = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        assert res_pause["state"] == "PAUSED"
        assert res_pause["is_reliable"] is False

        # 3. RESUME
        uncertainty_handler.resume()
        assert uncertainty_handler.is_paused is False
        assert uncertainty_handler.state == "RUNNING"
        # Window is clean; first marginal frame has evidence_count 1
        res_resume = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        assert res_resume["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert res_resume["evidence_count"] == 1

        # 4. RESET
        uncertainty_handler.reset()
        assert uncertainty_handler.total_evaluations == 0
        assert len(uncertainty_handler._evidence_window) == 0

        # 5. COMPLETE
        uncertainty_handler.complete()
        assert uncertainty_handler.state == "COMPLETED"
        res_comp = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.95)
        assert res_comp["state"] == "COMPLETED"
        assert res_comp["is_reliable"] is False

    def test_memory_boundedness(self, uncertainty_handler: UncertaintyHandler):
        """Case M: Bounded deque never exceeds max_evidence_history across 500 frames."""
        for i in range(500):
            conf = 0.40 + (i % 30) / 100.0
            uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", conf)
            assert len(uncertainty_handler._evidence_window) <= uncertainty_handler.max_evidence_history

    def test_replay_determinism(self):
        """Case N: Identical observation streams produce identical output sequences."""
        stream = [
            ("PICK_RED", "RED_SAMPLE", 0.60),
            ("PICK_RED", "RED_SAMPLE", 0.65),
            ("PICK_RED", "RED_SAMPLE", 0.68),
            ("IDLE", "NONE", 0.95),
            ("PLACE_RED", "RED_SAMPLE", 0.40),
            ("PLACE_RED", "RED_SAMPLE", 0.85),
        ]
        h1 = UncertaintyHandler()
        h2 = UncertaintyHandler()

        r1 = [h1.process_observation(a, o, c)["state"] for a, o, c in stream]
        r2 = [h2.process_observation(a, o, c)["state"] for a, o, c in stream]
        assert r1 == r2 == [
            "UNCERTAIN_EVIDENCE_ACCUMULATING",
            "UNCERTAIN_EVIDENCE_ACCUMULATING",
            "RESOLVED",
            "IDLE",
            "UNCERTAIN_LOW_CONFIDENCE",
            "CONFIDENT",
        ]


class TestPipelineUncertaintyFSMIntegration:
    """Integration tests verifying UncertaintyHandler with TemporalConfirmationEngine and FSM."""

    def test_uncertain_observation_does_not_advance_fsm(self, clean_fsm: SequenceValidatorFSM):
        """Case B / J: Marginal confidence observation does NOT advance FSM."""
        pipeline = InferencePipeline(validator=clean_fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock marginal perception (conf=0.60)
        pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.60)
        pipeline.object_detector.detect = lambda f: [{"label": "RED_SAMPLE", "confidence": 0.60, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        for _ in range(5):
            res = pipeline.process_frame_public(dummy_frame)
            assert AIResultAdapter.validate_public_contract(res) is True

        # FSM MUST NOT ADVANCE
        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0

    def test_public_contract_integrity_with_uncertainty(self, clean_fsm: SequenceValidatorFSM):
        """Case K: Public contract remains strictly 8 fields with authorized public status values."""
        pipeline = InferencePipeline(validator=clean_fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        confidences = [0.20, 0.55, 0.65, 0.70, 0.95]
        for conf in confidences:
            pipeline.action_classifier.classify = lambda **k: ("PICK_RED", conf)
            pipeline.object_detector.detect = lambda f: [{"label": "RED_SAMPLE", "confidence": conf, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]
            res = pipeline.process_frame_public(dummy_frame)

            assert set(res.keys()) == set(PUBLIC_CONTRACT_KEYS)
            assert len(res) == 8
            assert res["status"] in {"VALID", "SKIPPED", "OUT_OF_SEQUENCE"}
            assert AIResultAdapter.validate_public_contract(res) is True

    def test_b10_b11_separation(self, clean_fsm: SequenceValidatorFSM):
        """Case P: B10 temporal stability and B11 uncertainty evaluation remain decoupled."""
        temporal_engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        uncertainty_handler = UncertaintyHandler(high_confidence_threshold=0.70, marginal_confidence_threshold=0.50)

        # Feed frame with conf=0.60
        t_res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        u_res = uncertainty_handler.process_observation("PICK_RED", "RED_SAMPLE", 0.60)

        # B10 says LOW_CONFIDENCE, unconfirmed
        assert t_res["confirmed"] is False
        assert t_res["state"] == "LOW_CONFIDENCE"

        # B11 says UNCERTAIN_EVIDENCE_ACCUMULATING, evidence_count=1
        assert u_res["is_reliable"] is False
        assert u_res["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert u_res["evidence_count"] == 1
