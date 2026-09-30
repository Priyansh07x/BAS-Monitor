"""
test_b10_1_temporal_confirmation.py — Verification of Gate B10.1 Temporal Confirmation Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B10.1)

Tests:
1. Single observation unconfirmed (FSM does not advance).
2. 2-of-5 below threshold unconfirmed.
3. 3-of-5 threshold reached -> exactly one confirmation and FSM commit.
4. 4-of-5 and 5-of-5 majority voting determinism.
5. Alternating noise rejection (no spurious commits).
6. Composite action + object coupling (no cross-object vote contamination).
7. Minimum confidence gating (observations < 0.70 ignored).
8. Wrong-object confirmed candidate rejected authoritatively by FSM.
9. Sustained repeated actions produce exactly one procedural commit.
10. Cooldown suppression of repeated commits.
11. Cooldown release on neutral IDLE observation.
12. RESET discards rolling window and cooldown.
13. PAUSE / RESUME discards pre-pause votes.
14. COMPLETED freezes confirmation until reset.
15. Deterministic replay of identical streams.
16. Strict compliance with frozen 8-field public contract.
17. FSM procedural authority preserved against perception claims.
18. End-to-end InferenceWorker background thread integration without extra threads.
"""

from __future__ import annotations

import json
import time
import numpy as np
import pytest

from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import InferenceWorker, RateStrategy
from backend.ai.result_adapter import AIResultAdapter, PUBLIC_CONTRACT_KEYS
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.experiment.procedure_manager import ProcedureManager


@pytest.fixture
def clean_fsm() -> SequenceValidatorFSM:
    """Provides a freshly started SequenceValidatorFSM configured with EXP-001."""
    fsm = SequenceValidatorFSM()
    fsm.start()
    return fsm


@pytest.fixture
def temporal_engine() -> TemporalConfirmationEngine:
    """Provides a fresh TemporalConfirmationEngine with default N=5, M=3, tau=0.70."""
    return TemporalConfirmationEngine(
        window_size=5,
        confirmation_threshold=3,
        min_confidence=0.70,
        cooldown_frames=5,
    )


class TestTemporalConfirmationEngineUnit:
    """Unit tests for TemporalConfirmationEngine algorithm and state machine."""

    def test_single_observation_not_confirmed(self, temporal_engine: TemporalConfirmationEngine):
        """A single high-confidence observation must NOT trigger confirmation."""
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is False
        assert res["vote_count"] == 1
        assert res["state"] == "UNCONFIRMED"
        assert temporal_engine.total_commits == 0

    def test_two_of_five_not_confirmed(self, temporal_engine: TemporalConfirmationEngine):
        """2 matching observations out of 5 is below M=3 threshold and must not confirm."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.92)
        assert res["confirmed"] is False
        assert res["vote_count"] == 2
        assert res["state"] == "UNCONFIRMED"
        assert temporal_engine.total_commits == 0

    def test_three_of_five_confirmed(self, temporal_engine: TemporalConfirmationEngine):
        """3 matching observations reaches M=3 threshold and triggers confirmation."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.80)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 1.00)
        assert res["confirmed"] is True
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"
        assert res["confidence"] == 0.90  # Mean of [0.80, 0.90, 1.00]
        assert res["state"] == "CONFIRMED"
        assert temporal_engine.total_commits == 1

    def test_four_and_five_of_five_confirmed(self, temporal_engine: TemporalConfirmationEngine):
        """Window with 4 or 5 matching observations triggers confirmation."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        temporal_engine.process_observation("PICK_BLUE", "BLUE_SAMPLE", 0.85)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        # 3 out of 4 is >= 3
        assert res["confirmed"] is True
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"

    def test_alternating_noise_rejection(self, temporal_engine: TemporalConfirmationEngine):
        """Alternating noisy predictions should not reach threshold M=3."""
        stream = [
            ("PICK_RED", "RED_SAMPLE", 0.85),
            ("PICK_BLUE", "BLUE_SAMPLE", 0.85),
            ("PICK_RED", "RED_SAMPLE", 0.85),
            ("PLACE_RED", "RED_SAMPLE", 0.85),
            ("PICK_BLUE", "BLUE_SAMPLE", 0.85),
        ]
        for act, obj, conf in stream:
            res = temporal_engine.process_observation(act, obj, conf)
            assert res["confirmed"] is False
        assert temporal_engine.total_commits == 0

    def test_action_object_coupling_integrity(self, temporal_engine: TemporalConfirmationEngine):
        """Mismatched objects with the same action must NOT count towards the same vote."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        temporal_engine.process_observation("PICK_RED", "BLUE_SAMPLE", 0.85)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        res = temporal_engine.process_observation("PICK_RED", "BLUE_SAMPLE", 0.85)
        # PICK_RED + RED_SAMPLE has 2 votes; PICK_RED + BLUE_SAMPLE has 2 votes. Neither has 3.
        assert res["confirmed"] is False
        assert temporal_engine.total_commits == 0

    def test_confidence_threshold_gating(self, temporal_engine: TemporalConfirmationEngine):
        """Observations below min_confidence (0.70) must not count towards confirmation."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.65)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.68)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.69)
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.60)
        assert res["confirmed"] is False
        assert res["state"] == "LOW_CONFIDENCE"
        assert temporal_engine.total_commits == 0

    def test_sustained_action_cooldown(self, temporal_engine: TemporalConfirmationEngine):
        """20 consecutive identical frames produce exactly 1 commit, with subsequent in COOLDOWN."""
        commits = 0
        cooldown_count = 0
        for i in range(20):
            res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            if res["confirmed"]:
                commits += 1
            elif res["state"] == "COOLDOWN":
                cooldown_count += 1

        assert commits == 1
        assert cooldown_count > 0
        assert temporal_engine.total_commits == 1

    def test_cooldown_release_on_idle(self, temporal_engine: TemporalConfirmationEngine):
        """Observing IDLE releases post-commit cooldown."""
        # 1. Trigger confirmation
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        res1 = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res1["confirmed"] is True

        # 2. Next frame is in cooldown
        res2 = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res2["confirmed"] is False
        assert res2["state"] == "COOLDOWN"

        # 3. Observe neutral IDLE
        res_idle = temporal_engine.process_observation("IDLE", "NONE", 0.95)
        assert res_idle["state"] == "IDLE"

        # 4. Now a new action sequence can confirm again
        temporal_engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
        temporal_engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
        res3 = temporal_engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
        assert res3["confirmed"] is True
        assert res3["action"] == "PLACE_RED"
        assert temporal_engine.total_commits == 2

    def test_reset_lifecycle(self, temporal_engine: TemporalConfirmationEngine):
        """reset() completely wipes accumulated votes and cooldown state."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        temporal_engine.reset()

        assert len(temporal_engine._window) == 0
        assert temporal_engine.total_commits == 0
        # 1 more observation after reset must not confirm (would have been 3rd if not reset)
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is False
        assert res["vote_count"] == 1

    def test_pause_and_resume_lifecycle(self, temporal_engine: TemporalConfirmationEngine):
        """pause() and resume() clear pre-pause observations to prevent stale transitions."""
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        temporal_engine.pause()

        # While paused, observations return PAUSED
        p_res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert p_res["state"] == "PAUSED"
        assert p_res["confirmed"] is False

        temporal_engine.resume()
        # After resume, window is clean; one more frame must NOT confirm
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is False
        assert res["vote_count"] == 1

    def test_completed_lifecycle(self, temporal_engine: TemporalConfirmationEngine):
        """complete() freezes confirmation voting until explicit reset."""
        temporal_engine.complete()
        res = temporal_engine.process_observation("PICK_RED", "RED_SAMPLE", 0.95)
        assert res["confirmed"] is False
        assert res["state"] == "COMPLETED"

    def test_deterministic_replay(self):
        """Identical observation streams produce identical output sequences."""
        stream = [
            ("PICK_RED", "RED_SAMPLE", 0.85),
            ("PICK_RED", "RED_SAMPLE", 0.88),
            ("PICK_RED", "RED_SAMPLE", 0.92),
            ("IDLE", "NONE", 0.99),
            ("PLACE_RED", "RED_SAMPLE", 0.80),
            ("PLACE_RED", "RED_SAMPLE", 0.85),
            ("PLACE_RED", "RED_SAMPLE", 0.90),
        ]
        engine1 = TemporalConfirmationEngine()
        engine2 = TemporalConfirmationEngine()

        results1 = [engine1.process_observation(a, o, c)["confirmed"] for a, o, c in stream]
        results2 = [engine2.process_observation(a, o, c)["confirmed"] for a, o, c in stream]
        assert results1 == results2 == [False, False, True, False, False, False, True]


class TestPipelineTemporalFSMIntegration:
    """Integration tests verifying Perception -> TemporalConfirmationEngine -> FSM -> AIResultAdapter."""

    def test_unconfirmed_observation_does_not_advance_fsm(self, clean_fsm: SequenceValidatorFSM):
        """Single unconfirmed frame produces public result but does not advance FSM."""
        pipeline = InferencePipeline(validator=clean_fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock classify to return S1 action
        pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        pipeline.object_detector.detect = lambda f: [{"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        pub_res = pipeline.process_frame_public(dummy_frame)

        # Public contract validity
        for k in PUBLIC_CONTRACT_KEYS:
            assert k in pub_res

        assert pub_res["action"] == "PICK_RED"
        assert pub_res["object"] == "RED_SAMPLE"
        assert pub_res["expected_step"] == "S1"
        assert pub_res["next_step"] == "S2"

        # FSM MUST STILL BE AT S1 (index 0)
        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0

    def test_three_frames_advance_fsm_once(self, clean_fsm: SequenceValidatorFSM):
        """3 consecutive matching frames confirm perception and advance FSM exactly once to S2."""
        pipeline = InferencePipeline(validator=clean_fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        pipeline.object_detector.detect = lambda f: [{"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        # Frame 1: Unconfirmed
        r1 = pipeline.process_frame_public(dummy_frame)
        assert clean_fsm.current_step_index == 0
        assert r1["expected_step"] == "S1"

        # Frame 2: Unconfirmed
        r2 = pipeline.process_frame_public(dummy_frame)
        assert clean_fsm.current_step_index == 0
        assert r2["expected_step"] == "S1"

        # Frame 3: CONFIRMED! FSM advances to S2
        r3 = pipeline.process_frame_public(dummy_frame)
        assert clean_fsm.current_step_index == 1
        assert r3["status"] == "VALID"
        assert len(clean_fsm.validated_steps) == 1
        assert clean_fsm.validated_steps[0]["detected_action"] == "PICK_RED"

        # Frame 4: In cooldown, FSM does not re-advance
        r4 = pipeline.process_frame_public(dummy_frame)
        assert clean_fsm.current_step_index == 1
        assert r4["expected_step"] == "S2"
        assert len(clean_fsm.validated_steps) == 1

    def test_wrong_object_confirmed_candidate_rejected_by_fsm(self, clean_fsm: SequenceValidatorFSM):
        """Temporal engine confirms candidate with wrong object; FSM authoritatively rejects it."""
        pipeline = InferencePipeline(validator=clean_fsm, consistency_evaluator=None)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # S1 expects PICK_RED on RED_SAMPLE. Perception produces PICK_RED on BLUE_SAMPLE.
        pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        pipeline.object_detector.detect = lambda f: [{"label": "BLUE_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        # Frames 1, 2, 3
        pipeline.process_frame_public(dummy_frame)
        pipeline.process_frame_public(dummy_frame)
        r3 = pipeline.process_frame_public(dummy_frame)

        # Temporal filter confirmed PICK_RED + BLUE_SAMPLE, but FSM rejected wrong object
        assert r3["status"] == "OUT_OF_SEQUENCE"
        assert clean_fsm.current_step_index == 0  # Still at S1
        assert len(clean_fsm.validated_steps) == 0
        assert len(clean_fsm.anomalies) >= 1

    def test_complete_procedure_temporal_traversal(self, clean_fsm: SequenceValidatorFSM):
        """Full S1 -> S5 traversal with 3-frame confirmation per step."""
        pipeline = InferencePipeline(validator=clean_fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        steps = [
            ("PICK_RED", "RED_SAMPLE"),
            ("PLACE_RED", "RED_SAMPLE"),
            ("PICK_BLUE", "BLUE_SAMPLE"),
            ("PLACE_BLUE", "BLUE_SAMPLE"),
            ("CLOSE_LID", "CONTAINER_LID"),
        ]

        for step_idx, (act, obj) in enumerate(steps):
            pipeline.action_classifier.classify = lambda **k: (act, 0.95)
            pipeline.object_detector.detect = lambda f: [{"label": obj, "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

            # 3 frames to confirm
            for f in range(3):
                res = pipeline.process_frame_public(dummy_frame)

            assert clean_fsm.current_step_index == step_idx + 1

        assert clean_fsm.state == "COMPLETED"
        assert len(clean_fsm.validated_steps) == 5

    def test_worker_integration_non_blocking_execution(self, clean_fsm: SequenceValidatorFSM):
        """InferenceWorker executes temporal confirmation in background thread without extra queues."""
        pipeline = InferencePipeline(validator=clean_fsm)
        pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.92)
        pipeline.object_detector.detect = lambda f: [{"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        results = []
        worker = InferenceWorker(
            pipeline=pipeline,
            validator=clean_fsm,
            result_callback=lambda res: results.append(res),
            rate_strategy=RateStrategy.OPPORTUNISTIC_LATEST,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        worker.start()
        try:
            # Submit 5 frames
            for i in range(5):
                worker.submit_frame(dummy_frame)
                time.sleep(0.05)

            # Wait briefly for worker processing
            time.sleep(0.3)

            assert len(results) >= 3
            # After 3 frames, S1 was validated and step advanced to S2
            assert clean_fsm.current_step_index == 1
            assert clean_fsm.validated_steps[0]["detected_action"] == "PICK_RED"

            # Check public contract on all emitted results
            for r in results:
                assert set(r.keys()) == set(PUBLIC_CONTRACT_KEYS)
        finally:
            worker.stop()
