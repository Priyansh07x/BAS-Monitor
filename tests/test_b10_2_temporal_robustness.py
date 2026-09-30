"""
test_b10_2_temporal_robustness.py — Verification & Robustness of Temporal Confirmation Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B10.2.1)

Covers:
1. Parameter Sensitivity Matrix:
   - Configurable (N, M) combinations: (3, 2), (5, 3), (7, 4), (10, 6)
   - Invariant: 2 <= M <= N
   - Invalid configurations rejected with ValueError (N < 1, M < 1, M > N)
   - Fewer than M matching candidates does not confirm; exactly M confirms; more than M confirms
   - Unrelated candidates do not contribute votes; composite (action, object) identity intact
2. Confidence Boundaries:
   - Exact boundary precision around tau = 0.70 (0.69, 0.6999, 0.70, 0.7000, 0.71)
   - Sub-threshold observations cannot pollute valid vote count
3. Cooldown Boundaries:
   - Continuous 20+ identical observations -> exactly one procedural commit
   - Sustained candidate while cooldown active -> no second commit
   - Neutral release on IDLE/NONE
   - Different candidate switching releases lock
   - Reset completely clears cooldown
   - Lifecycle: START, PAUSE, RESUME, RESET
4. Rapid Non-Stationary Noise:
   - Rapid action flicker (PICK_RED / PICK_BLUE / PLACE_RED / PICK_RED / PICK_BLUE)
   - Cross-object separation (PICK_RED + RED_SAMPLE vs PICK_RED + BLUE_SAMPLE)
   - Valid candidate bursts separated by unrelated candidates
   - Sub-threshold bursts that never reach M
   - Long noisy streams with occasional valid M-of-N clusters
   - Deterministic replay
5. Multi-Step Procedure Noise:
   - Full EXP-001 (S1–S5) traversal with interleaved noise bursts
   - Noise before S1, between S1/S2, between S2/S3, around S4/S5
   - Wrong-object temporal candidate
   - Future-step candidate (skipped step)
   - Repeated previous-step candidate
   - FSM procedural authority preserved
6. Confirmation Latency / Frame Count:
   - Deterministic frame-count latency (M frames) across all (N, M) configurations
7. Rate Interaction:
   - Boundedness and determinism under 10 FPS, 15 FPS, and 30 FPS producer rate matching
8. Memory Boundedness:
   - Window size never exceeds configured N (1000+ frames)
   - Reset releases old candidate history
9. Public Contract:
   - Exact 8-field public AI contract on all results with zero contract leakage
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import InferenceWorker, RateStrategy
from backend.ai.result_adapter import AIResultAdapter, PUBLIC_CONTRACT_KEYS
from backend.experiment.sequence_validator import SequenceValidatorFSM


# =============================================================================
# 1. PARAMETER SENSITIVITY MATRIX & INVARIANTS
# =============================================================================

class TestParameterSensitivityMatrix:
    """Validates configurable (N, M) parameter combinations, invariants, and invalid configuration rejection."""

    @pytest.mark.parametrize(
        "n, m",
        [
            (3, 2),
            (5, 3),
            (7, 4),
            (10, 6),
        ],
    )
    def test_parameterized_m_of_n_confirmation_threshold(self, n: int, m: int):
        """Feeding M-1 candidates does not confirm; exactly Mth candidate confirms."""
        engine = TemporalConfirmationEngine(window_size=n, confirmation_threshold=m, min_confidence=0.70)

        # Feed M-1 matching observations
        for i in range(m - 1):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
            assert res["confirmed"] is False
            assert res["vote_count"] == i + 1
            assert res["state"] == "UNCONFIRMED"
            assert engine.total_commits == 0

        # Feed Mth matching observation -> triggers confirmation!
        res_m = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        assert res_m["confirmed"] is True
        assert res_m["vote_count"] == m
        assert res_m["state"] == "CONFIRMED"
        assert engine.total_commits == 1

    @pytest.mark.parametrize(
        "n, m",
        [
            (3, 2),
            (5, 3),
            (7, 4),
            (10, 6),
        ],
    )
    def test_more_than_m_candidates_behavior(self, n: int, m: int):
        """Feeding more than M candidates confirms at M, and subsequent identical frames enter cooldown."""
        engine = TemporalConfirmationEngine(window_size=n, confirmation_threshold=m, min_confidence=0.70)

        # Feed M matching observations -> confirms at M
        for _ in range(m):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)

        assert res["confirmed"] is True
        assert engine.total_commits == 1

        # Feed additional frames beyond M -> in cooldown, no duplicate commit
        for _ in range(3):
            res_extra = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            assert res_extra["confirmed"] is False
            assert res_extra["state"] == "COOLDOWN"

        assert engine.total_commits == 1

    @pytest.mark.parametrize(
        "invalid_n, invalid_m",
        [
            (0, 2),    # N < 1
            (-5, 2),   # N negative
            (5, 0),    # M < 1
            (5, -1),   # M negative
            (5, 6),    # M > N
            (3, 10),   # M > N
        ],
    )
    def test_invalid_parameter_configuration_rejection(self, invalid_n: int, invalid_m: int):
        """Invalid (N, M) configurations must be rejected deterministically with ValueError."""
        with pytest.raises(ValueError):
            TemporalConfirmationEngine(window_size=invalid_n, confirmation_threshold=invalid_m)

    def test_unrelated_candidates_do_not_contribute_votes(self):
        """In N=7, M=4 engine, unrelated actions cannot combine to reach M."""
        engine = TemporalConfirmationEngine(window_size=7, confirmation_threshold=4, min_confidence=0.70)

        stream = [
            ("PICK_RED", "RED_SAMPLE", 0.90),
            ("PICK_BLUE", "BLUE_SAMPLE", 0.90),
            ("PLACE_RED", "RED_SAMPLE", 0.90),
            ("PICK_RED", "RED_SAMPLE", 0.90),
            ("PICK_BLUE", "BLUE_SAMPLE", 0.90),
            ("CLOSE_LID", "CONTAINER_LID", 0.90),
            ("PICK_RED", "RED_SAMPLE", 0.90),
        ]
        # In 7 frames: PICK_RED has 3 votes, PICK_BLUE has 2, PLACE_RED has 1, CLOSE_LID has 1. None reaches 4.
        for act, obj, conf in stream:
            res = engine.process_observation(act, obj, conf)
            assert res["confirmed"] is False
        assert engine.total_commits == 0

    def test_composite_action_object_identity_integrity(self):
        """Composite (action, object) identity remains intact; different objects with same action do not share votes."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # 2 of PICK_RED + RED_SAMPLE
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.85)

        # 2 of PICK_RED + BLUE_SAMPLE
        engine.process_observation("PICK_RED", "BLUE_SAMPLE", 0.85)
        res = engine.process_observation("PICK_RED", "BLUE_SAMPLE", 0.85)

        # Neither composite candidate reached threshold 3
        assert res["confirmed"] is False
        assert engine.total_commits == 0


# =============================================================================
# 2. CONFIDENCE THRESHOLD BOUNDARY TESTS
# =============================================================================

class TestConfidenceBoundaries:
    """Verifies precision and behavior around the exact 0.70 confidence boundary."""

    @pytest.mark.parametrize(
        "sub_threshold_conf",
        [0.0, 0.50, 0.65, 0.69, 0.6999],
    )
    def test_sub_threshold_confidence_does_not_count(self, sub_threshold_conf: float):
        """Observations below 0.70 are categorized as LOW_CONFIDENCE and ignored in voting."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        for _ in range(5):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", sub_threshold_conf)
            assert res["confirmed"] is False
            assert res["state"] == "LOW_CONFIDENCE"
        assert engine.total_commits == 0

    @pytest.mark.parametrize(
        "qualifying_conf",
        [0.70, 0.7000, 0.7001, 0.71, 0.85, 1.0],
    )
    def test_qualifying_confidence_reaches_confirmation(self, qualifying_conf: float):
        """Observations >= 0.70 count towards confirmation."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        for _ in range(2):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", qualifying_conf)
            assert res["confirmed"] is False

        res3 = engine.process_observation("PICK_RED", "RED_SAMPLE", qualifying_conf)
        assert res3["confirmed"] is True
        assert res3["state"] == "CONFIRMED"
        assert engine.total_commits == 1

    def test_low_confidence_cannot_pollute_valid_vote_count(self):
        """Low-confidence observations cannot pollute valid vote count."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # 2 valid votes
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)

        # 4 low-confidence frames -> push old votes out of window
        for _ in range(4):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.50)

        # 1 valid vote -> window now has only 1 valid vote, so count is 1 (< 3)
        res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is False
        assert res["vote_count"] == 1


# =============================================================================
# 3. COOLDOWN & DEBOUNCE BOUNDARIES
# =============================================================================

class TestCooldownBoundaries:
    """Verifies post-commit cooldown, continuous gesture suppression, neutral release, and lifecycle."""

    def test_twenty_five_continuous_frames_single_commit(self):
        """25 continuous frames of identical action produce exactly 1 commit."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        commits = 0
        cooldown_responses = 0

        for _ in range(25):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            if res["confirmed"]:
                commits += 1
            elif res["state"] == "COOLDOWN":
                cooldown_responses += 1

        assert commits == 1
        assert cooldown_responses == 22
        assert engine.total_commits == 1

    def test_sustained_candidate_in_cooldown_no_second_commit(self):
        """Candidate remains identical while cooldown active -> no second commit."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # Trigger first commit
        for _ in range(3):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert engine.total_commits == 1

        # Feed 10 more identical frames
        for _ in range(10):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            assert res["confirmed"] is False
            assert res["state"] == "COOLDOWN"

        assert engine.total_commits == 1

    def test_neutral_idle_releases_cooldown(self):
        """Observing IDLE releases candidate cooldown, permitting new confirmation."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # Confirm step
        for _ in range(3):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert engine.total_commits == 1

        # While holding PICK_RED -> COOLDOWN
        res_cd = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res_cd["state"] == "COOLDOWN"

        # Operator returns to neutral IDLE
        res_idle = engine.process_observation("IDLE", "NONE", 0.99)
        assert res_idle["state"] == "IDLE"

        # Operator initiates next step (PLACE_RED) -> confirms
        for _ in range(2):
            engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
        res_place = engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
        assert res_place["confirmed"] is True
        assert res_place["action"] == "PLACE_RED"
        assert engine.total_commits == 2

    def test_candidate_switching_releases_previous_lock(self):
        """Transitioning directly to a new action candidate clears previous lock."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # Confirm PICK_RED
        for _ in range(3):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert engine.total_commits == 1

        # Direct transition to PLACE_RED
        for _ in range(2):
            res = engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
            assert res["confirmed"] is False
            assert res["state"] == "UNCONFIRMED"

        res_place = engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
        assert res_place["confirmed"] is True
        assert res_place["action"] == "PLACE_RED"
        assert engine.total_commits == 2

    def test_reset_completely_clears_cooldown_and_state(self):
        """reset() completely wipes accumulated votes and cooldown state."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # Trigger commit
        for _ in range(3):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert engine.total_commits == 1
        assert engine.last_committed_candidate == ("PICK_RED", "RED_SAMPLE")

        # Explicit reset
        engine.reset()

        assert len(engine._window) == 0
        assert engine.last_committed_candidate is None
        assert engine.total_commits == 0
        assert engine.total_observations == 0

        # After reset, PICK_RED can be confirmed again from fresh state
        for _ in range(2):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is True
        assert engine.total_commits == 1

    def test_lifecycle_start_pause_resume_reset(self):
        """Lifecycle transitions: START -> PAUSE -> RESUME -> RESET."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        assert engine.state == "RUNNING"

        # Accumulate 2 votes
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)

        # PAUSE
        engine.pause()
        assert engine.is_paused is True
        assert engine.state == "PAUSED"
        assert len(engine._window) == 0

        # Observations during PAUSE return PAUSED
        p_res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert p_res["state"] == "PAUSED"
        assert p_res["confirmed"] is False

        # RESUME
        engine.resume()
        assert engine.is_paused is False
        assert engine.state == "RUNNING"

        # Window is clean after resume; 1 frame does not confirm
        res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is False
        assert res["vote_count"] == 1


# =============================================================================
# 4. RAPID NON-STATIONARY NOISE SEQUENCES
# =============================================================================

class TestRapidNonStationaryNoise:
    """Verifies noise rejection and separation of complex non-stationary stream patterns."""

    def test_rapid_action_flicker_rejection(self):
        """Case A: Rapid alternating actions (PICK_RED / PICK_BLUE / PLACE_RED / PICK_RED / PICK_BLUE)."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        flicker_stream = [
            ("PICK_RED", "RED_SAMPLE", 0.95),
            ("PICK_BLUE", "BLUE_SAMPLE", 0.95),
            ("PLACE_RED", "RED_SAMPLE", 0.95),
            ("PICK_RED", "RED_SAMPLE", 0.95),
            ("PICK_BLUE", "BLUE_SAMPLE", 0.95),
        ]
        for act, obj, conf in flicker_stream:
            res = engine.process_observation(act, obj, conf)
            assert res["confirmed"] is False
        assert engine.total_commits == 0

    def test_cross_object_mixture_noise_rejection(self):
        """Case B: PICK_RED + RED_SAMPLE mixed with PICK_RED + BLUE_SAMPLE."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        mixed_stream = [
            ("PICK_RED", "RED_SAMPLE", 0.90),
            ("PICK_RED", "BLUE_SAMPLE", 0.90),
            ("PICK_RED", "RED_SAMPLE", 0.90),
            ("PICK_RED", "BLUE_SAMPLE", 0.90),
            ("PICK_RED", "RED_SAMPLE", 0.90),
        ]
        # At frame 5: PICK_RED + RED_SAMPLE reaches 3 votes in 5-frame window -> confirms!
        results = [engine.process_observation(a, o, c)["confirmed"] for a, o, c in mixed_stream]
        assert results == [False, False, False, False, True]
        assert engine.total_commits == 1

    def test_valid_bursts_separated_by_unrelated_candidates(self):
        """Case C: Valid bursts separated by unrelated candidates."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # 2 frames of PICK_RED
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)

        # 3 unrelated frames (flushes window)
        engine.process_observation("PICK_BLUE", "BLUE_SAMPLE", 0.90)
        engine.process_observation("PLACE_BLUE", "BLUE_SAMPLE", 0.90)
        engine.process_observation("CLOSE_LID", "CONTAINER_LID", 0.90)

        # Next frame of PICK_RED: 1st PICK_RED was pushed out of maxlen=5 deque,
        # so window has [PICK_RED, PICK_BLUE, PLACE_BLUE, CLOSE_LID, PICK_RED] -> 2 votes (< 3 threshold)
        res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
        assert res["confirmed"] is False
        assert res["vote_count"] == 2
        assert engine.total_commits == 0

    def test_sub_threshold_bursts_never_confirm(self):
        """Case D: Short valid bursts that never reach M."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        for _ in range(5):
            # 2 matching frames
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            # 3 distractor frames
            engine.process_observation("PICK_BLUE", "BLUE_SAMPLE", 0.90)
            engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
            engine.process_observation("IDLE", "NONE", 0.95)

        assert engine.total_commits == 0

    def test_long_noisy_stream_with_occasional_valid_clusters(self):
        """Case E: Long noisy stream containing occasional valid M-of-N clusters."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        commits = 0

        # Cluster 1: PICK_RED (frames 10-12)
        # Cluster 2: PLACE_RED (frames 40-42)
        for f in range(60):
            if 10 <= f <= 12:
                res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            elif 40 <= f <= 42:
                res = engine.process_observation("PLACE_RED", "RED_SAMPLE", 0.90)
            elif f % 2 == 0:
                res = engine.process_observation("IDLE", "NONE", 0.90)
            else:
                res = engine.process_observation("PICK_BLUE", "BLUE_SAMPLE", 0.50)  # Low conf

            if res["confirmed"]:
                commits += 1

        assert commits == 2
        assert engine.total_commits == 2

    def test_deterministic_replay_produces_identical_output(self):
        """Repeated runs with identical inputs produce bitwise identical confirmation outcomes."""
        stream = [
            ("PICK_RED", "RED_SAMPLE", 0.88),
            ("PICK_RED", "RED_SAMPLE", 0.91),
            ("PICK_RED", "RED_SAMPLE", 0.94),
            ("IDLE", "NONE", 0.99),
            ("PLACE_RED", "RED_SAMPLE", 0.82),
            ("PLACE_RED", "RED_SAMPLE", 0.86),
            ("PLACE_RED", "RED_SAMPLE", 0.90),
        ]
        e1 = TemporalConfirmationEngine()
        e2 = TemporalConfirmationEngine()

        r1 = [e1.process_observation(a, o, c)["confirmed"] for a, o, c in stream]
        r2 = [e2.process_observation(a, o, c)["confirmed"] for a, o, c in stream]
        assert r1 == r2 == [False, False, True, False, False, False, True]


# =============================================================================
# 5. MULTI-STEP PROCEDURE TRAVERSAL WITH INTERLEAVED NOISE
# =============================================================================

class TestMultiStepProcedureWithNoise:
    """Verifies complete EXP-001 execution through InferencePipeline with injected noise."""

    def test_full_procedure_with_interleaved_noise_bursts(self):
        """S1->S5 traversal with noise injected before, between, and around steps."""
        fsm = SequenceValidatorFSM()
        fsm.start()
        pipeline = InferencePipeline(validator=fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_perception(act: str, obj: str, conf: float = 0.95):
            pipeline.action_classifier.classify = lambda **k: (act, conf)
            pipeline.object_detector.detect = lambda f: [{"label": obj, "confidence": conf, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        # A/B. Noise before S1
        mock_perception("PICK_BLUE", "BLUE_SAMPLE", 0.50)
        pipeline.process_frame_public(dummy_frame)
        mock_perception("IDLE", "NONE", 0.90)
        pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 0

        # S1: PICK_RED + RED_SAMPLE -> 3 frames
        mock_perception("PICK_RED", "RED_SAMPLE", 0.92)
        for _ in range(3):
            pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 1

        # C. Noise between S1 and S2
        mock_perception("PICK_RED", "RED_SAMPLE", 0.60)  # Low conf
        pipeline.process_frame_public(dummy_frame)
        mock_perception("IDLE", "NONE", 0.90)
        pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 1

        # S2: PLACE_RED + RED_SAMPLE -> 3 frames
        mock_perception("PLACE_RED", "RED_SAMPLE", 0.92)
        for _ in range(3):
            pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 2

        # D. Noise between S2 and S3
        mock_perception("PLACE_RED", "RED_SAMPLE", 0.55)
        pipeline.process_frame_public(dummy_frame)
        mock_perception("IDLE", "NONE", 0.90)
        pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 2

        # S3: PICK_BLUE + BLUE_SAMPLE -> 3 frames
        mock_perception("PICK_BLUE", "BLUE_SAMPLE", 0.92)
        for _ in range(3):
            pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 3

        # S4: PLACE_BLUE + BLUE_SAMPLE -> 3 frames
        mock_perception("PLACE_BLUE", "BLUE_SAMPLE", 0.92)
        for _ in range(3):
            pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 4

        # E. Noise around S4/S5
        mock_perception("PLACE_BLUE", "BLUE_SAMPLE", 0.50)
        pipeline.process_frame_public(dummy_frame)
        mock_perception("IDLE", "NONE", 0.90)
        pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 4

        # S5: CLOSE_LID + CONTAINER_LID -> 3 frames
        mock_perception("CLOSE_LID", "CONTAINER_LID", 0.92)
        for _ in range(3):
            pipeline.process_frame_public(dummy_frame)

        # Full completion verified
        assert fsm.current_step_index == 5
        assert fsm.state == "COMPLETED"
        assert len(fsm.validated_steps) == 5

    def test_procedural_anomalies_and_fsm_authority(self):
        """FSM remains authoritative for wrong-object, future-step, and repeated previous-step candidates."""
        fsm = SequenceValidatorFSM()
        fsm.start()
        pipeline = InferencePipeline(validator=fsm, consistency_evaluator=None)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def mock_perception(act: str, obj: str, conf: float = 0.95):
            pipeline.action_classifier.classify = lambda **k: (act, conf)
            pipeline.object_detector.detect = lambda f: [{"label": obj, "confidence": conf, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        # F. Wrong-object temporal candidate at S1 (expects RED_SAMPLE)
        mock_perception("PICK_RED", "BLUE_SAMPLE", 0.95)
        # Frames 1 and 2: unconfirmed frames do not advance FSM or log anomaly
        pipeline.process_frame_public(dummy_frame)
        pipeline.process_frame_public(dummy_frame)
        assert fsm.current_step_index == 0
        assert len(fsm.anomalies) == 0

        # Frame 3: confirmed! FSM validates and rejects wrong object
        r_conf = pipeline.process_frame_public(dummy_frame)
        assert r_conf["status"] == "OUT_OF_SEQUENCE"
        assert fsm.current_step_index == 0  # Did not advance
        assert len(fsm.anomalies) >= 1

        # Clear cooldown via IDLE
        mock_perception("IDLE", "NONE", 0.95)
        pipeline.process_frame_public(dummy_frame)

        # G. Future-step candidate at S1 (feeding S3 PICK_BLUE + BLUE_SAMPLE) -> SKIPPED
        mock_perception("PICK_BLUE", "BLUE_SAMPLE", 0.95)
        for _ in range(3):
            r_skip = pipeline.process_frame_public(dummy_frame)
        assert r_skip["status"] == "SKIPPED"
        assert fsm.current_step_index == 3  # Auto-advanced to S4

        # Clear cooldown via IDLE
        mock_perception("IDLE", "NONE", 0.95)
        pipeline.process_frame_public(dummy_frame)

        # H. Repeated previous-step candidate (feeding S1 PICK_RED + RED_SAMPLE while at S4)
        mock_perception("PICK_RED", "RED_SAMPLE", 0.95)
        for _ in range(3):
            r_rep = pipeline.process_frame_public(dummy_frame)
        assert r_rep["status"] == "OUT_OF_SEQUENCE"
        assert fsm.current_step_index == 3  # Did not advance from S4


# =============================================================================
# 6. CONFIRMATION LATENCY / FRAME COUNT
# =============================================================================

class TestConfirmationLatencyFrameCount:
    """Verifies confirmation occurs at the exact deterministic frame-count latency."""

    @pytest.mark.parametrize(
        "n, m",
        [
            (3, 2),
            (5, 3),
            (7, 4),
            (10, 6),
        ],
    )
    def test_exact_frame_count_confirmation_latency(self, n: int, m: int):
        """Frame-count confirmation latency equals exactly M frames."""
        engine = TemporalConfirmationEngine(window_size=n, confirmation_threshold=m, min_confidence=0.70)

        latency_frames = 0
        confirmed = False

        for f in range(1, m + 2):
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)
            if res["confirmed"]:
                latency_frames = f
                confirmed = True
                break

        assert confirmed is True
        assert latency_frames == m


# =============================================================================
# 7. RATE INTERACTION & PACING
# =============================================================================

class TestRateInteraction:
    """Verifies temporal filtering determinism under synthetic worker rate consumption."""

    def test_ten_fps_synthetic_worker_pacing(self):
        """Simulates 10 FPS consumption (100 ms per step): 3 observations confirm in 3 frames."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        mock_mono = 100.0

        res1 = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90, timestamp=mock_mono)
        assert res1["confirmed"] is False

        mock_mono += 0.100  # +100 ms
        res2 = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90, timestamp=mock_mono)
        assert res2["confirmed"] is False

        mock_mono += 0.100  # +100 ms
        res3 = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90, timestamp=mock_mono)
        assert res3["confirmed"] is True

    def test_fifteen_fps_synthetic_worker_pacing(self):
        """Simulates 15 FPS consumption (66.6 ms per step): 3 observations confirm in 3 frames."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        mock_mono = 100.0

        for _ in range(2):
            mock_mono += 0.0666
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90, timestamp=mock_mono)
            assert res["confirmed"] is False

        mock_mono += 0.0666
        res3 = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90, timestamp=mock_mono)
        assert res3["confirmed"] is True

    def test_thirty_fps_producer_with_slower_ai_consumption(self):
        """Simulates 30 FPS camera with opportunistic latest-frame sampling by slower AI."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)

        # Producer generates 30 FPS frames (33 ms interval).
        # AI worker samples every 3rd frame (10 FPS effective rate).
        sampled_frames = [0, 3, 6]  # Frame indices sampled by worker

        for i, f_idx in enumerate(sampled_frames):
            ts = f_idx * 0.0333
            res = engine.process_observation("PICK_RED", "RED_SAMPLE", 0.92, timestamp=ts)
            if i < 2:
                assert res["confirmed"] is False
            else:
                assert res["confirmed"] is True


# =============================================================================
# 8. MEMORY BOUNDEDNESS & DETERMINISM
# =============================================================================

class TestMemoryBoundednessAndDeterminism:
    """Verifies fixed-bound memory consumption and strict 8-field public contract."""

    def test_rolling_window_strictly_bounded_over_one_thousand_frames(self):
        """Window deque length never exceeds configured N across 1000 observations."""
        engine = TemporalConfirmationEngine(window_size=5, confirmation_threshold=3, min_confidence=0.70)
        actions = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "IDLE"]
        objects = ["RED_SAMPLE", "BLUE_SAMPLE", "NONE"]

        for i in range(1000):
            act = actions[i % len(actions)]
            obj = objects[i % len(objects)]
            conf = 0.50 + (i % 50) / 100.0
            engine.process_observation(act, obj, conf)
            assert len(engine._window) <= 5

    def test_reset_releases_old_candidate_history(self):
        """Reset clears window buffer completely, freeing memory."""
        engine = TemporalConfirmationEngine(window_size=10, confirmation_threshold=6, min_confidence=0.70)
        # Feed 5 unconfirmed frames (< M=6)
        for _ in range(5):
            engine.process_observation("PICK_RED", "RED_SAMPLE", 0.90)

        assert len(engine._window) == 5
        engine.reset()
        assert len(engine._window) == 0

    def test_public_contract_integrity_across_robustness_runs(self):
        """Every generated result contains exactly the 8 frozen contract keys."""
        fsm = SequenceValidatorFSM()
        fsm.start()
        pipeline = InferencePipeline(validator=fsm)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(50):
            res = pipeline.process_frame_public(
                frame=dummy_frame,
                timestamp="2026-09-28T12:00:00.000Z",
            )
            assert set(res.keys()) == set(PUBLIC_CONTRACT_KEYS)
            assert len(res) == 8
            assert AIResultAdapter.validate_public_contract(res) is True
