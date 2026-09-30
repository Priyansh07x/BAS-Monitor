"""
test_b11_2_2_pipeline_uncertainty_integration.py — Verification of Gate B11.2.2 Pipeline & Uncertainty Integration
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B11.2.2)

Tests:
1. Reliable aligned observation throughput (RELIABLE_ALIGNED -> B10 Confirmation -> FSM Commit).
2. Cross-modal semantic conflict isolation (PICK_RED + BLUE_SAMPLE -> No FSM Call).
3. Missing object uncertainty isolation (PICK_RED + NONE -> No FSM Call).
4. Low object confidence uncertainty isolation (PICK_RED + RED_SAMPLE @ 0.35 -> No FSM Call).
5. Spatial geometric veto isolation (PICK_RED + distant hands -> UNCERTAIN_SPATIAL_CONTRADICTION -> No FSM Call).
6. Marginal confidence accumulation and resolution (conf=0.55 -> RESOLVED after required evidence).
7. Perceptual noise stream suppression (alternating conflict frames -> zero spurious commits).
8. Procedural authority separation (perceptually coherent wrong-step candidate evaluated and handled authoritatively by FSM).
9. Full end-to-end EXP-001 multi-step procedural traversal with 3-frame confirmation per step.
10. Strict 8-field public contract adherence and zero diagnostic leakage.
11. Pipeline lifecycle management (reset, pause, resume, completion sync).
12. Threading and non-blocking InferenceWorker integration with integrated pipeline.
13. Deterministic replay invariance over identical synthetic streams.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import InferenceWorker, RateStrategy
from backend.ai.multimodal_consistency import (
    CATEGORY_CROSS_MODAL_CONFLICT,
    CATEGORY_IDLE,
    CATEGORY_INSTABILITY_ACTION_FLICKER,
    CATEGORY_RELIABLE_ALIGNED,
    CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE,
    CATEGORY_UNCERTAIN_MISSING_OBJECT,
    CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION,
    MultimodalConsistencyEvaluator,
)
from backend.ai.result_adapter import PUBLIC_CONTRACT_KEYS
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.experiment.sequence_validator import SequenceValidatorFSM


@pytest.fixture
def clean_fsm() -> SequenceValidatorFSM:
    """Provides a freshly started SequenceValidatorFSM configured with EXP-001."""
    fsm = SequenceValidatorFSM()
    fsm.start()
    return fsm


@pytest.fixture
def integrated_pipeline(clean_fsm: SequenceValidatorFSM) -> InferencePipeline:
    """Provides an InferencePipeline configured with full B11.2 integration."""
    return InferencePipeline(
        validator=clean_fsm,
        temporal_window_size=5,
        temporal_threshold=3,
        min_confidence=0.70,
        cooldown_frames=5,
    )


class TestPipelineConsistencyUncertaintyRouting:
    """Tests the perception routing between MultimodalConsistencyEvaluator, UncertaintyHandler, B10, and FSM."""

    def test_reliable_aligned_observation_throughput(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """High-confidence semantically aligned observation confirms after 3 frames and advances FSM."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.92)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        # Frame 1: Unconfirmed
        r1 = integrated_pipeline.process_frame_public(dummy_frame)
        assert integrated_pipeline.last_consistency_result["category"] == CATEGORY_RELIABLE_ALIGNED
        assert integrated_pipeline.last_consistency_result["is_reliable"] is True
        assert integrated_pipeline.last_uncertainty_result["state"] == "CONFIDENT"
        assert clean_fsm.current_step_index == 0
        assert r1["expected_step"] == "S1"

        # Frame 2: Unconfirmed
        r2 = integrated_pipeline.process_frame_public(dummy_frame)
        assert clean_fsm.current_step_index == 0
        assert r2["expected_step"] == "S1"

        # Frame 3: Confirmed & FSM Validated!
        r3 = integrated_pipeline.process_frame_public(dummy_frame)
        assert clean_fsm.current_step_index == 1  # Advanced S1 -> S2
        assert r3["status"] == "VALID"
        assert len(clean_fsm.validated_steps) == 1
        assert clean_fsm.validated_steps[0]["detected_action"] == "PICK_RED"

    def test_cross_modal_semantic_conflict_isolation(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Cross-modal conflict (PICK_RED + BLUE_SAMPLE) is isolated and NEVER reaches FSM."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "BLUE_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        for _ in range(5):
            res = integrated_pipeline.process_frame_public(dummy_frame)
            assert integrated_pipeline.last_consistency_result["category"] == CATEGORY_CROSS_MODAL_CONFLICT
            assert integrated_pipeline.last_consistency_result["is_reliable"] is False
            assert res["expected_step"] == "S1"

        # FSM MUST NOT BE CALLED: zero step advance and zero anomalies
        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0
        assert len(clean_fsm.anomalies) == 0

    def test_missing_object_uncertainty_isolation(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Action without detected object is flagged as UNCERTAIN_MISSING_OBJECT and does not advance FSM."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        integrated_pipeline.object_detector.detect = lambda f: []

        for _ in range(5):
            res = integrated_pipeline.process_frame_public(dummy_frame)
            assert integrated_pipeline.last_consistency_result["category"] == CATEGORY_UNCERTAIN_MISSING_OBJECT
            assert integrated_pipeline.last_consistency_result["is_reliable"] is False

        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0
        assert len(clean_fsm.anomalies) == 0

    def test_low_object_confidence_isolation(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Action with low object confidence (< 0.50) is flagged as UNCERTAIN_LOW_OBJECT_CONFIDENCE."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.35, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        for _ in range(5):
            integrated_pipeline.process_frame_public(dummy_frame)
            assert integrated_pipeline.last_consistency_result["category"] == CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE
            assert integrated_pipeline.last_consistency_result["is_reliable"] is False

        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0

    def test_spatial_contradiction_veto_isolation(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Manipulation action with hands far away triggers spatial contradiction veto."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.88)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.90, "x1": 0.05, "y1": 0.05, "x2": 0.15, "y2": 0.15}
        ]
        integrated_pipeline.interaction_engine.evaluate_interaction = lambda **k: {
            "state": "IDLE",
            "normalized_scale_proximity": 2.50,
            "target_object": "RED_SAMPLE",
        }

        for _ in range(5):
            integrated_pipeline.process_frame_public(dummy_frame)
            assert integrated_pipeline.last_consistency_result["category"] == CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION
            assert integrated_pipeline.last_consistency_result["is_reliable"] is False

        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0


class TestMarginalEvidenceAndNoiseRejection:
    """Tests marginal evidence accumulation/resolution and noise rejection."""

    def test_marginal_evidence_accumulation_and_resolution(self, clean_fsm: SequenceValidatorFSM):
        """Marginal confidence (0.55) accumulates in UncertaintyHandler until resolved."""
        pipeline = InferencePipeline(
            validator=clean_fsm,
            temporal_window_size=5,
            temporal_threshold=3,
            min_confidence=0.70,
            cooldown_frames=5,
            uncertainty_handler=UncertaintyHandler(
                high_confidence_threshold=0.70,
                marginal_confidence_threshold=0.50,
            ),
            consistency_evaluator=MultimodalConsistencyEvaluator(
                action_confidence_threshold=0.50,
                object_confidence_threshold=0.50,
            ),
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.55)
        pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.85, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        # Frame 1: Accumulating evidence
        pipeline.process_frame_public(dummy_frame)
        assert pipeline.last_uncertainty_result["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert clean_fsm.current_step_index == 0

        # Frame 2: Accumulating evidence
        pipeline.process_frame_public(dummy_frame)
        assert pipeline.last_uncertainty_result["state"] == "UNCERTAIN_EVIDENCE_ACCUMULATING"
        assert clean_fsm.current_step_index == 0

        # Frame 3: Resolved! UncertaintyHandler marks reliable
        pipeline.process_frame_public(dummy_frame)
        assert pipeline.last_uncertainty_result["state"] == "RESOLVED"
        assert pipeline.last_uncertainty_result["is_reliable"] is True

    def test_perceptual_noise_stream_suppression(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Alternating conflicting frames never accumulate valid confirmation votes."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        noise_sequence = [
            ("PICK_RED", "BLUE_SAMPLE"),    # Cross-modal conflict
            ("PICK_BLUE", "RED_SAMPLE"),   # Cross-modal conflict
            ("PICK_RED", "BLUE_SAMPLE"),    # Cross-modal conflict
            ("PICK_BLUE", "RED_SAMPLE"),   # Cross-modal conflict
            ("CLOSE_LID", "RED_SAMPLE"),    # Cross-modal conflict
        ]

        for act, obj in noise_sequence:
            integrated_pipeline.action_classifier.classify = lambda a=act, **k: (a, 0.90)
            integrated_pipeline.object_detector.detect = lambda f, o=obj: [
                {"label": o, "confidence": 0.90, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
            ]
            integrated_pipeline.process_frame_public(dummy_frame)
            assert integrated_pipeline.last_consistency_result["is_reliable"] is False

        # FSM must remain at S1 without advancement or anomalies
        assert clean_fsm.current_step_index == 0
        assert len(clean_fsm.validated_steps) == 0
        assert len(clean_fsm.anomalies) == 0


class TestProceduralAuthorityAndMultiStepTraversal:
    """Tests separation of perceptual consistency authority from procedural FSM authority."""

    def test_perceptually_reliable_wrong_step_candidate_handled_by_fsm(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """A candidate that is perceptually coherent (PICK_BLUE + BLUE_SAMPLE) but procedurally out of order is evaluated by FSM."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # S1 expects PICK_RED + RED_SAMPLE. Perception produces PICK_BLUE + BLUE_SAMPLE (Step 3).
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_BLUE", 0.95)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "BLUE_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        # Frames 1, 2
        integrated_pipeline.process_frame_public(dummy_frame)
        integrated_pipeline.process_frame_public(dummy_frame)

        # Frame 3: Confirmed by B10 -> FSM evaluates and flags SKIPPED status
        r3 = integrated_pipeline.process_frame_public(dummy_frame)
        assert integrated_pipeline.last_consistency_result["is_reliable"] is True
        assert r3["status"] == "SKIPPED"
        assert r3["detected_step"] == "S3"
        assert r3["expected_step"] == "S1"
        assert clean_fsm.current_step_index == 3  # Advanced past skipped step S3

    def test_complete_exp001_procedure_traversal(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Full S1 -> S5 traversal through integrated pipeline with 3 frames per step."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        steps = [
            ("PICK_RED", "RED_SAMPLE"),
            ("PLACE_RED", "RED_SAMPLE"),
            ("PICK_BLUE", "BLUE_SAMPLE"),
            ("PLACE_BLUE", "BLUE_SAMPLE"),
            ("CLOSE_LID", "CONTAINER_LID"),
        ]

        for step_idx, (act, obj) in enumerate(steps):
            integrated_pipeline.action_classifier.classify = lambda a=act, **k: (a, 0.95)
            integrated_pipeline.object_detector.detect = lambda f, o=obj: [
                {"label": o, "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
            ]

            # 3 frames to confirm
            for _ in range(3):
                res = integrated_pipeline.process_frame_public(dummy_frame)

            assert clean_fsm.current_step_index == step_idx + 1

        assert clean_fsm.state == "COMPLETED"
        assert len(clean_fsm.validated_steps) == 5


class TestPublicContractAndLifecycle:
    """Tests strict 8-field public contract compliance and lifecycle operations."""

    def test_strict_8_field_public_contract_and_no_diagnostic_leakage(self, integrated_pipeline: InferencePipeline):
        """Public result contains strictly 8 canonical keys with zero diagnostic leakage."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.90, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        pub_res = integrated_pipeline.process_frame_public(dummy_frame)

        assert set(pub_res.keys()) == set(PUBLIC_CONTRACT_KEYS)
        assert len(pub_res) == 8
        # Diagnostic results must NOT be in public dict, but accessible via properties
        assert "consistency_result" not in pub_res
        assert "uncertainty_result" not in pub_res
        assert integrated_pipeline.last_consistency_result is not None
        assert integrated_pipeline.last_uncertainty_result is not None

    def test_pipeline_reset_pause_resume_lifecycle(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """Pipeline lifecycle methods reset, pause, and resume internal components consistently."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.90, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        # Submit 2 frames
        integrated_pipeline.process_frame_public(dummy_frame)
        integrated_pipeline.process_frame_public(dummy_frame)
        assert integrated_pipeline.last_consistency_result is not None

        # Reset
        integrated_pipeline.reset()
        assert integrated_pipeline.last_consistency_result is None
        assert integrated_pipeline.last_uncertainty_result is None
        assert integrated_pipeline.temporal_filter.state == "RUNNING"

        # Pause
        integrated_pipeline.pause()
        assert integrated_pipeline.temporal_filter.is_paused is True
        assert integrated_pipeline.uncertainty_handler.is_paused is True

        # Resume
        integrated_pipeline.resume()
        assert integrated_pipeline.temporal_filter.is_paused is False
        assert integrated_pipeline.uncertainty_handler.is_paused is False

    def test_deterministic_replay_invariance(self, clean_fsm: SequenceValidatorFSM):
        """Identical sequences produce identical public outputs and internal states."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def run_sequence():
            fsm = SequenceValidatorFSM()
            fsm.start()
            pipe = InferencePipeline(validator=fsm)
            pipe.action_classifier.classify = lambda **k: ("PICK_RED", 0.90)
            pipe.object_detector.detect = lambda f: [
                {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
            ]
            outputs = []
            for _ in range(4):
                outputs.append(pipe.process_frame_public(dummy_frame))
            return outputs, fsm.current_step_index

        out1, step1 = run_sequence()
        out2, step2 = run_sequence()

        assert step1 == step2 == 1
        for o1, o2 in zip(out1, out2):
            assert o1["action"] == o2["action"]
            assert o1["object"] == o2["object"]
            assert o1["status"] == o2["status"]
            assert o1["expected_step"] == o2["expected_step"]


class TestInferenceWorkerIntegration:
    """Tests background InferenceWorker execution with integrated pipeline."""

    def test_worker_background_execution(self, clean_fsm: SequenceValidatorFSM, integrated_pipeline: InferencePipeline):
        """InferenceWorker successfully runs integrated pipeline in background thread."""
        integrated_pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.92)
        integrated_pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
        ]

        results = []
        worker = InferenceWorker(
            pipeline=integrated_pipeline,
            validator=clean_fsm,
            result_callback=lambda res: results.append(res),
            rate_strategy=RateStrategy.OPPORTUNISTIC_LATEST,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        worker.start()
        try:
            for _ in range(5):
                worker.submit_frame(dummy_frame)
                time.sleep(0.04)

            time.sleep(0.25)
            assert len(results) >= 3
            assert clean_fsm.current_step_index == 1
            for r in results:
                assert set(r.keys()) == set(PUBLIC_CONTRACT_KEYS)
        finally:
            worker.stop()
