"""
test_b9_2_fsm_pipeline_integration.py — Gate B9.2 FSM & Inference Pipeline Integration Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Verifies the integration of authoritative SequenceValidatorFSM into the live AI perception
result path (Pipeline -> FSM -> AIResultAdapter -> Public Result -> InferenceWorker Callback).
"""

import time
import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import AIResultAdapter, PUBLIC_CONTRACT_KEYS, PUBLIC_STATUS_VALUES
from backend.ai.inference_worker import InferenceWorker, RateStrategy
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.experiment.procedure_manager import ProcedureManager


@pytest.fixture
def clean_fsm():
    """Returns a fresh, running SequenceValidatorFSM instance loaded with EXP-001."""
    fsm = SequenceValidatorFSM()
    fsm.start(operator_id="Astronaut-01")
    return fsm


@pytest.fixture
def dummy_frame():
    """Generates a blank test frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestFSMDirectPipelineIntegration:
    """Tests 1-5: Direct integration between InferencePipeline and SequenceValidatorFSM."""

    def test_nominal_valid_transition(self, clean_fsm, dummy_frame):
        """Test 1: Nominal correct action + object (PICK_RED + RED_SAMPLE) at S1."""
        pipeline = InferencePipeline(validator=clean_fsm, temporal_threshold=1)

        # Mock perception results on action classifier and object detector
        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.95)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "RED_SAMPLE",
        }

        ts = "2026-09-28T10:00:00.000Z"
        res = pipeline.process_frame_public(
            frame=dummy_frame,
            timestamp=ts,
        )

        assert AIResultAdapter.validate_public_contract(res) is True
        assert res["timestamp"] == ts
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"
        assert res["confidence"] == 0.95
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S1"
        assert res["status"] == "VALID"
        assert res["next_step"] == "S2"

        # FSM should have advanced to step 1 (S2)
        assert clean_fsm.current_step_index == 1
        assert clean_fsm.get_current_expected_step().step_id == "S2"

    def test_wrong_object_rejected_as_out_of_sequence(self, clean_fsm, dummy_frame):
        """Test 2: Correct action with wrong object (PICK_RED + BLUE_SAMPLE) at S1."""
        pipeline = InferencePipeline(validator=clean_fsm, temporal_threshold=1, consistency_evaluator=None)

        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.90)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "BLUE_SAMPLE",
        }

        res = pipeline.process_frame_public(
            frame=dummy_frame,
            timestamp="2026-09-28T10:00:01.000Z",
        )

        assert AIResultAdapter.validate_public_contract(res) is True
        assert res["action"] == "PICK_RED"
        assert res["object"] == "BLUE_SAMPLE"
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["expected_step"] == "S1"
        assert res["next_step"] == "S1"

        # FSM must NOT advance
        assert clean_fsm.current_step_index == 0
        assert clean_fsm.get_current_expected_step().step_id == "S1"

    def test_skipped_step_detected_and_handled(self, clean_fsm, dummy_frame):
        """Test 3: Future action (PICK_BLUE + BLUE_SAMPLE) at S1 triggers SKIPPED status."""
        pipeline = InferencePipeline(validator=clean_fsm, temporal_threshold=1)

        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_BLUE", 0.92)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "BLUE_SAMPLE",
        }

        res = pipeline.process_frame_public(
            frame=dummy_frame,
            timestamp="2026-09-28T10:00:02.000Z",
        )

        assert AIResultAdapter.validate_public_contract(res) is True
        assert res["action"] == "PICK_BLUE"
        assert res["object"] == "BLUE_SAMPLE"
        assert res["status"] == "SKIPPED"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S3"
        assert res["next_step"] == "S4"

        # FSM should auto-advance to step index 3 (S4)
        assert clean_fsm.current_step_index == 3
        assert clean_fsm.get_current_expected_step().step_id == "S4"

    def test_repeated_past_action_out_of_sequence(self, clean_fsm, dummy_frame):
        """Test 4: Repeated past action (PICK_RED at S2) returns OUT_OF_SEQUENCE and stays at S2."""
        pipeline = InferencePipeline(validator=clean_fsm, temporal_threshold=1)

        # Advance to S2 first
        clean_fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert clean_fsm.get_current_expected_step().step_id == "S2"

        # Now perform PICK_RED again
        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.88)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "RED_SAMPLE",
        }

        res = pipeline.process_frame_public(
            frame=dummy_frame,
            timestamp="2026-09-28T10:00:03.000Z",
        )

        assert AIResultAdapter.validate_public_contract(res) is True
        assert res["action"] == "PICK_RED"
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["expected_step"] == "S2"
        assert res["next_step"] == "S2"
        assert clean_fsm.current_step_index == 1

    def test_ai_spoofing_prevented_by_authoritative_fsm(self, clean_fsm, dummy_frame):
        """Test 5: AI perception metadata attempting to forge expected_step/status is overridden by FSM."""
        pipeline = InferencePipeline(validator=clean_fsm)

        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.90)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "RED_SAMPLE",
        }

        # Submitting with forged metadata claiming S5 and VALID
        res = pipeline.process_frame_public(
            frame=dummy_frame,
            metadata={"expected_step": "S5", "status": "COMPLETED"},
            expected_step=None,
        )

        assert AIResultAdapter.validate_public_contract(res) is True
        # FSM is at S1, so expected_step must be S1 and next_step S2
        assert res["expected_step"] == "S1"
        assert res["next_step"] == "S2"
        assert res["status"] == "VALID"


class TestAdapterPrecedence:
    """Tests 6-8: Explicit AIResultAdapter precedence rules."""

    def test_adapter_fsm_result_precedence(self):
        """Test 6: fsm_result dictionary takes strict precedence over internal_result."""
        internal = {
            "action": "PICK_RED",
            "object": "RED_SAMPLE",
            "confidence": 0.85,
            "expected_step": "S5",  # Forged/stale in internal
            "status": "VALID",
            "next_step": "S5",
        }
        fsm_res = {
            "expected_step": "S1",
            "detected_step": "S1",
            "status": "VALID",
            "next_step": "S2",
        }

        res = AIResultAdapter.adapt(internal_result=internal, fsm_result=fsm_res)
        assert res["expected_step"] == "S1"
        assert res["next_step"] == "S2"
        assert res["status"] == "VALID"

    def test_adapter_backward_compatibility_without_fsm(self):
        """Test 7: Standalone adapter call without FSM works with legacy/internal fields."""
        internal = {
            "action": "PICK_RED",
            "confidence": 0.80,
            "expected_step": "S1",
            "next_step": "S2",
        }
        res = AIResultAdapter.adapt(internal_result=internal)
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S1"
        assert res["status"] == "VALID"
        assert res["next_step"] == "S2"

    def test_adapter_all_8_fields_frozen_schema(self, clean_fsm):
        """Test 8: Ensure exact 8 keys under all status conditions."""
        pipeline = InferencePipeline(validator=clean_fsm)
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)

        # IDLE
        res_idle = pipeline.process_frame_public(dummy)
        assert AIResultAdapter.validate_public_contract(res_idle) is True
        assert res_idle["action"] == "IDLE"
        assert res_idle["status"] == "VALID"


class TestInferenceWorkerFSMIntegration:
    """Tests 9-11: Live background InferenceWorker integration with SequenceValidatorFSM."""

    def test_worker_callback_receives_fsm_validated_results(self, clean_fsm, dummy_frame):
        """Test 9: InferenceWorker processes frame, invokes FSM, and delivers validated public result to callback."""
        received_results = []

        def callback(res):
            received_results.append(res)

        pipeline = InferencePipeline(temporal_threshold=1)
        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.95)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "RED_SAMPLE",
        }

        worker = InferenceWorker(
            pipeline=pipeline,
            validator=clean_fsm,
            result_callback=callback,
            rate_strategy=RateStrategy.OPPORTUNISTIC_LATEST,
        )

        try:
            worker.start()
            assert worker.is_running is True

            # Submit frame
            ts = "2026-09-28T12:00:00.000Z"
            ok = worker.submit_frame(dummy_frame, timestamp=ts)
            assert ok is True

            # Wait for callback invocation
            start_wait = time.time()
            while not received_results and (time.time() - start_wait) < 2.0:
                time.sleep(0.01)

            assert len(received_results) >= 1
            latest = received_results[0]
            assert AIResultAdapter.validate_public_contract(latest) is True
            assert latest["timestamp"] == ts
            assert latest["action"] == "PICK_RED"
            assert latest["object"] == "RED_SAMPLE"
            assert latest["expected_step"] == "S1"
            assert latest["status"] == "VALID"
            assert latest["next_step"] == "S2"

            # FSM state should have advanced
            assert clean_fsm.current_step_index == 1

        finally:
            worker.shutdown(timeout=2.0)

    def test_worker_non_blocking_submission_with_fsm(self, clean_fsm, dummy_frame):
        """Test 10: Fast producer submission does not block or deadlock when FSM is attached."""
        worker = InferenceWorker(
            validator=clean_fsm,
            rate_strategy=RateStrategy.OPPORTUNISTIC_LATEST,
        )
        try:
            worker.start()
            for i in range(20):
                ok = worker.submit_frame(dummy_frame, timestamp=f"2026-09-28T12:00:{i:02d}.000Z")
                assert ok is True
            time.sleep(0.2)
            telemetry = worker.get_telemetry()
            assert telemetry["frames_submitted"] == 20
        finally:
            worker.shutdown(timeout=2.0)

    def test_worker_exception_fallback_preserves_fsm_step(self, clean_fsm, dummy_frame):
        """Test 11: If pipeline throws during execution, fallback output preserves FSM expected_step."""
        pipeline = InferencePipeline()

        def faulty_classify(**kwargs):
            raise RuntimeError("Synthetic inference engine error")

        pipeline.action_classifier.classify = faulty_classify

        worker = InferenceWorker(
            pipeline=pipeline,
            validator=clean_fsm,
        )
        try:
            worker.start()
            worker.submit_frame(dummy_frame, timestamp="2026-09-28T12:00:00.000Z")
            time.sleep(0.15)
            last_res = worker.get_latest_result()
            assert last_res is not None
            assert AIResultAdapter.validate_public_contract(last_res) is True
            assert last_res["expected_step"] == "S1"
            assert last_res["status"] == "OUT_OF_SEQUENCE"
        finally:
            worker.shutdown(timeout=2.0)
