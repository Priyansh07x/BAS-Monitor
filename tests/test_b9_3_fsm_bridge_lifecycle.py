"""
test_b9_3_fsm_bridge_lifecycle.py — Gate B9.3 FSM, AppState, and Bridge Lifecycle Integration Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Verifies:
1. AppState authoritatively owns the single SequenceValidatorFSM instance.
2. InferenceWorker / InferencePipeline consume the exact same authoritative FSM instance.
3. Bridge exposes Qt-facing lifecycle methods (start, pause, resume, reset, get progress).
4. Deterministic lifecycle transitions: START, PAUSE, RESUME, RESET, and COMPLETION.
5. Immunity against invalid lifecycle transitions.
6. Real-time perception validation advances the shared FSM state through Bridge signals.
7. Strict preservation of the frozen 8-field public AI contract and non-blocking threading.
"""

import json
import threading
import time
import numpy as np
import pytest

try:
    from PySide6.QtCore import QCoreApplication
except ImportError:
    QCoreApplication = None

from backend.app_state import AppState
from backend.bridge import Bridge
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.ai.inference_worker import InferenceWorker, RateStrategy
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import AIResultAdapter


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QCoreApplication exists if PySide6 is installed."""
    if QCoreApplication is not None:
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        return app
    return None


@pytest.fixture
def clean_app_state():
    """Returns a fresh AppState instance with clean FSM."""
    fsm = SequenceValidatorFSM()
    fsm.reset()
    state = AppState(sequence_validator=fsm)
    return state


@pytest.fixture
def clean_bridge(clean_app_state):
    """Returns a fresh Bridge connected to clean_app_state."""
    bridge = Bridge(clean_app_state)
    yield bridge
    bridge.shutdown()


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestAppStateAuthoritativeFSMSingleton:
    """Requirement A & K: AppState owns single authoritative FSM shared by Worker and Pipeline."""

    def test_app_state_fsm_ownership(self, clean_app_state):
        """AppState authoritatively owns exactly one SequenceValidatorFSM."""
        assert hasattr(clean_app_state, "sequence_validator")
        assert isinstance(clean_app_state.sequence_validator, SequenceValidatorFSM)

    def test_worker_and_pipeline_share_same_fsm_instance(self, clean_app_state):
        """InferenceWorker and its owned InferencePipeline share the AppState validator."""
        worker = clean_app_state.inference_worker
        assert worker.validator is clean_app_state.sequence_validator
        assert worker.pipeline.validator is clean_app_state.sequence_validator

    def test_no_duplicate_fsm_state(self, clean_bridge):
        """Bridge, AppState, and InferenceWorker observe identical FSM state."""
        assert clean_bridge.state.sequence_validator is clean_bridge.inference_worker.validator
        clean_bridge.startMonitoring()
        assert clean_bridge.state.monitoring_status == "RUNNING"
        assert clean_bridge.inference_worker.validator.state == "RUNNING"


class TestFSMLifecycleTransitions:
    """Requirements B–G: Deterministic lifecycle state machine behavior via Bridge and AppState."""

    def test_start_lifecycle(self, clean_bridge):
        """Requirement B: START transitions IDLE -> RUNNING with expected_step S1."""
        assert clean_bridge.getFSMState() == "IDLE"
        assert clean_bridge.state.current_step_index == 0

        ok = clean_bridge.startMonitoring()
        assert ok is True
        assert clean_bridge.getFSMState() == "RUNNING"
        assert clean_bridge.state.monitoring_status == "RUNNING"
        curr_step = clean_bridge.state.sequence_validator.get_current_expected_step()
        assert curr_step is not None
        assert curr_step.step_id == "S1"

    def test_start_procedure_with_operator_and_session(self, clean_bridge):
        """Requirement B: startProcedure returns structured session JSON and sets RUNNING."""
        res_json = clean_bridge.startProcedure("Astronaut-07", "Microgravity Test A")
        session = json.loads(res_json)
        assert isinstance(session, dict)
        assert clean_bridge.getFSMState() == "RUNNING"

    def test_pause_lifecycle_preserves_step_index(self, clean_bridge):
        """Requirement C: PAUSE transitions RUNNING -> PAUSED while preserving current step index."""
        clean_bridge.startMonitoring()
        # Advance step to S2
        clean_bridge.state.sequence_validator.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert clean_bridge.state.current_step_index == 1

        clean_bridge.pauseMonitoring()
        assert clean_bridge.getFSMState() == "PAUSED"
        assert clean_bridge.state.current_step_index == 1
        assert clean_bridge.state.sequence_validator.get_current_expected_step().step_id == "S2"

    def test_resume_lifecycle_continues_same_state(self, clean_bridge):
        """Requirement D: RESUME transitions PAUSED -> RUNNING without altering progress."""
        clean_bridge.startMonitoring()
        clean_bridge.state.sequence_validator.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        clean_bridge.pauseMonitoring()
        assert clean_bridge.getFSMState() == "PAUSED"

        clean_bridge.resumeMonitoring()
        assert clean_bridge.getFSMState() == "RUNNING"
        assert clean_bridge.state.current_step_index == 1
        assert clean_bridge.state.sequence_validator.get_current_expected_step().step_id == "S2"

    def test_reset_lifecycle_clears_all_progress(self, clean_bridge):
        """Requirement E: RESET transitions to IDLE, resets step to S1/index 0, and clears records."""
        clean_bridge.startMonitoring()
        clean_bridge.state.sequence_validator.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert len(clean_bridge.state.sequence_validator.validated_steps) == 1

        clean_bridge.resetMonitoring()
        assert clean_bridge.getFSMState() == "IDLE"
        assert clean_bridge.state.current_step_index == 0
        assert len(clean_bridge.state.sequence_validator.validated_steps) == 0
        assert len(clean_bridge.state.sequence_validator.anomalies) == 0
        assert clean_bridge.state.sequence_validator.get_current_expected_step().step_id == "S1"

    def test_completion_lifecycle_retention(self, clean_bridge):
        """Requirement F: Validating S1–S5 transitions FSM to COMPLETED and retains state."""
        clean_bridge.startMonitoring()
        fsm = clean_bridge.state.sequence_validator

        # Complete steps S1 through S5
        fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
        fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
        fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
        res5 = fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")

        assert res5["status"] == "VALID"
        assert fsm.state == "COMPLETED"
        assert clean_bridge.getFSMState() == "COMPLETED"
        progress = json.loads(clean_bridge.getProcedureProgress())
        assert progress["is_complete"] is True
        assert progress["validated_count"] == 5

    def test_invalid_lifecycle_calls_do_not_corrupt_fsm(self, clean_bridge):
        """Requirement G: Invalid transitions (pause when IDLE, resume when RUNNING) are safely ignored."""
        assert clean_bridge.getFSMState() == "IDLE"

        # Pausing an IDLE state does not corrupt it
        clean_bridge.pauseMonitoring()
        assert clean_bridge.getFSMState() == "IDLE"

        # Resuming an IDLE state does not corrupt it
        clean_bridge.resumeMonitoring()
        assert clean_bridge.getFSMState() == "IDLE"

        # Starting
        clean_bridge.startMonitoring()
        assert clean_bridge.getFSMState() == "RUNNING"

        # Resuming an already RUNNING state keeps it RUNNING
        clean_bridge.resumeMonitoring()
        assert clean_bridge.getFSMState() == "RUNNING"


class TestLiveBridgeWorkerFSMIntegration:
    """Requirements H, I, J: Live end-to-end perception -> FSM -> Bridge signals and contract."""

    def test_live_frame_advances_fsm_via_bridge_signals(self, clean_bridge, dummy_frame):
        """Requirement H: Submitting camera frame through Bridge/Worker validates and advances FSM."""
        emitted_results = []

        def on_result(res):
            emitted_results.append(res)

        clean_bridge.aiResultReady.connect(on_result)

        # Mock perception results on the worker's pipeline
        pipeline = clean_bridge.inference_worker.pipeline
        if pipeline.temporal_filter is not None:
            pipeline.temporal_filter._confirmation_threshold = 1
        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.95)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "RED_SAMPLE",
        }

        # Start procedure & AI
        clean_bridge.startMonitoring()
        assert clean_bridge.isAIRunning() is True

        # Submit frame through camera feed
        ts = "2026-09-28T14:00:00.000Z"
        ok = clean_bridge.inference_worker.submit_frame(dummy_frame, timestamp=ts)
        assert ok is True

        # Wait for result emission
        start_wait = time.time()
        while not emitted_results and (time.time() - start_wait) < 2.0:
            time.sleep(0.01)

        assert len(emitted_results) >= 1
        res = emitted_results[0]

        # Requirement I: Strict 8-field public AI contract compliance
        assert AIResultAdapter.validate_public_contract(res) is True
        assert res["timestamp"] == ts
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S1"
        assert res["status"] == "VALID"
        assert res["next_step"] == "S2"

        # Authoritative FSM state must have advanced to S2
        assert clean_bridge.state.current_step_index == 1
        assert clean_bridge.state.sequence_validator.get_current_expected_step().step_id == "S2"

    def test_gui_cannot_override_fsm_authoritative_state(self, clean_bridge, dummy_frame):
        """Requirement I: GUI / metadata claims cannot spoof expected_step or status."""
        emitted_results = []
        clean_bridge.aiResultReady.connect(lambda r: emitted_results.append(r))

        pipeline = clean_bridge.inference_worker.pipeline
        pipeline.action_classifier.classify = lambda **kwargs: ("PICK_RED", 0.90)
        pipeline.interaction_engine.evaluate_interaction = lambda **kwargs: {
            "state": "GRASPING",
            "target_object": "RED_SAMPLE",
        }

        clean_bridge.startMonitoring()

        # Submit frame with malicious/spoofed metadata claiming S5
        clean_bridge.inference_worker.submit_frame(
            frame=dummy_frame,
            metadata={"expected_step": "S5", "status": "COMPLETED"},
        )

        start_wait = time.time()
        while not emitted_results and (time.time() - start_wait) < 2.0:
            time.sleep(0.01)

        assert len(emitted_results) >= 1
        res = emitted_results[0]
        # FSM authoritative state (S1 -> S2) prevails
        assert res["expected_step"] == "S1"
        assert res["status"] == "VALID"
        assert res["next_step"] == "S2"

    def test_threading_boundary_zero_extra_fsm_threads(self, clean_bridge):
        """Requirement J: No separate FSM thread is created; worker remains the single background AI thread."""
        initial_threads = threading.enumerate()
        clean_bridge.startMonitoring()

        active_threads = threading.enumerate()
        worker_threads = [t for t in active_threads if t.name == "AIInferenceWorker"]
        fsm_threads = [t for t in active_threads if "FSM" in t.name]

        assert len(worker_threads) == 1
        assert len(fsm_threads) == 0  # FSM does NOT spawn an extra thread
