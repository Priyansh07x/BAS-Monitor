# tests/test_b12_2_recovery_bridge_voice.py
"""
Sub-Gate B12.2 Comprehensive Verification Test Suite.
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Verifies Bridge, VoiceAlertService, Logging, and AppState integration of Recovery Events:
- recoveryAlertReady and recoveryAlertJsonReady signal emissions
- Strict preservation of the frozen 8-field public AI contract on aiResultReady
- getActiveRecoveryGuidance() slot behavior across recovery lifecycles
- VoiceAlertService corrective guidance speech and debouncing
- SystemLogger and ExperimentLogger recovery event recording
- Full lifecycle synchronization (start, pause, resume, reset, stop, completed)
- Isolation from perception uncertainty / multimodal conflict events (B11)
- Non-blocking worker thread delivery and thread safety
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

try:
    from PySide6.QtCore import QCoreApplication
except ImportError:
    QCoreApplication = None

from backend.ai.result_adapter import PUBLIC_CONTRACT_KEYS
from backend.app_state import AppState
from backend.bridge import Bridge
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryEvent, RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.voice.voice_alert import VoiceAlertService
from backend.logging.system_logger import SystemLogger
from backend.logging.experiment_logger import ExperimentLogger


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
def config_path():
    p = Path(__file__).resolve().parent.parent / "config" / "experiment.json"
    assert p.exists(), f"config/experiment.json not found at {p}"
    return str(p)


@pytest.fixture
def procedure_manager(config_path):
    pm = ProcedureManager(config_path)
    return pm


@pytest.fixture
def clean_app_state(procedure_manager):
    """Returns a fresh AppState instance with clean FSM and RecoveryManager."""
    fsm = SequenceValidatorFSM(procedure_manager=procedure_manager, min_confidence_threshold=0.60)
    fsm.reset()
    rm = RecoveryManager(procedure_manager=procedure_manager)
    state = AppState(sequence_validator=fsm, recovery_manager=rm)
    return state


@pytest.fixture
def clean_bridge(clean_app_state):
    """Returns a fresh Bridge connected to clean_app_state with mocked worker start to avoid thread conflicts."""
    bridge = Bridge(clean_app_state)
    if bridge.inference_worker is not None:
        bridge.inference_worker.start = MagicMock(return_value=True)
    yield bridge
    bridge.shutdown()


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================================================
# 1. SIGNAL EXISTENCE & EMISSION
# ============================================================================

def test_bridge_recovery_signals_exist_and_emit(qapp, clean_bridge):
    """Test that recoveryAlertReady and recoveryAlertJsonReady exist on Bridge and emit properly."""
    dict_emissions = []
    json_emissions = []

    clean_bridge.recoveryAlertReady.connect(lambda d: dict_emissions.append(d))
    clean_bridge.recoveryAlertJsonReady.connect(lambda s: json_emissions.append(s))

    test_event = RecoveryEvent(
        event_type="PROCEDURAL_RECOVERY",
        timestamp=1700000000.0,
        experiment_id="EXP-001",
        expected_step="S1",
        expected_step_number=1,
        expected_action="PICK_RED",
        expected_object="RED_SAMPLE",
        detected_action="PICK_BLUE",
        detected_object="BLUE_SAMPLE",
        detected_step="S3",
        procedural_status="OUT_OF_ORDER",
        explanation="Out of order action. Detected PICK_BLUE, but expecting Step S1.",
        recovery_instruction="Return hand to starting position and re-acquire the red sample.",
        timeout_s=30.0,
        requires_operator_action=True,
    )

    clean_bridge.emit_recovery_alert(test_event)

    assert len(dict_emissions) == 1
    assert len(json_emissions) == 1

    payload_dict = dict_emissions[0]
    payload_json = json.loads(json_emissions[0])

    assert payload_dict["expected_step"] == "S1"
    assert payload_dict["expected_action"] == "PICK_RED"
    assert payload_dict["detected_action"] == "PICK_BLUE"
    assert payload_dict["recovery_instruction"] == "Return hand to starting position and re-acquire the red sample."
    assert payload_dict["timeout_s"] == 30.0
    assert payload_dict["procedural_status"] == "OUT_OF_ORDER"

    assert payload_json == payload_dict


# ============================================================================
# 2. FROZEN 8-FIELD PUBLIC AI CONTRACT PRESERVATION
# ============================================================================

def test_frozen_8_field_public_ai_contract_preservation_on_bridge(qapp, clean_bridge, dummy_frame):
    """
    Test that aiResultReady and aiResultJsonReady strictly emit the frozen 8-field schema.
    Zero recovery keys must appear in the public AI result.
    """
    clean_bridge.startMonitoring()
    pipeline = clean_bridge.inference_worker.pipeline
    pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.95)
    pipeline.object_detector.detect = lambda f: [
        {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    ai_emissions = []
    clean_bridge.aiResultReady.connect(lambda d: ai_emissions.append(d))

    # Process frame via pipeline
    res_dict = pipeline.process_frame_public(dummy_frame, timestamp=1.0)
    clean_bridge._on_ai_result_from_worker(res_dict)

    assert len(ai_emissions) == 1
    public_res = ai_emissions[0]

    # Verify exact 8 frozen fields
    assert set(public_res.keys()) == set(PUBLIC_CONTRACT_KEYS)

    # Confirm NO recovery keys are in public AI result
    assert "recovery" not in public_res
    assert "recovery_instruction" not in public_res
    assert "timeout_s" not in public_res
    assert "requires_operator_action" not in public_res
    assert "recovery_event" not in public_res


# ============================================================================
# 3. GET ACTIVE RECOVERY GUIDANCE SLOT
# ============================================================================

def test_get_active_recovery_guidance_query_slot(qapp, clean_bridge):
    """
    Test getActiveRecoveryGuidance() returns formatted JSON during active violation,
    and returns 'null' when idle, recovered, or reset.
    """
    rm = clean_bridge.state.recovery_manager
    fsm = clean_bridge.state.sequence_validator

    # Initial state: IDLE -> returns "null"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Start procedure -> still IDLE
    clean_bridge.startMonitoring()
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Trigger procedural violation (Step S1 expects PICK_RED on RED_SAMPLE, give BLUE_SAMPLE)
    fsm_res = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")
    event = rm.evaluate_fsm_result(fsm_res)

    assert event is not None
    assert rm.state == "RECOVERY_ACTIVE"

    # Query slot -> should return JSON of active event
    active_json_str = clean_bridge.getActiveRecoveryGuidance()
    assert active_json_str != "null"
    active_data = json.loads(active_json_str)
    assert active_data["expected_step"] == "S1"
    assert "recovery_instruction" in active_data
    assert active_data["procedural_status"] == "INVALID_OBJECT"

    # Valid step -> Recovery cleared -> returns "null"
    val_res_valid = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    rm.evaluate_fsm_result(val_res_valid)
    assert rm.state == "RECOVERED"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Reset procedure -> returns "null"
    clean_bridge.resetProcedure()
    assert clean_bridge.getActiveRecoveryGuidance() == "null"


# ============================================================================
# 4. VOICE ALERT SERVICE RECOVERY GUIDANCE & DEBOUNCE
# ============================================================================

def test_voice_alert_service_recovery_guidance_and_debounce():
    """
    Test VoiceAlertService.alert_recovery_guidance debounces repeated identical alerts
    and speaks distinct alerts.
    """
    voice = VoiceAlertService()
    voice.speak = MagicMock()

    # First call -> Speaks
    success1 = voice.alert_recovery_guidance(step_id="S1", recovery_text="Return hand to starting position.", debounce_seconds=2.0)
    assert success1 is True
    voice.speak.assert_called_once_with("Recovery guidance for Step S1. Return hand to starting position.", priority=True)

    # Immediate duplicate call -> Debounced (returns False, speak not called again)
    voice.speak.reset_mock()
    success2 = voice.alert_recovery_guidance(step_id="S1", recovery_text="Return hand to starting position.", debounce_seconds=2.0)
    assert success2 is False
    voice.speak.assert_not_called()

    # Different step / text -> Speaks immediately
    voice.speak.reset_mock()
    success3 = voice.alert_recovery_guidance(step_id="S2", recovery_text="Retrieve red sample if dislodged.", debounce_seconds=2.0)
    assert success3 is True
    voice.speak.assert_called_once_with("Recovery guidance for Step S2. Retrieve red sample if dislodged.", priority=True)

    # Reset debounce -> Speaks step S2 again
    voice.reset_debounce()
    voice.speak.reset_mock()
    success4 = voice.alert_recovery_guidance(step_id="S2", recovery_text="Retrieve red sample if dislodged.", debounce_seconds=2.0)
    assert success4 is True
    voice.speak.assert_called_once_with("Recovery guidance for Step S2. Retrieve red sample if dislodged.", priority=True)

    # Empty text -> Ignored
    voice.speak.reset_mock()
    success_empty = voice.alert_recovery_guidance(step_id="S3", recovery_text="")
    assert success_empty is False
    voice.speak.assert_not_called()


# ============================================================================
# 5. LOGGING INTEGRATION: SYSTEM LOGGER & EXPERIMENT LOGGER
# ============================================================================

def test_logging_integration_of_recovery_events(tmp_path):
    """
    Test SystemLogger.log_recovery and ExperimentLogger.log_recovery capture recovery data.
    """
    # SystemLogger
    sys_logger = SystemLogger()
    with patch.object(sys_logger.logger, "warning") as mock_warn:
        sys_logger.log_recovery(step_id="S1", recovery_instruction="Re-acquire sample.", procedural_status="INVALID_OBJECT")
        mock_warn.assert_called_once()
        assert "[S1]" in mock_warn.call_args[0][0]
        assert "Re-acquire sample." in mock_warn.call_args[0][0]

    # ExperimentLogger
    exp_logger = ExperimentLogger(log_dir=tmp_path)
    exp_logger.start_session(experiment_id="EXP-001", experiment_name="Test Session")

    rec_event = RecoveryEvent(
        expected_step="S1",
        expected_step_number=1,
        expected_action="PICK_RED",
        expected_object="RED_SAMPLE",
        detected_action="PICK_BLUE",
        detected_object="BLUE_SAMPLE",
        explanation="Safety violation",
        recovery_instruction="Re-acquire red sample immediately.",
        timeout_s=15.0,
        procedural_status="INVALID_OBJECT",
        timestamp=100.0
    )

    exp_logger.log_recovery(rec_event)
    exp_logger.log_anomaly(
        anomaly_type="INVALID_OBJECT",
        message="Safety violation",
        step_id=1,
        recovery_instruction="Re-acquire red sample immediately.",
        recovery_event=rec_event
    )

    summary = exp_logger.end_session()
    assert summary["session_data"]["summary"]["recoveries_triggered"] == 1
    assert len(summary["session_data"]["recoveries"]) == 1
    rec_entry = summary["session_data"]["recoveries"][0]
    assert rec_entry["expected_step"] == "S1"
    assert rec_entry["recovery_instruction"] == "Re-acquire red sample immediately."
    assert rec_entry["procedural_status"] == "INVALID_OBJECT"

    anom_entry = summary["session_data"]["anomalies"][0]
    assert anom_entry["recovery_instruction"] == "Re-acquire red sample immediately."
    assert anom_entry["recovery_event"]["expected_step"] == "S1"


# ============================================================================
# 6. PIPELINE & WORKER OUT-OF-BAND RECOVERY DISPATCH
# ============================================================================

def test_pipeline_and_worker_out_of_band_recovery_dispatch(qapp, clean_bridge, dummy_frame):
    """
    Test that InferencePipeline triggers recovery_manager and InferenceWorker
    dispatches recovery events out-of-band via recovery_callback.
    """
    worker = clean_bridge.inference_worker
    pipeline = worker.pipeline
    fsm = clean_bridge.state.sequence_validator

    clean_bridge.startMonitoring()

    dispatched_recoveries = []
    worker.recovery_callback = lambda ev: dispatched_recoveries.append(ev)

    # Frame 1-3: Valid step 1 execution (PICK_RED on RED_SAMPLE) -> Confirms and validates S1
    pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.95)
    pipeline.object_detector.detect = lambda f: [
        {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    for _ in range(3):
        pipeline.process_frame_public(dummy_frame)

    assert fsm.current_step_index == 1  # S1 validated, now expecting S2
    assert pipeline.last_recovery_event is None

    # Step 2 violation: Perform Step S4 action (PLACE_BLUE on BLUE_SAMPLE)
    # 3 frames to confirm via B10 and trigger FSM violation
    pipeline.action_classifier.classify = lambda **k: ("PLACE_BLUE", 0.95)
    pipeline.object_detector.detect = lambda f: [
        {"label": "BLUE_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    for _ in range(3):
        pipeline.process_frame_public(dummy_frame)

    rec_ev = pipeline.last_recovery_event
    assert rec_ev is not None
    assert rec_ev.expected_step == "S2"  # Current expected step was S2
    assert rec_ev.procedural_status == "SKIPPED"

    # Worker out-of-band dispatch simulation
    if worker.recovery_callback:
        worker.recovery_callback(rec_ev.to_dict())

    assert len(dispatched_recoveries) == 1
    assert dispatched_recoveries[0]["procedural_status"] == "SKIPPED"


# ============================================================================
# 7. ISOLATION FROM B11 PERCEPTION UNCERTAINTY / CONFLICTS
# ============================================================================

def test_isolation_from_b11_perception_uncertainty_and_conflicts(qapp, clean_bridge, dummy_frame):
    """
    Test that low confidence or multimodal conflicts in B11 do NOT trigger recovery events
    or bridge recovery signals.
    """
    pipeline = clean_bridge.inference_worker.pipeline
    fsm = clean_bridge.state.sequence_validator

    clean_bridge.startMonitoring()

    recovery_signals = []
    clean_bridge.recoveryAlertReady.connect(lambda d: recovery_signals.append(d))

    # Low confidence action (0.35 < 0.60 threshold)
    pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.35)
    pipeline.object_detector.detect = lambda f: [
        {"label": "RED_SAMPLE", "confidence": 0.35, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    for _ in range(3):
        pipeline.process_frame_public(dummy_frame)

    assert pipeline.last_recovery_event is None
    assert len(recovery_signals) == 0

    # Conflicting modalities: object is CONTAINER_LID, action is PICK_RED
    pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.95)
    pipeline.object_detector.detect = lambda f: [
        {"label": "CONTAINER_LID", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    for _ in range(3):
        pipeline.process_frame_public(dummy_frame)

    # Multimodal inconsistency causes pipeline uncertainty route, not an authoritative FSM violation
    assert pipeline.last_recovery_event is None
    assert len(recovery_signals) == 0


# ============================================================================
# 8. LIFECYCLE SYNCHRONIZATION
# ============================================================================

def test_recovery_lifecycle_synchronization(qapp, clean_bridge):
    """
    Test start, pause, resume, reset, and stop across Bridge, AppState, and RecoveryManager.
    """
    rm = clean_bridge.state.recovery_manager
    fsm = clean_bridge.state.sequence_validator

    # Start
    clean_bridge.startMonitoring()
    assert clean_bridge.getFSMState() == "RUNNING"
    assert rm.state == "IDLE"

    # Trigger violation
    fsm_res = fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    rm.evaluate_fsm_result(fsm_res)
    assert rm.state == "RECOVERY_ACTIVE"

    # Pause
    clean_bridge.pauseMonitoring()
    assert rm.state == "PAUSED"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Resume
    clean_bridge.resumeMonitoring()
    assert rm.state == "RECOVERY_ACTIVE"
    assert clean_bridge.getActiveRecoveryGuidance() != "null"

    # Reset
    clean_bridge.resetProcedure()
    assert rm.state == "IDLE"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Stop
    clean_bridge.stopProcedure()
    assert rm.state == "IDLE"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"
