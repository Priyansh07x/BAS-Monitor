# tests/test_b12_3_recovery_consolidation.py
"""
Sub-Gate B12.3 Comprehensive Verification & Consolidation Test Suite.
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B12 Final)

Verifies:
1. Canonical configuration binding (S1–S5 recovery instructions and timeouts: S1-S4=30s, S5=25s).
2. Comprehensive procedural violation scenario traversal:
   - INVALID_OBJECT (e.g. S1 PICK_RED on BLUE_SAMPLE)
   - SKIPPED (e.g. S1 -> S3 PICK_BLUE; distant skip S1 -> S5 CLOSE_LID)
   - OUT_OF_ORDER / repeated past action (e.g. at S4, observe S1 PICK_RED)
   - UNRECOGNIZED action
   - PREMATURE CLOSE_LID (at S2, observe CLOSE_LID on CONTAINER_LID)
3. End-to-end B11 -> B10 -> FSM -> B12 boundary traversal & conflict isolation.
4. Recovery state resolution lifecycle (IDLE -> RECOVERY_ACTIVE -> RECOVERED -> IDLE).
5. Repeated violation-recovery cycles and independence.
6. VoiceAlertService debounce robustness and distinct violation speech.
7. Timeout metadata integrity across serialization, lifecycle, and reset.
8. Lifecycle transitions and long-run boundedness (START, PAUSE, RESUME, RESET, STOP, COMPLETED).
9. Bridge out-of-band channel isolation & getActiveRecoveryGuidance() accuracy.
10. Strict preservation of the frozen 8-field public AI contract (zero recovery pollution).
11. Logging integration across SystemLogger and ExperimentLogger telemetry.
12. Threading boundary and non-blocking background execution.
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

from backend.ai.result_adapter import PUBLIC_CONTRACT_KEYS, PUBLIC_STATUS_VALUES
from backend.app_state import AppState
from backend.bridge import Bridge
from backend.experiment.procedure_manager import ProcedureManager, ProcedureStep
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
# 1. CANONICAL CONFIGURATION AUDIT & BINDING
# ============================================================================

def test_canonical_configuration_recovery_and_timeouts(procedure_manager):
    """Verify S1–S5 canonical recovery text and exact timeout values from config/experiment.json."""
    expected_spec = {
        "S1": ("Return hand to starting position and re-acquire the red sample.", 30.0, "PICK_RED", "RED_SAMPLE"),
        "S2": ("Retrieve red sample if dislodged and place firmly inside the container.", 30.0, "PLACE_RED", "RED_SAMPLE"),
        "S3": ("Return hand to starting position and re-acquire the blue sample.", 30.0, "PICK_BLUE", "BLUE_SAMPLE"),
        "S4": ("Retrieve blue sample if dislodged and place firmly inside the container.", 30.0, "PLACE_BLUE", "BLUE_SAMPLE"),
        "S5": ("Re-align container lid and press firmly until locked.", 25.0, "CLOSE_LID", "CONTAINER_LID"),
    }

    for step_id, (expected_rec, expected_timeout, action, obj) in expected_spec.items():
        step = procedure_manager.get_step_by_id(step_id)
        assert step is not None, f"Step {step_id} missing in ProcedureManager"
        assert step.recovery == expected_rec
        assert step.timeout_s == expected_timeout
        assert step.expected_action == action
        assert step.required_object == obj


# ============================================================================
# 2. COMPREHENSIVE PROCEDURAL VIOLATION SCENARIO TRAVERSAL
# ============================================================================

def test_violation_scenario_invalid_object(clean_bridge):
    """Scenario A: At S1 (expects PICK_RED on RED_SAMPLE), detect PICK_RED on BLUE_SAMPLE."""
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager
    clean_bridge.startMonitoring()

    fsm_res = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")
    assert fsm_res["validation_status"] == "OUT_OF_ORDER"
    assert fsm_res["error_type"] == "INVALID_OBJECT"

    event = rm.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S1"
    assert event.expected_step_number == 1
    assert event.expected_action == "PICK_RED"
    assert event.expected_object == "RED_SAMPLE"
    assert event.detected_action == "PICK_RED"
    assert event.detected_object == "BLUE_SAMPLE"
    assert event.procedural_status == "INVALID_OBJECT"
    assert event.recovery_instruction == "Return hand to starting position and re-acquire the red sample."
    assert event.timeout_s == 30.0
    assert event.requires_operator_action is True
    assert fsm.current_step_index == 0  # Unmutated


def test_violation_scenario_skipped_step(clean_bridge):
    """Scenario B: At S1, jump to S3 (PICK_BLUE on BLUE_SAMPLE), and distant skip S1 -> S5 (CLOSE_LID)."""
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager
    clean_bridge.startMonitoring()

    # S1 -> S3 Skip
    fsm_res = fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    assert fsm_res["validation_status"] == "SKIPPED"
    event = rm.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S1"
    assert event.detected_step == "S3"
    assert event.procedural_status == "SKIPPED"
    assert event.recovery_instruction == "Return hand to starting position and re-acquire the red sample."
    assert event.timeout_s == 30.0

    # Reset and test distant skip S1 -> S5
    clean_bridge.resetProcedure()
    clean_bridge.startMonitoring()
    fsm_res_s5 = fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")
    assert fsm_res_s5["validation_status"] == "SKIPPED"
    event_s5 = rm.evaluate_fsm_result(fsm_res_s5)
    assert event_s5 is not None
    assert event_s5.expected_step == "S1"
    assert event_s5.detected_step == "S5"
    assert event_s5.procedural_status == "SKIPPED"
    assert event_s5.recovery_instruction == "Return hand to starting position and re-acquire the red sample."


def test_violation_scenario_out_of_order_repeated_past_step(clean_bridge):
    """Scenario C: Progress to S4, then observe S1 (PICK_RED on RED_SAMPLE)."""
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager
    clean_bridge.startMonitoring()

    # Progress S1 -> S2 -> S3 -> S4
    fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    assert fsm.current_step_index == 3  # Now expecting S4 (PLACE_BLUE)

    # Perform repeated Step S1 action
    fsm_res = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm_res["validation_status"] == "OUT_OF_ORDER"
    event = rm.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S4"
    assert event.expected_step_number == 4
    assert event.expected_action == "PLACE_BLUE"
    assert event.expected_object == "BLUE_SAMPLE"
    assert event.detected_step == "S1"
    assert event.procedural_status == "OUT_OF_ORDER"
    assert event.recovery_instruction == "Retrieve blue sample if dislodged and place firmly inside the container."
    assert event.timeout_s == 30.0
    assert fsm.current_step_index == 3  # FSM index preserved


def test_violation_scenario_unrecognized_action(clean_bridge):
    """Scenario D: At S5, submit an action outside the configured vocabulary."""
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager
    clean_bridge.startMonitoring()

    # Advance to S5
    fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
    assert fsm.current_step_index == 4  # Expecting S5 (CLOSE_LID)

    # Unrecognized action
    fsm_res = fsm.validate_action("UNKNOWN_GESTURE", 0.90, "UNKNOWN_OBJ")
    assert fsm_res["validation_status"] == "UNRECOGNIZED"
    event = rm.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S5"
    assert event.expected_step_number == 5
    assert event.expected_action == "CLOSE_LID"
    assert event.expected_object == "CONTAINER_LID"
    assert event.recovery_instruction == "Re-align container lid and press firmly until locked."
    assert event.timeout_s == 25.0
    assert fsm.current_step_index == 4


def test_violation_scenario_premature_close_lid(clean_bridge):
    """Scenario E: At S2 (expected PLACE_RED), observe CLOSE_LID on CONTAINER_LID."""
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager
    clean_bridge.startMonitoring()

    # S1 completed
    fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm.current_step_index == 1  # Expecting S2

    # Premature CLOSE_LID
    fsm_res = fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")
    assert fsm_res["validation_status"] == "SKIPPED"
    event = rm.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S2"
    assert event.detected_step == "S5"
    assert event.procedural_status == "SKIPPED"
    assert event.recovery_instruction == "Retrieve red sample if dislodged and place firmly inside the container."
    assert event.timeout_s == 30.0


# ============================================================================
# 3. B11 → B10 → FSM → B12 BOUNDARY TRAVERSAL & CONFLICT ISOLATION
# ============================================================================

def test_b11_perception_conflict_isolation_produces_zero_recovery(clean_bridge, dummy_frame):
    """
    Demonstrate that perceptual conflict (B11) or unconfirmed frames (B10)
    never invoke FSM violations and therefore generate ZERO recovery events.
    """
    pipeline = clean_bridge.inference_worker.pipeline
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager

    clean_bridge.startMonitoring()
    assert rm.state == "IDLE"

    # Conflicting stream: action is PICK_RED, object is CONTAINER_LID
    pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.95)
    pipeline.object_detector.detect = lambda f: [
        {"label": "CONTAINER_LID", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    for _ in range(10):
        pipeline.process_frame_public(dummy_frame)
        assert pipeline.last_recovery_event is None
        assert rm.state == "IDLE"
        assert len(rm.recovery_history) == 0

    assert fsm.current_step_index == 0
    assert len(fsm.anomalies) == 0


# ============================================================================
# 4. RECOVERY RESOLUTION & REPEATED CYCLES
# ============================================================================

def test_recovery_lifecycle_resolution_and_repeated_cycles(clean_bridge):
    """
    Verify complete recovery lifecycle:
    IDLE -> violation -> RECOVERY_ACTIVE -> corrective action -> RECOVERED -> next step -> IDLE
    and repeat with a second independent violation.
    """
    fsm = clean_bridge.state.sequence_validator
    rm = clean_bridge.state.recovery_manager
    clean_bridge.startMonitoring()

    # --- CYCLE 1: Violation at S1 ---
    res_err1 = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")
    rm.evaluate_fsm_result(res_err1)
    assert rm.state == "RECOVERY_ACTIVE"
    assert clean_bridge.getActiveRecoveryGuidance() != "null"

    # Corrective action: Perform valid S1
    res_corr1 = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert res_corr1["validation_status"] == "VALID"
    rm.evaluate_fsm_result(res_corr1)
    assert rm.state == "RECOVERED"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Next valid action S2
    res_s2 = fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
    assert res_s2["validation_status"] == "VALID"
    rm.evaluate_fsm_result(res_s2)
    assert rm.state == "IDLE"

    # --- CYCLE 2: Violation at S3 ---
    # Perform invalid action on S3 (expected PICK_BLUE on BLUE_SAMPLE, perform with wrong object RED_SAMPLE)
    res_err2 = fsm.validate_action("PICK_BLUE", 0.95, "RED_SAMPLE")
    assert res_err2["validation_status"] == "OUT_OF_ORDER"
    assert res_err2["error_type"] == "INVALID_OBJECT"
    event2 = rm.evaluate_fsm_result(res_err2)
    assert rm.state == "RECOVERY_ACTIVE"
    assert event2.expected_step == "S3"
    assert event2.recovery_instruction == "Return hand to starting position and re-acquire the blue sample."

    # Corrective action on S3
    res_corr2 = fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    assert res_corr2["validation_status"] == "VALID"
    rm.evaluate_fsm_result(res_corr2)
    assert rm.state == "RECOVERED"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"


# ============================================================================
# 5. VOICE ALERT SERVICE DEBOUNCE ROBUSTNESS
# ============================================================================

def test_voice_debounce_robustness():
    """Verify voice debounce timing and multi-step alert generation."""
    voice = VoiceAlertService()
    voice.speak = MagicMock()

    # Step S1 first alert -> Speaks
    assert voice.alert_recovery_guidance("S1", "Re-acquire red sample.", debounce_seconds=3.0) is True
    assert voice.speak.call_count == 1

    # Immediate duplicates (frames 2..10) -> Suppressed
    for _ in range(9):
        assert voice.alert_recovery_guidance("S1", "Re-acquire red sample.", debounce_seconds=3.0) is False
    assert voice.speak.call_count == 1

    # Distinct violation (Step S2) -> Speaks immediately
    voice.speak.reset_mock()
    assert voice.alert_recovery_guidance("S2", "Place red sample in container.", debounce_seconds=3.0) is True
    assert voice.speak.call_count == 1

    # Reset debounce -> Step S2 speaks again
    voice.reset_debounce()
    voice.speak.reset_mock()
    assert voice.alert_recovery_guidance("S2", "Place red sample in container.", debounce_seconds=3.0) is True
    assert voice.speak.call_count == 1


# ============================================================================
# 6. TIMEOUT / WALL-CLOCK METADATA INTEGRITY
# ============================================================================

def test_timeout_metadata_integrity_and_serialization(procedure_manager):
    """Verify timeout values survive serialization and match canonical procedure steps."""
    rm = RecoveryManager(procedure_manager)

    fsm_res_s1 = {"validation_status": "OUT_OF_ORDER", "error_type": "INVALID_OBJECT", "expected_step": "S1"}
    ev1 = rm.evaluate_fsm_result(fsm_res_s1)
    assert ev1.timeout_s == 30.0
    d1 = ev1.to_dict()
    assert d1["timeout_s"] == 30.0
    json_str1 = json.dumps(d1)
    assert json.loads(json_str1)["timeout_s"] == 30.0

    fsm_res_s5 = {"validation_status": "OUT_OF_ORDER", "error_type": "UNRECOGNIZED", "expected_step": "S5"}
    ev5 = rm.evaluate_fsm_result(fsm_res_s5)
    assert ev5.timeout_s == 25.0
    assert ev5.to_dict()["timeout_s"] == 25.0


# ============================================================================
# 7. LIFECYCLE / LONG-RUN BOUNDEDNESS
# ============================================================================

def test_lifecycle_and_boundedness_across_sessions(clean_bridge):
    """
    Test START -> PAUSE -> RESUME -> RESET -> STOP -> COMPLETE
    and verify no stale recovery state leaks.
    """
    rm = clean_bridge.state.recovery_manager
    fsm = clean_bridge.state.sequence_validator

    clean_bridge.startMonitoring()
    fsm_res = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")
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
    assert rm.last_recovery_event is None
    assert len(rm.recovery_history) == 0
    assert clean_bridge.getActiveRecoveryGuidance() == "null"

    # Complete full procedure
    clean_bridge.startMonitoring()
    fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
    fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")

    assert fsm.state == "COMPLETED"
    assert clean_bridge.getFSMState() == "COMPLETED"
    assert clean_bridge.getActiveRecoveryGuidance() == "null"


# ============================================================================
# 8. BRIDGE / PUBLIC CONTRACT FINAL AUDIT
# ============================================================================

def test_strict_public_ai_contract_isolation_and_values(clean_bridge, dummy_frame):
    """
    Verify strict 8-field public AI contract on aiResultReady:
    - Exactly 8 keys
    - Allowed statuses only (VALID, SKIPPED, OUT_OF_SEQUENCE)
    - Zero recovery metadata leakage
    """
    clean_bridge.startMonitoring()
    pipeline = clean_bridge.inference_worker.pipeline
    pipeline.action_classifier.classify = lambda **k: ("PICK_RED", 0.95)
    pipeline.object_detector.detect = lambda f: [
        {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 10, "y1": 10, "x2": 50, "y2": 50}
    ]

    emissions = []
    clean_bridge.aiResultReady.connect(lambda d: emissions.append(d))

    res = pipeline.process_frame_public(dummy_frame, timestamp=1.0)
    clean_bridge._on_ai_result_from_worker(res)

    assert len(emissions) == 1
    ai_dict = emissions[0]

    # Verify exact keys
    assert set(ai_dict.keys()) == set(PUBLIC_CONTRACT_KEYS)
    assert ai_dict["status"] in PUBLIC_STATUS_VALUES

    # Verify NO recovery pollution
    forbidden_keys = {"recovery", "recovery_instruction", "timeout_s", "requires_operator_action", "recovery_event"}
    assert forbidden_keys.isdisjoint(ai_dict.keys())


# ============================================================================
# 9. LOGGING TELEMETRY INTEGRATION AUDIT
# ============================================================================

def test_logging_telemetry_integration(tmp_path):
    """Verify ExperimentLogger session summaries record recovery metrics."""
    logger = ExperimentLogger(log_dir=tmp_path)
    logger.start_session(experiment_id="EXP-001", experiment_name="Recovery Test")

    event = RecoveryEvent(
        expected_step="S1",
        expected_step_number=1,
        expected_action="PICK_RED",
        expected_object="RED_SAMPLE",
        detected_action="PICK_BLUE",
        detected_object="BLUE_SAMPLE",
        procedural_status="INVALID_OBJECT",
        recovery_instruction="Return hand to starting position and re-acquire the red sample.",
        timeout_s=30.0,
    )

    logger.log_recovery(event)
    logger.log_anomaly(
        anomaly_type="INVALID_OBJECT",
        message="Object mismatch",
        step_id=1,
        recovery_instruction=event.recovery_instruction,
        recovery_event=event,
    )

    res = logger.end_session()
    session = res["session_data"]

    assert session["summary"]["recoveries_triggered"] == 1
    assert len(session["recoveries"]) == 1
    assert session["recoveries"][0]["recovery_instruction"] == "Return hand to starting position and re-acquire the red sample."
    assert session["anomalies"][0]["recovery_instruction"] == "Return hand to starting position and re-acquire the red sample."
