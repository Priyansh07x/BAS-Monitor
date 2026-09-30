"""
test_b12_1_recovery_core.py — Unit & Integration Tests for Workstream B Gate B12.1
ISRO SIH26174 BAS Experiment Monitor

Verifies:
1. ProcedureStep binding & loading of canonical recovery metadata from config/experiment.json.
2. Structured RecoveryEvent dataclass schema, fields, and to_dict() serialization.
3. Deterministic RecoveryManager event generation for authoritative FSM violations:
   - INVALID_OBJECT
   - SKIPPED
   - OUT_OF_ORDER
   - UNRECOGNIZED
4. Strict non-violation exclusion:
   - VALID
   - LOW_CONFIDENCE
   - IDLE
   - COMPLETED
   - IGNORED
   - Multimodal perception conflict / uncertainty outputs
5. FSM Authority Invariance: RecoveryManager observation never mutates FSM state, step index, or history.
6. RecoveryManager lifecycle state transitions (IDLE -> RECOVERY_ACTIVE -> RECOVERED -> IDLE, pause, resume, reset).
7. Frozen 8-field public AI contract invariance.
"""

from pathlib import Path
import pytest

from backend.ai.result_adapter import AIResultAdapter
from backend.experiment.procedure_manager import ProcedureManager, ProcedureStep
from backend.experiment.recovery_manager import RecoveryEvent, RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM


@pytest.fixture
def config_path():
    p = Path(__file__).resolve().parent.parent / "config" / "experiment.json"
    assert p.exists(), f"config/experiment.json not found at {p}"
    return str(p)


@pytest.fixture
def procedure_manager(config_path):
    pm = ProcedureManager(config_path)
    assert pm.total_steps == 5
    return pm


@pytest.fixture
def fsm(procedure_manager):
    validator = SequenceValidatorFSM(procedure_manager, min_confidence_threshold=0.60)
    validator.start()
    return validator


@pytest.fixture
def recovery_manager(procedure_manager):
    return RecoveryManager(procedure_manager)


# =============================================================================
# 1. ProcedureStep Binding & Config Loading
# =============================================================================

def test_procedure_step_recovery_attributes():
    """Verify ProcedureStep dataclass accepts and preserves recovery and timeout_s."""
    step = ProcedureStep(
        step_id="S1",
        step_number=1,
        expected_action="PICK_RED",
        instruction="Pick up the red marker.",
        description="Pick up the red marker.",
        required_object="RED_SAMPLE",
        duration_est_seconds=30.0,
        recovery="Return hand to starting position and re-acquire the red sample.",
        timeout_s=30.0,
    )
    assert step.step_id == "S1"
    assert step.recovery == "Return hand to starting position and re-acquire the red sample."
    assert step.timeout_s == 30.0

    d = step.to_dict()
    assert d["recovery"] == "Return hand to starting position and re-acquire the red sample."
    assert d["timeout_s"] == 30.0


def test_procedure_manager_loads_canonical_recovery_from_config(procedure_manager):
    """Verify ProcedureManager extracts exact canonical recovery strings and timeouts for S1–S5 from config/experiment.json."""
    expected_recoveries = {
        "S1": ("Return hand to starting position and re-acquire the red sample.", 30.0),
        "S2": ("Retrieve red sample if dislodged and place firmly inside the container.", 30.0),
        "S3": ("Return hand to starting position and re-acquire the blue sample.", 30.0),
        "S4": ("Retrieve blue sample if dislodged and place firmly inside the container.", 30.0),
        "S5": ("Re-align container lid and press firmly until locked.", 25.0),
    }

    for step_id, (expected_rec, expected_timeout) in expected_recoveries.items():
        step = procedure_manager.get_step_by_id(step_id)
        assert step is not None, f"Step {step_id} not found in ProcedureManager"
        assert step.recovery == expected_rec, f"Step {step_id} recovery mismatch: expected '{expected_rec}', got '{step.recovery}'"
        assert step.timeout_s == expected_timeout, f"Step {step_id} timeout mismatch: expected {expected_timeout}, got {step.timeout_s}"


# =============================================================================
# 2. RecoveryEvent Schema & Serialization
# =============================================================================

def test_recovery_event_schema_and_serialization():
    """Verify RecoveryEvent schema fields and to_dict() serialization."""
    event = RecoveryEvent(
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

    d = event.to_dict()
    assert d["event_type"] == "PROCEDURAL_RECOVERY"
    assert d["timestamp"] == 1700000000.0
    assert d["experiment_id"] == "EXP-001"
    assert d["expected_step"] == "S1"
    assert d["expected_step_number"] == 1
    assert d["expected_action"] == "PICK_RED"
    assert d["expected_object"] == "RED_SAMPLE"
    assert d["detected_action"] == "PICK_BLUE"
    assert d["detected_object"] == "BLUE_SAMPLE"
    assert d["detected_step"] == "S3"
    assert d["procedural_status"] == "OUT_OF_ORDER"
    assert "Out of order" in d["explanation"]
    assert d["recovery_instruction"] == "Return hand to starting position and re-acquire the red sample."
    assert d["timeout_s"] == 30.0
    assert d["requires_operator_action"] is True


# =============================================================================
# 3. Violation Event Generation for Authoritative FSM Violations
# =============================================================================

def test_recovery_on_invalid_object(fsm, recovery_manager):
    """Verify RecoveryManager creates recovery event on INVALID_OBJECT error."""
    # Step S1 expects PICK_RED on RED_SAMPLE. Pass wrong object.
    fsm_res = fsm.validate_action(
        detected_action="PICK_RED",
        confidence=0.95,
        object_name="BLUE_SAMPLE",
    )
    assert fsm_res["validation_status"] == "OUT_OF_ORDER"
    assert fsm_res["error_type"] == "INVALID_OBJECT"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.event_type == "PROCEDURAL_RECOVERY"
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
    assert recovery_manager.state == "RECOVERY_ACTIVE"


def test_recovery_on_skipped_step(fsm, recovery_manager):
    """Verify RecoveryManager creates recovery event on SKIPPED step."""
    # Currently at S1. Jump directly to S3 (PICK_BLUE).
    fsm_res = fsm.validate_action(
        detected_action="PICK_BLUE",
        confidence=0.92,
        object_name="BLUE_SAMPLE",
    )
    assert fsm_res["validation_status"] == "SKIPPED"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S1"
    assert event.detected_step == "S3"
    assert event.procedural_status == "SKIPPED"
    assert event.recovery_instruction == "Return hand to starting position and re-acquire the red sample."
    assert recovery_manager.state == "RECOVERY_ACTIVE"


def test_recovery_on_out_of_order(fsm, recovery_manager):
    """Verify RecoveryManager creates recovery event on repeated / OUT_OF_ORDER step."""
    # First complete S1 validly
    fsm_res1 = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm_res1["validation_status"] == "VALID"
    assert fsm.current_step_index == 1  # Now at S2

    # Now detect repeated S1 (PICK_RED)
    fsm_res2 = fsm.validate_action("PICK_RED", 0.90, "RED_SAMPLE")
    assert fsm_res2["validation_status"] == "OUT_OF_ORDER"

    event = recovery_manager.evaluate_fsm_result(fsm_res2)
    assert event is not None
    assert event.expected_step == "S2"
    assert event.expected_action == "PLACE_RED"
    assert event.detected_action == "PICK_RED"
    assert event.procedural_status == "OUT_OF_ORDER"
    assert event.recovery_instruction == "Retrieve red sample if dislodged and place firmly inside the container."
    assert recovery_manager.state == "RECOVERY_ACTIVE"


def test_recovery_on_unrecognized_action(fsm, recovery_manager):
    """Verify RecoveryManager creates recovery event on UNRECOGNIZED action."""
    fsm_res = fsm.validate_action("UNKNOWN_GESTURE", 0.85, "wrench")
    assert fsm_res["validation_status"] == "UNRECOGNIZED"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is not None
    assert event.expected_step == "S1"
    assert event.procedural_status == "UNRECOGNIZED"
    assert event.recovery_instruction == "Return hand to starting position and re-acquire the red sample."
    assert recovery_manager.state == "RECOVERY_ACTIVE"


# =============================================================================
# 4. Strict Non-Violation Exclusion
# =============================================================================

def test_no_recovery_on_valid_actions(fsm, recovery_manager):
    """Verify VALID actions do not trigger recovery events."""
    fsm_res = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm_res["validation_status"] == "VALID"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is None
    assert recovery_manager.state == "IDLE"
    assert recovery_manager.last_recovery_event is None


def test_no_recovery_on_low_confidence(fsm, recovery_manager):
    """Verify LOW_CONFIDENCE actions do not trigger recovery events."""
    fsm_res = fsm.validate_action("PICK_RED", 0.30, "RED_SAMPLE")
    assert fsm_res["validation_status"] == "LOW_CONFIDENCE"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is None
    assert recovery_manager.state == "IDLE"
    assert recovery_manager.last_recovery_event is None


def test_no_recovery_on_idle(fsm, recovery_manager):
    """Verify IDLE actions do not trigger recovery events."""
    fsm_res = fsm.validate_action("IDLE", 0.99)
    assert fsm_res["validation_status"] == "IDLE"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is None
    assert recovery_manager.state == "IDLE"


def test_no_recovery_on_completed_procedure(procedure_manager, recovery_manager):
    """Verify completed procedure evaluation does not trigger recovery."""
    fsm = SequenceValidatorFSM(procedure_manager)
    fsm.start()
    fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
    fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
    fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
    fsm_res_final = fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")
    assert fsm_res_final.get("is_complete") is True or fsm.state == "COMPLETED"

    # Validate once more when completed
    fsm_res_post = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm_res_post["validation_status"] in ("IGNORED", "COMPLETED")

    event = recovery_manager.evaluate_fsm_result(fsm_res_post)
    assert event is None

    # Test explicit COMPLETED payload
    completed_payload = {"status": "COMPLETED", "validation_status": "COMPLETED", "message": "All steps done"}
    event_comp = recovery_manager.evaluate_fsm_result(completed_payload)
    assert event_comp is None


def test_no_recovery_on_ignored_when_fsm_not_running(procedure_manager, recovery_manager):
    """Verify non-running FSM outputs (IGNORED) do not trigger recovery."""
    fsm = SequenceValidatorFSM(procedure_manager)  # Not started
    fsm_res = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm_res["validation_status"] == "IGNORED"

    event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert event is None


def test_no_recovery_on_perception_uncertainty_or_conflicts(recovery_manager):
    """Verify perception-level uncertainty/conflict payloads without authoritative FSM violation return None."""
    perception_evals = [
        {"action": "PICK_RED", "confidence": 0.40, "evaluation": "MARGINAL", "validation_status": "LOW_CONFIDENCE"},
        {"action": "PICK_RED", "confidence": 0.20, "conflict": "MULTIMODAL_MISMATCH", "validation_status": "LOW_CONFIDENCE"},
        {"action": "IDLE", "confidence": 0.99, "validation_status": "IDLE"},
        {"status": "VALID", "action": "PICK_RED", "confidence": 0.90},
    ]
    for p in perception_evals:
        assert recovery_manager.evaluate_fsm_result(p) is None


# =============================================================================
# 5. FSM Authority & Invariance (Zero Mutation Guarantee)
# =============================================================================

def test_fsm_authority_invariance(fsm, recovery_manager):
    """Verify evaluating recovery never alters FSM step index, state, or validated steps."""
    initial_step_index = fsm.current_step_index
    initial_state = fsm.state
    initial_validated_count = len(fsm.validated_steps)
    initial_anomalies_count = len(fsm.anomalies)

    # Perform violation action
    fsm_res = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")  # INVALID_OBJECT
    fsm_step_index_after_violation = fsm.current_step_index
    fsm_anomalies_after_violation = len(fsm.anomalies)

    # RecoveryManager evaluates the result
    recovery_event = recovery_manager.evaluate_fsm_result(fsm_res)
    assert recovery_event is not None

    # Verify FSM internal state is completely untouched by recovery evaluation
    assert fsm.current_step_index == fsm_step_index_after_violation == initial_step_index
    assert fsm.state == initial_state
    assert len(fsm.validated_steps) == initial_validated_count
    assert len(fsm.anomalies) == fsm_anomalies_after_violation


# =============================================================================
# 6. Lifecycle Transitions (IDLE -> RECOVERY_ACTIVE -> RECOVERED -> IDLE)
# =============================================================================

def test_recovery_lifecycle_state_transitions(fsm, recovery_manager):
    """Verify complete lifecycle of recovery manager states across violation and recovery."""
    assert recovery_manager.state == "IDLE"

    # Step 1: Procedural violation -> RECOVERY_ACTIVE
    fsm_res_violation = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")
    ev1 = recovery_manager.evaluate_fsm_result(fsm_res_violation)
    assert ev1 is not None
    assert recovery_manager.state == "RECOVERY_ACTIVE"
    assert recovery_manager.last_recovery_event == ev1
    assert len(recovery_manager.recovery_history) == 1

    # Step 2: Another violation while active -> stays RECOVERY_ACTIVE, updates history
    fsm_res_violation2 = fsm.validate_action("UNKNOWN_ACTION", 0.90)
    ev2 = recovery_manager.evaluate_fsm_result(fsm_res_violation2)
    assert ev2 is not None
    assert recovery_manager.state == "RECOVERY_ACTIVE"
    assert recovery_manager.last_recovery_event == ev2
    assert len(recovery_manager.recovery_history) == 2

    # Step 3: Operator performs correct action -> RECOVERED
    fsm_res_valid = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    assert fsm_res_valid["validation_status"] == "VALID"
    ev3 = recovery_manager.evaluate_fsm_result(fsm_res_valid)
    assert ev3 is None
    assert recovery_manager.state == "RECOVERED"

    # Step 4: Next valid action -> IDLE
    fsm_res_valid2 = fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
    assert fsm_res_valid2["validation_status"] == "VALID"
    ev4 = recovery_manager.evaluate_fsm_result(fsm_res_valid2)
    assert ev4 is None
    assert recovery_manager.state == "IDLE"


def test_recovery_pause_resume_and_reset(fsm, recovery_manager):
    """Verify pause, resume, and reset methods on RecoveryManager."""
    # Trigger active recovery
    fsm_res_violation = fsm.validate_action("PICK_RED", 0.95, "BLUE_SAMPLE")
    recovery_manager.evaluate_fsm_result(fsm_res_violation)
    assert recovery_manager.state == "RECOVERY_ACTIVE"

    # Pause
    recovery_manager.pause()
    assert recovery_manager.state == "PAUSED"
    # While paused, evaluate returns None
    fsm_res_violation2 = fsm.validate_action("UNKNOWN", 0.90)
    assert recovery_manager.evaluate_fsm_result(fsm_res_violation2) is None

    # Resume
    recovery_manager.resume()
    assert recovery_manager.state == "RECOVERY_ACTIVE"

    # Reset
    recovery_manager.reset()
    assert recovery_manager.state == "IDLE"
    assert recovery_manager.last_recovery_event is None
    assert len(recovery_manager.recovery_history) == 0


# =============================================================================
# 7. Frozen 8-Field Public AI Contract Invariance
# =============================================================================

def test_frozen_8_field_public_ai_contract_invariance(procedure_manager):
    """Verify that AIResultAdapter outputs strictly conform to the 8-field contract."""
    fsm = SequenceValidatorFSM(procedure_manager)
    fsm.start()
    adapter = AIResultAdapter()

    fsm_res = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
    adapted = adapter.adapt(fsm_res)

    expected_keys = {
        "timestamp",
        "action",
        "object",
        "confidence",
        "expected_step",
        "detected_step",
        "status",
        "next_step",
    }
    assert set(adapted.keys()) == expected_keys
    assert len(adapted) == 8
