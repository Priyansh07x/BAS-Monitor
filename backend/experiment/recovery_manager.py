"""
recovery_manager.py — Procedural Recovery Handling & Event Generation
ISRO SIH26174 BAS Experiment Monitor — Workstream B Gate B12.1

Observes authoritative SequenceValidatorFSM outputs, generates structured
RecoveryEvent instances with step-bound recovery instructions upon procedural
violations, and tracks internal recovery state without mutating procedural FSM state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional

from backend.experiment.procedure_manager import ProcedureManager, ProcedureStep


@dataclass
class RecoveryEvent:
    """
    Structured procedural recovery event emitted when SequenceValidatorFSM
    identifies an authoritative sequence or object violation.
    """

    event_type: str = "PROCEDURAL_RECOVERY"
    timestamp: float = field(default_factory=time.time)
    experiment_id: str = "EXP-001"
    expected_step: Optional[str] = None
    expected_step_number: Optional[int] = None
    expected_action: Optional[str] = None
    expected_object: Optional[str] = None
    detected_action: Optional[str] = None
    detected_object: Optional[str] = None
    detected_step: Optional[str] = None
    procedural_status: str = "UNKNOWN"
    explanation: str = ""
    recovery_instruction: str = ""
    timeout_s: Optional[float] = 30.0
    requires_operator_action: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert recovery event to dictionary representation."""
        return asdict(self)


class RecoveryManager:
    """
    Deterministic, bounded recovery manager for experiment procedures.

    Observes authoritative SequenceValidatorFSM results, binds canonical step-level
    recovery metadata from ProcedureManager / config, generates RecoveryEvents upon
    genuine procedural violations, and tracks internal recovery lifecycle state.

    State Model:
      - 'IDLE': Normal operation, no active recovery.
      - 'RECOVERY_ACTIVE': A procedural violation has occurred; operator recovery required.
      - 'RECOVERED': Subsequent valid action confirmed following an active recovery.
      - 'PAUSED': Recovery monitoring temporarily paused.
    """

    # Procedural violation statuses that trigger recovery
    VIOLATION_STATUSES = frozenset({
        "OUT_OF_ORDER",
        "SKIPPED",
        "UNRECOGNIZED",
    })

    # Error types that trigger recovery
    VIOLATION_ERROR_TYPES = frozenset({
        "INVALID_OBJECT",
        "SKIPPED_STEP",
        "OUT_OF_ORDER",
        "UNRECOGNIZED",
    })

    # Non-violation statuses that MUST NOT trigger recovery
    NON_VIOLATION_STATUSES = frozenset({
        "VALID",
        "LOW_CONFIDENCE",
        "IDLE",
        "COMPLETED",
        "IGNORED",
    })

    def __init__(self, procedure_manager: Optional[ProcedureManager] = None) -> None:
        self._pm: Optional[ProcedureManager] = procedure_manager
        self._state: str = "IDLE"
        self._previous_state: str = "IDLE"
        self._last_recovery_event: Optional[RecoveryEvent] = None
        self._recovery_history: List[RecoveryEvent] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Current internal recovery state."""
        with self._lock:
            return self._state

    @property
    def last_recovery_event(self) -> Optional[RecoveryEvent]:
        """Most recent recovery event generated, if any."""
        with self._lock:
            return self._last_recovery_event

    @property
    def recovery_history(self) -> List[RecoveryEvent]:
        """Immutable copy of recovery events generated in this session."""
        with self._lock:
            return list(self._recovery_history)

    def set_procedure_manager(self, pm: ProcedureManager) -> None:
        """Bind or update the procedure manager reference."""
        with self._lock:
            self._pm = pm

    def evaluate_fsm_result(
        self,
        fsm_result: Dict[str, Any],
        procedure_manager: Optional[ProcedureManager] = None,
    ) -> Optional[RecoveryEvent]:
        """
        Evaluate an authoritative FSM evaluation result.

        If a genuine procedural violation is identified:
          1. Constructs a RecoveryEvent with step-bound recovery instructions.
          2. Transitions internal state to 'RECOVERY_ACTIVE'.
          3. Returns the RecoveryEvent.

        If the result is VALID:
          1. If currently 'RECOVERY_ACTIVE', transitions to 'RECOVERED'.
          2. If currently 'RECOVERED', transitions to 'IDLE'.
          3. Returns None.

        If the result is LOW_CONFIDENCE, IDLE, IGNORED, COMPLETED, or perception uncertainty:
          - Returns None without creating a recovery event.
        """
        if not isinstance(fsm_result, dict):
            return None

        with self._lock:
            if self._state == "PAUSED":
                return None

            pm = procedure_manager or self._pm

            validation_status = str(fsm_result.get("validation_status") or fsm_result.get("status") or "").upper()
            error_type = str(fsm_result.get("error_type") or "").upper()

            # Check for non-violation statuses first (LOW_CONFIDENCE, IDLE, COMPLETED, IGNORED)
            if validation_status == "VALID" and not error_type:
                if self._state == "RECOVERY_ACTIVE":
                    self._state = "RECOVERED"
                elif self._state == "RECOVERED":
                    self._state = "IDLE"
                return None

            if validation_status in self.NON_VIOLATION_STATUSES and not error_type:
                return None

            # Determine if this constitutes a true procedural violation
            is_violation = (
                validation_status in self.VIOLATION_STATUSES
                or error_type in self.VIOLATION_ERROR_TYPES
            )

            if not is_violation:
                return None

            # Extract step identifiers
            raw_expected_step = fsm_result.get("expected_step") or fsm_result.get("step_id")
            norm_expected_step = ProcedureManager.normalize_step_id(raw_expected_step) if raw_expected_step else None

            # Look up step definition from procedure manager
            step: Optional[ProcedureStep] = None
            if pm:
                if norm_expected_step:
                    step = pm.get_step_by_id(norm_expected_step)
                if not step and isinstance(raw_expected_step, int):
                    step = pm.get_step_by_number(raw_expected_step)

            # Resolve expected metadata
            exp_step_id = (step.step_id if step else norm_expected_step) or str(raw_expected_step or "")
            exp_step_num = (step.step_number if step else fsm_result.get("step_number"))
            exp_action = (step.expected_action if step else fsm_result.get("expected_action"))
            exp_object = (step.required_object if step else fsm_result.get("expected_object"))

            # Resolve canonical recovery instruction
            recovery_instruction = ""
            if step and step.recovery:
                recovery_instruction = step.recovery
            elif exp_action and exp_object:
                recovery_instruction = f"Perform {exp_action} on {exp_object} to continue procedure."
            elif exp_action:
                recovery_instruction = f"Perform {exp_action} to continue procedure."
            else:
                recovery_instruction = "Resume standard procedure flow."

            # Resolve timeout
            timeout_s: Optional[float] = None
            if step and step.timeout_s is not None:
                timeout_s = step.timeout_s
            elif step and step.duration_est_seconds:
                timeout_s = step.duration_est_seconds
            elif "timeout_s" in fsm_result:
                try:
                    timeout_s = float(fsm_result["timeout_s"])
                except (ValueError, TypeError):
                    timeout_s = 30.0
            else:
                timeout_s = 30.0

            # Determine effective procedural status
            effective_status = error_type or validation_status or "OUT_OF_SEQUENCE"

            # Resolve explanation message
            explanation = str(
                fsm_result.get("message")
                or fsm_result.get("reason")
                or f"Procedural violation: {effective_status} at Step {exp_step_id}."
            ).strip()

            detected_act = fsm_result.get("detected_action") or fsm_result.get("action")
            detected_obj = fsm_result.get("detected_object") or fsm_result.get("object")
            detected_step_val = fsm_result.get("detected_step")

            exp_id_str = pm.experiment_id if pm else "EXP-001"

            # Build RecoveryEvent
            event = RecoveryEvent(
                event_type="PROCEDURAL_RECOVERY",
                timestamp=time.time(),
                experiment_id=exp_id_str,
                expected_step=exp_step_id or None,
                expected_step_number=exp_step_num,
                expected_action=exp_action,
                expected_object=exp_object,
                detected_action=detected_act,
                detected_object=detected_obj,
                detected_step=detected_step_val,
                procedural_status=effective_status,
                explanation=explanation,
                recovery_instruction=recovery_instruction,
                timeout_s=timeout_s,
                requires_operator_action=True,
            )

            # Update internal state
            self._state = "RECOVERY_ACTIVE"
            self._last_recovery_event = event
            self._recovery_history.append(event)

            return event

    def reset(self) -> None:
        """Reset internal recovery state and clear history."""
        with self._lock:
            self._state = "IDLE"
            self._previous_state = "IDLE"
            self._last_recovery_event = None
            self._recovery_history.clear()

    def pause(self) -> None:
        """Pause recovery monitoring."""
        with self._lock:
            if self._state != "PAUSED":
                self._previous_state = self._state
                self._state = "PAUSED"

    def resume(self) -> None:
        """Resume recovery monitoring from previous state."""
        with self._lock:
            if self._state == "PAUSED":
                self._state = self._previous_state
