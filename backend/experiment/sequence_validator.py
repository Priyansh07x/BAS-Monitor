"""
sequence_validator.py — Deterministic Finite State Machine (FSM) Engine
ISRO SIH26174 BAS Experiment Monitor

Enforces deterministic protocol adherence for astronaut experiment steps.
Evaluates incoming detected actions against the active procedure sequence,
detects skipped steps (e.g. S_n+2 before S_n+1), out-of-order execution,
triggers voice warnings via VoiceAlertService, and writes structured records via ExperimentLogger.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from .procedure_manager import ProcedureManager, ProcedureStep
from ..voice.voice_alert import get_voice_service
from ..logging.experiment_logger import ExperimentLogger
from ..logging.system_logger import get_system_logger


class SequenceValidatorFSM:
    """
    Deterministic sequence validation engine.
    """

    def __init__(
        self,
        procedure_manager: Optional[ProcedureManager] = None,
        min_confidence_threshold: float = 0.70,
    ):
        self.pm: ProcedureManager = procedure_manager or ProcedureManager()
        self.min_confidence = min_confidence_threshold

        self.current_step_index: int = 0
        self.state: str = "IDLE"  # 'IDLE', 'RUNNING', 'COMPLETED', 'PAUSED'
        self.validated_steps: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []

        self._voice = get_voice_service()
        self._slog = get_system_logger()
        self._exp_logger = ExperimentLogger()
        self._session_active: bool = False

    def load_procedure(self, pm_or_path: ProcedureManager | str) -> bool:
        """Load or replace the current active procedure manager."""
        if isinstance(pm_or_path, ProcedureManager):
            self.pm = pm_or_path
            return True
        elif isinstance(pm_or_path, str):
            self.pm = ProcedureManager(pm_or_path)
            return self.pm.total_steps > 0
        return False

    def start(self, operator_id: str = "Astronaut-01", notes: str = "") -> Dict[str, Any]:
        """Start a new monitored experiment sequence."""
        self.reset()
        self.state = "RUNNING"

        exp_id = self.pm.experiment_id
        exp_name = self.pm.experiment_name

        session = self._exp_logger.start_session(
            experiment_id=exp_id,
            experiment_name=exp_name,
            operator_id=operator_id,
            notes=notes,
        )
        self._session_active = True

        first_step = self.get_current_expected_step()
        if first_step:
            self._voice.alert_next_step(first_step.step_number, first_step.description)
            self._slog.info(f"FSM Started. Expecting Step {first_step.step_number}: {first_step.expected_action}")

        return session

    def validate_action(
        self,
        detected_action: str,
        confidence: float,
        object_name: Optional[str] = None,
        details: str = "",
    ) -> Dict[str, Any]:
        """
        Process a detected action event through the deterministic state machine.
        Returns evaluation dict with validation_status: 'VALID', 'SKIPPED', 'OUT_OF_ORDER', 'UNRECOGNIZED'.
        """
        if self.state != "RUNNING":
            return {
                "status": "IGNORED",
                "reason": f"Validator not in RUNNING state (current: {self.state})",
                "detected_action": detected_action,
            }

        detected_action = detected_action.strip()
        expected_step = self.get_current_expected_step()

        if not expected_step:
            self.state = "COMPLETED"
            return {"status": "COMPLETED", "message": "All procedure steps already validated."}

        # Find if detected action belongs to ANY step in this experiment
        matching_step = self.pm.get_step_by_action(detected_action)

        # -------------------------------------------------------------
        # CASE 1: MATCHES CURRENT EXPECTED STEP (VALID)
        # -------------------------------------------------------------
        if detected_action == expected_step.expected_action:
            if confidence < self.min_confidence:
                result = {
                    "validation_status": "LOW_CONFIDENCE",
                    "step_id": expected_step.step_number,
                    "expected_action": expected_step.expected_action,
                    "detected_action": detected_action,
                    "confidence": confidence,
                    "message": f"Action matches Step {expected_step.step_number} but confidence {int(confidence*100)}% is below threshold.",
                }
                if self._session_active:
                    self._exp_logger.log_anomaly("LOW_CONFIDENCE", result["message"], expected_step.step_number)
                return result

            # Valid Step Transition
            step_record = {
                "step_id": expected_step.step_number,
                "title": expected_step.description,
                "expected_action": expected_step.expected_action,
                "detected_action": detected_action,
                "confidence": confidence,
                "validation_status": "VALID",
                "timestamp": datetime.now().isoformat(),
            }
            self.validated_steps.append(step_record)

            if self._session_active:
                self._exp_logger.log_step(
                    step_id=expected_step.step_number,
                    step_title=expected_step.description,
                    expected_action=expected_step.expected_action,
                    detected_action=detected_action,
                    confidence=confidence,
                    validation_status="VALID",
                    details=details,
                )

            self._slog.info(f"FSM VALID: Step {expected_step.step_number} ({expected_step.expected_action}) confirmed.")

            # Advance to next step
            self.current_step_index += 1
            next_step = self.get_current_expected_step()

            if next_step:
                self._voice.alert_next_step(next_step.step_number, next_step.description)
            else:
                self.state = "COMPLETED"
                self._voice.alert_procedure_completed(self.pm.experiment_name)
                if self._session_active:
                    self._exp_logger.end_session(status="COMPLETED")
                    self._session_active = False

            return {
                "validation_status": "VALID",
                "step_id": expected_step.step_number,
                "step_title": expected_step.description,
                "next_step_id": next_step.step_number if next_step else None,
                "is_complete": self.state == "COMPLETED",
                "confidence": confidence,
            }

        # -------------------------------------------------------------
        # CASE 2: STEP WAS SKIPPED (Detected future step > current_step)
        # -------------------------------------------------------------
        elif matching_step and matching_step.step_number > expected_step.step_number:
            skipped_steps = [
                s for s in self.pm.steps
                if expected_step.step_number <= s.step_number < matching_step.step_number
            ]
            skipped_names = ", ".join([f"Step {s.step_number} ({s.description})" for s in skipped_steps])

            msg = f"Skipped step detected! Performed Step {matching_step.step_number} before completing {skipped_names}."
            self._slog.warn(msg)

            # High priority voice warning
            for s in skipped_steps:
                self._voice.alert_skipped_step(s.step_number, s.description)

            if self._session_active:
                self._exp_logger.log_anomaly("SKIPPED_STEP", msg, expected_step.step_number)
                self._exp_logger.log_step(
                    step_id=matching_step.step_number,
                    step_title=matching_step.description,
                    expected_action=expected_step.expected_action,
                    detected_action=detected_action,
                    confidence=confidence,
                    validation_status="SKIPPED",
                    details=msg,
                )

            # Auto-advance past skipped step to maintain flow
            self.current_step_index = matching_step.step_number

            return {
                "validation_status": "SKIPPED",
                "detected_action": detected_action,
                "skipped_step_numbers": [s.step_number for s in skipped_steps],
                "message": msg,
                "confidence": confidence,
            }

        # -------------------------------------------------------------
        # CASE 3: OUT OF ORDER (Detected previous or out-of-sequence step)
        # -------------------------------------------------------------
        elif matching_step:
            msg = f"Out of order action. Detected {detected_action} (Step {matching_step.step_number}), but expecting Step {expected_step.step_number} ({expected_step.expected_action})."
            self._slog.warn(msg)
            self._voice.alert_out_of_order(expected_step.description, detected_action)

            if self._session_active:
                self._exp_logger.log_anomaly("OUT_OF_ORDER", msg, expected_step.step_number)
                self._exp_logger.log_step(
                    step_id=expected_step.step_number,
                    step_title=expected_step.description,
                    expected_action=expected_step.expected_action,
                    detected_action=detected_action,
                    confidence=confidence,
                    validation_status="OUT_OF_ORDER",
                    details=msg,
                )

            return {
                "validation_status": "OUT_OF_ORDER",
                "expected_action": expected_step.expected_action,
                "detected_action": detected_action,
                "expected_step": expected_step.step_number,
                "detected_step": matching_step.step_number,
                "message": msg,
                "confidence": confidence,
            }

        # -------------------------------------------------------------
        # CASE 4: UNRECOGNIZED ACTION
        # -------------------------------------------------------------
        else:
            return {
                "validation_status": "UNRECOGNIZED",
                "detected_action": detected_action,
                "expected_action": expected_step.expected_action,
                "confidence": confidence,
            }

    def get_current_expected_step(self) -> Optional[ProcedureStep]:
        """Return the step currently expected by the FSM."""
        return self.pm.get_step(self.current_step_index)

    def get_next_expected_step(self) -> Optional[ProcedureStep]:
        """Return the step after the currently expected one."""
        return self.pm.get_step(self.current_step_index + 1)

    def get_progress(self) -> Dict[str, Any]:
        """Return summary progress metrics."""
        total = self.pm.total_steps
        current = self.current_step_index
        percent = int((current / total) * 100) if total > 0 else 0
        return {
            "state": self.state,
            "current_step_index": current,
            "current_step_number": current + 1 if current < total else total,
            "total_steps": total,
            "progress_percent": percent,
            "validated_count": len(self.validated_steps),
            "anomalies_count": len(self.anomalies),
            "is_complete": self.state == "COMPLETED",
        }

    def pause(self) -> None:
        if self.state == "RUNNING":
            self.state = "PAUSED"
            self._slog.info("FSM Sequence paused.")

    def resume(self) -> None:
        if self.state == "PAUSED":
            self.state = "RUNNING"
            self._slog.info("FSM Sequence resumed.")

    def reset(self) -> None:
        """Reset sequence validator back to initial state."""
        self.current_step_index = 0
        self.state = "IDLE"
        self.validated_steps.clear()
        self.anomalies.clear()
        if self._session_active:
            try:
                self._exp_logger.end_session(status="ABORTED")
            except Exception:
                pass
            self._session_active = False


# Module singleton
_validator_instance: Optional[SequenceValidatorFSM] = None


def get_sequence_validator(**kwargs) -> SequenceValidatorFSM:
    """Return or initialize global SequenceValidator singleton."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SequenceValidatorFSM(**kwargs)
    return _validator_instance
