"""
sequence_validator.py — Deterministic Finite State Machine (FSM) Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B9.1)

Enforces deterministic protocol adherence for astronaut experiment steps.
Evaluates incoming detected actions and target objects against the active procedure sequence,
detects skipped steps (e.g. S_n+2 before S_n+1), out-of-order execution, object mismatches,
triggers voice warnings via VoiceAlertService, and writes structured records via ExperimentLogger.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from .procedure_manager import ProcedureManager, ProcedureStep
from ..voice.voice_alert import get_voice_service
from ..logging.experiment_logger import ExperimentLogger
from ..logging.system_logger import get_system_logger


CANONICAL_ACTIONS = {
    "PICK_RED",
    "PLACE_RED",
    "PICK_BLUE",
    "PLACE_BLUE",
    "CLOSE_LID",
    "IDLE",
}

CANONICAL_OBJECTS = {
    "RED_SAMPLE",
    "BLUE_SAMPLE",
    "SAMPLE_CONTAINER",
    "CONTAINER_LID",
    "NONE",
}


class SequenceValidatorFSM:
    """
    Deterministic sequence validation engine for EXP-001 protocol adherence.
    Authoritatively governs expected_step, procedural status, and next_step.
    """

    def __init__(
        self,
        procedure_manager: Optional[ProcedureManager] = None,
        min_confidence_threshold: float = 0.70,
    ):
        if procedure_manager is not None:
            self.pm = procedure_manager
        else:
            from pathlib import Path
            cfg_p = Path("config/experiment.json")
            if cfg_p.exists():
                self.pm = ProcedureManager(cfg_p)
            else:
                self.pm = ProcedureManager()

        self.min_confidence = min_confidence_threshold

        self.current_step_index: int = 0
        self.state: str = "IDLE"  # 'IDLE', 'RUNNING', 'COMPLETED', 'PAUSED'
        self.validated_steps: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []

        self._voice = get_voice_service()
        self._slog = get_system_logger()
        self._exp_logger = ExperimentLogger()
        self._session_active: bool = False
        self._lock = threading.Lock()

    def load_procedure(self, pm_or_path: ProcedureManager | str) -> bool:
        """Load or replace the current active procedure manager."""
        with self._lock:
            if isinstance(pm_or_path, ProcedureManager):
                self.pm = pm_or_path
                return True
            elif isinstance(pm_or_path, str):
                self.pm = ProcedureManager(pm_or_path)
                return self.pm.total_steps > 0
            return False

    def start(self, operator_id: str = "Astronaut-01", notes: str = "") -> Dict[str, Any]:
        """Start a new monitored experiment sequence."""
        with self._lock:
            self._reset_internal()
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

            first_step = self._get_current_expected_step_internal()
            if first_step:
                self._voice.alert_next_step(first_step.step_number, first_step.description)
                self._slog.info(
                    f"FSM Started. Expecting Step {first_step.step_id}: {first_step.expected_action} on {first_step.required_object}"
                )

            return session

    def validate_action(
        self,
        detected_action: str,
        confidence: float,
        object_name: Optional[str] = None,
        details: str = "",
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a detected action event through the deterministic state machine.
        Validates BOTH action AND target object against expected protocol step.

        Returns authoritative evaluation dict:
          - validation_status: 'VALID', 'SKIPPED', 'OUT_OF_ORDER', 'LOW_CONFIDENCE', 'IDLE', 'UNRECOGNIZED'
          - status: 'VALID', 'SKIPPED', 'OUT_OF_SEQUENCE'
          - expected_step: Canonical 'S1'–'S5'
          - detected_step: Canonical 'S1'–'S5' or None
          - next_step: Canonical 'S1'–'S5' or None
        """
        with self._lock:
            ts = timestamp or datetime.now().isoformat()
            conf = float(confidence) if confidence is not None else 0.0

            if self.state != "RUNNING":
                exp_step = self._get_current_expected_step_internal()
                exp_id = exp_step.step_id if exp_step else None
                return {
                    "status": "IGNORED",
                    "validation_status": "IGNORED",
                    "reason": f"Validator not in RUNNING state (current: {self.state})",
                    "detected_action": detected_action,
                    "detected_object": object_name,
                    "expected_step": exp_id,
                    "detected_step": None,
                    "next_step": exp_id,
                    "confidence": conf,
                    "timestamp": ts,
                }

            norm_action = str(detected_action).strip().upper() if detected_action else "IDLE"
            norm_object = str(object_name).strip().upper() if object_name else None

            expected_step = self._get_current_expected_step_internal()

            if not expected_step:
                self.state = "COMPLETED"
                return {
                    "status": "COMPLETED",
                    "validation_status": "COMPLETED",
                    "message": "All procedure steps already validated.",
                    "expected_step": None,
                    "detected_step": None,
                    "next_step": None,
                    "is_complete": True,
                    "confidence": conf,
                    "timestamp": ts,
                }

            # Find if detected action belongs to ANY step in this experiment
            matching_step = self.pm.get_step_by_action(norm_action)

            # -------------------------------------------------------------
            # CASE 1: MATCHES CURRENT EXPECTED ACTION
            # -------------------------------------------------------------
            if norm_action == expected_step.expected_action:
                # 1.1 Object Verification Check
                expected_obj = expected_step.required_object
                if norm_object is not None and expected_obj is not None and norm_object != expected_obj:
                    msg = (
                        f"Action {norm_action} performed on incorrect object '{norm_object}' "
                        f"(expected '{expected_obj}' for Step {expected_step.step_id}: {expected_step.description})."
                    )
                    self._slog.warn(msg)
                    self._voice.alert_out_of_order(expected_step.description, f"{norm_action} on {norm_object}")

                    self.anomalies.append({
                        "type": "INVALID_OBJECT",
                        "message": msg,
                        "step_number": expected_step.step_number,
                        "timestamp": ts,
                    })

                    if self._session_active:
                        self._exp_logger.log_anomaly("INVALID_OBJECT", msg, expected_step.step_number)

                    return {
                        "status": "OUT_OF_SEQUENCE",
                        "validation_status": "OUT_OF_ORDER",
                        "error_type": "INVALID_OBJECT",
                        "step_id": expected_step.step_id,
                        "step_number": expected_step.step_number,
                        "expected_step": expected_step.step_id,
                        "detected_step": expected_step.step_id,
                        "next_step": expected_step.step_id,
                        "expected_action": expected_step.expected_action,
                        "detected_action": norm_action,
                        "expected_object": expected_obj,
                        "detected_object": norm_object,
                        "confidence": conf,
                        "message": msg,
                        "timestamp": ts,
                    }

                # 1.2 Confidence Threshold Check
                if conf < self.min_confidence:
                    msg = (
                        f"Action matches Step {expected_step.step_id} ({expected_step.expected_action}) "
                        f"but confidence {int(conf*100)}% is below threshold {int(self.min_confidence*100)}%."
                    )
                    if self._session_active:
                        self._exp_logger.log_anomaly("LOW_CONFIDENCE", msg, expected_step.step_number)
                    return {
                        "status": "OUT_OF_SEQUENCE",
                        "validation_status": "LOW_CONFIDENCE",
                        "step_id": expected_step.step_id,
                        "step_number": expected_step.step_number,
                        "expected_step": expected_step.step_id,
                        "detected_step": expected_step.step_id,
                        "next_step": expected_step.step_id,
                        "expected_action": expected_step.expected_action,
                        "detected_action": norm_action,
                        "expected_object": expected_obj,
                        "detected_object": norm_object or expected_obj,
                        "confidence": conf,
                        "message": msg,
                        "timestamp": ts,
                    }

                # 1.3 Valid Step Transition (Both Action and Object confirmed)
                resolved_object = norm_object or expected_obj
                step_record = {
                    "step_id": expected_step.step_id,
                    "step_number": expected_step.step_number,
                    "title": expected_step.description,
                    "expected_action": expected_step.expected_action,
                    "detected_action": norm_action,
                    "required_object": expected_obj,
                    "detected_object": resolved_object,
                    "confidence": conf,
                    "validation_status": "VALID",
                    "timestamp": ts,
                }
                self.validated_steps.append(step_record)

                if self._session_active:
                    self._exp_logger.log_step(
                        step_id=expected_step.step_number,
                        step_title=expected_step.description,
                        expected_action=expected_step.expected_action,
                        detected_action=norm_action,
                        confidence=conf,
                        validation_status="VALID",
                        details=details,
                    )

                self._slog.info(
                    f"FSM VALID: Step {expected_step.step_id} ({expected_step.expected_action} on {resolved_object}) confirmed."
                )

                # Advance to next step
                self.current_step_index += 1
                next_step = self._get_current_expected_step_internal()

                if next_step:
                    self._voice.alert_next_step(next_step.step_number, next_step.description)
                else:
                    self.state = "COMPLETED"
                    self._voice.alert_procedure_completed(self.pm.experiment_name)
                    if self._session_active:
                        self._exp_logger.end_session(status="COMPLETED")
                        self._session_active = False

                return {
                    "status": "VALID",
                    "validation_status": "VALID",
                    "step_id": expected_step.step_id,
                    "step_number": expected_step.step_number,
                    "step_title": expected_step.description,
                    "expected_step": expected_step.step_id,
                    "detected_step": expected_step.step_id,
                    "next_step": next_step.step_id if next_step else None,
                    "next_step_id": next_step.step_id if next_step else None,
                    "next_step_number": next_step.step_number if next_step else None,
                    "action": norm_action,
                    "object": resolved_object,
                    "is_complete": self.state == "COMPLETED",
                    "confidence": conf,
                    "timestamp": ts,
                }

            # -------------------------------------------------------------
            # CASE 2: STEP WAS SKIPPED (Detected future step > current_step)
            # -------------------------------------------------------------
            elif matching_step and matching_step.step_number > expected_step.step_number:
                skipped_steps = [
                    s for s in self.pm.steps
                    if expected_step.step_number <= s.step_number < matching_step.step_number
                ]
                skipped_names = ", ".join([f"Step {s.step_id} ({s.description})" for s in skipped_steps])

                msg = f"Skipped step detected! Performed Step {matching_step.step_id} before completing {skipped_names}."
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
                        detected_action=norm_action,
                        confidence=conf,
                        validation_status="SKIPPED",
                        details=msg,
                    )

                # Auto-advance past skipped step to maintain flow
                self.current_step_index = matching_step.step_number
                next_step = self._get_current_expected_step_internal()

                return {
                    "status": "SKIPPED",
                    "validation_status": "SKIPPED",
                    "detected_action": norm_action,
                    "detected_object": norm_object or matching_step.required_object,
                    "expected_step": expected_step.step_id,
                    "detected_step": matching_step.step_id,
                    "next_step": next_step.step_id if next_step else None,
                    "step_id": matching_step.step_id,
                    "step_number": matching_step.step_number,
                    "skipped_step_ids": [s.step_id for s in skipped_steps],
                    "skipped_step_numbers": [s.step_number for s in skipped_steps],
                    "message": msg,
                    "confidence": conf,
                    "timestamp": ts,
                }

            # -------------------------------------------------------------
            # CASE 3: OUT OF ORDER (Detected previous or repeated step)
            # -------------------------------------------------------------
            elif matching_step and matching_step.step_number <= expected_step.step_number:
                msg = (
                    f"Out of order action. Detected {norm_action} (Step {matching_step.step_id}), "
                    f"but expecting Step {expected_step.step_id} ({expected_step.expected_action})."
                )
                self._slog.warn(msg)
                self._voice.alert_out_of_order(expected_step.description, norm_action)

                if self._session_active:
                    self._exp_logger.log_anomaly("OUT_OF_ORDER", msg, expected_step.step_number)
                    self._exp_logger.log_step(
                        step_id=expected_step.step_number,
                        step_title=expected_step.description,
                        expected_action=expected_step.expected_action,
                        detected_action=norm_action,
                        confidence=conf,
                        validation_status="OUT_OF_ORDER",
                        details=msg,
                    )

                return {
                    "status": "OUT_OF_SEQUENCE",
                    "validation_status": "OUT_OF_ORDER",
                    "expected_action": expected_step.expected_action,
                    "detected_action": norm_action,
                    "expected_object": expected_step.required_object,
                    "detected_object": norm_object,
                    "expected_step": expected_step.step_id,
                    "detected_step": matching_step.step_id,
                    "next_step": expected_step.step_id,
                    "step_id": expected_step.step_id,
                    "step_number": expected_step.step_number,
                    "message": msg,
                    "confidence": conf,
                    "timestamp": ts,
                }

            # -------------------------------------------------------------
            # CASE 4: IDLE STEADY-STATE ACTION
            # -------------------------------------------------------------
            elif norm_action == "IDLE":
                return {
                    "status": "VALID",
                    "validation_status": "IDLE",
                    "detected_action": "IDLE",
                    "detected_object": norm_object or "NONE",
                    "expected_action": expected_step.expected_action,
                    "expected_object": expected_step.required_object,
                    "expected_step": expected_step.step_id,
                    "detected_step": None,
                    "next_step": expected_step.step_id,
                    "confidence": conf,
                    "timestamp": ts,
                }

            # -------------------------------------------------------------
            # CASE 5: UNRECOGNIZED / OUT-OF-VOCABULARY ACTION
            # -------------------------------------------------------------
            else:
                return {
                    "status": "OUT_OF_SEQUENCE",
                    "validation_status": "UNRECOGNIZED",
                    "detected_action": norm_action,
                    "detected_object": norm_object,
                    "expected_action": expected_step.expected_action,
                    "expected_object": expected_step.required_object,
                    "expected_step": expected_step.step_id,
                    "detected_step": None,
                    "next_step": expected_step.step_id,
                    "confidence": conf,
                    "timestamp": ts,
                }

    def validate_ai_result(self, ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience validator accepting an internal or public AI result payload.
        Ensures AI perception does not override authoritative FSM state.
        """
        if not isinstance(ai_result, dict):
            return self.validate_action(detected_action="IDLE", confidence=0.0)

        action = ai_result.get("action", "IDLE")
        conf = ai_result.get("confidence", 0.0)
        obj = ai_result.get("object") or (ai_result.get("interaction") or {}).get("target_object")
        timestamp = ai_result.get("timestamp")
        return self.validate_action(
            detected_action=action,
            confidence=conf,
            object_name=obj,
            timestamp=timestamp,
        )

    def _get_current_expected_step_internal(self) -> Optional[ProcedureStep]:
        """Unsynchronized internal helper for current step."""
        return self.pm.get_step(self.current_step_index)

    def get_current_expected_step(self) -> Optional[ProcedureStep]:
        """Return the step currently expected by the FSM."""
        with self._lock:
            return self._get_current_expected_step_internal()

    def get_next_expected_step(self) -> Optional[ProcedureStep]:
        """Return the step after the currently expected one."""
        with self._lock:
            return self.pm.get_step(self.current_step_index + 1)

    def get_progress(self) -> Dict[str, Any]:
        """Return summary progress metrics with canonical step representations."""
        with self._lock:
            total = self.pm.total_steps
            current = self.current_step_index
            exp_step = self._get_current_expected_step_internal()
            percent = int((current / total) * 100) if total > 0 else 0
            return {
                "state": self.state,
                "current_step_index": current,
                "current_step_id": exp_step.step_id if exp_step else None,
                "current_step_number": current + 1 if current < total else total,
                "total_steps": total,
                "progress_percent": percent,
                "validated_count": len(self.validated_steps),
                "anomalies_count": len(self.anomalies),
                "is_complete": self.state == "COMPLETED",
            }

    def pause(self) -> None:
        with self._lock:
            if self.state == "RUNNING":
                self.state = "PAUSED"
                self._slog.info("FSM Sequence paused.")

    def resume(self) -> None:
        with self._lock:
            if self.state == "PAUSED":
                self.state = "RUNNING"
                self._slog.info("FSM Sequence resumed.")

    def _reset_internal(self) -> None:
        """Unsynchronized internal reset."""
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

    def reset(self) -> None:
        """Reset sequence validator back to initial state."""
        with self._lock:
            self._reset_internal()


# Module singleton
_validator_instance: Optional[SequenceValidatorFSM] = None
_instance_lock = threading.Lock()


def get_sequence_validator(**kwargs) -> SequenceValidatorFSM:
    """Return or initialize global SequenceValidator singleton."""
    global _validator_instance
    with _instance_lock:
        if _validator_instance is None:
            _validator_instance = SequenceValidatorFSM(**kwargs)
        return _validator_instance
