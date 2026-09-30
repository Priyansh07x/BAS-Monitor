"""
procedure_manager.py — Experiment Protocol & Step Definition Manager
ISRO SIH26174 BAS Experiment Monitor

Loads, normalizes, and manages structured experiment procedures from JSON configurations,
supplying expected actions, step instructions, and validation requirements to the FSM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProcedureStep:
    """Represents a discrete protocol step within an experiment."""

    def __init__(
        self,
        step_id: str,
        step_number: int,
        expected_action: str,
        instruction: str = "",
        description: str = "",
        required_object: Optional[str] = None,
        duration_est_seconds: float = 30.0,
        recovery: str = "",
        timeout_s: Optional[float] = None,
    ):
        self.step_id = str(step_id)
        self.step_number = int(step_number)
        self.expected_action = str(expected_action).strip()
        self.instruction = str(instruction).strip()
        self.description = str(description).strip() or self.instruction
        self.required_object = required_object
        self.duration_est_seconds = float(duration_est_seconds)
        self.recovery = str(recovery).strip()
        self.timeout_s = float(timeout_s) if timeout_s is not None else self.duration_est_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "expected_action": self.expected_action,
            "instruction": self.instruction,
            "description": self.description,
            "required_object": self.required_object,
            "duration_est_seconds": self.duration_est_seconds,
            "recovery": self.recovery,
            "timeout_s": self.timeout_s,
        }


class ProcedureManager:
    """
    Manages loading, parsing, and step navigation for astronaut experiments.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        self.experiment_id: str = "DEFAULT"
        self.experiment_name: str = "Untitled Experiment"
        self.steps: List[ProcedureStep] = []
        self._action_to_step: Dict[str, ProcedureStep] = {}

        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, filepath: str | Path) -> bool:
        """Load experiment protocol from a JSON configuration file."""
        path = Path(filepath)
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.load_from_dict(data)
        except Exception:
            return False

    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """Parse structured dictionary into ProcedureStep list."""
        self.experiment_id = data.get("id", data.get("version", "EXP-01"))
        self.experiment_name = data.get("name", "BAS Experiment")
        raw_steps = data.get("steps", [])

        self.steps.clear()
        self._action_to_step.clear()

        for idx, s in enumerate(raw_steps, start=1):
            s_id = s.get("id", f"S{idx}")
            action = s.get("action", s.get("expected_action", ""))
            instruction = s.get("instruction", s.get("description", ""))
            desc = s.get("description", instruction)
            req_obj = s.get("required_object") or s.get("object")
            duration = s.get("duration_est", s.get("duration", 30))

            # Convert string duration like "30s" to float
            if isinstance(duration, str):
                duration = float(duration.replace("s", "").strip())

            recovery = s.get("recovery", "")
            raw_timeout = s.get("timeout_s", s.get("timeout", duration))
            if isinstance(raw_timeout, str):
                raw_timeout = float(raw_timeout.replace("s", "").strip())
            timeout_s = float(raw_timeout) if raw_timeout is not None else float(duration)

            step = ProcedureStep(
                step_id=str(s_id),
                step_number=idx,
                expected_action=action,
                instruction=instruction,
                description=desc,
                required_object=req_obj,
                duration_est_seconds=float(duration),
                recovery=recovery,
                timeout_s=timeout_s,
            )
            self.steps.append(step)
            if action:
                self._action_to_step[action] = step

        return len(self.steps) > 0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_step(self, index: int) -> Optional[ProcedureStep]:
        """Get step by 0-based index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_step_by_number(self, step_number: int) -> Optional[ProcedureStep]:
        """Get step by 1-based step number."""
        return self.get_step(step_number - 1)

    def get_step_by_action(self, action: str) -> Optional[ProcedureStep]:
        """Lookup step by its expected action string."""
        return self._action_to_step.get(action.strip())

    def get_step_by_id(self, step_id: str) -> Optional[ProcedureStep]:
        """Lookup step by its canonical step ID (e.g. 'S1')."""
        norm_id = str(step_id).strip().upper()
        for s in self.steps:
            if s.step_id.upper() == norm_id:
                return s
        return None

    @staticmethod
    def normalize_step_id(step_val: Any) -> Optional[str]:
        """Normalize various step representations ('S1', 1, '1', 's1') to canonical 'S1'–'S5' or None."""
        if step_val is None:
            return None
        s = str(step_val).strip().upper()
        if s in {"S1", "S2", "S3", "S4", "S5"}:
            return s
        if s.isdigit():
            num = int(s)
            if 1 <= num <= 5:
                return f"S{num}"
        if s.startswith("S") and s[1:].isdigit():
            num = int(s[1:])
            if 1 <= num <= 5:
                return f"S{num}"
        return None

    def get_next_step(self, current_index: int) -> Optional[ProcedureStep]:
        """Return the subsequent step given current index."""
        return self.get_step(current_index + 1)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export all steps as list of dicts for UI/QWebChannel consumption."""
        return [step.to_dict() for step in self.steps]
