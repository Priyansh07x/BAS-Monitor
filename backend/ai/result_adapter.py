"""
result_adapter.py — Public AI Result Contract Adapter
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B6.3)

Converts internal perception and inference pipeline outputs into the exact 8-field
frozen public contract defined in docs/architecture.md §2:
{
    "timestamp": "...",
    "action": "...",
    "object": "...",
    "confidence": 0.0,
    "expected_step": "...",
    "detected_step": "...",
    "status": "...",
    "next_step": "..."
}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Union


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

LEGACY_ACTIONS = {
    "PICK_CONTAINER",
    "PIPETTE_TRANSFER",
    "INSERT_ANALYZER",
    "SEAL_CONTAINER",
    "INITIATE_SCAN",
}

LEGACY_OBJECTS = {
    "CONTAINER",
    "SAMPLE_VIAL",
    "PIPETTE",
    "WELL_PLATE",
    "CENTRIFUGE_TUBE",
    "FORCEPS",
    "INCUBATOR_DOOR",
}

ACTION_TO_STEP = {
    "PICK_RED": "S1",
    "PLACE_RED": "S2",
    "PICK_BLUE": "S3",
    "PLACE_BLUE": "S4",
    "CLOSE_LID": "S5",
}

STEP_TO_ACTION = {
    "S1": "PICK_RED",
    "S2": "PLACE_RED",
    "S3": "PICK_BLUE",
    "S4": "PLACE_BLUE",
    "S5": "CLOSE_LID",
}

STEP_TO_NEXT = {
    "S1": "S2",
    "S2": "S3",
    "S3": "S4",
    "S4": "S5",
    "S5": None,
}

ACTION_TO_DEFAULT_OBJECT = {
    "PICK_RED": "RED_SAMPLE",
    "PLACE_RED": "RED_SAMPLE",
    "PICK_BLUE": "BLUE_SAMPLE",
    "PLACE_BLUE": "BLUE_SAMPLE",
    "CLOSE_LID": "CONTAINER_LID",
    "IDLE": "NONE",
}

PUBLIC_STATUS_VALUES = {"VALID", "SKIPPED", "OUT_OF_SEQUENCE"}

STATUS_MAP = {
    "VALID": "VALID",
    "SKIPPED": "SKIPPED",
    "OUT_OF_SEQUENCE": "OUT_OF_SEQUENCE",
    "OUT_OF_ORDER": "OUT_OF_SEQUENCE",
    "UNCERTAIN": "OUT_OF_SEQUENCE",
    "LOW_CONFIDENCE": "OUT_OF_SEQUENCE",
    "UNRECOGNIZED": "OUT_OF_SEQUENCE",
    "ANOMALY": "OUT_OF_SEQUENCE",
    "IGNORED": "OUT_OF_SEQUENCE",
    "IDLE": "VALID",
    "RUNNING": "VALID",
    "COMPLETED": "VALID",
}

PUBLIC_CONTRACT_KEYS = (
    "timestamp",
    "action",
    "object",
    "confidence",
    "expected_step",
    "detected_step",
    "status",
    "next_step",
)


def normalize_step_id(step_val: Any) -> Optional[str]:
    """Normalizes step identifiers (e.g., 'S1', 1, '1', 's1') to canonical 'S1'–'S5' or None."""
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


def normalize_timestamp(ts_val: Any) -> str:
    """Normalizes timestamp values to ISO-8601 string representation."""
    if ts_val is None:
        return datetime.now().isoformat()
    if isinstance(ts_val, (int, float)):
        # If reasonable unix epoch timestamp in seconds
        if ts_val > 100_000_000:
            try:
                return datetime.fromtimestamp(ts_val).isoformat()
            except Exception:
                pass
        return datetime.now().isoformat()
    if isinstance(ts_val, str) and ts_val.strip():
        return ts_val.strip()
    return datetime.now().isoformat()


def normalize_confidence(conf_val: Any) -> float:
    """Clamps confidence values to float in [0.0, 1.0]."""
    if conf_val is None:
        return 0.0
    try:
        c = float(conf_val)
        if c < 0.0:
            return 0.0
        if c > 1.0:
            return 1.0
        return round(c, 2)
    except (ValueError, TypeError):
        return 0.0


def normalize_status(status_val: Any) -> str:
    """Maps internal FSM and pipeline statuses to frozen public status vocabulary."""
    if not status_val:
        return "VALID"
    s = str(status_val).strip().upper()
    return STATUS_MAP.get(s, "OUT_OF_SEQUENCE")


def normalize_action(action_val: Any) -> str:
    """Validates action label against canonical EXP-001 vocabulary, rejecting legacy strings."""
    if not action_val:
        return "IDLE"
    a = str(action_val).strip().upper()
    if a in CANONICAL_ACTIONS:
        return a
    return "IDLE"


def normalize_object(obj_val: Any, action: str) -> str:
    """Validates object label against canonical EXP-001 vocabulary, defaulting based on action."""
    if obj_val:
        o = str(obj_val).strip().upper()
        if o in CANONICAL_OBJECTS and o not in LEGACY_OBJECTS:
            return o
    return ACTION_TO_DEFAULT_OBJECT.get(action, "NONE")


class AIResultAdapter:
    """
    Converts internal perception results to the frozen 8-field public AI contract.
    """

    @staticmethod
    def adapt(
        internal_result: Optional[Dict[str, Any]] = None,
        fsm_result: Optional[Dict[str, Any]] = None,
        expected_step: Optional[Union[str, int]] = None,
        fsm_status: Optional[str] = None,
        next_step: Optional[Union[str, int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transforms an internal pipeline result dict into the exact 8-field public AI contract.
        Authoritative FSM results (or explicit FSM parameters) take precedence over perception claims.
        """
        res = internal_result or {}
        fsm = fsm_result or {}

        # 1. Action (from kwargs or perception result)
        raw_action = kwargs.get("action", res.get("action"))
        action = normalize_action(raw_action)

        # 2. Object (from kwargs or perception result)
        raw_obj = kwargs.get("object", res.get("object") or (res.get("interaction") or {}).get("target_object"))
        obj = normalize_object(raw_obj, action=action)

        # 3. Confidence (from kwargs or perception result)
        raw_conf = kwargs.get("confidence", res.get("confidence"))
        conf = normalize_confidence(raw_conf)

        # 4. Timestamp (from kwargs, perception result, or FSM result)
        raw_ts = kwargs.get("timestamp", res.get("timestamp") or (fsm.get("timestamp") if fsm else None))
        ts = normalize_timestamp(raw_ts)

        # 5. Detected Step (from kwargs, FSM, perception result, or canonical action mapping)
        if "detected_step" in kwargs:
            raw_det_step = kwargs["detected_step"]
        elif fsm and fsm.get("detected_step") is not None:
            raw_det_step = fsm["detected_step"]
        elif res.get("detected_step") is not None:
            raw_det_step = res["detected_step"]
        else:
            raw_det_step = ACTION_TO_STEP.get(action)
        detected_step = normalize_step_id(raw_det_step)

        # 6. Expected Step (authoritative from explicit arg, FSM, or fallback)
        if expected_step is not None:
            raw_exp_step = expected_step
        elif fsm and "expected_step" in fsm:
            raw_exp_step = fsm.get("expected_step")
        elif "expected_step" in kwargs:
            raw_exp_step = kwargs["expected_step"]
        elif "expected_step" in res:
            raw_exp_step = res["expected_step"]
        else:
            raw_exp_step = detected_step
        exp_step = normalize_step_id(raw_exp_step)

        # 7. Next Step (authoritative from explicit arg, FSM, or fallback)
        if next_step is not None:
            raw_next_step = next_step
        elif fsm and "next_step" in fsm:
            raw_next_step = fsm.get("next_step")
        elif "next_step" in kwargs:
            raw_next_step = kwargs["next_step"]
        elif "next_step" in res:
            raw_next_step = res["next_step"]
        else:
            raw_next_step = STEP_TO_NEXT.get(exp_step) if exp_step else None
        nxt_step = normalize_step_id(raw_next_step)

        # 8. Status (authoritative from explicit arg, FSM, or fallback)
        if fsm_status is not None:
            raw_status = fsm_status
        elif fsm and ("status" in fsm or "validation_status" in fsm):
            raw_status = fsm.get("status") or fsm.get("validation_status")
        elif "status" in kwargs:
            raw_status = kwargs["status"]
        elif "status" in res:
            raw_status = res["status"]
        else:
            if exp_step and detected_step and exp_step != detected_step:
                raw_status = "OUT_OF_SEQUENCE"
            else:
                raw_status = "VALID"
        status = normalize_status(raw_status)

        # Construct exact 8-field public dictionary
        public_result: Dict[str, Any] = {
            "timestamp": ts,
            "action": action,
            "object": obj,
            "confidence": conf,
            "expected_step": exp_step,
            "detected_step": detected_step,
            "status": status,
            "next_step": nxt_step,
        }

        return public_result

    @staticmethod
    def validate_public_contract(result: Dict[str, Any]) -> bool:
        """
        Validates that a result dictionary strictly complies with the frozen public contract:
        - Exactly the 8 required keys (no extra keys, no missing keys)
        - Correct field types and value constraints
        """
        if not isinstance(result, dict):
            return False
        if set(result.keys()) != set(PUBLIC_CONTRACT_KEYS):
            return False
        if not isinstance(result["timestamp"], str) or not result["timestamp"]:
            return False
        if result["action"] not in CANONICAL_ACTIONS:
            return False
        if result["object"] not in CANONICAL_OBJECTS:
            return False
        if not isinstance(result["confidence"], (float, int)) or not (0.0 <= result["confidence"] <= 1.0):
            return False
        if result["status"] not in PUBLIC_STATUS_VALUES:
            return False
        if result["detected_step"] is not None and result["detected_step"] not in {"S1", "S2", "S3", "S4", "S5"}:
            return False
        if result["expected_step"] is not None and result["expected_step"] not in {"S1", "S2", "S3", "S4", "S5"}:
            return False
        if result["next_step"] is not None and result["next_step"] not in {"S1", "S2", "S3", "S4", "S5"}:
            return False
        return True
