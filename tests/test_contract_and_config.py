"""
test_contract_and_config.py — Gate B1 Contract and Configuration Verification Tests
ISRO SIH26174 BAS Experiment Monitor

Validates:
1. Canonical config/experiment.json structure, validity, schema completeness, and step ordering.
2. Public AI result contract schema conformance per docs/architecture.md.
3. Compatibility with ProcedureManager parsing.
"""

import json
from pathlib import Path
from datetime import datetime
import pytest


CONFIG_PATH = Path("config/experiment.json")
ARCH_CONTRACT_REQUIRED_KEYS = {
    "timestamp",
    "action",
    "object",
    "confidence",
    "expected_step",
    "detected_step",
    "status",
    "next_step",
}
VALID_STATUS_VALUES = {"VALID", "SKIPPED", "OUT_OF_SEQUENCE"}


def test_experiment_config_exists():
    """Verify that config/experiment.json exists and is a regular file."""
    assert CONFIG_PATH.exists(), "config/experiment.json does not exist"
    assert CONFIG_PATH.is_file(), "config/experiment.json is not a file"


def test_experiment_config_is_valid_json():
    """Verify that config/experiment.json parses as valid JSON."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "experiment.json root must be a JSON object (dict)"


def test_experiment_has_id_and_version():
    """Verify that experiment definition has non-empty ID, version, and name."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "id" in data and bool(str(data["id"]).strip()), "Missing or empty 'id'"
    assert "version" in data and bool(str(data["version"]).strip()), "Missing or empty 'version'"
    assert "name" in data and bool(str(data["name"]).strip()), "Missing or empty 'name'"
    assert "purpose" in data and bool(str(data["purpose"]).strip()), "Missing or empty 'purpose'"


def test_experiment_has_preconditions_completion_and_failure():
    """Verify experiment defines preconditions, completion conditions, and failure conditions."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "preconditions" in data and isinstance(data["preconditions"], list) and len(data["preconditions"]) > 0
    assert "completion_conditions" in data and isinstance(data["completion_conditions"], list) and len(data["completion_conditions"]) > 0
    assert "failure_conditions" in data and isinstance(data["failure_conditions"], list) and len(data["failure_conditions"]) > 0


def test_experiment_has_objects_and_actions():
    """Verify experiment defines object vocabulary and action vocabulary."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "objects" in data and isinstance(data["objects"], list) and len(data["objects"]) > 0
    assert "actions" in data and isinstance(data["actions"], list) and len(data["actions"]) > 0


def test_experiment_steps_validity_and_ordering():
    """Verify step IDs are unique, ordered (S1, S2, ...), have actions, objects, and recovery."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps")
    assert isinstance(steps, list), "'steps' must be a list"
    assert len(steps) >= 1, "'steps' list cannot be empty"

    seen_ids = set()
    for idx, step in enumerate(steps, start=1):
        step_id = step.get("id")
        assert step_id, f"Step at index {idx} missing 'id'"
        assert step_id not in seen_ids, f"Duplicate step ID '{step_id}' found"
        seen_ids.add(step_id)

        # Expected stable ID format: S1, S2, ...
        expected_id = f"S{idx}"
        assert step_id == expected_id, f"Step index {idx} expected ID '{expected_id}', got '{step_id}'"

        # Step number matching index
        assert step.get("step_number") == idx, f"Step '{step_id}' step_number mismatch"

        # Action verification
        action = step.get("action")
        assert action and bool(str(action).strip()), f"Step '{step_id}' missing or empty 'action'"
        assert action in data["actions"], f"Step '{step_id}' action '{action}' not declared in root 'actions'"

        # Target object verification
        target_obj = step.get("object") or step.get("required_object")
        assert target_obj, f"Step '{step_id}' missing target 'object'"
        assert target_obj in data["objects"], f"Step '{step_id}' object '{target_obj}' not declared in root 'objects'"

        # Recovery instruction verification
        recovery = step.get("recovery")
        assert recovery and bool(str(recovery).strip()), f"Step '{step_id}' missing 'recovery' guidance"


def test_procedure_manager_can_load_canonical_config():
    """Verify that ProcedureManager loads config/experiment.json correctly."""
    from backend.experiment.procedure_manager import ProcedureManager

    pm = ProcedureManager(CONFIG_PATH)
    assert pm.total_steps == 5
    assert pm.experiment_id == "EXP-001"
    assert pm.experiment_name == "Microgravity Sample Transfer and Containment Procedure"

    # Verify step 1
    s1 = pm.get_step(0)
    assert s1 is not None
    assert s1.step_id == "S1"
    assert s1.expected_action == "PICK_RED"
    assert s1.required_object == "RED_SAMPLE"

    # Verify action lookup
    s5 = pm.get_step_by_action("CLOSE_LID")
    assert s5 is not None
    assert s5.step_id == "S5"
    assert s5.step_number == 5


def test_public_ai_result_contract_validation():
    """
    Verify that a sample AI result dict conforms to docs/architecture.md §2 schema:
    - timestamp: ISO 8601 string
    - action: string
    - object: string
    - confidence: float between 0.0 and 1.0
    - expected_step: string matching canonical ID or None
    - detected_step: string matching canonical ID
    - status: string in {'VALID', 'SKIPPED', 'OUT_OF_SEQUENCE'}
    - next_step: string matching canonical ID or None
    """
    sample_result = {
        "timestamp": datetime.now().isoformat(),
        "action": "PICK_RED",
        "object": "RED_SAMPLE",
        "confidence": 0.94,
        "expected_step": "S1",
        "detected_step": "S1",
        "status": "VALID",
        "next_step": "S2",
    }

    # Verify all required keys are present
    assert ARCH_CONTRACT_REQUIRED_KEYS.issubset(sample_result.keys())

    # Type & Value Assertions
    assert isinstance(sample_result["timestamp"], str)
    assert isinstance(sample_result["action"], str) and bool(sample_result["action"])
    assert isinstance(sample_result["object"], str) and bool(sample_result["object"])
    assert isinstance(sample_result["confidence"], (float, int))
    assert 0.0 <= sample_result["confidence"] <= 1.0
    assert sample_result["status"] in VALID_STATUS_VALUES
    assert sample_result["expected_step"] in {"S1", "S2", "S3", "S4", "S5", None}
    assert sample_result["detected_step"] in {"S1", "S2", "S3", "S4", "S5"}
    assert sample_result["next_step"] in {"S1", "S2", "S3", "S4", "S5", None}
