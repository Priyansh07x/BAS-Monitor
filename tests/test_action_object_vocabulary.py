"""
test_action_object_vocabulary.py — Gate B2 Procedure Action and Object Vocabulary Tests
ISRO SIH26174 BAS Experiment Monitor

Validates:
1. Canonical procedure actions: PICK_RED, PLACE_RED, PICK_BLUE, PLACE_BLUE, CLOSE_LID
2. Canonical objects: RED_SAMPLE, BLUE_SAMPLE, SAMPLE_CONTAINER, CONTAINER_LID
3. Step sequence integrity (S1 -> S2 -> S3 -> S4 -> S5)
4. Decoupling from Part-1 ML classifier classes ('catch', 'not-catch')
5. Standalone procedure loading without ML model files
"""

import json
from pathlib import Path
import pytest

from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.sequence_validator import SequenceValidatorFSM


CONFIG_PATH = Path("config/experiment.json")

CANONICAL_ACTIONS = [
    "PICK_RED",
    "PLACE_RED",
    "PICK_BLUE",
    "PLACE_BLUE",
    "CLOSE_LID",
]

CANONICAL_OBJECTS = [
    "RED_SAMPLE",
    "BLUE_SAMPLE",
    "SAMPLE_CONTAINER",
    "CONTAINER_LID",
]

CANONICAL_STEP_SEQUENCE = ["S1", "S2", "S3", "S4", "S5"]

CURRENT_PART1_MODEL_CLASSES = ["catch", "not-catch"]


def test_canonical_actions_defined_and_unique():
    """Verify all 5 canonical procedure actions exist and are unique."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = data.get("actions", [])
    assert isinstance(actions, list)
    assert len(actions) == 5, f"Expected 5 actions, got {len(actions)}"
    assert len(actions) == len(set(actions)), "Duplicate action IDs detected"

    for action in CANONICAL_ACTIONS:
        assert action in actions, f"Missing canonical action '{action}' in config"


def test_canonical_objects_defined_and_unique():
    """Verify all 4 canonical objects exist and are unique."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    assert isinstance(objects, list)
    assert len(objects) == 4, f"Expected 4 objects, got {len(objects)}"
    assert len(objects) == len(set(objects)), "Duplicate object IDs detected"

    for obj in CANONICAL_OBJECTS:
        assert obj in objects, f"Missing canonical object '{obj}' in config"


def test_steps_reference_canonical_actions_and_objects():
    """Verify every step in the sequence references a canonical action and canonical object."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    assert len(steps) == 5, f"Expected 5 steps, got {len(steps)}"

    for step in steps:
        step_id = step.get("id")
        action = step.get("action")
        obj = step.get("object") or step.get("required_object")

        assert action in CANONICAL_ACTIONS, f"Step {step_id} has non-canonical action '{action}'"
        assert obj in CANONICAL_OBJECTS, f"Step {step_id} has non-canonical object '{obj}'"


def test_step_sequence_order_is_s1_through_s5():
    """Verify step sequence strictly follows S1 -> S2 -> S3 -> S4 -> S5."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    step_ids = [s.get("id") for s in steps]
    assert step_ids == CANONICAL_STEP_SEQUENCE, f"Step sequence mismatch: expected {CANONICAL_STEP_SEQUENCE}, got {step_ids}"


def test_procedure_decoupled_from_current_model_classes():
    """
    Verify that current Part-1 model classes ('catch', 'not-catch')
    are NOT required to equal procedure actions, confirming strict architectural decoupling.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = set(data.get("actions", []))
    part1_classes = set(CURRENT_PART1_MODEL_CLASSES)

    # They should not be equal; procedure actions are specific to EXP-001
    assert actions != part1_classes, "Procedure actions must not be constrained by 2-class ML model"
    assert not part1_classes.issubset(actions), "catch/not-catch must not be forced into procedure actions"


def test_procedure_loads_without_ml_model_files():
    """
    Verify that ProcedureManager and SequenceValidatorFSM can initialize,
    load the canonical config, and validate a sequence purely deterministically
    without requiring any physical .pt, .tflite, or .hef model files.
    """
    pm = ProcedureManager(CONFIG_PATH)
    assert pm.total_steps == 5

    fsm = SequenceValidatorFSM(procedure_manager=pm)
    session = fsm.start(operator_id="ASTRONAUT_TEST")
    assert fsm.state == "RUNNING"

    # Step S1: PICK_RED
    res1 = fsm.validate_action("PICK_RED", confidence=0.95)
    assert res1.get("validation_status") == "VALID"
    assert res1.get("step_id") in ("S1", 1)
    assert res1.get("step_number") == 1

    # Step S2: PLACE_RED
    res2 = fsm.validate_action("PLACE_RED", confidence=0.92)
    assert res2.get("validation_status") == "VALID"
    assert res2.get("step_id") in ("S2", 2)
    assert res2.get("step_number") == 2

    # Step S3: PICK_BLUE
    res3 = fsm.validate_action("PICK_BLUE", confidence=0.90)
    assert res3.get("validation_status") == "VALID"
    assert res3.get("step_id") in ("S3", 3)
    assert res3.get("step_number") == 3

    # Step S4: PLACE_BLUE
    res4 = fsm.validate_action("PLACE_BLUE", confidence=0.94)
    assert res4.get("validation_status") == "VALID"
    assert res4.get("step_id") in ("S4", 4)
    assert res4.get("step_number") == 4

    # Step S5: CLOSE_LID
    res5 = fsm.validate_action("CLOSE_LID", confidence=0.96)
    assert res5.get("validation_status") == "VALID"
    assert res5.get("step_id") in ("S5", 5)
    assert res5.get("step_number") == 5
    assert res5.get("is_complete") is True
    assert fsm.state == "COMPLETED"
