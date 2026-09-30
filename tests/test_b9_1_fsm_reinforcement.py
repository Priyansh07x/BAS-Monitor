"""
test_b9_1_fsm_reinforcement.py — Gate B9.1 FSM Reinforcement & Action-Object Binding Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B9.1)

Validates:
1. Strict Action + Object binding validation in SequenceValidatorFSM against canonical EXP-001.
2. Canonical step identifiers consistently exposed as S1–S5.
3. Sole authority of FSM over expected_step, procedural status, and next_step.
4. Deterministic handling of:
   - Nominal correct action + correct object (S1 through S5)
   - Correct action + wrong object (INVALID_OBJECT / OUT_OF_SEQUENCE)
   - Future action / skipped steps (SKIPPED)
   - Repeated past action (OUT_OF_ORDER / OUT_OF_SEQUENCE)
   - Unrecognized action / object (UNRECOGNIZED / OUT_OF_SEQUENCE)
   - Low confidence below threshold (LOW_CONFIDENCE)
   - IDLE and non-running states (IGNORED / IDLE)
5. Decoupling and immunity from AI-provided claimed status or step overrides.
6. Thread-safety across concurrent validation requests.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Dict
import pytest

from backend.experiment.procedure_manager import ProcedureManager, ProcedureStep
from backend.experiment.sequence_validator import SequenceValidatorFSM


CONFIG_PATH = Path("config/experiment.json")


@pytest.fixture
def procedure_manager() -> ProcedureManager:
    """Fixture providing ProcedureManager initialized with canonical EXP-001 config."""
    pm = ProcedureManager(CONFIG_PATH)
    assert pm.total_steps == 5
    return pm


@pytest.fixture
def fsm(procedure_manager: ProcedureManager) -> SequenceValidatorFSM:
    """Fixture providing SequenceValidatorFSM in RUNNING state."""
    fsm_inst = SequenceValidatorFSM(procedure_manager=procedure_manager, min_confidence_threshold=0.70)
    fsm_inst.start(operator_id="Astronaut-B9-Test")
    assert fsm_inst.state == "RUNNING"
    return fsm_inst


# ==============================================================================
# 1. Action + Object Binding Tests
# ==============================================================================

class TestActionObjectBinding:
    """Validates that SequenceValidatorFSM validates both action AND target object."""

    def test_nominal_step1_valid_action_and_object(self, fsm: SequenceValidatorFSM):
        """S1: PICK_RED with RED_SAMPLE succeeds and advances to S2."""
        res = fsm.validate_action(
            detected_action="PICK_RED",
            confidence=0.95,
            object_name="RED_SAMPLE",
        )
        assert res["status"] == "VALID"
        assert res["validation_status"] == "VALID"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S1"
        assert res["next_step"] == "S2"
        assert res["step_id"] == "S1"
        assert res["step_number"] == 1
        assert res["action"] == "PICK_RED"
        assert res["object"] == "RED_SAMPLE"
        assert fsm.current_step_index == 1

    def test_step1_correct_action_wrong_object_rejected(self, fsm: SequenceValidatorFSM):
        """S1: PICK_RED with BLUE_SAMPLE fails (INVALID_OBJECT) and does NOT advance step."""
        res = fsm.validate_action(
            detected_action="PICK_RED",
            confidence=0.95,
            object_name="BLUE_SAMPLE",
        )
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["validation_status"] == "OUT_OF_ORDER"
        assert res.get("error_type") == "INVALID_OBJECT"
        assert res["expected_step"] == "S1"
        assert res["next_step"] == "S1"
        assert res["expected_action"] == "PICK_RED"
        assert res["expected_object"] == "RED_SAMPLE"
        assert res["detected_object"] == "BLUE_SAMPLE"
        # Must stay at S1
        assert fsm.current_step_index == 0

    def test_step1_correct_action_container_object_rejected(self, fsm: SequenceValidatorFSM):
        """S1: PICK_RED with SAMPLE_CONTAINER fails and preserves S1."""
        res = fsm.validate_action(
            detected_action="PICK_RED",
            confidence=0.92,
            object_name="SAMPLE_CONTAINER",
        )
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["validation_status"] == "OUT_OF_ORDER"
        assert res.get("error_type") == "INVALID_OBJECT"
        assert fsm.current_step_index == 0

    def test_full_nominal_five_step_sequence(self, fsm: SequenceValidatorFSM):
        """Full nominal sequence S1->S2->S3->S4->S5 with explicit action and object pairs."""
        # S1: PICK_RED + RED_SAMPLE
        r1 = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert r1["status"] == "VALID"
        assert r1["expected_step"] == "S1"
        assert r1["next_step"] == "S2"
        assert fsm.current_step_index == 1

        # S2: PLACE_RED + RED_SAMPLE
        r2 = fsm.validate_action("PLACE_RED", 0.92, "RED_SAMPLE")
        assert r2["status"] == "VALID"
        assert r2["expected_step"] == "S2"
        assert r2["next_step"] == "S3"
        assert fsm.current_step_index == 2

        # S3: PICK_BLUE + BLUE_SAMPLE
        r3 = fsm.validate_action("PICK_BLUE", 0.91, "BLUE_SAMPLE")
        assert r3["status"] == "VALID"
        assert r3["expected_step"] == "S3"
        assert r3["next_step"] == "S4"
        assert fsm.current_step_index == 3

        # S4: PLACE_BLUE + BLUE_SAMPLE
        r4 = fsm.validate_action("PLACE_BLUE", 0.94, "BLUE_SAMPLE")
        assert r4["status"] == "VALID"
        assert r4["expected_step"] == "S4"
        assert r4["next_step"] == "S5"
        assert fsm.current_step_index == 4

        # S5: CLOSE_LID + CONTAINER_LID
        r5 = fsm.validate_action("CLOSE_LID", 0.96, "CONTAINER_LID")
        assert r5["status"] == "VALID"
        assert r5["expected_step"] == "S5"
        assert r5["next_step"] is None
        assert r5["is_complete"] is True
        assert fsm.state == "COMPLETED"

    def test_step5_correct_action_wrong_object_rejected(self, fsm: SequenceValidatorFSM):
        """S5: CLOSE_LID on RED_SAMPLE fails and does NOT complete procedure."""
        # Fast forward to S5
        fsm.current_step_index = 4
        assert fsm.get_current_expected_step().step_id == "S5"

        res = fsm.validate_action("CLOSE_LID", 0.95, "RED_SAMPLE")
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["validation_status"] == "OUT_OF_ORDER"
        assert res.get("error_type") == "INVALID_OBJECT"
        assert res["expected_step"] == "S5"
        assert res["next_step"] == "S5"
        assert fsm.state == "RUNNING"


# ==============================================================================
# 2. Canonical Step Representation Tests (S1–S5)
# ==============================================================================

class TestCanonicalStepRepresentation:
    """Validates step normalization and S1–S5 representation across FSM and ProcedureManager."""

    def test_procedure_manager_step_id_lookup(self, procedure_manager: ProcedureManager):
        """ProcedureManager returns correct step for canonical IDs."""
        assert procedure_manager.get_step_by_id("S1").expected_action == "PICK_RED"
        assert procedure_manager.get_step_by_id("s2").expected_action == "PLACE_RED"
        assert procedure_manager.get_step_by_id("S3").required_object == "BLUE_SAMPLE"
        assert procedure_manager.get_step_by_id("S4").step_number == 4
        assert procedure_manager.get_step_by_id("S5").expected_action == "CLOSE_LID"
        assert procedure_manager.get_step_by_id("S6") is None

    def test_procedure_manager_step_id_normalization(self):
        """ProcedureManager normalizes diverse step representations to S1–S5."""
        assert ProcedureManager.normalize_step_id("S1") == "S1"
        assert ProcedureManager.normalize_step_id("s2") == "S2"
        assert ProcedureManager.normalize_step_id(3) == "S3"
        assert ProcedureManager.normalize_step_id("4") == "S4"
        assert ProcedureManager.normalize_step_id("S5") == "S5"
        assert ProcedureManager.normalize_step_id(0) is None
        assert ProcedureManager.normalize_step_id(6) is None
        assert ProcedureManager.normalize_step_id("INVALID") is None

    def test_fsm_progress_reports_canonical_step_id(self, fsm: SequenceValidatorFSM):
        """FSM progress reports canonical current_step_id."""
        prog0 = fsm.get_progress()
        assert prog0["current_step_id"] == "S1"
        assert prog0["current_step_index"] == 0

        fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        prog1 = fsm.get_progress()
        assert prog1["current_step_id"] == "S2"
        assert prog1["current_step_index"] == 1


# ==============================================================================
# 3. Procedural Status & Anomaly Transitions
# ==============================================================================

class TestProceduralStatusTransitions:
    """Validates deterministic FSM state handling for all anomaly conditions."""

    def test_skipped_step_detection_and_auto_advance(self, fsm: SequenceValidatorFSM):
        """At S1, detecting S3 (PICK_BLUE) flags SKIPPED [S1, S2] and advances to S4."""
        res = fsm.validate_action("PICK_BLUE", 0.90, "BLUE_SAMPLE")
        assert res["status"] == "SKIPPED"
        assert res["validation_status"] == "SKIPPED"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S3"
        assert res["next_step"] == "S4"
        assert res["skipped_step_ids"] == ["S1", "S2"]
        assert res["skipped_step_numbers"] == [1, 2]
        # FSM auto-advances past S3 to S4 (index 3)
        assert fsm.current_step_index == 3
        assert fsm.get_current_expected_step().step_id == "S4"

    def test_out_of_order_repeated_past_step(self, fsm: SequenceValidatorFSM):
        """At S2, performing S1 (PICK_RED) triggers OUT_OF_ORDER and stays at S2."""
        # Advance to S2
        fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert fsm.current_step_index == 1

        res = fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["validation_status"] == "OUT_OF_ORDER"
        assert res["expected_step"] == "S2"
        assert res["detected_step"] == "S1"
        assert res["next_step"] == "S2"
        assert fsm.current_step_index == 1

    def test_unrecognized_action(self, fsm: SequenceValidatorFSM):
        """Out-of-vocabulary action triggers UNRECOGNIZED and maintains state."""
        res = fsm.validate_action("UNKNOWN_GESTURE", 0.85, "RED_SAMPLE")
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["validation_status"] == "UNRECOGNIZED"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] is None
        assert res["next_step"] == "S1"
        assert fsm.current_step_index == 0

    def test_idle_steady_state_action(self, fsm: SequenceValidatorFSM):
        """IDLE action is valid steady-state without advancing step."""
        res = fsm.validate_action("IDLE", 0.90, "NONE")
        assert res["status"] == "VALID"
        assert res["validation_status"] == "IDLE"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] is None
        assert res["next_step"] == "S1"
        assert fsm.current_step_index == 0

    def test_low_confidence_blocks_step_transition(self, fsm: SequenceValidatorFSM):
        """Correct action+object with confidence < 0.70 does not advance step."""
        res = fsm.validate_action("PICK_RED", 0.55, "RED_SAMPLE")
        assert res["status"] == "OUT_OF_SEQUENCE"
        assert res["validation_status"] == "LOW_CONFIDENCE"
        assert res["expected_step"] == "S1"
        assert res["next_step"] == "S1"
        assert fsm.current_step_index == 0

    def test_non_running_fsm_ignores_action(self, procedure_manager: ProcedureManager):
        """FSM in IDLE state returns IGNORED."""
        idle_fsm = SequenceValidatorFSM(procedure_manager=procedure_manager)
        assert idle_fsm.state == "IDLE"
        res = idle_fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        assert res["status"] == "IGNORED"
        assert res["validation_status"] == "IGNORED"
        assert res["expected_step"] == "S1"


# ==============================================================================
# 4. Authoritative FSM State vs AI Override Immunity
# ==============================================================================

class TestAuthoritativeStateOwnership:
    """Validates that FSM strictly overrides any AI-claimed procedural states."""

    def test_fsm_ignores_ai_claimed_status_and_steps(self, fsm: SequenceValidatorFSM):
        """Passing an AI result with bogus expected_step, status, next_step is ignored."""
        bogus_ai_result = {
            "timestamp": "2026-09-28T00:00:00",
            "action": "PICK_RED",
            "object": "RED_SAMPLE",
            "confidence": 0.95,
            # AI attempts to claim false procedural metadata:
            "expected_step": "S5",
            "detected_step": "S5",
            "status": "COMPLETED",
            "next_step": "S5",
        }

        # Validate via validate_ai_result convenience method
        res = fsm.validate_ai_result(bogus_ai_result)

        # FSM must determine authoritative values based on current state (S1):
        assert res["status"] == "VALID"
        assert res["expected_step"] == "S1"
        assert res["detected_step"] == "S1"
        assert res["next_step"] == "S2"
        assert fsm.current_step_index == 1


# ==============================================================================
# 5. Thread Safety & Concurrency Verification
# ==============================================================================

class TestFSMThreadSafety:
    """Validates thread-safe execution of FSM validation and state queries."""

    def test_concurrent_action_validation(self, procedure_manager: ProcedureManager):
        """Concurrent validation requests execute safely without state corruption."""
        concurrent_fsm = SequenceValidatorFSM(procedure_manager=procedure_manager)
        concurrent_fsm.start(operator_id="ThreadSafetyTest")

        results = []

        def worker_task(idx: int) -> Dict[str, Any]:
            if idx % 2 == 0:
                return concurrent_fsm.validate_action("IDLE", 0.90, "NONE")
            else:
                return concurrent_fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i) for i in range(20)]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        assert len(results) == 20
        # Step S1 should have transitioned exactly once to S2 (or higher)
        assert concurrent_fsm.current_step_index >= 1
        assert len(concurrent_fsm.validated_steps) >= 1
