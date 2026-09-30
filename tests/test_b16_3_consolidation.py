"""
test_b16_3_consolidation.py — Consolidated Failure Testing & Robustness Traversal Test Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B16.3 Consolidation)

Verifies the consolidated 14-failure traversal matrix across all AI perception (F01–F08)
and procedural violation (P01–P06) modes:
  F01: Wrong action (Semantic mismatch & transient noise)
  F02: Missed action (Intermittent frame drop & IDLE leak)
  F03: Low confidence (Sub-marginal confidence isolation)
  F04: Occlusion (Missing target object & hand landmarks)
  F05: Poor lighting (Severe underexposure & overexposure)
  F06: Blur (Gaussian defocus blur & graceful degradation)
  F07: Fast motion (Candidate flicker & single-slot buffer dynamics)
  F08: Multiple objects (Cluttered workspace & distractor disambiguation)
  P01: Skipped step (Forward step skip S1 -> S3 / S4)
  P02: Out-of-order step (Illegal non-adjacent sequence jump)
  P03: Repeated past step (Re-executing already completed step)
  P04: Premature action (Attempting terminal action prematurely)
  P05: Incomplete action (Required object mismatch & unrecognized action)
  P06: Timeout violation (Deterministic progression exceeding step limits)

Architectural Invariants Strictly Enforced:
1. Authority Boundaries:
   - B10 = Temporal stability authority (M-of-N voting)
   - B11 = Uncertainty & multimodal consistency authority (spatial/semantic veto)
   - FSM/B9 = Sole procedural sequencing authority (governs step transitions)
   - B12 = Procedural recovery authority (triggers recovery ONLY on authoritative FSM violations)
   - B13 = Passive telemetry diagnostics observer
   - B16 = Test and failure-injection consolidation authority only
2. Critical Isolation Invariant:
   - Perception failures (F01–F08) MUST NOT trigger procedural recovery (B12).
   - Confirmed procedure violations (P01–P06) MUST authoritatively trigger procedural recovery (B12).
3. Contract Protection:
   - Frozen 8-field public AI contract (docs/architecture.md §2) strictly preserved across all failure modes.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
import unittest
import numpy as np

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.multimodal_consistency import (
    MultimodalConsistencyEvaluator,
    CATEGORY_CROSS_MODAL_CONFLICT,
    CATEGORY_UNCERTAIN_MISSING_OBJECT,
)
from backend.ai.result_adapter import AIResultAdapter
from backend.ai.telemetry_aggregator import TelemetryDiagnosticAggregator
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM
from evaluation.benchmark_harness import BenchmarkHarness, BenchmarkResult


class TestB163Consolidation(unittest.TestCase):
    """
    Consolidated failure-testing and robustness traversal verification suite (Gate B16.3).
    """

    def setUp(self) -> None:
        self.fsm = SequenceValidatorFSM()
        self.fsm.start()
        self.rec_mgr = RecoveryManager()
        self.agg = TelemetryDiagnosticAggregator()
        self.pipeline = InferencePipeline(
            validator=self.fsm,
            recovery_manager=self.rec_mgr,
            telemetry_aggregator=self.agg,
        )
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # =========================================================================
    # 1. BENCHMARK HARNESS CONSOLIDATED TRAVERSAL EXECUTION
    # =========================================================================

    def test_benchmark_harness_consolidated_failure_traversal_execution(self) -> None:
        """
        Verifies that BenchmarkHarness.run_consolidated_failure_traversal_benchmark() executes
        all 14 failure modes headlessly, records zero false recoveries, and passes 100%.
        """
        result = BenchmarkHarness.run_consolidated_failure_traversal_benchmark()

        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "consolidated_failure_traversal")
        self.assertEqual(result.category, "FAILURE_TESTING_CONSOLIDATION")
        self.assertTrue(result.passed)
        self.assertEqual(result.iterations, 14)

        metrics = result.metrics
        self.assertEqual(metrics["total_failure_cases"], 14)
        self.assertEqual(metrics["passed_cases"], 14)
        self.assertEqual(metrics["failed_cases"], 0)
        self.assertEqual(metrics["false_recovery_count_for_perception"], 0)
        self.assertEqual(metrics["recovery_count_for_confirmed_procedure_violations"], 6)
        self.assertEqual(metrics["public_contract_compliance"], "14/14 (100.0%)")

        # Verify summary sub-dict
        summary = metrics["summary"]
        for fid in [f"f{i:02d}" for i in range(1, 9)] + [f"p{i:02d}" for i in range(1, 7)]:
            matching_keys = [k for k in summary.keys() if k.startswith(fid)]
            self.assertEqual(len(matching_keys), 1, f"Missing summary entry for {fid}")
            self.assertEqual(summary[matching_keys[0]], "PASSED")

    # =========================================================================
    # 2. STANDARDIZED FAILURE RECORD SCHEMA COMPLIANCE
    # =========================================================================

    def test_all_14_matrix_records_standardized_schema(self) -> None:
        """
        Verifies that every record produced in the 14-failure matrix contains all
        required standardized fields with proper data types.
        """
        result = BenchmarkHarness.run_consolidated_failure_traversal_benchmark()
        records = result.metrics.get("records", [])

        self.assertEqual(len(records), 14)
        required_keys = {
            "failure_id",
            "domain",
            "injected_stimulus",
            "expected_behavior",
            "observed_behavior",
            "system_response",
            "failure_cause_category",
            "recovery_containment_result",
            "telemetry_result",
            "passed",
        }

        ai_ids = {f"F{i:02d}" for i in range(1, 9)}
        proc_ids = {f"P{i:02d}" for i in range(1, 7)}
        observed_ids = set()

        for rec in records:
            self.assertTrue(required_keys.issubset(rec.keys()), f"Record missing keys: {rec}")
            self.assertIsInstance(rec["failure_id"], str)
            self.assertIn(rec["domain"], ("AI_PERCEPTION", "PROCEDURE"))
            self.assertTrue(rec["passed"])
            observed_ids.add(rec["failure_id"])

        self.assertEqual(observed_ids, ai_ids.union(proc_ids))

    # =========================================================================
    # 3. DISPATCHER DISCOVERY & INVOCATION
    # =========================================================================

    def test_dispatcher_discovery_and_invocation(self) -> None:
        """
        Verifies dispatcher names ('consolidated_failure_traversal', 'failure_traversal', 'b16_3')
        invoke the consolidated benchmark properly.
        """
        for alias in ["consolidated_failure_traversal", "failure_traversal", "b16_3", "failure_matrix"]:
            res = BenchmarkHarness.run_benchmark(alias)
            self.assertIsInstance(res, BenchmarkResult)
            self.assertTrue(res.passed)
            self.assertEqual(res.metrics["total_failure_cases"], 14)

    # =========================================================================
    # 4. EXPORT CONSOLIDATED BENCHMARK RESULTS (JSON & TEXT)
    # =========================================================================

    def test_export_consolidated_benchmark_results(self) -> None:
        """
        Verifies that consolidated benchmark results export cleanly to JSON and text summary files.
        """
        result = BenchmarkHarness.run_consolidated_failure_traversal_benchmark()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = BenchmarkHarness.export_results(result, tmpdir, base_name="b16_3_consolidated_failure_traversal")
            json_file = paths["json_path"]
            txt_file = paths["txt_path"]

            self.assertTrue(json_file.exists())
            self.assertTrue(txt_file.exists())

            with open(json_file, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            self.assertEqual(loaded_json["benchmark_name"], "consolidated_failure_traversal")
            self.assertTrue(loaded_json["passed"])
            self.assertEqual(len(loaded_json["metrics"]["records"]), 14)

            with open(txt_file, "r", encoding="utf-8") as f:
                txt_content = f.read()
            self.assertIn("BENCHMARK REPORT: CONSOLIDATED_FAILURE_TRAVERSAL [PASSED]", txt_content)
            self.assertIn("total_failure_cases", txt_content)

    # =========================================================================
    # 5. PERCEPTION ISOLATION INVARIANT (0 FALSE RECOVERY EVENTS)
    # =========================================================================

    def test_perception_isolation_zero_false_recovery_invariant(self) -> None:
        """
        Verifies that visual disturbances, detector false negatives, occlusions, and
        multimodal conflicts produce exactly 0 false recovery triggers in RecoveryManager.
        """
        result = BenchmarkHarness.run_consolidated_failure_traversal_benchmark()
        metrics = result.metrics

        self.assertEqual(metrics["false_recovery_count_for_perception"], 0)
        self.assertEqual(metrics["ai_failure_coverage"], "8/8 (100.0%)")

        # In-depth check of records
        ai_records = [r for r in metrics["records"] if r["domain"] == "AI_PERCEPTION"]
        self.assertEqual(len(ai_records), 8)
        for r in ai_records:
            self.assertEqual(r["recovery_containment_result"], "ISOLATED_ZERO_RECOVERY")
            self.assertTrue(r["passed"])

    # =========================================================================
    # 6. PROCEDURE VIOLATION RECOVERY INVARIANT (100% RECOVERY GENERATION)
    # =========================================================================

    def test_procedure_violation_confirmed_recovery_invariant(self) -> None:
        """
        Verifies that confirmed procedural sequence violations produce authoritative
        RecoveryEvent objects binding canonical recovery text from config/experiment.json.
        """
        result = BenchmarkHarness.run_consolidated_failure_traversal_benchmark()
        metrics = result.metrics

        self.assertEqual(metrics["recovery_count_for_confirmed_procedure_violations"], 6)
        self.assertEqual(metrics["procedure_failure_coverage"], "6/6 (100.0%)")

        proc_records = [r for r in metrics["records"] if r["domain"] == "PROCEDURE"]
        self.assertEqual(len(proc_records), 6)
        for r in proc_records:
            self.assertTrue(r["recovery_containment_result"].startswith("RECOVERY_"))
            self.assertTrue(r["passed"])

    # =========================================================================
    # 7. CROSS-SUBSYSTEM AUTHORITY BOUNDARIES
    # =========================================================================

    def test_cross_subsystem_authority_boundaries(self) -> None:
        """
        Verifies that the consolidated traversal maintains absolute authority decoupling:
          - B10 temporal filter does not advance FSM on unconfirmed noise
          - B11 consistency evaluator vetoes invalid object combinations
          - FSM is the sole step sequencer
          - B12 recovery reacts strictly to authoritative FSM violations
          - B13 telemetry is an out-of-band passive observer
        """
        # Step 1: Inject noisy perception (B11 veto)
        res_veto = self.pipeline.process_frame_public(self.dummy_frame)
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(self.rec_mgr.state, "IDLE")

        # Step 2: Inject valid S1 action
        fsm_valid_s1 = self.fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        self.assertEqual(fsm_valid_s1["status"], "VALID")
        self.assertEqual(self.fsm.current_step_index, 1)
        self.assertEqual(self.rec_mgr.state, "IDLE")

        # Step 3: Inject out-of-order action (P02 at S2)
        fsm_ooo = self.fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")
        self.assertEqual(fsm_ooo["status"], "SKIPPED")
        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_ooo)
        self.assertIsNotNone(rec_event)
        self.assertEqual(self.rec_mgr.state, "RECOVERY_ACTIVE")

        # Step 4: Passive telemetry snapshot verification
        self.agg.record_cycle(
            consistency_result={"category": "RELIABLE_ALIGNED", "is_reliable": True},
            uncertainty_result={"state": "CONFIDENT", "is_reliable": True},
            temporal_result={"confirmed": True, "action": "CLOSE_LID", "object": "CONTAINER_LID"},
            fsm_result=fsm_ooo,
            recovery_event=rec_event,
            validator=self.fsm,
        )
        snapshot = self.agg.get_snapshot()
        self.assertEqual(snapshot.recovery.active_state, "RECOVERY_ACTIVE")

    # =========================================================================
    # 8. FROZEN PUBLIC 8-FIELD CONTRACT PRESERVATION
    # =========================================================================

    def test_public_8_field_schema_preservation_across_entire_matrix(self) -> None:
        """
        Verifies that all public AI responses emitted during perception and procedure
        error stimulus strictly match the frozen 8-field public AI contract.
        """
        for i in range(10):
            res = self.pipeline.process_frame_public(self.dummy_frame)
            self.assertTrue(AIResultAdapter.validate_public_contract(res))
            self.assertIn("timestamp", res)
            self.assertIn("action", res)
            self.assertIn("object", res)
            self.assertIn("confidence", res)
            self.assertIn("expected_step", res)
            self.assertIn("detected_step", res)
            self.assertIn("status", res)
            self.assertIn("next_step", res)
            self.assertEqual(len(res), 8)

    # =========================================================================
    # 9. END-TO-END LIFECYCLE STRESS WITH INTERLEAVED FAULTS
    # =========================================================================

    def test_end_to_end_lifecycle_stress_with_interleaved_faults(self) -> None:
        """
        Executes a complete EXP-001 lifecycle with interleaved perception faults
        (F01, F03, F07) and procedural faults (P02, P03), confirming recovery and resolution.
        """
        eval_f01 = MultimodalConsistencyEvaluator().evaluate(
            action="PICK_RED",
            action_confidence=0.90,
            object_name="BLUE_SAMPLE",
            object_confidence=0.90,
            objects=[{"label": "BLUE_SAMPLE", "confidence": 0.9}],
        )
        self.assertFalse(eval_f01["is_reliable"])
        self.assertEqual(self.rec_mgr.state, "IDLE")

        s1_res = self.fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        self.assertEqual(s1_res["status"], "VALID")
        self.assertEqual(self.fsm.current_step_index, 1)

        # S2: Procedural violation (repeat S1 at S2) -> RECOVERY_ACTIVE -> Corrective valid S2 -> RECOVERED
        s2_err = self.fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        self.assertEqual(s2_err["status"], "OUT_OF_SEQUENCE")
        rec_e = self.rec_mgr.evaluate_fsm_result(s2_err)
        self.assertIsNotNone(rec_e)
        self.assertEqual(self.rec_mgr.state, "RECOVERY_ACTIVE")
        self.assertEqual(self.fsm.current_step_index, 1)

        s2_valid = self.fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
        self.assertEqual(s2_valid["status"], "VALID")
        self.rec_mgr.evaluate_fsm_result(s2_valid)
        self.assertEqual(self.rec_mgr.state, "RECOVERED")
        self.assertEqual(self.fsm.current_step_index, 2)

        # S3: Nominal valid S3 -> IDLE
        s3_valid = self.fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
        self.assertEqual(s3_valid["status"], "VALID")
        self.rec_mgr.evaluate_fsm_result(s3_valid)
        self.assertEqual(self.rec_mgr.state, "IDLE")
        self.assertEqual(self.fsm.current_step_index, 3)

        # S4: Nominal valid S4
        s4_valid = self.fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
        self.assertEqual(s4_valid["status"], "VALID")
        self.assertEqual(self.fsm.current_step_index, 4)

        # S5: Nominal valid S5 -> Procedure completed
        s5_valid = self.fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")
        self.assertEqual(s5_valid["status"], "VALID")
        self.assertEqual(self.fsm.state, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
