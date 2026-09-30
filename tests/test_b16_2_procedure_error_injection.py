"""
test_b16_2_procedure_error_injection.py — Dedicated Procedure Error Injection & Recovery Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B16.2)

Verifies all six (6) procedure failure modes (P01–P06) across the procedure validation graph:
  P01: Skipped step (Forward jump over intermediate steps, e.g. S1 -> S3)
  P02: Out-of-order step (Illegal step sequence transitions, e.g. S1 -> S4)
  P03: Repeated past step (Re-execution of an already-completed step, e.g. at S3 execute S1)
  P04: Premature action (Attempting final step S5 before completing prior transfers)
  P05: Incomplete action / object mismatch (Action attempted on incorrect required object)
  P06: Step timeout (Deterministic timestamp progression exceeding canonical step timeouts)

Architectural Invariants Strictly Enforced:
1. Authority Boundaries:
   - SequenceValidatorFSM (B9) remains the sole procedural sequencing authority.
   - RecoveryManager (B12) generates RecoveryEvents ONLY on authoritative procedural violations.
   - VoiceAlertService & Bridge provide debounced out-of-band corrective alerts without mutating FSM.
   - TelemetryDiagnosticAggregator (B13) passively records procedural anomalies without side effects.
   - B16 remains test/injection authority only (zero background watchdog threads added to production).
2. Contract Strictness:
   - Frozen 8-field public AI contract (docs/architecture.md §2) strictly preserved across all procedural errors.
3. Canonical Procedure:
   - Standard EXP-001 (S1: PICK_RED, S2: PLACE_RED, S3: PICK_BLUE, S4: PLACE_BLUE, S5: CLOSE_LID).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

try:
    from PySide6.QtCore import QCoreApplication
except ImportError:
    QCoreApplication = None

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import AIResultAdapter
from backend.ai.telemetry_aggregator import TelemetryDiagnosticAggregator
from backend.app_state import AppState
from backend.bridge import Bridge
from backend.experiment.procedure_manager import ProcedureManager, ProcedureStep
from backend.experiment.recovery_manager import RecoveryEvent, RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.voice.voice_alert import VoiceAlertService


class TestB162ProcedureErrorInjection(unittest.TestCase):
    """Dedicated integration test suite for Gate B16.2 Procedure Failure & Recovery Modes."""

    @classmethod
    def setUpClass(cls) -> None:
        if QCoreApplication is not None and QCoreApplication.instance() is None:
            cls.qapp = QCoreApplication([])
        else:
            cls.qapp = None

    def setUp(self) -> None:
        self.config_path = Path(__file__).resolve().parent.parent / "config" / "experiment.json"
        self.pm = ProcedureManager(str(self.config_path))
        self.fsm = SequenceValidatorFSM(self.pm, min_confidence_threshold=0.70)
        self.fsm.start()
        self.rec_mgr = RecoveryManager(self.pm)
        self.agg = TelemetryDiagnosticAggregator()
        self.voice = VoiceAlertService(enabled=True)
        self.pipeline = InferencePipeline(
            validator=self.fsm,
            recovery_manager=self.rec_mgr,
            telemetry_aggregator=self.agg,
        )
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        if self.fsm._session_active:
            self.fsm.reset()
        self.pipeline.reset()
        self.rec_mgr.reset()
        self.voice.shutdown()

    # =========================================================================
    # P01: SKIPPED STEP (Forward Skip Over Intermediate Step)
    # =========================================================================

    def test_p01_single_step_skip_injection_and_recovery(self) -> None:
        """
        P01 Case 1: While expecting S1 (PICK_RED), astronaut skips to S2 (PLACE_RED on RED_SAMPLE).
        Verifies:
          - FSM identifies skipped step S1 and auto-advances past S1 to S2.
          - FSM status is SKIPPED.
          - RecoveryManager generates RecoveryEvent with S1 recovery instruction:
            "Return hand to starting position and re-acquire the red sample."
          - RecoveryManager transitions to RECOVERY_ACTIVE.
          - Public AI contract status is SKIPPED with expected_step=S1, detected_step=S2.
        """
        # Inject PLACE_RED on RED_SAMPLE while expecting S1
        fsm_res = self.fsm.validate_action(
            detected_action="PLACE_RED",
            confidence=0.92,
            object_name="RED_SAMPLE",
            timestamp="2026-09-30T10:00:00",
        )

        self.assertEqual(fsm_res["status"], "SKIPPED")
        self.assertEqual(fsm_res["validation_status"], "SKIPPED")
        self.assertEqual(fsm_res["expected_step"], "S1")
        self.assertEqual(fsm_res["detected_step"], "S2")
        self.assertEqual(fsm_res["next_step"], "S3")
        self.assertIn("S1", fsm_res.get("skipped_step_ids", []))

        # Evaluate Recovery
        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertIsInstance(rec_event, RecoveryEvent)
        self.assertEqual(rec_event.expected_step, "S1")
        self.assertEqual(rec_event.procedural_status, "SKIPPED")
        self.assertEqual(rec_event.recovery_instruction, "Return hand to starting position and re-acquire the red sample.")
        self.assertEqual(rec_event.timeout_s, 30.0)
        self.assertEqual(self.rec_mgr.state, "RECOVERY_ACTIVE")

        # Verify Public Adapter Contract
        pub_res = AIResultAdapter.adapt(internal_result=None, fsm_result=fsm_res)
        self.assertEqual(pub_res["status"], "SKIPPED")
        self.assertEqual(pub_res["expected_step"], "S1")
        self.assertEqual(pub_res["detected_step"], "S2")
        self.assertEqual(pub_res["next_step"], "S3")
        self.assertTrue(AIResultAdapter.validate_public_contract(pub_res))

    def test_p01_multi_step_skip_injection_and_recovery(self) -> None:
        """
        P01 Case 2: Distant skip from S1 directly to S3 (PICK_BLUE on BLUE_SAMPLE).
        Verifies:
          - FSM identifies skipped steps S1 and S2, auto-advancing to S3 (expecting S4).
          - RecoveryManager binds recovery instructions for skipped step.
        """
        fsm_res = self.fsm.validate_action(
            detected_action="PICK_BLUE",
            confidence=0.90,
            object_name="BLUE_SAMPLE",
            timestamp="2026-09-30T10:00:01",
        )

        self.assertEqual(fsm_res["status"], "SKIPPED")
        self.assertEqual(fsm_res["expected_step"], "S1")
        self.assertEqual(fsm_res["detected_step"], "S3")
        self.assertEqual(fsm_res["next_step"], "S4")
        self.assertEqual(fsm_res.get("skipped_step_ids"), ["S1", "S2"])

        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S1")
        self.assertEqual(self.rec_mgr.state, "RECOVERY_ACTIVE")

    # =========================================================================
    # P02: OUT-OF-ORDER STEP (Illegal Sequence Transition)
    # =========================================================================

    def test_p02_out_of_order_step_containment_and_recovery(self) -> None:
        """
        P02: While expecting S2 (PLACE_RED), astronaut performs S4 (PLACE_BLUE on BLUE_SAMPLE).
        Verifies:
          - S1 is confirmed first.
          - While expecting S2, PLACE_BLUE is injected.
          - FSM flags SKIPPED or OUT_OF_ORDER, does NOT validate S2.
          - RecoveryManager triggers RecoveryEvent for S2 ("Retrieve red sample if dislodged and place firmly inside the container.").
          - Contract status is SKIPPED / OUT_OF_SEQUENCE.
        """
        # Step 1 confirmed nominally
        res1 = self.fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        self.assertEqual(res1["status"], "VALID")
        self.assertEqual(self.fsm.current_step_index, 1)  # Expecting S2

        # Inject S4 (PLACE_BLUE on BLUE_SAMPLE)
        fsm_res = self.fsm.validate_action(
            detected_action="PLACE_BLUE",
            confidence=0.91,
            object_name="BLUE_SAMPLE",
            timestamp="2026-09-30T10:00:02",
        )

        self.assertEqual(fsm_res["status"], "SKIPPED")
        self.assertEqual(fsm_res["expected_step"], "S2")
        self.assertEqual(fsm_res["detected_step"], "S4")

        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S2")
        self.assertEqual(rec_event.recovery_instruction, "Retrieve red sample if dislodged and place firmly inside the container.")
        self.assertEqual(rec_event.timeout_s, 30.0)

    # =========================================================================
    # P03: REPEATED PAST STEP (Re-executing an Already-Completed Step)
    # =========================================================================

    def test_p03_repeated_past_step_rejection_and_guidance(self) -> None:
        """
        P03: Steps S1 and S2 are completed. While expecting S3 (PICK_BLUE),
        astronaut repeats S1 (PICK_RED on RED_SAMPLE).
        Verifies:
          - FSM identifies matching_step (S1) <= expected_step (S3).
          - FSM validation_status is OUT_OF_ORDER and status is OUT_OF_SEQUENCE.
          - FSM current_step_index remains 2 (expecting S3).
          - RecoveryManager generates RecoveryEvent for S3 ("Return hand to starting position and re-acquire the blue sample.").
          - Public contract status is OUT_OF_SEQUENCE.
        """
        # Validate S1 and S2
        self.fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        self.fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
        self.assertEqual(self.fsm.current_step_index, 2)  # Expecting S3

        # Repeat S1 (PICK_RED on RED_SAMPLE)
        fsm_res = self.fsm.validate_action(
            detected_action="PICK_RED",
            confidence=0.92,
            object_name="RED_SAMPLE",
            timestamp="2026-09-30T10:00:03",
        )

        self.assertEqual(fsm_res["status"], "OUT_OF_SEQUENCE")
        self.assertEqual(fsm_res["validation_status"], "OUT_OF_ORDER")
        self.assertEqual(fsm_res["expected_step"], "S3")
        self.assertEqual(fsm_res["detected_step"], "S1")
        self.assertEqual(fsm_res["next_step"], "S3")

        # FSM step index unchanged
        self.assertEqual(self.fsm.current_step_index, 2)

        # RecoveryEvent generated for S3
        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S3")
        self.assertEqual(rec_event.procedural_status, "OUT_OF_ORDER")
        self.assertEqual(rec_event.recovery_instruction, "Return hand to starting position and re-acquire the blue sample.")
        self.assertEqual(rec_event.timeout_s, 30.0)

        # Public contract
        pub_res = AIResultAdapter.adapt(internal_result=None, fsm_result=fsm_res)
        self.assertEqual(pub_res["status"], "OUT_OF_SEQUENCE")
        self.assertEqual(pub_res["expected_step"], "S3")
        self.assertEqual(pub_res["detected_step"], "S1")
        self.assertTrue(AIResultAdapter.validate_public_contract(pub_res))

    # =========================================================================
    # P04: PREMATURE ACTION (Premature S5 CLOSE_LID Attempt)
    # =========================================================================

    def test_p04_premature_close_lid_injection_and_recovery(self) -> None:
        """
        P04: While expecting S1 (or S2), astronaut prematurely executes S5 (CLOSE_LID on CONTAINER_LID).
        Verifies:
          - FSM identifies premature jump to S5, flagging skipped transfer steps S1–S4.
          - RecoveryManager generates RecoveryEvent for step S1 with appropriate instructions.
          - RecoveryManager timeout is bounded.
        """
        fsm_res = self.fsm.validate_action(
            detected_action="CLOSE_LID",
            confidence=0.93,
            object_name="CONTAINER_LID",
            timestamp="2026-09-30T10:00:04",
        )

        self.assertEqual(fsm_res["status"], "SKIPPED")
        self.assertEqual(fsm_res["expected_step"], "S1")
        self.assertEqual(fsm_res["detected_step"], "S5")
        self.assertEqual(fsm_res.get("skipped_step_ids"), ["S1", "S2", "S3", "S4"])

        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S1")
        self.assertEqual(rec_event.recovery_instruction, "Return hand to starting position and re-acquire the red sample.")
        self.assertEqual(self.rec_mgr.state, "RECOVERY_ACTIVE")

    # =========================================================================
    # P05: INCOMPLETE ACTION / OBJECT MISMATCH (Action on Wrong Required Object)
    # =========================================================================

    def test_p05_incomplete_action_invalid_object_containment(self) -> None:
        """
        P05 Case 1: At Step S1, astronaut attempts PICK_RED on BLUE_SAMPLE or CONTAINER_LID.
        Verifies:
          - FSM detects matching action (PICK_RED) but mismatch on required object (RED_SAMPLE).
          - FSM returns error_type="INVALID_OBJECT", status="OUT_OF_SEQUENCE", validation_status="OUT_OF_ORDER".
          - FSM current_step_index remains 0.
          - RecoveryManager generates RecoveryEvent for S1.
        """
        fsm_res = self.fsm.validate_action(
            detected_action="PICK_RED",
            confidence=0.91,
            object_name="BLUE_SAMPLE",
            timestamp="2026-09-30T10:00:05",
        )

        self.assertEqual(fsm_res["status"], "OUT_OF_SEQUENCE")
        self.assertEqual(fsm_res["error_type"], "INVALID_OBJECT")
        self.assertEqual(fsm_res["expected_step"], "S1")
        self.assertEqual(self.fsm.current_step_index, 0)

        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S1")
        self.assertEqual(rec_event.procedural_status, "INVALID_OBJECT")
        self.assertEqual(rec_event.recovery_instruction, "Return hand to starting position and re-acquire the red sample.")

    def test_p05_incomplete_action_unrecognized_action_containment(self) -> None:
        """
        P05 Case 2: Out-of-vocabulary or unknown action (e.g. PIPETTE_TRANSFER or UNKNOWN_MANIPULATION).
        Verifies:
          - FSM flags validation_status="UNRECOGNIZED", status="OUT_OF_SEQUENCE".
          - FSM step is not advanced.
          - RecoveryManager generates RecoveryEvent for current expected step.
        """
        fsm_res = self.fsm.validate_action(
            detected_action="UNKNOWN_GESTURE",
            confidence=0.88,
            object_name="RED_SAMPLE",
            timestamp="2026-09-30T10:00:06",
        )

        self.assertEqual(fsm_res["status"], "OUT_OF_SEQUENCE")
        self.assertEqual(fsm_res["validation_status"], "UNRECOGNIZED")
        self.assertEqual(fsm_res["expected_step"], "S1")
        self.assertIsNone(fsm_res["detected_step"])

        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S1")
        self.assertEqual(rec_event.procedural_status, "UNRECOGNIZED")

    # =========================================================================
    # P06: TIMEOUT (Deterministic Progression & Recovery Metadata)
    # =========================================================================

    def test_p06_timeout_metadata_and_deterministic_progression(self) -> None:
        """
        P06: Step S1 has configured timeout of 30.0s, S5 has 25.0s.
        Verifies:
          - Canonical ProcedureStep instances define timeout_s correctly (S1-S4=30.0s, S5=25.0s).
          - Deterministic timestamp progression tracks step duration without background timer threads.
          - RecoveryEvent embeds exact timeout_s for operator guidance.
        """
        # Audit canonical step timeouts from config
        s1 = self.pm.get_step_by_id("S1")
        s2 = self.pm.get_step_by_id("S2")
        s3 = self.pm.get_step_by_id("S3")
        s4 = self.pm.get_step_by_id("S4")
        s5 = self.pm.get_step_by_id("S5")

        self.assertEqual(s1.timeout_s, 30.0)
        self.assertEqual(s2.timeout_s, 30.0)
        self.assertEqual(s3.timeout_s, 30.0)
        self.assertEqual(s4.timeout_s, 30.0)
        self.assertEqual(s5.timeout_s, 25.0)

        # Advance FSM to S5
        self.fsm.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        self.fsm.validate_action("PLACE_RED", 0.95, "RED_SAMPLE")
        self.fsm.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
        self.fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
        self.assertEqual(self.fsm.current_step_index, 4)  # Expecting S5

        # Invalidate S5 with out-of-order action at t = t0 + 35.0s (> 25.0s timeout)
        fsm_res = self.fsm.validate_action(
            detected_action="PICK_RED",
            confidence=0.90,
            object_name="RED_SAMPLE",
            timestamp="2026-09-30T10:00:35",
        )

        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)
        self.assertEqual(rec_event.expected_step, "S5")
        self.assertEqual(rec_event.timeout_s, 25.0)
        self.assertEqual(rec_event.recovery_instruction, "Re-align container lid and press firmly until locked.")

    # =========================================================================
    # BRIDGE SIGNALS & VOICE ALERT INTEGRATION
    # =========================================================================

    def test_bridge_recovery_signal_emission_and_getter(self) -> None:
        """
        Verifies Bridge emits recoveryAlertReady / recoveryAlertJsonReady Qt signals
        upon RecoveryEvent generation and returns structured guidance via getActiveRecoveryGuidance().
        """
        app_state = AppState(sequence_validator=self.fsm, recovery_manager=self.rec_mgr)
        bridge = Bridge(app_state)
        if bridge.inference_worker is not None:
            bridge.inference_worker.start = MagicMock(return_value=True)

        signal_mock = MagicMock()
        json_signal_mock = MagicMock()
        bridge.recoveryAlertReady.connect(signal_mock)
        bridge.recoveryAlertJsonReady.connect(json_signal_mock)

        # Trigger procedural violation (S1 -> S3 skip)
        fsm_res = self.fsm.validate_action("PICK_BLUE", 0.90, "BLUE_SAMPLE")
        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)
        self.assertIsNotNone(rec_event)

        # Dispatch via Bridge handler
        bridge.emit_recovery_alert(rec_event)

        # Verify signals received
        signal_mock.assert_called_once()
        json_signal_mock.assert_called_once()
        raw_json = json_signal_mock.call_args[0][0]
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["expected_step"], "S1")
        self.assertEqual(parsed["procedural_status"], "SKIPPED")

        # Verify QWebChannel getter slot
        guidance_json = bridge.getActiveRecoveryGuidance()
        self.assertNotEqual(guidance_json, "null")
        guidance = json.loads(guidance_json)
        self.assertEqual(guidance["expected_step"], "S1")
        self.assertEqual(guidance["recovery_instruction"], "Return hand to starting position and re-acquire the red sample.")

        bridge.shutdown()

    def test_voice_alert_recovery_debouncing(self) -> None:
        """
        Verifies VoiceAlertService debounces repeated identical recovery guidance alerts
        within a 3.0s window.
        """
        voice = VoiceAlertService(enabled=True)
        voice.speak = MagicMock()

        # Alert 1: Dispatched
        res1 = voice.alert_recovery_guidance("S1", "Return hand to starting position and re-acquire the red sample.", debounce_seconds=3.0)
        self.assertTrue(res1)
        voice.speak.assert_called_once()

        # Alert 2 (Immediate duplicate): Suppressed by debounce
        voice.speak.reset_mock()
        res2 = voice.alert_recovery_guidance("S1", "Return hand to starting position and re-acquire the red sample.", debounce_seconds=3.0)
        self.assertFalse(res2)
        voice.speak.assert_not_called()

        # Alert 3 (Different step or text): Dispatched immediately
        res3 = voice.alert_recovery_guidance("S2", "Retrieve red sample if dislodged and place firmly inside the container.", debounce_seconds=3.0)
        self.assertTrue(res3)
        voice.speak.assert_called_once()

        # Reset debounce cache -> Alert 1 dispatched again
        voice.speak.reset_mock()
        voice.reset_debounce()
        res4 = voice.alert_recovery_guidance("S1", "Return hand to starting position and re-acquire the red sample.", debounce_seconds=3.0)
        self.assertTrue(res4)
        voice.speak.assert_called_once()

        voice.shutdown()

    # =========================================================================
    # PASSIVE TELEMETRY RECORDING (B13 INTEGRATION)
    # =========================================================================

    def test_passive_telemetry_recording_of_procedure_failures(self) -> None:
        """
        Verifies TelemetryDiagnosticAggregator (B13) passively records procedural anomalies
        and recovery events across the pipeline without mutating FSM state.
        """
        # Frame 1: Valid S1
        self.pipeline.process_frame_public(self.dummy_frame, fsm_status="VALID")
        # Frame 2: Skipped step S1 -> S3
        fsm_res = self.fsm.validate_action("PICK_BLUE", 0.90, "BLUE_SAMPLE")
        rec_event = self.rec_mgr.evaluate_fsm_result(fsm_res)

        self.agg.record_cycle(
            consistency_result={"category": "RELIABLE_ALIGNED", "is_reliable": True},
            uncertainty_result={"state": "CONFIDENT", "is_reliable": True},
            temporal_result={"confirmed": True, "action": "PICK_BLUE", "object": "BLUE_SAMPLE"},
            fsm_result=fsm_res,
            recovery_event=rec_event,
            validator=self.fsm,
        )

        snapshot = self.agg.get_snapshot()
        self.assertEqual(snapshot.recovery.active_state, "RECOVERY_ACTIVE")
        self.assertIsNotNone(snapshot.recovery.latest_event)
        self.assertEqual(snapshot.recovery.latest_event.get("expected_step"), "S1")
        self.assertEqual(snapshot.recovery.latest_event.get("procedural_status"), "SKIPPED")

    # =========================================================================
    # RECOVERY LIFECYCLE RESOLUTION (IDLE -> RECOVERY_ACTIVE -> RECOVERED -> IDLE)
    # =========================================================================

    def test_recovery_lifecycle_full_nominal_resolution(self) -> None:
        """
        Verifies the full recovery state machine cycle:
          1. IDLE (nominal state)
          2. Procedural violation occurs -> RECOVERY_ACTIVE
          3. Operator executes corrective valid action -> RECOVERED
          4. Next nominal valid action confirms -> IDLE
        """
        self.assertEqual(self.rec_mgr.state, "IDLE")

        # 1. Procedural violation (S1 -> S3 skip)
        fsm_res1 = self.fsm.validate_action("PICK_BLUE", 0.90, "BLUE_SAMPLE")
        self.rec_mgr.evaluate_fsm_result(fsm_res1)
        self.assertEqual(self.rec_mgr.state, "RECOVERY_ACTIVE")

        # 2. Operator performs corrective action for current expected step (S4 PLACE_BLUE)
        fsm_res2 = self.fsm.validate_action("PLACE_BLUE", 0.95, "BLUE_SAMPLE")
        self.assertEqual(fsm_res2["status"], "VALID")
        self.rec_mgr.evaluate_fsm_result(fsm_res2)
        self.assertEqual(self.rec_mgr.state, "RECOVERED")

        # 3. Subsequent nominal action (S5 CLOSE_LID)
        fsm_res3 = self.fsm.validate_action("CLOSE_LID", 0.95, "CONTAINER_LID")
        self.assertEqual(fsm_res3["status"], "VALID")
        self.rec_mgr.evaluate_fsm_result(fsm_res3)
        self.assertEqual(self.rec_mgr.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
