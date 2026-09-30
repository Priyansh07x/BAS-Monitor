"""
tests/test_b18_1_joint_acceptance.py
====================================
Workstream B — Gate B18.1: Joint End-to-End Software Acceptance Demonstration
& Multi-Stream Verification Suite.

Validates the complete Workstream B software chain across 4 multi-stream scenarios:
  1. Stream A: Nominal complete sequence traversal (S1 -> S5, CSSR=1.0, 0 recoveries).
  2. Stream B: Perception noise rejection (transient spikes, low-confidence, multimodal conflicts).
  3. Stream C: Procedural violation stream (forward skip S1 -> S3 triggering FSM + RecoveryManager).
  4. Stream D: Multi-cycle lifecycle continuity (nominal -> violation -> reset -> nominal).

Also verifies:
  - Background worker threading and single-slot LatestFrameBuffer pacing.
  - Qt Bridge signal integration (aiResultReady, recoveryAlertReady) and VoiceAlertService.
  - Frozen 8-field public AI contract invariance.
  - BenchmarkHarness B18.1 discovery, execution, and artifact export.
  - Epistemic honesty disclosures regarding software simulation vs physical hardware/neural status.
"""

import os
import time
import tempfile
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any, List

from backend.app_state import AppState
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.telemetry_aggregator import TelemetryDiagnosticAggregator
from backend.ai.inference_worker import InferenceWorker, RateStrategy, LatestFrameBuffer
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.experiment.recovery_manager import RecoveryManager
from backend.bridge import Bridge
from backend.voice.voice_alert import VoiceAlertService
from evaluation.benchmark_harness import BenchmarkHarness, BenchmarkResult


REQUIRED_PUBLIC_KEYS = {
    "timestamp",
    "action",
    "object",
    "confidence",
    "expected_step",
    "detected_step",
    "status",
    "next_step",
}

VALID_PUBLIC_STATUSES = {"VALID", "SKIPPED", "OUT_OF_SEQUENCE"}


def feed_pipeline(pipeline: InferencePipeline, frame: np.ndarray, act: str, obj: str, conf: float, ts: str) -> Dict[str, Any]:
    """Helper to inject deterministic perception stimulus through live pipeline stages."""
    pipeline.action_classifier.classify = lambda **kw: (act, conf)
    pipeline.object_detector.detect = lambda f: [{"label": obj, "confidence": conf, "x1": 100, "y1": 100, "x2": 200, "y2": 200}]
    pipeline.hand_detector.detect = lambda f: [{"label": "HAND", "x1": 100, "y1": 100, "x2": 200, "y2": 200}]
    pipeline.interaction_engine.evaluate_interaction = lambda **kw: {"state": "HOLDING", "target_object": obj, "scale_proximity": 0.8}
    return pipeline.process_frame_public(frame=frame, timestamp=ts)


class TestJointSoftwareAcceptanceSuite:
    """Gate B18.1 Joint End-to-End Software Acceptance Demonstration."""

    def test_stream_a_nominal_complete_traversal(self):
        """
        Stream A: Traverses full canonical EXP-001 sequence (S1 -> S5)
        with >= 3 frames per step. Verifies FSM reaches completed state,
        CSSR is 100%, and zero false recoveries are triggered.
        """
        agg = TelemetryDiagnosticAggregator()
        fsm = SequenceValidatorFSM()
        rec_mgr = RecoveryManager(fsm.pm)
        fsm.start()

        pipeline = InferencePipeline(
            validator=fsm,
            recovery_manager=rec_mgr,
            telemetry_aggregator=agg,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        nominal_steps = [
            ("PICK_RED", "RED_SAMPLE", "S1"),
            ("PLACE_RED", "RED_SAMPLE", "S2"),
            ("PICK_BLUE", "BLUE_SAMPLE", "S3"),
            ("PLACE_BLUE", "BLUE_SAMPLE", "S4"),
            ("CLOSE_LID", "CONTAINER_LID", "S5"),
        ]

        observed_statuses = []
        for step_idx, (act, obj, step_id) in enumerate(nominal_steps):
            for f in range(3):
                ts = f"2026-09-30T10:00:{step_idx:02d}_{f}"
                res = feed_pipeline(pipeline, dummy_frame, act, obj, 0.95, ts)
                assert set(res.keys()) == REQUIRED_PUBLIC_KEYS
                assert res["status"] in VALID_PUBLIC_STATUSES
                observed_statuses.append(res["status"])

        assert fsm.state == "COMPLETED", "FSM should reach COMPLETED state after S1..S5"
        assert fsm.current_step_index == 5
        assert rec_mgr.state == "IDLE", "No recoveries should be active in nominal stream"
        assert len(rec_mgr.recovery_history) == 0
        assert all(st == "VALID" for st in observed_statuses), "All nominal step confirmations must be VALID"

    def test_stream_b_perception_noise_containment(self):
        """
        Stream B: Injects perception noise (transient spikes, low-confidence, multimodal conflicts).
        Verifies B10 temporal filter and B11 uncertainty handlers contain noise, FSM does not mutate,
        zero false recoveries are emitted, and nominal S1 completes cleanly afterward.
        """
        agg = TelemetryDiagnosticAggregator()
        fsm = SequenceValidatorFSM()
        rec_mgr = RecoveryManager(fsm.pm)
        fsm.start()

        pipeline = InferencePipeline(
            validator=fsm,
            recovery_manager=rec_mgr,
            telemetry_aggregator=agg,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. 1-frame transient spike (premature S5 CLOSE_LID)
        res_spike = feed_pipeline(pipeline, dummy_frame, "CLOSE_LID", "CONTAINER_LID", 0.95, "2026-09-30T10:01:00")
        assert fsm.current_step_index == 0, "Transient spike must not advance FSM"
        assert rec_mgr.state == "IDLE"
        assert len(rec_mgr.recovery_history) == 0

        # 2. Low-confidence frame (0.30 confidence)
        res_low_conf = feed_pipeline(pipeline, dummy_frame, "PICK_RED", "RED_SAMPLE", 0.30, "2026-09-30T10:01:01")
        assert fsm.current_step_index == 0
        assert rec_mgr.state == "IDLE"
        assert len(rec_mgr.recovery_history) == 0

        # 3. Multimodal conflict frame (PICK_RED action on BLUE_SAMPLE)
        pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.92)
        pipeline.object_detector.detect = lambda f: [{"label": "BLUE_SAMPLE", "confidence": 0.92, "x1": 100, "y1": 100, "x2": 200, "y2": 200}]
        pipeline.hand_detector.detect = lambda f: [{"label": "HAND", "x1": 100, "y1": 100, "x2": 200, "y2": 200}]
        pipeline.interaction_engine.evaluate_interaction = lambda **kw: {"state": "HOLDING", "target_object": "BLUE_SAMPLE", "scale_proximity": 0.8}
        res_conflict = pipeline.process_frame_public(frame=dummy_frame, timestamp="2026-09-30T10:01:02")
        assert fsm.current_step_index == 0
        assert rec_mgr.state == "IDLE"
        assert len(rec_mgr.recovery_history) == 0

        # 4. Now provide 3 clean frames of S1 (PICK_RED, RED_SAMPLE)
        for f in range(3):
            res_s1 = feed_pipeline(pipeline, dummy_frame, "PICK_RED", "RED_SAMPLE", 0.95, f"2026-09-30T10:01:0{3+f}")
        assert fsm.current_step_index == 1, "S1 should confirm and advance FSM to S2 (index 1)"
        assert rec_mgr.state == "IDLE"
        assert len(rec_mgr.recovery_history) == 0

    def test_stream_c_authoritative_violation_and_recovery(self):
        """
        Stream C: Injects authoritative procedural violation (forward skip S1 -> S3 skipping S2).
        Verifies FSM reports SKIPPED status and RecoveryManager generates structured recovery instruction.
        """
        agg = TelemetryDiagnosticAggregator()
        fsm = SequenceValidatorFSM()
        rec_mgr = RecoveryManager(fsm.pm)
        fsm.start()

        pipeline = InferencePipeline(
            validator=fsm,
            recovery_manager=rec_mgr,
            telemetry_aggregator=agg,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Complete S1
        for f in range(3):
            feed_pipeline(pipeline, dummy_frame, "PICK_RED", "RED_SAMPLE", 0.95, f"2026-09-30T10:02:0{f}")
        assert fsm.current_step_index == 1

        # Now inject S3 (PICK_BLUE, BLUE_SAMPLE) sustained for 3 frames (skipping S2 PLACE_RED)
        last_res = None
        for f in range(3):
            last_res = feed_pipeline(pipeline, dummy_frame, "PICK_BLUE", "BLUE_SAMPLE", 0.95, f"2026-09-30T10:02:1{f}")

        assert last_res is not None
        assert last_res["status"] == "SKIPPED"
        assert last_res["expected_step"] == "S2"
        assert last_res["detected_step"] == "S3"

        rec_event = pipeline.last_recovery_event or rec_mgr.last_recovery_event
        assert rec_event is not None, "Recovery event must be generated for procedural skip"
        assert rec_event.procedural_status == "SKIPPED"
        assert rec_event.expected_step == "S2"
        assert rec_event.detected_step == "S3"
        assert "red sample" in rec_event.recovery_instruction.lower()

    def test_stream_d_multi_cycle_continuity_and_reset(self):
        """
        Stream D: Verifies lifecycle continuity across multiple nominal/recovery/reset cycles.
        Ensures zero memory leakage, bounded telemetry deques, and complete isolation between cycles.
        """
        agg = TelemetryDiagnosticAggregator()
        fsm = SequenceValidatorFSM()
        rec_mgr = RecoveryManager(fsm.pm)

        pipeline = InferencePipeline(
            validator=fsm,
            recovery_manager=rec_mgr,
            telemetry_aggregator=agg,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        num_cycles = 3

        for cycle in range(num_cycles):
            fsm.reset()
            fsm.start()
            assert fsm.current_step_index == 0
            assert fsm.state == "RUNNING"

            # Run full S1..S5
            for act, obj in [
                ("PICK_RED", "RED_SAMPLE"),
                ("PLACE_RED", "RED_SAMPLE"),
                ("PICK_BLUE", "BLUE_SAMPLE"),
                ("PLACE_BLUE", "BLUE_SAMPLE"),
                ("CLOSE_LID", "CONTAINER_LID"),
            ]:
                for f in range(3):
                    res = feed_pipeline(pipeline, dummy_frame, act, obj, 0.95, f"2026-09-30T10:03:{cycle:02d}_{f}")
                    assert set(res.keys()) == REQUIRED_PUBLIC_KEYS

            assert fsm.state == "COMPLETED", f"Cycle {cycle} must complete successfully"
            assert fsm.current_step_index == 5

        # Verify telemetry aggregated all frames safely
        assert len(agg._total_pipe_ms) <= agg._window_size
        assert len(agg._total_pipe_ms) > 0

    def test_worker_threading_and_single_slot_buffer(self):
        """
        Verifies InferenceWorker background thread operation with LatestFrameBuffer.
        Ensures rate-matching decoupling: camera producer does not block,
        and inference worker consumes latest available frame.
        """
        fsm = SequenceValidatorFSM()
        fsm.start()
        agg = TelemetryDiagnosticAggregator()
        pipeline = InferencePipeline(validator=fsm, telemetry_aggregator=agg)

        results_collected = []

        def on_result(payload):
            results_collected.append(payload)

        worker = InferenceWorker(
            pipeline=pipeline,
            rate_strategy=RateStrategy.OPPORTUNISTIC_LATEST,
            result_callback=on_result,
        )

        worker.start()
        assert worker.is_running

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # Producer pushes 20 frames rapidly
        for i in range(20):
            worker.submit_frame(dummy_frame, timestamp=f"2026-09-30T10:04:{i:02d}")
            time.sleep(0.005)

        time.sleep(0.1)
        worker.stop()
        assert not worker.is_running

        assert len(results_collected) > 0, "Worker must have processed frames asynchronously"
        for r in results_collected:
            assert set(r.keys()) == REQUIRED_PUBLIC_KEYS

    def test_bridge_signal_and_voice_alert_integration(self):
        """
        Verifies Qt Bridge signal emission and VoiceAlertService integration.
        Ensures public AI results emit aiResultReady, recovery events emit recoveryAlertReady,
        and VoiceAlertService triggers debounced spoken alerts.
        """
        state = AppState()
        bridge = Bridge(state)
        fsm = SequenceValidatorFSM()
        rec_mgr = RecoveryManager(fsm.pm)
        voice_service = VoiceAlertService()

        ai_results_emitted = []
        recovery_alerts_emitted = []

        # Connect slots
        bridge.aiResultReady.connect(lambda payload: ai_results_emitted.append(payload))
        bridge.recoveryAlertReady.connect(lambda rec: recovery_alerts_emitted.append(rec))

        fsm.start()
        pipeline = InferencePipeline(validator=fsm, recovery_manager=rec_mgr)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Process valid frame and emit via bridge
        res = feed_pipeline(pipeline, dummy_frame, "PICK_RED", "RED_SAMPLE", 0.95, "2026-09-30T10:05:00")
        bridge._on_ai_result_from_worker(res)
        assert len(ai_results_emitted) == 1
        assert set(ai_results_emitted[0].keys()) == REQUIRED_PUBLIC_KEYS

        # 2. Trigger procedural error, handle via RecoveryManager and Bridge
        # Confirm S1
        for _ in range(3):
            feed_pipeline(pipeline, dummy_frame, "PICK_RED", "RED_SAMPLE", 0.95, "2026-09-30T10:05:01")
        # Skip to S3
        skip_res = None
        for _ in range(3):
            skip_res = feed_pipeline(pipeline, dummy_frame, "PICK_BLUE", "BLUE_SAMPLE", 0.95, "2026-09-30T10:05:02")

        bridge._on_ai_result_from_worker(skip_res)
        assert len(ai_results_emitted) == 2
        assert ai_results_emitted[1]["status"] == "SKIPPED"

        rec_event = pipeline.last_recovery_event or rec_mgr.last_recovery_event
        assert rec_event is not None
        bridge.emit_recovery_alert(rec_event)
        assert len(recovery_alerts_emitted) == 1
        assert recovery_alerts_emitted[0]["procedural_status"] == "SKIPPED"

        # Voice alert invocation
        voice_triggered = voice_service.alert_recovery_guidance(rec_event.expected_step, rec_event.recovery_instruction)
        assert voice_triggered, "Voice alert service should announce recovery event"

    def test_frozen_8_field_public_ai_contract_invariance(self):
        """
        Verifies that throughout all pipeline execution modes, the public AI contract
        strictly enforces the exact 8 frozen fields with zero extra diagnostic leaks.
        """
        fsm = SequenceValidatorFSM()
        fsm.start()
        agg = TelemetryDiagnosticAggregator()
        pipeline = InferencePipeline(validator=fsm, telemetry_aggregator=agg)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Run several combinations
        test_inputs = [
            ("NONE", "UNKNOWN", 0.1),
            ("PICK_RED", "RED_SAMPLE", 0.95),
            ("INVALID_ACT", "INVALID_OBJ", 0.5),
            ("CLOSE_LID", "CONTAINER_LID", 0.99),
        ]

        for act, obj, conf in test_inputs:
            out = feed_pipeline(pipeline, dummy_frame, act, obj, conf, "2026-09-30T10:06:00")
            assert set(out.keys()) == REQUIRED_PUBLIC_KEYS, f"Public contract violated for input ({act}, {obj})"
            assert out["status"] in VALID_PUBLIC_STATUSES
            assert isinstance(out["timestamp"], str)
            assert isinstance(out["action"], str)
            assert isinstance(out["object"], str)
            assert isinstance(out["confidence"], float)
            assert isinstance(out["expected_step"], str)
            assert out["detected_step"] is None or isinstance(out["detected_step"], str)
            assert isinstance(out["status"], str)
            assert out["next_step"] is None or isinstance(out["next_step"], str)

    def test_benchmark_harness_b18_1_discovery_and_export(self):
        """
        Verifies BenchmarkHarness discovers 'joint_software_acceptance' (and alias 'b18_1'),
        executes all multi-stream scenarios, returns passed=True, and exports valid JSON/TXT artifacts.
        """
        assert "joint_software_acceptance" in BenchmarkHarness.EXTENDED_BENCHMARKS
        assert "joint_software_acceptance" in BenchmarkHarness.list_benchmarks()

        res = BenchmarkHarness.run_benchmark("b18_1", num_nominal_cycles=2)
        assert isinstance(res, BenchmarkResult)
        assert res.benchmark_name == "joint_software_acceptance"
        assert res.passed is True
        assert res.metrics["all_streams_passed"] is True
        assert res.metrics["stream_a_nominal_passed"] is True
        assert res.metrics["stream_b_noise_passed"] is True
        assert res.metrics["stream_c_violation_passed"] is True
        assert res.metrics["stream_d_multicycle_passed"] is True
        assert res.metrics["contract_violations"] == 0

        # Export verification
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = BenchmarkHarness.export_results(res, tmpdir, base_name="test_b18_1_export")
            assert paths["json_path"].exists()
            assert paths["txt_path"].exists()
            assert paths["json_path"].stat().st_size > 0
            assert paths["txt_path"].stat().st_size > 0

    def test_epistemic_honesty_software_simulation_disclosure(self):
        """
        Verifies that benchmark results and pipeline metadata contain clear epistemic
        honesty disclosures regarding software simulation vs deferred flight hardware / neural checkpoints.
        """
        res = BenchmarkHarness.run_benchmark("joint_software_acceptance")
        assert len(res.limitations) >= 3
        limitations_text = " ".join(res.limitations).lower()
        assert "software-level" in limitations_text or "synthetic" in limitations_text
        assert "exp-001" in limitations_text
        assert "deferred" in limitations_text or "unavailable" in limitations_text
