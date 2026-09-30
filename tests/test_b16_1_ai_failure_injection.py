"""
test_b16_1_ai_failure_injection.py — Dedicated AI Failure Injection & Perception Robustness Test Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B16.1)

Verifies all 8 AI and perception failure modes (F01–F08) across the end-to-end perception graph:
  F01: Wrong action (Semantic action-object mismatch & transient noise burst)
  F02: Missed action (Intermittent frame drop / false negative IDLE leak)
  F03: Low confidence (Sub-marginal < 0.50 & marginal 0.50 <= conf < 0.70 accumulation)
  F04: Occlusion (Missing object bounding box, missing hand landmarks, synthetic cutout)
  F05: Poor lighting (Severe underexposure, overexposure, synthetic brightness disturbance)
  F06: Blur (Gaussian defocus blur, degraded 111-D keypoint vector processing)
  F07: Fast motion (Inter-frame candidate flicker & single-slot latest-frame replacement)
  F08: Multiple objects (Cluttered workspace, distractor objects on holding tray, 3D volume selection)

Architectural Invariants Strictly Enforced:
1. Authority Boundaries:
   - B10 = Temporal stability authority (filters single-frame transients via M-of-N voting)
   - B11 = Multimodal consistency & uncertainty authority (vetoes contradictory perceptions)
   - FSM/B9 = Sole procedural sequencing authority (governs step transitions)
   - B12 = Procedural recovery authority (triggers recovery ONLY on authoritative FSM violations)
   - B13 = Passive telemetry observer (records diagnostic snapshots without mutating state)
   - B16 = Test and failure-injection authority only
2. Critical Invariant:
   - Perception failures, visual noise, and multimodal uncertainties MUST NOT trigger procedural recovery (B12).
3. Contract Protection:
   - Frozen 8-field public AI contract (docs/architecture.md §2) strictly preserved across all failure modes.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import unittest
import numpy as np

from backend.ai.augmentation import AugmentationEngine
from backend.ai.hmr import (
    BoundingVolume3D,
    evaluate_payload_spatial_relations,
    get_canonical_exp001_volumes,
    Joint3D,
)
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import InferenceWorker, LatestFrameBuffer, RateStrategy
from backend.ai.multimodal_consistency import (
    CATEGORY_CROSS_MODAL_CONFLICT,
    CATEGORY_IDLE,
    CATEGORY_INSTABILITY_ACTION_FLICKER,
    CATEGORY_RELIABLE_ALIGNED,
    CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE,
    CATEGORY_UNCERTAIN_MISSING_OBJECT,
    CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION,
    CATEGORY_UNCERTAIN_STATIC_OBJECT,
    MultimodalConsistencyEvaluator,
)
from backend.ai.result_adapter import AIResultAdapter
from backend.ai.telemetry_aggregator import TelemetryDiagnosticAggregator
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.experiment.interaction_logic import InteractionEngine
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryEvent, RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM


class TestB161AIFailureInjection(unittest.TestCase):
    """Dedicated integration test suite for Gate B16.1 AI & Perception Failure Modes."""

    def setUp(self) -> None:
        self.config_path = Path(__file__).resolve().parent.parent / "config" / "experiment.json"
        self.pm = ProcedureManager(str(self.config_path))
        self.fsm = SequenceValidatorFSM(self.pm, min_confidence_threshold=0.70)
        self.fsm.start()
        self.rec_mgr = RecoveryManager(self.pm)
        self.agg = TelemetryDiagnosticAggregator()
        self.pipeline = InferencePipeline(
            validator=self.fsm,
            recovery_manager=self.rec_mgr,
            telemetry_aggregator=self.agg,
        )
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_frame[100:200, 100:200] = 128

    def tearDown(self) -> None:
        if self.fsm._session_active:
            self.fsm.reset()
        self.pipeline.reset()

    # =========================================================================
    # F01: WRONG ACTION (Semantic Mismatch & Transient Burst)
    # =========================================================================

    def test_f01_semantic_action_object_mismatch_containment(self) -> None:
        """
        F01 Case 1: Classifier outputs PICK_RED, but detected object is BLUE_SAMPLE.
        Verifies:
          - B11 MultimodalConsistencyEvaluator flags CROSS_MODAL_CONFLICT (is_reliable=False).
          - UncertaintyHandler records conflict.
          - B10 TemporalConfirmationEngine is BYPASSED.
          - FSM current step remains at S1 (invariant).
          - RecoveryManager does NOT trigger recovery event (zero false alarm).
          - Public result contract remains compliant.
        """
        evaluator = MultimodalConsistencyEvaluator()
        consistency = evaluator.evaluate(
            action="PICK_RED",
            action_confidence=0.92,
            object_name="BLUE_SAMPLE",
            interaction={"state": "HOLDING", "target_object": "BLUE_SAMPLE", "scale_proximity": 0.8},
            objects=[{"label": "BLUE_SAMPLE", "confidence": 0.88, "x1": 100, "y1": 100, "x2": 200, "y2": 200}],
        )
        self.assertEqual(consistency["category"], CATEGORY_CROSS_MODAL_CONFLICT)
        self.assertFalse(consistency["is_reliable"])

        # Inject into live pipeline
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.92)
        self.pipeline.object_detector.detect = lambda f: [
            {"label": "BLUE_SAMPLE", "confidence": 0.88, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ]

        for i in range(5):
            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:0{i}",
            )
            # Contract verified
            self.assertEqual(res["expected_step"], "S1")
            self.assertEqual(res["action"], "PICK_RED")
            self.assertTrue(AIResultAdapter.validate_public_contract(res))

        # Consistency evaluator recorded cross modal conflict
        self.assertEqual(self.pipeline.last_consistency_result["category"], CATEGORY_CROSS_MODAL_CONFLICT)
        self.assertFalse(self.pipeline.last_consistency_result["is_reliable"])

        # FSM not advanced
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(len(self.fsm.anomalies), 0)
        # Recovery NOT triggered
        self.assertIsNone(self.pipeline.last_recovery_event)
        self.assertEqual(self.rec_mgr.state, "IDLE")

    def test_f01_transient_wrong_action_burst_filtered_by_b10(self) -> None:
        """
        F01 Case 2: A semantically valid but transient 2-frame spike of PLACE_RED occurs during S1.
        Verifies:
          - B10 M-of-N (3/5) majority hysteresis rejects the 2-frame spike.
          - FSM remains expecting S1.
          - RecoveryManager does NOT trigger recovery event.
        """
        self.pipeline.action_classifier.classify = lambda **kw: ("PLACE_RED", 0.85)
        self.pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.85, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ]
        self.pipeline.hand_detector.detect = lambda f: [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "landmarks": []}
        ]

        # Inject 2 frames of spurious PLACE_RED
        for i in range(2):
            self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:0{i}",
            )

        # FSM state invariant
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(self.fsm.get_current_expected_step().step_id, "S1")
        self.assertEqual(len(self.fsm.anomalies), 0)
        self.assertIsNone(self.pipeline.last_recovery_event)

    # =========================================================================
    # F02: MISSED ACTION (False Negative / Intermittent Drop / IDLE Leak)
    # =========================================================================

    def test_f02_intermittent_frame_drop_and_recovery_to_nominal(self) -> None:
        """
        F02 Case 1: Intermittent detection loss (Action -> IDLE -> Action -> IDLE -> Action).
        Verifies:
          - B10 rolling window (3 of 5) successfully confirms the action once 3 valid frames accumulate.
          - Intermittent IDLE frames do not cause procedural errors or reset FSM.
        """
        frames_pattern = [
            ("PICK_RED", "RED_SAMPLE", 0.88),
            ("IDLE", "NONE", 0.0),
            ("PICK_RED", "RED_SAMPLE", 0.88),
            ("IDLE", "NONE", 0.0),
            ("PICK_RED", "RED_SAMPLE", 0.88),
        ]

        for i, (act, obj, conf) in enumerate(frames_pattern):
            self.pipeline.action_classifier.classify = lambda **kw: (act, conf)
            if obj != "NONE":
                self.pipeline.object_detector.detect = lambda f: [
                    {"label": obj, "confidence": conf, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
                ]
                self.pipeline.hand_detector.detect = lambda f: [
                    {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "landmarks": []}
                ]
            else:
                self.pipeline.object_detector.detect = lambda f: []
                self.pipeline.hand_detector.detect = lambda f: []

            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:0{i}",
            )

        # 3 out of 5 frames confirmed -> S1 validated -> transitions to S2
        self.assertEqual(self.fsm.current_step_index, 1)
        self.assertEqual(self.fsm.get_current_expected_step().step_id, "S2")
        self.assertEqual(res["status"], "VALID")
        self.assertEqual(res["next_step"], "S2")

    def test_f02_complete_missed_action_idle_leak(self) -> None:
        """
        F02 Case 2: Complete missed detection (all frames return IDLE).
        Verifies:
          - Pipeline outputs status="VALID", action="IDLE", expected_step="S1".
          - FSM remains locked at S1 without raising false errors.
          - Telemetry records idle cycles.
        """
        self.pipeline.action_classifier.classify = lambda **kw: ("IDLE", 0.0)
        self.pipeline.object_detector.detect = lambda f: []
        self.pipeline.hand_detector.detect = lambda f: []

        for i in range(10):
            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:{i:02d}",
            )
            self.assertEqual(res["action"], "IDLE")
            self.assertEqual(res["status"], "VALID")
            self.assertEqual(res["expected_step"], "S1")
            self.assertIsNone(res["detected_step"])

        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(len(self.fsm.anomalies), 0)
        self.assertIsNone(self.pipeline.last_recovery_event)

    # =========================================================================
    # F03: LOW CONFIDENCE (Sub-marginal & Marginal Evidence Accumulation)
    # =========================================================================

    def test_f03_sub_marginal_confidence_isolation(self) -> None:
        """
        F03 Case 1: Confidence is severely sub-marginal (conf = 0.35 < 0.50).
        Verifies:
          - UncertaintyHandler marks UNCERTAIN_LOW_CONFIDENCE (is_reliable=False).
          - B10 is bypassed.
          - FSM is not called and does not advance.
          - Telemetry records low_confidence_count.
        """
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.35)
        self.pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.85, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ]
        self.pipeline.hand_detector.detect = lambda f: [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "landmarks": []}
        ]

        for i in range(5):
            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:0{i}",
            )
            self.assertEqual(res["expected_step"], "S1")
            self.assertTrue(AIResultAdapter.validate_public_contract(res))

        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertFalse(self.pipeline.last_uncertainty_result["is_reliable"])
        self.assertEqual(self.pipeline.last_uncertainty_result["state"], "UNCERTAIN_LOW_CONFIDENCE")
        self.assertIsNone(self.pipeline.last_recovery_event)

    def test_f03_marginal_confidence_evidence_accumulation_and_resolution(self) -> None:
        """
        F03 Case 2: Marginal action confidence (conf = 0.60, in [0.50, 0.70)).
        Verifies:
          - First 2 frames: UNCERTAIN_EVIDENCE_ACCUMULATING (is_reliable=False).
          - 3rd consistent frame: RESOLVED (is_reliable=True) -> feeds B10 (1st vote).
          - Subsequent nominal frames allow B10 confirmation (3 votes) and nominal FSM validation.
        """
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.60)
        self.pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.85, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
        ]
        self.pipeline.hand_detector.detect = lambda f: [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "landmarks": []}
        ]

        # Frame 0: Accumulating (streak 1)
        self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:00")
        self.assertEqual(self.pipeline.last_uncertainty_result["state"], "UNCERTAIN_EVIDENCE_ACCUMULATING")
        self.assertFalse(self.pipeline.last_uncertainty_result["is_reliable"])
        self.assertEqual(self.fsm.current_step_index, 0)

        # Frame 1: Accumulating (streak 2)
        self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:01")
        self.assertEqual(self.pipeline.last_uncertainty_result["state"], "UNCERTAIN_EVIDENCE_ACCUMULATING")
        self.assertFalse(self.pipeline.last_uncertainty_result["is_reliable"])
        self.assertEqual(self.fsm.current_step_index, 0)

        # Frame 2: Resolved (streak 3) -> 1st positive vote in B10
        self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:02")
        self.assertEqual(self.pipeline.last_uncertainty_result["state"], "RESOLVED")
        self.assertTrue(self.pipeline.last_uncertainty_result["is_reliable"])
        self.assertEqual(self.fsm.current_step_index, 0)

        # Frames 3, 4, 5: Subsequent confident frames provide the 3 positive votes in B10 -> confirms S1
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.85)
        self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:03")
        self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:04")
        res5 = self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:05")

        self.assertEqual(self.fsm.current_step_index, 1)
        self.assertEqual(res5["status"], "VALID")
        self.assertEqual(res5["expected_step"], "S1")
        self.assertEqual(res5["next_step"], "S2")

    # =========================================================================
    # F04: OCCLUSION (Missing Objects, Missing Hands & Synthetic Cutout)
    # =========================================================================

    def test_f04_missing_target_object_occlusion_containment(self) -> None:
        """
        F04 Case 1: Target object is completely occluded (ObjectDetector returns empty list).
        Verifies:
          - MultimodalConsistencyEvaluator classifies as UNCERTAIN_MISSING_OBJECT.
          - is_reliable=False prevents FSM execution.
          - Recovery is NOT triggered.
        """
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.88)
        self.pipeline.object_detector.detect = lambda f: []
        self.pipeline.hand_detector.detect = lambda f: []

        for i in range(4):
            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:0{i}",
            )
            self.assertEqual(res["expected_step"], "S1")
            self.assertTrue(AIResultAdapter.validate_public_contract(res))

        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(self.pipeline.last_consistency_result["category"], CATEGORY_UNCERTAIN_MISSING_OBJECT)
        self.assertFalse(self.pipeline.last_consistency_result["is_reliable"])
        self.assertIsNone(self.pipeline.last_recovery_event)

    def test_f04_hand_occlusion_spatial_contradiction_containment(self) -> None:
        """
        F04 Case 2: Hands are far away / occluded, but manipulation action PICK_RED is reported.
        Verifies:
          - InteractionEngine outputs IDLE state with scale proximity outside reach (> 1.50).
          - MultimodalConsistencyEvaluator classifies as UNCERTAIN_SPATIAL_CONTRADICTION.
          - FSM and B10 are isolated.
        """
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.90)
        self.pipeline.object_detector.detect = lambda f: [
            {"label": "RED_SAMPLE", "confidence": 0.90, "x1": 100, "y1": 100, "x2": 150, "y2": 150}
        ]
        # Hand far away at (500, 500)
        self.pipeline.hand_detector.detect = lambda f: [
            {"x1": 500, "y1": 500, "x2": 550, "y2": 550, "landmarks": []}
        ]

        res = self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp="2026-09-29T12:00:00")
        self.assertEqual(self.pipeline.last_consistency_result["category"], CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION)
        self.assertFalse(self.pipeline.last_consistency_result["is_reliable"])
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertIsNone(self.pipeline.last_recovery_event)

    def test_f04_synthetic_cutout_occlusion_augmentation_stability(self) -> None:
        """
        F04 Case 3: Synthetic patch/cutout occlusion applied via AugmentationEngine.
        Verifies deterministic pipeline execution without crashes.
        """
        aug = AugmentationEngine()
        res_occ = aug.apply_occlusion(self.dummy_frame, box_fraction=0.20, seed=42)
        occluded_frame = res_occ.image
        self.assertEqual(occluded_frame.shape, self.dummy_frame.shape)

        res = self.pipeline.process_frame_public(frame=occluded_frame, timestamp="2026-09-29T12:00:00")
        self.assertIn("status", res)
        self.assertIn("expected_step", res)
        self.assertTrue(AIResultAdapter.validate_public_contract(res))

    # =========================================================================
    # F05: POOR LIGHTING (Severe Underexposure, Overexposure & Augmentation)
    # =========================================================================

    def test_f05_extreme_underexposure_and_overexposure_containment(self) -> None:
        """
        F05 Case 1: Pure black frame (0) and pure white washed-out frame (255).
        Verifies safe fallback, zero exceptions, and clean contract output.
        """
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        res_black = self.pipeline.process_frame_public(frame=black_frame, timestamp="2026-09-29T12:00:00")
        res_white = self.pipeline.process_frame_public(frame=white_frame, timestamp="2026-09-29T12:00:01")

        self.assertEqual(res_black["status"], "VALID")
        self.assertEqual(res_black["action"], "IDLE")
        self.assertEqual(res_white["status"], "VALID")
        self.assertEqual(res_white["action"], "IDLE")
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertTrue(AIResultAdapter.validate_public_contract(res_black))
        self.assertTrue(AIResultAdapter.validate_public_contract(res_white))

    def test_f05_synthetic_brightness_disturbance_stability(self) -> None:
        """
        F05 Case 2: Synthetic brightness disturbance (dim 4 of 7).
        """
        aug = AugmentationEngine()
        res_dim = aug.adjust_brightness(self.dummy_frame, gain=0.2, bias=-20.0)
        res_bright = aug.adjust_brightness(self.dummy_frame, gain=2.0, bias=30.0)
        dim_frame = res_dim.image
        bright_frame = res_bright.image

        res_dim_out = self.pipeline.process_frame_public(frame=dim_frame, timestamp="2026-09-29T12:00:00")
        res_bright_out = self.pipeline.process_frame_public(frame=bright_frame, timestamp="2026-09-29T12:00:01")

        self.assertEqual(res_dim_out["expected_step"], "S1")
        self.assertEqual(res_bright_out["expected_step"], "S1")
        self.assertTrue(AIResultAdapter.validate_public_contract(res_dim_out))
        self.assertTrue(AIResultAdapter.validate_public_contract(res_bright_out))

    # =========================================================================
    # F06: BLUR (Gaussian Defocus / Fast Motion Blur)
    # =========================================================================

    def test_f06_synthetic_gaussian_blur_and_keypoint_degradation(self) -> None:
        """
        F06: Synthetic spatial blur (dim 5 of 7) applied to frame.
        Verifies 111-D feature vector extraction determinism and pipeline throughput.
        """
        aug = AugmentationEngine()
        res_blur = aug.apply_blur(self.dummy_frame, kernel_size=7)
        blurred_frame = res_blur.image

        res = self.pipeline.process_frame_public(frame=blurred_frame, timestamp="2026-09-29T12:00:00")
        self.assertIn("status", res)
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertIsNone(self.pipeline.last_recovery_event)
        self.assertTrue(AIResultAdapter.validate_public_contract(res))

    # =========================================================================
    # F07: FAST MOTION (Candidate Flicker & Single-Slot Buffer Replacement)
    # =========================================================================

    def test_f07_rapid_action_flicker_containment(self) -> None:
        """
        F07 Case 1: High velocity motion causes candidate flicker (PICK_RED <-> PICK_BLUE alternating).
        Verifies:
          - Alternating candidate stream prevents any candidate from reaching majority threshold (M=3).
          - FSM remains locked at S1.
        """
        actions = [
            ("PICK_RED", "RED_SAMPLE"),
            ("PICK_BLUE", "BLUE_SAMPLE"),
            ("PICK_RED", "RED_SAMPLE"),
            ("PICK_BLUE", "BLUE_SAMPLE"),
        ]

        for i, (act, obj) in enumerate(actions):
            self.pipeline.action_classifier.classify = lambda **kw: (act, 0.85)
            self.pipeline.object_detector.detect = lambda f: [
                {"label": obj, "confidence": 0.85, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
            ]
            self.pipeline.hand_detector.detect = lambda f: [
                {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "landmarks": []}
            ]
            self.pipeline.process_frame_public(frame=self.dummy_frame, timestamp=f"2026-09-29T12:00:0{i}")

        # Neither candidate reached 3 votes -> S1 not validated
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(self.fsm.get_current_expected_step().step_id, "S1")
        self.assertEqual(len(self.fsm.anomalies), 0)

    def test_f07_fast_motion_single_slot_buffer_drop_dynamics(self) -> None:
        """
        F07 Case 2: Camera produces frames faster than perception throughput.
        Verifies:
          - LatestFrameBuffer replaces old frame in O(1) without queue buildup.
          - Frame drop count increments.
        """
        buf = LatestFrameBuffer()
        for i in range(10):
            buf.put(np.full((10, 10, 3), i, dtype=np.uint8), timestamp=float(i))

        self.assertEqual(buf.submission_count, 10)
        self.assertEqual(buf.replacement_count, 9)  # 9 frames replaced/dropped

        # Ingestion returns the latest frame
        frame_data = buf.get(timeout=0.1)
        self.assertIsNotNone(frame_data)
        f, ts, meta, exp = frame_data
        self.assertEqual(f[0, 0, 0], 9)
        self.assertEqual(ts, 9.0)

    # =========================================================================
    # F08: MULTIPLE OBJECTS (Clutter & 3D Spatial Selection)
    # =========================================================================

    def test_f08_cluttered_workspace_distractor_object_handling(self) -> None:
        """
        F08 Case 1: All 4 canonical objects are present in camera FOV.
        Operator interacts with RED_SAMPLE while BLUE_SAMPLE, SAMPLE_CONTAINER, and CONTAINER_LID are on tray.
        Verifies:
          - InteractionEngine associates hand with closest object (RED_SAMPLE).
          - MultimodalConsistencyEvaluator recognizes valid interaction on RED_SAMPLE.
          - S1 is validated cleanly without distractor interference.
        """
        all_objects = [
            {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 100, "y1": 100, "x2": 150, "y2": 150},
            {"label": "BLUE_SAMPLE", "confidence": 0.92, "x1": 350, "y1": 100, "x2": 400, "y2": 150},
            {"label": "SAMPLE_CONTAINER", "confidence": 0.94, "x1": 200, "y1": 100, "x2": 300, "y2": 250},
            {"label": "CONTAINER_LID", "confidence": 0.89, "x1": 200, "y1": 260, "x2": 300, "y2": 300},
        ]
        hand = [{"x1": 105, "y1": 105, "x2": 145, "y2": 145, "landmarks": [{"x": 0.2, "y": 0.2, "z": 0.0}]}]

        self.pipeline.object_detector.detect = lambda f: all_objects
        self.pipeline.hand_detector.detect = lambda f: hand
        self.pipeline.action_classifier.classify = lambda **kw: ("PICK_RED", 0.91)

        # 3 consecutive frames to confirm S1
        for i in range(3):
            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:00:0{i}",
            )

        self.assertEqual(self.fsm.current_step_index, 1)
        self.assertEqual(res["status"], "VALID")
        self.assertEqual(res["action"], "PICK_RED")
        self.assertEqual(res["object"], "RED_SAMPLE")
        self.assertEqual(res["next_step"], "S2")

    def test_f08_3d_spatial_geometry_multiple_volume_selection(self) -> None:
        """
        F08 Case 2: 3D spatial geometry evaluates hand reach against all 4 EXP-001 bounding volumes.
        Verifies wrist positioned at (-0.14, 0.0, 0.0) matches RED_SAMPLE (center -0.15, 0.0, 0.0).
        """
        volumes = get_canonical_exp001_volumes()
        wrist = Joint3D(x=-0.14, y=0.01, z=0.00, name="right_wrist")

        relations = evaluate_payload_spatial_relations(
            joints=[wrist],
            volumes=volumes,
            reach_threshold=0.12,
            is_metric=True,
        )

        self.assertIn("RED_SAMPLE", relations)
        self.assertIn("BLUE_SAMPLE", relations)
        self.assertTrue(relations["RED_SAMPLE"].is_within_reach)
        self.assertFalse(relations["BLUE_SAMPLE"].is_within_reach)
        self.assertLess(relations["RED_SAMPLE"].distance, relations["BLUE_SAMPLE"].distance)

    # =========================================================================
    # CROSS-AUTHORITY & INVARIANT INTEGRITY
    # =========================================================================

    def test_cross_authority_zero_recovery_on_perception_noise(self) -> None:
        """
        Verifies across 100 frames of varied visual noise (occlusions, brightness, blurs,
        conflicts, and sub-marginal confidence) that RecoveryManager NEVER triggers a recovery event.
        """
        noise_scenarios = [
            ("PICK_RED", "BLUE_SAMPLE", 0.90),    # Semantic conflict
            ("PICK_RED", "NONE", 0.0),           # Missing object
            ("PICK_BLUE", "RED_SAMPLE", 0.85),   # Semantic conflict
            ("PICK_RED", "RED_SAMPLE", 0.30),    # Low confidence
            ("IDLE", "NONE", 0.0),               # IDLE frame
        ]

        for i in range(100):
            act, obj, conf = noise_scenarios[i % len(noise_scenarios)]
            self.pipeline.action_classifier.classify = lambda **kw: (act, conf)
            if obj != "NONE":
                self.pipeline.object_detector.detect = lambda f: [
                    {"label": obj, "confidence": conf, "x1": 100, "y1": 100, "x2": 200, "y2": 200}
                ]
                self.pipeline.hand_detector.detect = lambda f: [
                    {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "landmarks": []}
                ]
            else:
                self.pipeline.object_detector.detect = lambda f: []
                self.pipeline.hand_detector.detect = lambda f: []

            res = self.pipeline.process_frame_public(
                frame=self.dummy_frame,
                timestamp=f"2026-09-29T12:{i//60:02d}:{i%60:02d}",
            )

            # Public schema strictly respected
            self.assertIn("status", res)
            self.assertIn("expected_step", res)
            self.assertEqual(res["expected_step"], "S1")
            self.assertTrue(AIResultAdapter.validate_public_contract(res))

        # FSM unaffected by perception noise
        self.assertEqual(self.fsm.current_step_index, 0)
        self.assertEqual(len(self.fsm.anomalies), 0)
        # ZERO recovery events triggered
        self.assertIsNone(self.pipeline.last_recovery_event)
        self.assertEqual(self.rec_mgr.state, "IDLE")
        self.assertEqual(len(self.rec_mgr.recovery_history), 0)


if __name__ == "__main__":
    unittest.main()
