"""
inference_pipeline.py — Master Perception Pipeline Coordinator
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B6.1)

Coordinates the end-to-end edge inference sequence for each incoming video frame:
  Frame ➔ Optional Camera Rectification Preprocessing Hook (Gate B4.1c / B6.1)
        ➔ Objects (Canonical EXP-001) + 3D Pose Mesh + Hand Tracking
        ➔ Interaction Logic (Rotation-Invariant Distance / Proximity)
        ➔ Keypoint Vector Flattening (111-D Contract)
        ➔ 1D-TCN Action Recognition (Canonical EXP-001 Actions)
        ➔ Annotation Rendering & Output Payload
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .object_detector import ObjectDetector
from .pose_detector import PoseDetector
from .hand_detector import HandDetector
from .action_classifier import ActionClassifier
from .hailo_inference import HailoInferenceEngine
from .result_adapter import AIResultAdapter
from .temporal_filter import TemporalConfirmationEngine
from .uncertainty_handler import UncertaintyHandler
from .multimodal_consistency import MultimodalConsistencyEvaluator
from ..video.frame_processor import FrameProcessor
from ..video.camera_rectification import CameraRectifier, RectificationConfig
from ..experiment.interaction_logic import InteractionEngine
from ..experiment.recovery_manager import RecoveryEvent, RecoveryManager


_DEFAULT_EVALUATOR = object()


class InferencePipeline:
    """
    Unified multimodal AI inference coordinator.
    """

    def __init__(
        self,
        hef_path: Optional[str | Path] = None,
        tflite_path: Optional[str | Path] = None,
        detection_threshold: float = 0.50,
        pose_threshold: float = 0.50,
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
        validator: Optional[Any] = None,
        temporal_filter: Optional[TemporalConfirmationEngine] = None,
        temporal_window_size: int = 5,
        temporal_threshold: int = 3,
        min_confidence: float = 0.70,
        cooldown_frames: int = 5,
        uncertainty_handler: Optional[UncertaintyHandler] = None,
        consistency_evaluator: Optional[Any] = _DEFAULT_EVALUATOR,
        recovery_manager: Optional[RecoveryManager] = None,
        telemetry_aggregator: Optional[Any] = None,
    ):
        self.frame_processor = FrameProcessor()
        self.rectifier = rectifier
        self.validator = validator
        self.recovery_manager = recovery_manager
        self.telemetry_aggregator = telemetry_aggregator
        self.object_detector = ObjectDetector(confidence_threshold=detection_threshold)
        self.pose_detector = PoseDetector(confidence_threshold=pose_threshold)
        self.hand_detector = HandDetector(confidence_threshold=pose_threshold)
        self.interaction_engine = InteractionEngine()
        self.action_classifier = ActionClassifier(model_path=tflite_path)
        self.hailo_engine = HailoInferenceEngine(hef_path=hef_path)
        self.temporal_filter = temporal_filter if temporal_filter is not None else TemporalConfirmationEngine(
            window_size=temporal_window_size,
            confirmation_threshold=temporal_threshold,
            min_confidence=min_confidence,
            cooldown_frames=cooldown_frames,
        )
        self.uncertainty_handler = uncertainty_handler if uncertainty_handler is not None else UncertaintyHandler(
            high_confidence_threshold=min_confidence,
            marginal_confidence_threshold=0.50,
        )
        if consistency_evaluator is _DEFAULT_EVALUATOR:
            self.consistency_evaluator = MultimodalConsistencyEvaluator(
                action_confidence_threshold=min_confidence,
                object_confidence_threshold=detection_threshold,
            )
        else:
            self.consistency_evaluator = consistency_evaluator

        self._frame_index = 0
        self._last_raw_action: Optional[str] = None
        self._last_raw_object: Optional[str] = None
        self._last_consistency_result: Optional[Dict[str, Any]] = None
        self._last_uncertainty_result: Optional[Dict[str, Any]] = None
        self._last_recovery_event: Optional[RecoveryEvent] = None

    @property
    def last_consistency_result(self) -> Optional[Dict[str, Any]]:
        """Latest internal multimodal consistency evaluation dictionary."""
        return self._last_consistency_result

    @property
    def last_uncertainty_result(self) -> Optional[Dict[str, Any]]:
        """Latest internal uncertainty handling evaluation dictionary."""
        return self._last_uncertainty_result

    @property
    def last_recovery_event(self) -> Optional[RecoveryEvent]:
        """Latest internal procedural recovery event emitted during validation."""
        return self._last_recovery_event

    def process_frame(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
        timestamp: Optional[Union[float, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete visual and temporal perception graph on one frame.
        """
        if frame is None or frame.size == 0:
            return {
                "frame_index": self._frame_index,
                "timestamp": timestamp,
                "metadata": metadata or {},
                "rectification_applied": False,
                "objects": [],
                "pose": [],
                "hands": [],
                "interaction": {"state": "IDLE"},
                "action": "IDLE",
                "confidence": 0.0,
                "annotated_frame": None,
            }

        self._frame_index += 1

        t_rect_start = time.perf_counter()
        # 0. Camera Mounting Calibration & Rectification Preprocessing Hook (Gate B4.1c.2 / B6.1)
        active_rectifier = rectifier if rectifier is not None else self.rectifier
        rect_applied = False
        proc_frame = frame
        if active_rectifier is not None:
            try:
                from ..video.frame_processor import _resolve_rectifier
                r_obj = _resolve_rectifier(active_rectifier)
                if r_obj is not None and not r_obj.is_identity():
                    proc_frame = r_obj.rectify_frame(frame)
                    rect_applied = True
            except Exception:
                proc_frame = frame
                rect_applied = False
        t_rect_end = time.perf_counter()
        rect_duration_ms = (t_rect_end - t_rect_start) * 1000.0

        t_det_start = time.perf_counter()
        # 1. Object Detection (Hailo NPU or YOLO/Fallback with Canonical Vocabulary)
        objects = self.object_detector.detect(proc_frame) or []

        # 2. 3D Body Pose Landmark Recovery
        pose_landmarks = self.pose_detector.detect(proc_frame) or []

        # 3. Fine-grained Hand Tracking
        hands = self.hand_detector.detect(proc_frame) or []

        # 4. Hand-Object Geometric Interaction Evaluation (Rotation-Invariant)
        hand_boxes = [
            {"x1": h["x1"], "y1": h["y1"], "x2": h["x2"], "y2": h["y2"]}
            for h in hands if "x1" in h and "y1" in h and "x2" in h and "y2" in h
        ]
        obj_boxes = [
            {"x1": o["x1"], "y1": o["y1"], "x2": o["x2"], "y2": o["y2"], "label": o.get("label", "OBJECT")}
            for o in objects if "x1" in o and "y1" in o and "x2" in o and "y2" in o
        ]
        hand_landmarks = [lm for h in hands for lm in h.get("landmarks", [])] if hands else None
        interaction = self.interaction_engine.evaluate_interaction(
            hand_boxes=hand_boxes,
            object_boxes=obj_boxes,
            hand_landmarks=hand_landmarks,
        )

        # 5. Extract 1D Keypoint Vector for Temporal Classifier (111-D contract)
        keypoint_vec = self.frame_processor.extract_keypoint_vector(pose_landmarks, objects)
        self.action_classifier.push_frame_vector(keypoint_vec)

        # 6. Action Recognition (Canonical EXP-001 actions with rich geometric context)
        detected_action, action_conf = self.action_classifier.classify(
            interaction_state=interaction.get("state"),
            target_object=interaction.get("target_object"),
            interaction=interaction,
            objects=objects,
            hands=hands,
        )

        # 7. Render Annotated HUD Frame on processed/rectified frame if requested
        annotated_frame = None
        if annotate:
            annotated_frame = self.frame_processor.draw_annotations(
                frame=proc_frame,
                detections=objects,
                pose_keypoints=pose_landmarks,
                action_text=detected_action,
                confidence=action_conf,
            )
        t_det_end = time.perf_counter()
        det_duration_ms = (t_det_end - t_det_start) * 1000.0

        return {
            "frame_index": self._frame_index,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "rectification_applied": rect_applied,
            "objects": objects,
            "pose": pose_landmarks,
            "hands": hands,
            "interaction": interaction,
            "action": detected_action,
            "confidence": action_conf,
            "annotated_frame": annotated_frame,
            "_stage_a_rectification_ms": rect_duration_ms,
            "_stage_b_detection_ms": det_duration_ms,
        }

    def process_frame_public(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
        timestamp: Optional[Union[float, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_step: Optional[Union[str, int]] = None,
        fsm_status: Optional[str] = None,
        next_step: Optional[Union[str, int]] = None,
        validator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Runs perception on a frame, performs multimodal consistency evaluation (B11.2),
        uncertainty handling (B11.1), temporal confirmation filtering (B10),
        submits only confirmed reliable actions to the authoritative SequenceValidatorFSM,
        and adapts the output to the frozen 8-field public AI contract (docs/architecture.md §2).
        """
        t_pub_start = time.perf_counter()

        # 1. Visual & Physical Perception (Stages A and B)
        internal_res = self.process_frame(
            frame=frame,
            annotate=annotate,
            rectifier=rectifier,
            timestamp=timestamp,
            metadata=metadata,
        )
        stage_a_ms = internal_res.get("_stage_a_rectification_ms", 0.0)
        stage_b_ms = internal_res.get("_stage_b_detection_ms", 0.0)

        detected_action = internal_res.get("action", "IDLE")
        detected_obj = (
            internal_res.get("object")
            or (internal_res.get("interaction") or {}).get("target_object")
            or (internal_res.get("objects", [{}])[0].get("label") if internal_res.get("objects") else None)
        )
        detected_conf = internal_res.get("confidence", 0.0)
        interaction = internal_res.get("interaction") or {}
        objects = internal_res.get("objects") or []
        ts = timestamp or internal_res.get("timestamp")

        # 2. Multimodal Consistency Evaluation (B11.2.1) (Stage C)
        t_c_start = time.perf_counter()
        consistency_res = None
        if self.consistency_evaluator is not None:
            consistency_res = self.consistency_evaluator.evaluate(
                action=detected_action,
                action_confidence=detected_conf,
                object_name=detected_obj,
                object_confidence=None,
                interaction=interaction,
                objects=objects,
            )
        self._last_consistency_result = consistency_res
        t_c_end = time.perf_counter()
        stage_c_ms = (t_c_end - t_c_start) * 1000.0

        # 3. Uncertainty Handling (B11.1) & Temporal Confirmation (B10) Routing (Stage D)
        t_d_start = time.perf_counter()
        uncertainty_res = None
        temporal_res = None
        is_reliable_observation = consistency_res.get("is_reliable", True) if consistency_res is not None else True

        if is_reliable_observation:
            # Observation is multimodally coherent
            if self.uncertainty_handler is not None:
                uncertainty_res = self.uncertainty_handler.process_observation(
                    action=detected_action,
                    object_name=detected_obj,
                    confidence=detected_conf,
                    timestamp=ts,
                )

            is_uncertainty_reliable = (
                uncertainty_res.get("is_reliable", True)
                if uncertainty_res is not None
                else True
            )

            if is_uncertainty_reliable:
                # Observation is both coherent and confident/resolved -> feed B10 temporal filter
                eval_conf = (
                    uncertainty_res.get("confidence", detected_conf)
                    if uncertainty_res is not None
                    else detected_conf
                )
                if self.temporal_filter is not None:
                    temporal_res = self.temporal_filter.process_observation(
                        action=detected_action,
                        object_name=detected_obj,
                        confidence=eval_conf,
                        timestamp=ts,
                    )
            else:
                # Observation is marginal and accumulating evidence -> do not record positive vote in B10
                temporal_res = None
        else:
            # Unreliable observation (cross-modal conflict, missing object, spatial veto, flicker, or idle)
            if self.uncertainty_handler is not None:
                uncertainty_res = self.uncertainty_handler.process_observation(
                    action=detected_action,
                    object_name=detected_obj,
                    confidence=detected_conf,
                    timestamp=ts,
                    is_conflict=(detected_action != "IDLE"),
                )
            if detected_action == "IDLE" and self.temporal_filter is not None:
                # Clear cooldown on neutral IDLE observation
                temporal_res = self.temporal_filter.process_observation(
                    action="IDLE",
                    object_name="NONE",
                    confidence=0.0,
                    timestamp=ts,
                )
            else:
                temporal_res = None

        self._last_uncertainty_result = uncertainty_res
        t_d_end = time.perf_counter()
        stage_d_ms = (t_d_end - t_d_start) * 1000.0

        # 4. Authoritative Procedural Validation (FSM) (Stage E)
        t_e_start = time.perf_counter()
        active_validator = validator if validator is not None else getattr(self, "validator", None)
        fsm_res = None
        self._last_recovery_event = None

        if active_validator is not None:
            # Sync procedure completion
            if getattr(active_validator, "state", None) == "COMPLETED":
                if self.temporal_filter is not None:
                    self.temporal_filter.complete()
                if self.uncertainty_handler is not None:
                    self.uncertainty_handler.complete()

            # Pass to FSM ONLY when temporally confirmed and perceptually reliable
            if is_reliable_observation:
                if self.temporal_filter is not None:
                    if temporal_res is not None and temporal_res.get("confirmed", False):
                        confirmed_payload = {
                            "action": temporal_res.get("action", detected_action),
                            "object": temporal_res.get("object", detected_obj),
                            "confidence": temporal_res.get("confidence", detected_conf),
                            "timestamp": ts,
                        }
                        fsm_res = active_validator.validate_ai_result(confirmed_payload)
                else:
                    # Direct passthrough fallback if temporal filter is explicitly None
                    fsm_res = active_validator.validate_ai_result(internal_res)

            # Evaluate recovery via RecoveryManager if present and FSM evaluated
            if self.recovery_manager is not None and fsm_res is not None:
                self._last_recovery_event = self.recovery_manager.evaluate_fsm_result(
                    fsm_res, procedure_manager=getattr(active_validator, "pm", None)
                )

        # 5. Resolve authoritative expected and next steps from FSM for unconfirmed states
        curr_exp_step = expected_step
        curr_next_step = next_step
        if active_validator is not None and fsm_res is None:
            if curr_exp_step is None:
                exp_obj = active_validator.get_current_expected_step()
                curr_exp_step = exp_obj.step_id if exp_obj else None
            if curr_next_step is None:
                next_obj = active_validator.get_next_expected_step()
                curr_next_step = next_obj.step_id if next_obj else None

        t_e_end = time.perf_counter()
        stage_e_ms = (t_e_end - t_e_start) * 1000.0

        # 7. Public Result Adapter (Frozen 8-field contract) (Stage F)
        t_f_start = time.perf_counter()
        adapted_pub_res = AIResultAdapter.adapt(
            internal_result=internal_res,
            fsm_result=fsm_res,
            expected_step=curr_exp_step,
            fsm_status=fsm_status,
            next_step=curr_next_step,
            timestamp=ts,
        )
        t_f_end = time.perf_counter()
        stage_f_ms = (t_f_end - t_f_start) * 1000.0
        total_pipe_ms = (t_f_end - t_pub_start) * 1000.0

        # 6. Passive Diagnostic Ingestion (Gate B13.1 / B17.1)
        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.record_stage_latencies(
                stage_a_rectification_ms=stage_a_ms,
                stage_b_detection_ms=stage_b_ms,
                stage_c_consistency_ms=stage_c_ms,
                stage_d_temporal_ms=stage_d_ms,
                stage_e_fsm_ms=stage_e_ms,
                stage_f_adapter_ms=stage_f_ms,
                total_pipeline_ms=total_pipe_ms,
            )
            self.telemetry_aggregator.record_cycle(
                consistency_result=consistency_res,
                uncertainty_result=uncertainty_res,
                temporal_result=temporal_res,
                fsm_result=fsm_res,
                recovery_event=self._last_recovery_event,
                validator=active_validator,
            )

        return adapted_pub_res

    def reset(self) -> None:
        """Reset temporal state, confirmation window, uncertainty handler, recovery manager, aggregator, and counters."""
        self._frame_index = 0
        self._last_raw_action = None
        self._last_raw_object = None
        self._last_consistency_result = None
        self._last_uncertainty_result = None
        self._last_recovery_event = None
        self.action_classifier.reset_buffer()
        self.interaction_engine.reset()
        if self.temporal_filter is not None:
            self.temporal_filter.reset()
        if self.uncertainty_handler is not None:
            self.uncertainty_handler.reset()
        if self.recovery_manager is not None:
            self.recovery_manager.reset()
        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.reset()

    def pause(self) -> None:
        """Pause temporal confirmation voting, uncertainty evaluation, recovery monitoring, and telemetry."""
        if self.temporal_filter is not None:
            self.temporal_filter.pause()
        if self.uncertainty_handler is not None:
            self.uncertainty_handler.pause()
        if self.recovery_manager is not None:
            self.recovery_manager.pause()
        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.pause_session()

    def resume(self) -> None:
        """Resume temporal confirmation voting, uncertainty evaluation, recovery monitoring, and telemetry."""
        self._last_raw_action = None
        self._last_raw_object = None
        if self.temporal_filter is not None:
            self.temporal_filter.resume()
        if self.uncertainty_handler is not None:
            self.uncertainty_handler.resume()
        if self.recovery_manager is not None:
            self.recovery_manager.resume()
        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.resume_session()

    def release(self) -> None:
        """Release underlying hardware and detector resources."""
        self.pose_detector.close()
        self.hand_detector.close()
        self.hailo_engine.release()


