"""
inference_pipeline.py — Master Perception Pipeline Coordinator
ISRO SIH26174 BAS Experiment Monitor

Coordinates the end-to-end edge inference sequence for each incoming video frame:
  Frame ➔ Hailo-8L / YOLOv8 Objects + 3D Pose Mesh + Hand Tracking
        ➔ Interaction Logic (IoU / Proximity)
        ➔ Keypoint Vector Flattening
        ➔ 1D-TCN Action Recognition
        ➔ Annotation Rendering & Output Payload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .object_detector import ObjectDetector
from .pose_detector import PoseDetector
from .hand_detector import HandDetector
from .action_classifier import ActionClassifier
from .hailo_inference import HailoInferenceEngine
from ..video.frame_processor import FrameProcessor
from ..experiment.interaction_logic import InteractionEngine


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
    ):
        self.frame_processor = FrameProcessor()
        self.object_detector = ObjectDetector(confidence_threshold=detection_threshold)
        self.pose_detector = PoseDetector(confidence_threshold=pose_threshold)
        self.hand_detector = HandDetector(confidence_threshold=pose_threshold)
        self.interaction_engine = InteractionEngine()
        self.action_classifier = ActionClassifier(model_path=tflite_path)
        self.hailo_engine = HailoInferenceEngine(hef_path=hef_path)

        self._frame_index = 0

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> Dict[str, Any]:
        """
        Runs the complete visual and temporal perception graph on one frame.
        """
        if frame is None or frame.size == 0:
            return {
                "objects": [],
                "pose": [],
                "hands": [],
                "interaction": {"state": "IDLE"},
                "action": "IDLE",
                "confidence": 0.0,
                "annotated_frame": None,
            }

        self._frame_index += 1

        # 1. Object Detection (Hailo NPU or YOLO/Fallback)
        objects = self.object_detector.detect(frame)

        # 2. 3D Body Pose Landmark Recovery
        pose_landmarks = self.pose_detector.detect(frame)

        # 3. Fine-grained Hand Tracking
        hands = self.hand_detector.detect(frame)

        # 4. Hand-Object Geometric Interaction Evaluation
        hand_boxes = [{"x1": h["x1"], "y1": h["y1"], "x2": h["x2"], "y2": h["y2"]} for h in hands]
        obj_boxes = [{"x1": o["x1"], "y1": o["y1"], "x2": o["x2"], "y2": o["y2"], "label": o["label"]} for o in objects]
        interaction = self.interaction_engine.evaluate_interaction(hand_boxes, obj_boxes)

        # 5. Extract 1D Keypoint Vector for Temporal Classifier
        keypoint_vec = self.frame_processor.extract_keypoint_vector(pose_landmarks, objects)
        self.action_classifier.push_frame_vector(keypoint_vec)

        # 6. Action Recognition
        detected_action, action_conf = self.action_classifier.classify(
            interaction_state=interaction.get("state")
        )

        # 7. Render Annotated HUD Frame if requested
        annotated_frame = None
        if annotate:
            annotated_frame = self.frame_processor.draw_annotations(
                frame=frame,
                detections=objects,
                pose_keypoints=pose_landmarks,
                action_text=detected_action,
                confidence=action_conf,
            )

        return {
            "frame_index": self._frame_index,
            "objects": objects,
            "pose": pose_landmarks,
            "hands": hands,
            "interaction": interaction,
            "action": detected_action,
            "confidence": action_conf,
            "annotated_frame": annotated_frame,
        }

    def reset(self) -> None:
        """Reset temporal state and counters."""
        self._frame_index = 0
        self.action_classifier.reset_buffer()

    def release(self) -> None:
        """Release underlying hardware and detector resources."""
        self.pose_detector.close()
        self.hand_detector.close()
        self.hailo_engine.release()
