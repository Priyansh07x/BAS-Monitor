"""
frame_processor.py — Vision Preprocessing & Annotation Engine
ISRO SIH26174 BAS Experiment Monitor

Normalizes, resizes, letterboxes, and prepares video frames for
Hailo-8L INT8 / YOLOv8 / 3D HMR / TFLite action models and renders
real-time visual bounding overlays.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class FrameProcessor:
    """
    High-performance image preprocessing and annotation utility.
    """

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scale_up: bool = True,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize and pad image while meeting stride-multiple constraints.
        Returns: (padded_image, scale_ratio, (pad_w, pad_h))
        """
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            r = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return image, r, (dw, dh)

    @staticmethod
    def preprocess_for_detector(
        frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Prepares BGR OpenCV frame for YOLO object detector.
        Converts BGR -> RGB and applies letterbox.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return FrameProcessor.letterbox(rgb, new_shape=target_size, auto=False)

    @staticmethod
    def preprocess_for_pose(
        frame: np.ndarray, target_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Prepares BGR frame for lightweight 3D pose / landmark model.
        Returns float32 normalized image [0, 1].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    @staticmethod
    def extract_keypoint_vector(
        pose_landmarks: Optional[List[Dict[str, float]]],
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Converts 3D joint coordinates and object centroids into a flat 1D vector
        for the temporal action classifier (1D-TCN).
        """
        vector: List[float] = []

        # 1. 3D Body Landmarks (typically 33 points x 3 = 99 features)
        if pose_landmarks:
            for pt in pose_landmarks[:33]:
                vector.extend([
                    float(pt.get("x", 0.0)),
                    float(pt.get("y", 0.0)),
                    float(pt.get("z", 0.0)),
                ])
        else:
            vector.extend([0.0] * (33 * 3))

        # 2. Object centroid offsets (up to 4 primary detected objects x 3 = 12 features)
        if bounding_boxes:
            for box in bounding_boxes[:4]:
                cx = (box.get("x1", 0) + box.get("x2", 0)) / 2.0
                cy = (box.get("y1", 0) + box.get("y2", 0)) / 2.0
                conf = box.get("confidence", 0.0)
                vector.extend([cx, cy, conf])
            # Pad if fewer than 4 objects
            remaining = 4 - len(bounding_boxes[:4])
            if remaining > 0:
                vector.extend([0.0] * (remaining * 3))
        else:
            vector.extend([0.0] * 12)

        return np.array(vector, dtype=np.float32)

    @staticmethod
    def draw_annotations(
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        pose_keypoints: Optional[List[Dict[str, float]]] = None,
        action_text: Optional[str] = None,
        confidence: Optional[float] = None,
        status_color: Tuple[int, int, int] = (243, 218, 0),  # Cyan in BGR
    ) -> np.ndarray:
        """
        Draws glowing sci-fi HUD bounding boxes, landmark points, and labels
        onto the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw Detections (bounding boxes)
        if detections:
            for det in detections:
                x1 = int(det.get("x1", 0) * w if det.get("x1", 0) <= 1.0 else det.get("x1", 0))
                y1 = int(det.get("y1", 0) * h if det.get("y1", 0) <= 1.0 else det.get("y1", 0))
                x2 = int(det.get("x2", 0) * w if det.get("x2", 0) <= 1.0 else det.get("x2", 0))
                y2 = int(det.get("y2", 0) * h if det.get("y2", 0) <= 1.0 else det.get("y2", 0))
                label = det.get("label", "OBJECT")
                conf = det.get("confidence", 0.0)

                # Draw dashed/corner box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), status_color, 2)
                # Corner accents
                c_len = 10
                cv2.line(annotated, (x1, y1), (x1 + c_len, y1), status_color, 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + c_len), status_color, 3)
                cv2.line(annotated, (x2, y1), (x2 - c_len, y1), status_color, 3)
                cv2.line(annotated, (x2, y1), (x2, y1 + c_len), status_color, 3)

                # Label tag
                tag = f"{label} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), status_color, -1)
                cv2.putText(
                    annotated,
                    tag,
                    (x1 + 3, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )

        # Draw Pose Landmarks
        if pose_keypoints:
            for pt in pose_keypoints:
                px = int(pt.get("x", 0) * w if pt.get("x", 0) <= 1.0 else pt.get("x", 0))
                py = int(pt.get("y", 0) * h if pt.get("y", 0) <= 1.0 else pt.get("y", 0))
                cv2.circle(annotated, (px, py), 3, (0, 255, 128), -1)

        # Draw Global Action Readout
        if action_text:
            text = f"ACTION: {action_text}"
            if confidence is not None:
                text += f" ({int(confidence * 100)}%)"
            cv2.putText(
                annotated,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

        return annotated
