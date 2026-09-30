"""
frame_processor.py — Vision Preprocessing & Annotation Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B4.1c.2)

Normalizes, resizes, letterboxes, and prepares video frames for
Hailo-8L INT8 / YOLOv8 / 3D HMR / TFLite action models and renders
real-time visual bounding overlays.

Incorporates optional Camera Rectification hook (Gate B4.1c.2):
  Raw Camera Frame
         ↓
  Optional Camera Rectification (B4.1c.1 Core: mounting tilt / roll correction)
         ↓
  Existing Letterbox / Resizing / Normalization
         ↓
  Downstream Perception (YOLO / Pose / 1D-TCN)

When rectification is disabled (the default), existing preprocessing behavior
remains strictly unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Optional OpenCV import with pure NumPy fallbacks
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

from backend.video.camera_rectification import (
    CameraRectifier,
    RectificationConfig,
    RectificationResult,
)


def _resolve_rectifier(
    rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None
) -> Optional[CameraRectifier]:
    """
    Helper function to safely instantiate or pass through a CameraRectifier.
    Fails safely on invalid configuration types.
    """
    if rectifier is None:
        return None
    if isinstance(rectifier, CameraRectifier):
        return rectifier
    if isinstance(rectifier, RectificationConfig):
        return CameraRectifier(rectifier)
    if isinstance(rectifier, dict):
        return CameraRectifier.from_dict(rectifier)
    raise TypeError(
        f"Invalid rectifier type: {type(rectifier)}. Expected CameraRectifier, "
        f"RectificationConfig, dict, or None."
    )


def _pure_numpy_resize(image: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """Pure NumPy bilinear resize fallback when cv2 is not available."""
    target_w, target_h = target_wh
    in_h, in_w = image.shape[:2]
    if in_w == target_w and in_h == target_h:
        return image.copy()

    y_coords = np.linspace(0, in_h - 1, target_h)
    x_coords = np.linspace(0, in_w - 1, target_w)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    x0 = np.floor(x_grid).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.floor(y_grid).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)

    wa = (x1 - x_grid) * (y1 - y_grid)
    wb = (x1 - x_grid) * (y_grid - y0)
    wc = (x_grid - x0) * (y1 - y_grid)
    wd = (x_grid - x0) * (y_grid - y0)

    if image.ndim == 3:
        out = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        for c in range(image.shape[2]):
            chan = image[:, :, c].astype(np.float64)
            interp = (
                wa * chan[y0, x0]
                + wb * chan[y1, x0]
                + wc * chan[y0, x1]
                + wd * chan[y1, x1]
            )
            out[:, :, c] = np.clip(interp, 0, 255).astype(image.dtype)
        return out
    else:
        img_f = image.astype(np.float64)
        interp = (
            wa * img_f[y0, x0]
            + wb * img_f[y1, x0]
            + wc * img_f[y0, x1]
            + wd * img_f[y1, x1]
        )
        return np.clip(interp, 0, 255).astype(image.dtype)


def _pure_numpy_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Pure NumPy BGR -> RGB color conversion."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()
    return image.copy()


class FrameProcessor:
    """
    High-performance image preprocessing, annotation, and camera rectification utility.
    """

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Union[int, Tuple[int, int]] = (640, 640),
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
            if cv2 is not None:
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            else:
                image = _pure_numpy_resize(image, new_unpad)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        if cv2 is not None:
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
            )
        else:
            if image.ndim == 3:
                image = np.pad(
                    image,
                    ((top, bottom), (left, right), (0, 0)),
                    mode="constant",
                    constant_values=color[0],
                )
            else:
                image = np.pad(
                    image,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=color[0],
                )

        return image, r, (dw, dh)

    @staticmethod
    def rectify_frame(
        frame: np.ndarray,
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Optional camera mounting tilt & roll rectification hook.
        
        If rectifier is None or disabled, returns the original frame unmodified (identity bypass).
        """
        r_obj = _resolve_rectifier(rectifier)
        if r_obj is None or r_obj.is_identity():
            return frame
        return r_obj.rectify_frame(frame)

    @staticmethod
    def preprocess_for_detector(
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Prepares BGR frame for YOLO object detector.
        
        Preprocessing flow:
          1. Optional Camera Rectification (if configured and enabled)
          2. BGR -> RGB color conversion
          3. Letterboxing to target_size
          
        Returns: (letterboxed_rgb_image, scale_ratio, (pad_w, pad_h))
        """
        # Step 1: Optional camera rectification
        rectified = FrameProcessor.rectify_frame(frame, rectifier=rectifier)

        # Step 2: BGR -> RGB conversion
        if cv2 is not None:
            rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
        else:
            rgb = _pure_numpy_bgr_to_rgb(rectified)

        # Step 3: Letterboxing
        return FrameProcessor.letterbox(rgb, new_shape=target_size, auto=False)

    @staticmethod
    def preprocess_for_pose(
        frame: np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Prepares BGR frame for lightweight 3D pose / landmark model.
        
        Preprocessing flow:
          1. Optional Camera Rectification (if configured and enabled)
          2. BGR -> RGB color conversion
          3. Resizing to target_size
          4. Normalization to float32 [0.0, 1.0]
          
        Returns: normalized float32 RGB array [0.0, 1.0] of shape (target_size[1], target_size[0], 3).
        """
        # Step 1: Optional camera rectification
        rectified = FrameProcessor.rectify_frame(frame, rectifier=rectifier)

        # Step 2: BGR -> RGB conversion
        if cv2 is not None:
            rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            rgb = _pure_numpy_bgr_to_rgb(rectified)
            resized = _pure_numpy_resize(rgb, target_size)

        # Step 3 & 4: Normalization
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    @staticmethod
    def process_frame_pipeline(
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        rectifier: Optional[Union[CameraRectifier, RectificationConfig, Dict[str, Any]]] = None,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Unified preprocessing pipeline coordinating optional camera rectification,
        detector letterboxing, pose normalization, and synchronized annotation mapping.
        """
        r_obj = _resolve_rectifier(rectifier)
        if r_obj is not None and not r_obj.is_identity():
            rect_res = r_obj.rectify(frame, bounding_boxes=bounding_boxes, keypoints=keypoints)
            rect_frame = rect_res.image
            sync_boxes = rect_res.bounding_boxes
            sync_kps = rect_res.keypoints
            rect_applied = True
        else:
            rect_frame = frame
            sync_boxes = [dict(b) for b in bounding_boxes] if bounding_boxes else []
            sync_kps = [dict(kp) for kp in keypoints] if keypoints else []
            rect_applied = False

        # Preprocess for detector
        detector_img, scale, padding = FrameProcessor.preprocess_for_detector(
            rect_frame, target_size=target_size, rectifier=None
        )

        # Preprocess for pose
        pose_img = FrameProcessor.preprocess_for_pose(
            rect_frame, target_size=(256, 256), rectifier=None
        )

        return {
            "rectified_frame": rect_frame,
            "rectification_applied": rect_applied,
            "detector_input": detector_img,
            "detector_scale": scale,
            "detector_padding": padding,
            "pose_input": pose_img,
            "bounding_boxes": sync_boxes,
            "keypoints": sync_kps,
        }

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
        if cv2 is None:
            # Fallback if cv2 is not installed: return unmodified frame
            return frame.copy()

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
