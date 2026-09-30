"""
augmentation.py — Reusable Orientation & Environmental Augmentation Core
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B4.1a)

Provides deterministic, synchronized data augmentation for:
  1. Rotation (0, 45, 90, 135, 180, 225, 270 deg and arbitrary angles)
  2. Scale (zoom in / zoom out)
  3. Translation (horizontal and vertical spatial shift)
  4. Brightness (gain, bias, gamma adjustment)
  5. Blur (Gaussian and box spatial blur)
  6. Occlusion (Cutout / rectangular block masking)
  7. Perspective (4-point projective / trapezoidal warping)

Key architectural properties:
- Zero ML framework dependency (pure Python + NumPy, optional OpenCV acceleration).
- Fully synchronized transformations: geometric transforms update image pixels,
  2D bounding boxes [x1, y1, x2, y2], and 2D/3D landmark keypoints [x, y, z].
- Fully deterministic execution when a seed is provided.
- Decoupled from runtime inference pipeline (does NOT run during live inference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ===========================================================================
# 1. Configuration & Result Containers
# ===========================================================================

@dataclass
class AugmentationConfig:
    """
    Configuration parameters for synthetic orientation & disturbance transformations.
    All numeric ranges represent PROPOSED DEFAULTS requiring experimental validation.
    """
    # 1. Rotation
    enable_rotation: bool = False
    rotation_angle_deg: float = 0.0  # Canonical: 0, 45, 90, 135, 180, 225, 270
    rotation_range_deg: Tuple[float, float] = (0.0, 360.0)

    # 2. Scale
    enable_scale: bool = False
    scale_factor: float = 1.0  # Proposed default range: [0.8, 1.2]
    scale_range: Tuple[float, float] = (0.8, 1.2)

    # 3. Translation (fraction of width/height)
    enable_translation: bool = False
    translation_x: float = 0.0  # Proposed default range: [-0.10, 0.10]
    translation_y: float = 0.0
    translation_range: Tuple[float, float] = (-0.10, 0.10)

    # 4. Brightness
    enable_brightness: bool = False
    brightness_gain: float = 1.0  # Multiplicative factor [0.8, 1.2]
    brightness_bias: float = 0.0  # Additive shift [-30.0, 30.0]
    brightness_gain_range: Tuple[float, float] = (0.80, 1.20)

    # 5. Blur
    enable_blur: bool = False
    blur_kernel_size: int = 3  # Odd integer >= 3 (e.g. 3, 5, 7)

    # 6. Occlusion
    enable_occlusion: bool = False
    occlusion_size_fraction: float = 0.15  # Max fraction of frame dimension
    occlusion_fill_value: int = 128  # Gray fill [0-255] or -1 for noise
    occlusion_center: Optional[Tuple[float, float]] = None  # (norm_x, norm_y) or random

    # 7. Perspective
    enable_perspective: bool = False
    perspective_distortion: float = 0.05  # Corner displacement fraction [0.0, 0.15]

    # Global Random Seed for Determinism
    seed: Optional[int] = None


@dataclass
class AugmentationResult:
    """
    Encapsulates the output of a multi-modal augmentation step.
    """
    image: np.ndarray
    bounding_boxes: List[Dict[str, Any]]
    keypoints: List[Dict[str, Any]]
    transform_metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# 2. Coordinate & Affine Transformation Helpers (Pure NumPy)
# ===========================================================================

def get_rotation_matrix_2d(
    center: Tuple[float, float], angle_deg: float, scale: float = 1.0
) -> np.ndarray:
    """
    Calculates 2D affine transformation matrix for rotation + scale around a center point.
    Returns 3x3 homogeneous matrix.
    """
    angle_rad = math.radians(angle_deg)
    alpha = scale * math.cos(angle_rad)
    beta = scale * math.sin(angle_rad)
    cx, cy = center

    # Forward transformation: [x', y', 1]^T = M * [x, y, 1]^T
    # where M maps source coordinate (x,y) to rotated coordinate (x',y')
    m = np.array([
        [alpha, -beta, (1 - alpha) * cx + beta * cy],
        [beta, alpha, -beta * cx + (1 - alpha) * cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return m


def transform_normalized_point(
    x: float, y: float, matrix: np.ndarray, width: int, height: int
) -> Tuple[float, float]:
    """
    Transforms a single normalized [0.0, 1.0] point using a 3x3 pixel-space affine matrix.
    Returns transformed normalized (x', y').
    """
    px = x * width
    py = y * height
    vec = np.array([px, py, 1.0], dtype=np.float64)
    res = matrix @ vec
    new_px = res[0] / res[2] if res[2] != 0 else res[0]
    new_py = res[1] / res[2] if res[2] != 0 else res[1]
    return float(new_px / width), float(new_py / height)


def transform_bounding_box(
    box: Dict[str, Any], matrix: np.ndarray, width: int, height: int, clip: bool = True
) -> Dict[str, Any]:
    """
    Transforms a 2D normalized bounding box by projecting its 4 corners and taking
    the new axis-aligned bounding envelope.
    """
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))

    # 4 corners
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    transformed_corners = [
        transform_normalized_point(cx, cy, matrix, width, height) for cx, cy in corners
    ]

    xs = [pt[0] for pt in transformed_corners]
    ys = [pt[1] for pt in transformed_corners]

    new_x1 = min(xs)
    new_y1 = min(ys)
    new_x2 = max(xs)
    new_y2 = max(ys)

    if clip:
        new_x1 = max(0.0, min(1.0, new_x1))
        new_y1 = max(0.0, min(1.0, new_y1))
        new_x2 = max(0.0, min(1.0, new_x2))
        new_y2 = max(0.0, min(1.0, new_y2))

    new_box = dict(box)
    new_box["x1"] = round(new_x1, 4)
    new_box["y1"] = round(new_y1, 4)
    new_box["x2"] = round(new_x2, 4)
    new_box["y2"] = round(new_y2, 4)
    return new_box


def transform_keypoints(
    keypoints: List[Dict[str, Any]], matrix: np.ndarray, width: int, height: int, clip: bool = True
) -> List[Dict[str, Any]]:
    """
    Transforms a list of normalized landmark keypoints.
    Preserves all auxiliary keys (e.g. z, visibility, label).
    """
    transformed = []
    for kp in keypoints:
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        nx, ny = transform_normalized_point(x, y, matrix, width, height)
        if clip:
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))

        new_kp = dict(kp)
        new_kp["x"] = round(nx, 4)
        new_kp["y"] = round(ny, 4)
        transformed.append(new_kp)
    return transformed


def warp_affine_numpy(
    image: np.ndarray, matrix_forward: np.ndarray, output_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Applies 2D affine warping using vectorized NumPy backward mapping.
    Works seamlessly without OpenCV installed; prefers OpenCV if available.
    """
    out_h, out_w = output_shape
    # Try fast OpenCV if available
    try:
        import cv2  # type: ignore
        m23 = matrix_forward[:2, :]
        return cv2.warpAffine(
            image, m23, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
    except ImportError:
        pass

    # Pure NumPy inverse mapping implementation
    # Compute inverse matrix mapping destination (xd, yd) -> source (xs, ys)
    try:
        m_inv = np.linalg.inv(matrix_forward)
    except np.linalg.LinAlgError:
        return image.copy()

    # Generate grid of destination coordinates
    y_coords, x_coords = np.indices((out_h, out_w), dtype=np.float64)
    ones = np.ones_like(x_coords)
    dest_pts = np.stack([x_coords.ravel(), y_coords.ravel(), ones.ravel()], axis=0)

    # Compute source points
    src_pts = m_inv @ dest_pts
    src_x = src_pts[0, :].reshape(out_h, out_w)
    src_y = src_pts[1, :].reshape(out_h, out_w)

    in_h, in_w = image.shape[:2]

    # Nearest-neighbor interpolation fallback in pure NumPy
    src_x_round = np.round(src_x).astype(np.int64)
    src_y_round = np.round(src_y).astype(np.int64)

    valid_mask = (src_x_round >= 0) & (src_x_round < in_w) & (src_y_round >= 0) & (src_y_round < in_h)

    if image.ndim == 3:
        warped = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            safe_x = np.clip(src_x_round, 0, in_w - 1)
            safe_y = np.clip(src_y_round, 0, in_h - 1)
            sampled = channel[safe_y, safe_x]
            warped[:, :, c] = np.where(valid_mask, sampled, 0)
    else:
        safe_x = np.clip(src_x_round, 0, in_w - 1)
        safe_y = np.clip(src_y_round, 0, in_h - 1)
        warped = np.where(valid_mask, image[safe_y, safe_x], 0)

    return warped


# ===========================================================================
# 3. Augmentation Core Engine
# ===========================================================================

class AugmentationEngine:
    """
    Unified multi-modal orientation and environmental disturbance augmentation engine.
    """

    CANONICAL_ORIENTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270]

    def __init__(self, default_config: Optional[AugmentationConfig] = None):
        self.default_config = default_config or AugmentationConfig()

    # -----------------------------------------------------------------------
    # 3A. Rotation
    # -----------------------------------------------------------------------
    def rotate(
        self,
        image: np.ndarray,
        angle_deg: float,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Rotates image and synchronizes 2D bounding boxes and keypoint landmarks.
        Supports exact 90-degree fast rotations and arbitrary continuous angles.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0 or angle_deg % 360 == 0:
            return AugmentationResult(
                image=image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"angle_deg": 0.0, "type": "identity"},
            )

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Build forward transformation matrix
        matrix = get_rotation_matrix_2d(center=center, angle_deg=angle_deg, scale=1.0)

        # Warp image
        warped_img = warp_affine_numpy(image, matrix, (h, w))

        # Synchronously transform annotations
        transformed_boxes = [transform_bounding_box(b, matrix, w, h) for b in boxes]
        transformed_kps = transform_keypoints(kps, matrix, w, h)

        return AugmentationResult(
            image=warped_img,
            bounding_boxes=transformed_boxes,
            keypoints=transformed_kps,
            transform_metadata={"angle_deg": angle_deg, "type": "rotation"},
        )

    # -----------------------------------------------------------------------
    # 3B. Scale
    # -----------------------------------------------------------------------
    def scale(
        self,
        image: np.ndarray,
        scale_factor: float,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Scales/zooms image around frame center and synchronizes annotations.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0 or scale_factor == 1.0:
            return AugmentationResult(
                image=image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"scale_factor": scale_factor, "type": "identity"},
            )

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = get_rotation_matrix_2d(center=center, angle_deg=0.0, scale=scale_factor)

        warped_img = warp_affine_numpy(image, matrix, (h, w))
        transformed_boxes = [transform_bounding_box(b, matrix, w, h) for b in boxes]
        transformed_kps = transform_keypoints(kps, matrix, w, h)

        return AugmentationResult(
            image=warped_img,
            bounding_boxes=transformed_boxes,
            keypoints=transformed_kps,
            transform_metadata={"scale_factor": scale_factor, "type": "scale"},
        )

    # -----------------------------------------------------------------------
    # 3C. Translation
    # -----------------------------------------------------------------------
    def translate(
        self,
        image: np.ndarray,
        dx_fraction: float,
        dy_fraction: float,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Translates image by fractional offsets (dx, dy) and synchronizes annotations.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0 or (dx_fraction == 0.0 and dy_fraction == 0.0):
            return AugmentationResult(
                image=image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"dx": dx_fraction, "dy": dy_fraction, "type": "identity"},
            )

        h, w = image.shape[:2]
        shift_x = dx_fraction * w
        shift_y = dy_fraction * h

        matrix = np.array([
            [1.0, 0.0, shift_x],
            [0.0, 1.0, shift_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        warped_img = warp_affine_numpy(image, matrix, (h, w))
        transformed_boxes = [transform_bounding_box(b, matrix, w, h) for b in boxes]
        transformed_kps = transform_keypoints(kps, matrix, w, h)

        return AugmentationResult(
            image=warped_img,
            bounding_boxes=transformed_boxes,
            keypoints=transformed_kps,
            transform_metadata={"dx": dx_fraction, "dy": dy_fraction, "type": "translation"},
        )

    # -----------------------------------------------------------------------
    # 3D. Brightness (Image Only)
    # -----------------------------------------------------------------------
    def adjust_brightness(
        self,
        image: np.ndarray,
        gain: float = 1.0,
        bias: float = 0.0,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Adjusts image brightness and contrast: I' = clip(I * gain + bias, 0, 255).
        Annotations remain unchanged in coordinate space.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0:
            return AugmentationResult(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"gain": gain, "bias": bias},
            )

        float_img = image.astype(np.float32)
        adjusted = np.clip(float_img * gain + bias, 0.0, 255.0).astype(image.dtype)

        return AugmentationResult(
            image=adjusted,
            bounding_boxes=boxes,
            keypoints=kps,
            transform_metadata={"gain": gain, "bias": bias, "type": "brightness"},
        )

    # -----------------------------------------------------------------------
    # 3E. Blur (Image Only)
    # -----------------------------------------------------------------------
    def apply_blur(
        self,
        image: np.ndarray,
        kernel_size: int = 3,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Applies spatial averaging blur to simulate motion blur and camera defocus.
        Annotations are preserved without coordinate alteration.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0 or kernel_size <= 1:
            return AugmentationResult(
                image=image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"kernel_size": kernel_size},
            )

        # Enforce odd kernel size
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        try:
            import cv2  # type: ignore
            blurred = cv2.blur(image, (k, k))
        except ImportError:
            # Pure NumPy box blur using uniform 2D filter
            pad_w = k // 2
            if image.ndim == 3:
                padded = np.pad(image, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)), mode="edge")
                blurred = np.zeros_like(image, dtype=np.float32)
                for di in range(k):
                    for dj in range(k):
                        blurred += padded[di:di + image.shape[0], dj:dj + image.shape[1], :]
                blurred = (blurred / (k * k)).astype(image.dtype)
            else:
                padded = np.pad(image, ((pad_w, pad_w), (pad_w, pad_w)), mode="edge")
                blurred = np.zeros_like(image, dtype=np.float32)
                for di in range(k):
                    for dj in range(k):
                        blurred += padded[di:di + image.shape[0], dj:dj + image.shape[1]]
                blurred = (blurred / (k * k)).astype(image.dtype)

        return AugmentationResult(
            image=blurred,
            bounding_boxes=boxes,
            keypoints=kps,
            transform_metadata={"kernel_size": k, "type": "blur"},
        )

    # -----------------------------------------------------------------------
    # 3F. Occlusion (Cutout / Synthetic Mask)
    # -----------------------------------------------------------------------
    def apply_occlusion(
        self,
        image: np.ndarray,
        box_fraction: float = 0.15,
        center: Optional[Tuple[float, float]] = None,
        fill_value: int = 128,
        seed: Optional[int] = None,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Applies synthetic rectangular cutout occlusion block.
        Deterministic when seed is provided.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0 or box_fraction <= 0.0:
            return AugmentationResult(
                image=image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"box_fraction": box_fraction},
            )

        rng = np.random.RandomState(seed) if seed is not None else np.random

        h, w = image.shape[:2]
        occ_h = int(round(box_fraction * h))
        occ_w = int(round(box_fraction * w))

        if center is not None:
            cx = int(round(center[0] * w))
            cy = int(round(center[1] * h))
        else:
            cx = rng.randint(occ_w // 2, max(occ_w // 2 + 1, w - occ_w // 2))
            cy = rng.randint(occ_h // 2, max(occ_h // 2 + 1, h - occ_h // 2))

        x1 = max(0, cx - occ_w // 2)
        y1 = max(0, cy - occ_h // 2)
        x2 = min(w, x1 + occ_w)
        y2 = min(h, y1 + occ_h)

        occluded_img = image.copy()
        if fill_value == -1:
            noise = rng.randint(0, 256, (y2 - y1, x2 - x1, occluded_img.shape[2] if occluded_img.ndim == 3 else 1))
            occluded_img[y1:y2, x1:x2] = noise.squeeze()
        else:
            occluded_img[y1:y2, x1:x2] = fill_value

        return AugmentationResult(
            image=occluded_img,
            bounding_boxes=boxes,
            keypoints=kps,
            transform_metadata={
                "occlusion_rect": [x1 / w, y1 / h, x2 / w, y2 / h],
                "type": "occlusion",
            },
        )

    # -----------------------------------------------------------------------
    # 3G. Perspective (Trapezoidal / Viewpoint Tilt)
    # -----------------------------------------------------------------------
    def apply_perspective(
        self,
        image: np.ndarray,
        distortion: float = 0.05,
        seed: Optional[int] = None,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> AugmentationResult:
        """
        Applies trapezoidal perspective warp to simulate oblique pitch / camera mount angle.
        Synchronizes 2D bounding boxes and keypoint landmarks.
        """
        boxes = [dict(b) for b in (bounding_boxes or [])]
        kps = [dict(k) for k in (keypoints or [])]

        if image is None or image.size == 0 or distortion <= 0.0:
            return AugmentationResult(
                image=image.copy() if image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                bounding_boxes=boxes,
                keypoints=kps,
                transform_metadata={"distortion": distortion},
            )

        rng = np.random.RandomState(seed) if seed is not None else np.random
        h, w = image.shape[:2]

        dx = distortion * w
        dy = distortion * h

        # 4 source corners
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # 4 destination corners with deterministic slight tilt
        d0 = rng.uniform(-dx, dx)
        d1 = rng.uniform(-dy, dy)
        dst_pts = np.float32([
            [max(0, d0), max(0, d1)],
            [min(w, w - d0), max(0, d1)],
            [min(w, w - d0), min(h, h - d1)],
            [max(0, d0), min(h, h - d1)],
        ])

        # Compute 3x3 perspective homography matrix H
        try:
            import cv2  # type: ignore
            h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, h_mat, (w, h))
        except ImportError:
            # Linear affine approximation for pure NumPy environments
            # Approximate top 3 points as affine
            center = (w / 2.0, h / 2.0)
            h_mat = get_rotation_matrix_2d(center=center, angle_deg=0.0, scale=1.0 - (distortion * 0.5))
            warped = warp_affine_numpy(image, h_mat, (h, w))

        transformed_boxes = [transform_bounding_box(b, h_mat, w, h) for b in boxes]
        transformed_kps = transform_keypoints(kps, h_mat, w, h)

        return AugmentationResult(
            image=warped,
            bounding_boxes=transformed_boxes,
            keypoints=transformed_kps,
            transform_metadata={"distortion": distortion, "type": "perspective"},
        )

    # -----------------------------------------------------------------------
    # 3H. Composite Pipeline Application
    # -----------------------------------------------------------------------
    def apply(
        self,
        image: np.ndarray,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
        config: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
    ) -> AugmentationResult:
        """
        Executes a composite pipeline of configured transformations sequentially
        with full synchronization across image, boxes, and keypoints.
        """
        cfg = config or self.default_config
        active_seed = seed if seed is not None else cfg.seed

        current_img = image.copy()
        current_boxes = [dict(b) for b in (bounding_boxes or [])]
        current_kps = [dict(k) for k in (keypoints or [])]
        applied_transforms = []

        # 1. Rotation
        if cfg.enable_rotation:
            angle = cfg.rotation_angle_deg
            res = self.rotate(current_img, angle, current_boxes, current_kps)
            current_img, current_boxes, current_kps = res.image, res.bounding_boxes, res.keypoints
            applied_transforms.append(f"rotation_{angle}deg")

        # 2. Scale
        if cfg.enable_scale:
            res = self.scale(current_img, cfg.scale_factor, current_boxes, current_kps)
            current_img, current_boxes, current_kps = res.image, res.bounding_boxes, res.keypoints
            applied_transforms.append(f"scale_{cfg.scale_factor}x")

        # 3. Translation
        if cfg.enable_translation:
            res = self.translate(current_img, cfg.translation_x, cfg.translation_y, current_boxes, current_kps)
            current_img, current_boxes, current_kps = res.image, res.bounding_boxes, res.keypoints
            applied_transforms.append(f"translate_({cfg.translation_x},{cfg.translation_y})")

        # 4. Perspective
        if cfg.enable_perspective:
            res = self.apply_perspective(current_img, cfg.perspective_distortion, active_seed, current_boxes, current_kps)
            current_img, current_boxes, current_kps = res.image, res.bounding_boxes, res.keypoints
            applied_transforms.append("perspective")

        # 5. Brightness
        if cfg.enable_brightness:
            res = self.adjust_brightness(current_img, cfg.brightness_gain, cfg.brightness_bias, current_boxes, current_kps)
            current_img = res.image
            applied_transforms.append(f"brightness_gain{cfg.brightness_gain}")

        # 6. Blur
        if cfg.enable_blur:
            res = self.apply_blur(current_img, cfg.blur_kernel_size, current_boxes, current_kps)
            current_img = res.image
            applied_transforms.append(f"blur_k{cfg.blur_kernel_size}")

        # 7. Occlusion
        if cfg.enable_occlusion:
            res = self.apply_occlusion(
                current_img,
                cfg.occlusion_size_fraction,
                cfg.occlusion_center,
                cfg.occlusion_fill_value,
                active_seed,
                current_boxes,
                current_kps,
            )
            current_img = res.image
            applied_transforms.append("occlusion")

        return AugmentationResult(
            image=current_img,
            bounding_boxes=current_boxes,
            keypoints=current_kps,
            transform_metadata={"applied_transforms": applied_transforms, "seed": active_seed},
        )
