"""
camera_rectification.py — Fixed Camera Mounting Calibration & View Rectification Core
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B4.1c.1)

Provides a deterministic, reusable camera rectification and calibration transform component:
  Known camera/view calibration parameters
                ↓
  Deterministic geometric transform (Rotation, Affine, Homography)
                ↓
  Rectified frame + Synchronized Annotations (Bounding Boxes & Keypoints)

=============================================================================
CRITICAL ARCHITECTURAL DISTINCTION:
1. CAMERA CALIBRATION / MOUNTING TILT (Addressed by this module):
   A fixed, known geometric transformation required because the physical camera
   is mounted at a known angle/tilt/perspective relative to the payload rack.
2. SCENE / ASTRONAUT ORIENTATION (Addressed in B4/B15):
   The dynamic physical orientation of the astronaut or objects in microgravity.
   This module does NOT dynamically estimate astronaut pose or rotate arbitrary
   frames based on human posture.

CURRENT CALIBRATION STATUS:
   All physical calibration values are PROPOSED DEFAULTS requiring experimental
   calibration once real flight/rig camera geometry is physically measured.
   Default behavior is IDENTITY / NO-OP when unconfigured or disabled.
=============================================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ===========================================================================
# 1. Configuration & Result Dataclasses
# ===========================================================================

@dataclass
class RectificationConfig:
    """
    Configuration parameters for fixed camera mounting calibration and rectification.
    
    All numeric parameters default to identity / no-op.
    Non-identity defaults are labeled: PROPOSED DEFAULT — REQUIRES EXPERIMENTAL CALIBRATION.
    """
    # Enable / disable rectification (default False = identity / no-op)
    enabled: bool = False

    # Fixed mounting rotation in degrees (canonical: 0, 45, 90, 135, 180, 225, 270)
    rotation_deg: float = 0.0

    # Scale / zoom correction factor (1.0 = identity)
    scale: float = 1.0

    # Normalized translational offset [-1.0, 1.0] along X and Y axes
    translation_x: float = 0.0
    translation_y: float = 0.0

    # Optional explicit 2x3 or 3x3 affine transformation matrix (overrides rotation/scale/trans)
    affine_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None

    # Optional explicit 3x3 homography / perspective transformation matrix
    homography_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None

    # Optional output frame size (width, height). If None, preserves input frame dimensions.
    output_size: Optional[Tuple[int, int]] = None

    # Rotation center offset (dx, dy) relative to image center (0.5, 0.5) in normalized coords
    center_offset: Optional[Tuple[float, float]] = None

    # Border padding mode: "constant", "replicate", "reflect"
    border_mode: str = "constant"

    # Border padding value for "constant" mode (e.g. black = (0, 0, 0))
    border_value: Union[int, Tuple[int, int, int]] = (0, 0, 0)

    # Interpolation mode: "bilinear", "nearest"
    interpolation: str = "bilinear"

    # Calibration traceability metadata
    calibration_status: str = "UNSPECIFIED"  # "UNSPECIFIED", "PROPOSED_DEFAULT", "CALIBRATED"
    calibration_notes: str = "PROPOSED DEFAULT — REQUIRES EXPERIMENTAL CALIBRATION"


@dataclass
class RectificationResult:
    """
    Encapsulates the output of a deterministic frame and annotation rectification step.
    """
    image: np.ndarray
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    keypoints: List[Dict[str, Any]] = field(default_factory=list)
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    inverse_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    applied: bool = False
    input_shape: Tuple[int, int] = (0, 0)  # (height, width)
    output_shape: Tuple[int, int] = (0, 0)  # (height, width)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# 2. Geometric Transform Helper Functions (Pure NumPy)
# ===========================================================================

def build_rectification_matrix(
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    center_offset: Optional[Tuple[float, float]] = None,
    affine_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
    homography_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None,
) -> np.ndarray:
    """
    Constructs a 3x3 homogeneous pixel-space forward transformation matrix:
        [x_dst, y_dst, 1]^T = M @ [x_src, y_src, 1]^T
    
    Order of operations for canonical parameters:
    1. Translate center of rotation to origin
    2. Apply rotation and scale
    3. Translate back to destination center + translational offset
    """
    # 1. Custom Homography Matrix takes precedence
    if homography_matrix is not None:
        h_mat = np.array(homography_matrix, dtype=np.float64)
        if h_mat.shape == (3, 3):
            return h_mat
        raise ValueError(f"Invalid homography matrix shape: {h_mat.shape}. Expected (3, 3).")

    # 2. Custom Affine Matrix takes next precedence
    if affine_matrix is not None:
        a_mat = np.array(affine_matrix, dtype=np.float64)
        if a_mat.shape == (2, 3):
            return np.vstack([a_mat, [0.0, 0.0, 1.0]])
        elif a_mat.shape == (3, 3):
            return a_mat
        raise ValueError(f"Invalid affine matrix shape: {a_mat.shape}. Expected (2, 3) or (3, 3).")

    # 3. Canonical Parameterized Transform
    # Determine center of rotation in source coordinates
    cx_src = (in_w / 2.0)
    cy_src = (in_h / 2.0)
    if center_offset is not None:
        cx_src += center_offset[0] * in_w
        cy_src += center_offset[1] * in_h

    # Destination center
    cx_dst = out_w / 2.0
    cy_dst = out_h / 2.0

    # Compute rotation components
    rad = math.radians(rotation_deg)
    cos_a = math.cos(rad) * scale
    sin_a = math.sin(rad) * scale

    # Shift to origin: T_neg
    t_neg = np.array([
        [1.0, 0.0, -cx_src],
        [0.0, 1.0, -cy_src],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Rotation & Scale: R
    r_mat = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    # Shift to destination center + translation: T_pos
    tx_px = translation_x * out_w
    ty_px = translation_y * out_h
    t_pos = np.array([
        [1.0, 0.0, cx_dst + tx_px],
        [0.0, 1.0, cy_dst + ty_px],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Composite forward matrix: M = T_pos @ R @ T_neg
    m_forward = t_pos @ r_mat @ t_neg
    return m_forward


def transform_normalized_point(
    x: float,
    y: float,
    matrix: np.ndarray,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
) -> Tuple[float, float]:
    """
    Transforms a normalized [0.0, 1.0] coordinate from input frame space
    to output frame space using the 3x3 pixel-space forward matrix.
    """
    px = x * in_w
    py = y * in_h
    vec = np.array([px, py, 1.0], dtype=np.float64)
    res = matrix @ vec
    z = res[2]
    if abs(z) > 1e-12:
        dst_px = res[0] / z
        dst_py = res[1] / z
    else:
        dst_px = res[0]
        dst_py = res[1]

    norm_x = dst_px / out_w if out_w > 0 else 0.0
    norm_y = dst_py / out_h if out_h > 0 else 0.0
    return float(norm_x), float(norm_y)


def transform_bounding_box(
    box: Dict[str, Any],
    matrix: np.ndarray,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    clip: bool = True,
) -> Dict[str, Any]:
    """
    Transforms a 2D bounding box by transforming its 4 corners and taking
    the new axis-aligned bounding envelope in rectified output coordinates.
    """
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))

    # 4 corners in normalized coordinates
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    transformed_corners = [
        transform_normalized_point(cx, cy, matrix, in_w, in_h, out_w, out_h)
        for cx, cy in corners
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
    keypoints: List[Dict[str, Any]],
    matrix: np.ndarray,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    clip: bool = True,
) -> List[Dict[str, Any]]:
    """
    Transforms a list of normalized landmark keypoints.
    Preserves all auxiliary keys (e.g. z, visibility, label) untouched.
    """
    transformed: List[Dict[str, Any]] = []
    for kp in keypoints:
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        nx, ny = transform_normalized_point(x, y, matrix, in_w, in_h, out_w, out_h)
        if clip:
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))

        new_kp = dict(kp)
        new_kp["x"] = round(nx, 4)
        new_kp["y"] = round(ny, 4)
        transformed.append(new_kp)
    return transformed


def warp_image_numpy(
    image: np.ndarray,
    matrix_forward: np.ndarray,
    out_h: int,
    out_w: int,
    border_mode: str = "constant",
    border_value: Union[int, Tuple[int, int, int]] = (0, 0, 0),
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Warps an image array using inverse transformation mapping.
    Uses OpenCV if installed; otherwise executes vectorized pure NumPy inverse mapping.
    """
    # 1. Try fast OpenCV if available
    try:
        import cv2  # type: ignore
        cv_border = cv2.BORDER_CONSTANT
        if border_mode == "replicate":
            cv_border = cv2.BORDER_REPLICATE
        elif border_mode == "reflect":
            cv_border = cv2.BORDER_REFLECT

        cv_interp = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        b_val = border_value if isinstance(border_value, tuple) else (border_value, border_value, border_value)

        # Check if matrix is affine (row 3 is [0, 0, 1])
        if np.allclose(matrix_forward[2, :], [0.0, 0.0, 1.0], atol=1e-6):
            m23 = matrix_forward[:2, :]
            return cv2.warpAffine(
                image, m23, (out_w, out_h), flags=cv_interp, borderMode=cv_border, borderValue=b_val
            )
        else:
            return cv2.warpPerspective(
                image, matrix_forward, (out_w, out_h), flags=cv_interp, borderMode=cv_border, borderValue=b_val
            )
    except ImportError:
        pass

    # 2. Pure NumPy Vectorized Inverse Mapping Fallback
    try:
        m_inv = np.linalg.inv(matrix_forward)
    except np.linalg.LinAlgError:
        # Singular matrix fallback: return zero/blank frame
        if image.ndim == 3:
            return np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
        return np.zeros((out_h, out_w), dtype=image.dtype)

    in_h, in_w = image.shape[:2]

    # Destination coordinate grid
    y_coords, x_coords = np.indices((out_h, out_w), dtype=np.float64)
    ones = np.ones_like(x_coords)
    dest_pts = np.stack([x_coords.ravel(), y_coords.ravel(), ones.ravel()], axis=0)

    # Map to source coordinates
    src_pts = m_inv @ dest_pts
    z = src_pts[2, :]
    z_safe = np.where(np.abs(z) > 1e-12, z, 1.0)
    src_x = (src_pts[0, :] / z_safe).reshape(out_h, out_w)
    src_y = (src_pts[1, :] / z_safe).reshape(out_h, out_w)

    if interpolation == "bilinear":
        # Bilinear interpolation
        x0 = np.floor(src_x).astype(np.int64)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(np.int64)
        y1 = y0 + 1

        wa = ((x1 - src_x) * (y1 - src_y))
        wb = ((x1 - src_x) * (src_y - y0))
        wc = ((src_x - x0) * (y1 - src_y))
        wd = ((src_x - x0) * (src_y - y0))

        valid_mask = (x0 >= 0) & (x1 < in_w) & (y0 >= 0) & (y1 < in_h)

        x0_c = np.clip(x0, 0, in_w - 1)
        x1_c = np.clip(x1, 0, in_w - 1)
        y0_c = np.clip(y0, 0, in_h - 1)
        y1_c = np.clip(y1, 0, in_h - 1)

        if image.ndim == 3:
            num_channels = image.shape[2]
            warped = np.zeros((out_h, out_w, num_channels), dtype=image.dtype)
            b_tuple = border_value if isinstance(border_value, tuple) else (border_value,) * num_channels
            for c in range(num_channels):
                chan = image[:, :, c].astype(np.float64)
                ia = chan[y0_c, x0_c]
                ib = chan[y1_c, x0_c]
                ic = chan[y0_c, x1_c]
                id_val = chan[y1_c, x1_c]
                interp_val = wa * ia + wb * ib + wc * ic + wd * id_val
                fill_v = b_tuple[c] if c < len(b_tuple) else 0
                warped[:, :, c] = np.where(valid_mask, np.clip(interp_val, 0, 255).astype(image.dtype), fill_v)
            return warped
        else:
            img_f = image.astype(np.float64)
            ia = img_f[y0_c, x0_c]
            ib = img_f[y1_c, x0_c]
            ic = img_f[y0_c, x1_c]
            id_val = img_f[y1_c, x1_c]
            interp_val = wa * ia + wb * ib + wc * ic + wd * id_val
            fill_v = border_value if isinstance(border_value, int) else border_value[0]
            return np.where(valid_mask, np.clip(interp_val, 0, 255).astype(image.dtype), fill_v)

    else:
        # Nearest-neighbor interpolation
        src_x_round = np.round(src_x).astype(np.int64)
        src_y_round = np.round(src_y).astype(np.int64)
        valid_mask = (src_x_round >= 0) & (src_x_round < in_w) & (src_y_round >= 0) & (src_y_round < in_h)

        safe_x = np.clip(src_x_round, 0, in_w - 1)
        safe_y = np.clip(src_y_round, 0, in_h - 1)

        if image.ndim == 3:
            num_channels = image.shape[2]
            warped = np.zeros((out_h, out_w, num_channels), dtype=image.dtype)
            b_tuple = border_value if isinstance(border_value, tuple) else (border_value,) * num_channels
            for c in range(num_channels):
                chan = image[:, :, c]
                fill_v = b_tuple[c] if c < len(b_tuple) else 0
                warped[:, :, c] = np.where(valid_mask, chan[safe_y, safe_x], fill_v)
            return warped
        else:
            fill_v = border_value if isinstance(border_value, int) else border_value[0]
            return np.where(valid_mask, image[safe_y, safe_x], fill_v)


# ===========================================================================
# 3. Main CameraRectifier Class
# ===========================================================================

class CameraRectifier:
    """
    Deterministic Camera Rectification and Mounting Calibration Core.
    
    Transforms physical camera frames to standard upright coordinates based on
    known physical mounting parameters.
    """

    def __init__(self, config: Optional[RectificationConfig] = None) -> None:
        self.config = config if config is not None else RectificationConfig()

    def is_identity(self) -> bool:
        """
        Returns True if rectification is disabled or configured as pure identity.
        """
        if not self.config.enabled:
            return True

        if (
            abs(self.config.rotation_deg) < 1e-6
            and abs(self.config.scale - 1.0) < 1e-6
            and abs(self.config.translation_x) < 1e-6
            and abs(self.config.translation_y) < 1e-6
            and self.config.affine_matrix is None
            and self.config.homography_matrix is None
            and self.config.output_size is None
            and self.config.center_offset is None
        ):
            return True

        return False

    def get_forward_matrix(
        self,
        in_w: int,
        in_h: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computes the 3x3 pixel-space forward transformation matrix.
        """
        effective_out_w = out_w if out_w is not None else (
            self.config.output_size[0] if self.config.output_size else in_w
        )
        effective_out_h = out_h if out_h is not None else (
            self.config.output_size[1] if self.config.output_size else in_h
        )

        return build_rectification_matrix(
            in_w=in_w,
            in_h=in_h,
            out_w=effective_out_w,
            out_h=effective_out_h,
            rotation_deg=self.config.rotation_deg,
            scale=self.config.scale,
            translation_x=self.config.translation_x,
            translation_y=self.config.translation_y,
            center_offset=self.config.center_offset,
            affine_matrix=self.config.affine_matrix,
            homography_matrix=self.config.homography_matrix,
        )

    def get_inverse_matrix(
        self,
        in_w: int,
        in_h: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computes the 3x3 pixel-space inverse transformation matrix.
        """
        m_fwd = self.get_forward_matrix(in_w, in_h, out_w, out_h)
        try:
            return np.linalg.inv(m_fwd)
        except np.linalg.LinAlgError:
            return np.eye(3, dtype=np.float64)

    def rectify(
        self,
        image: np.ndarray,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        keypoints: Optional[List[Dict[str, Any]]] = None,
    ) -> RectificationResult:
        """
        Performs full rectification on an image and optional annotations.
        
        If disabled or identity, returns the original image and annotations (no-op).
        """
        in_h, in_w = image.shape[:2]
        out_w = self.config.output_size[0] if self.config.output_size else in_w
        out_h = self.config.output_size[1] if self.config.output_size else in_h

        # Identity / No-op bypass
        if self.is_identity():
            return RectificationResult(
                image=image.copy(),
                bounding_boxes=[dict(b) for b in bounding_boxes] if bounding_boxes else [],
                keypoints=[dict(kp) for kp in keypoints] if keypoints else [],
                transform_matrix=np.eye(3, dtype=np.float64),
                inverse_matrix=np.eye(3, dtype=np.float64),
                applied=False,
                input_shape=(in_h, in_w),
                output_shape=(in_h, in_w),
                metadata={
                    "calibration_status": self.config.calibration_status,
                    "calibration_notes": self.config.calibration_notes,
                    "enabled": False,
                },
            )

        # Compute forward matrix
        m_fwd = self.get_forward_matrix(in_w, in_h, out_w, out_h)
        try:
            m_inv = np.linalg.inv(m_fwd)
        except np.linalg.LinAlgError:
            m_inv = np.eye(3, dtype=np.float64)

        # Warp image
        warped_img = warp_image_numpy(
            image=image,
            matrix_forward=m_fwd,
            out_h=out_h,
            out_w=out_w,
            border_mode=self.config.border_mode,
            border_value=self.config.border_value,
            interpolation=self.config.interpolation,
        )

        # Transform Bounding Boxes
        rectified_boxes: List[Dict[str, Any]] = []
        if bounding_boxes:
            for box in bounding_boxes:
                rectified_boxes.append(
                    transform_bounding_box(box, m_fwd, in_w, in_h, out_w, out_h, clip=True)
                )

        # Transform Keypoints
        rectified_keypoints: List[Dict[str, Any]] = []
        if keypoints:
            rectified_keypoints = transform_keypoints(
                keypoints, m_fwd, in_w, in_h, out_w, out_h, clip=True
            )

        return RectificationResult(
            image=warped_img,
            bounding_boxes=rectified_boxes,
            keypoints=rectified_keypoints,
            transform_matrix=m_fwd,
            inverse_matrix=m_inv,
            applied=True,
            input_shape=(in_h, in_w),
            output_shape=(out_h, out_w),
            metadata={
                "calibration_status": self.config.calibration_status,
                "calibration_notes": self.config.calibration_notes,
                "rotation_deg": self.config.rotation_deg,
                "scale": self.config.scale,
                "enabled": True,
            },
        )

    def rectify_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper returning only the rectified frame array.
        """
        return self.rectify(frame).image

    def transform_bounding_boxes(
        self,
        boxes: List[Dict[str, Any]],
        in_w: int,
        in_h: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        matrix: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Standalone method to transform a list of normalized bounding boxes.
        """
        if self.is_identity() and matrix is None:
            return [dict(b) for b in boxes]

        eff_out_w = out_w if out_w is not None else in_w
        eff_out_h = out_h if out_h is not None else in_h
        m = matrix if matrix is not None else self.get_forward_matrix(in_w, in_h, eff_out_w, eff_out_h)

        return [
            transform_bounding_box(b, m, in_w, in_h, eff_out_w, eff_out_h, clip=True)
            for b in boxes
        ]

    def transform_keypoints(
        self,
        keypoints: List[Dict[str, Any]],
        in_w: int,
        in_h: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        matrix: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Standalone method to transform a list of normalized landmark keypoints.
        """
        if self.is_identity() and matrix is None:
            return [dict(kp) for kp in keypoints]

        eff_out_w = out_w if out_w is not None else in_w
        eff_out_h = out_h if out_h is not None else in_h
        m = matrix if matrix is not None else self.get_forward_matrix(in_w, in_h, eff_out_w, eff_out_h)

        return transform_keypoints(keypoints, m, in_w, in_h, eff_out_w, eff_out_h, clip=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Exports configuration to a serializable dictionary.
        """
        affine_list = (
            self.config.affine_matrix.tolist()
            if isinstance(self.config.affine_matrix, np.ndarray)
            else self.config.affine_matrix
        )
        homography_list = (
            self.config.homography_matrix.tolist()
            if isinstance(self.config.homography_matrix, np.ndarray)
            else self.config.homography_matrix
        )

        return {
            "enabled": self.config.enabled,
            "rotation_deg": self.config.rotation_deg,
            "scale": self.config.scale,
            "translation_x": self.config.translation_x,
            "translation_y": self.config.translation_y,
            "affine_matrix": affine_list,
            "homography_matrix": homography_list,
            "output_size": list(self.config.output_size) if self.config.output_size else None,
            "center_offset": list(self.config.center_offset) if self.config.center_offset else None,
            "border_mode": self.config.border_mode,
            "border_value": self.config.border_value,
            "interpolation": self.config.interpolation,
            "calibration_status": self.config.calibration_status,
            "calibration_notes": self.config.calibration_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraRectifier:
        """
        Instantiates a CameraRectifier from a configuration dictionary.
        """
        out_size = tuple(data["output_size"]) if data.get("output_size") else None
        center_off = tuple(data["center_offset"]) if data.get("center_offset") else None

        config = RectificationConfig(
            enabled=data.get("enabled", False),
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            scale=float(data.get("scale", 1.0)),
            translation_x=float(data.get("translation_x", 0.0)),
            translation_y=float(data.get("translation_y", 0.0)),
            affine_matrix=data.get("affine_matrix"),
            homography_matrix=data.get("homography_matrix"),
            output_size=out_size,
            center_offset=center_off,
            border_mode=data.get("border_mode", "constant"),
            border_value=data.get("border_value", (0, 0, 0)),
            interpolation=data.get("interpolation", "bilinear"),
            calibration_status=data.get("calibration_status", "UNSPECIFIED"),
            calibration_notes=data.get(
                "calibration_notes", "PROPOSED DEFAULT — REQUIRES EXPERIMENTAL CALIBRATION"
            ),
        )
        return cls(config)

    @classmethod
    def from_settings(cls, settings_dict: Dict[str, Any]) -> CameraRectifier:
        """
        Instantiates CameraRectifier from a general settings dictionary (e.g. settings.json).
        If 'rectification' section is absent, defaults to identity / disabled rectifier.
        """
        rect_data = settings_dict.get("rectification", {})
        return cls.from_dict(rect_data)

    @classmethod
    def from_config_file(cls, filepath: Union[str, Path]) -> CameraRectifier:
        """
        Loads rectification configuration from a JSON configuration file.
        """
        p = Path(filepath)
        if not p.exists():
            # Graceful fallback to default disabled rectifier
            return cls(RectificationConfig())
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_settings(data) if "rectification" in data else cls.from_dict(data)
        except Exception:
            return cls(RectificationConfig())
