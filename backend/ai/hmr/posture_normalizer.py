"""
posture_normalizer.py — Microgravity Posture Normalization & Roll-Invariance Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.2)

Provides geometric body-frame normalization for free-floating astronauts in microgravity.
Eliminates dependence on camera orientation, roll angles, or terrestrial floor priors.

Mathematical Formulation:
  1. Pelvis Root Centering: P_rel = P - P_pelvis
  2. Cranial Torso Axis: y_body = normalize(P_neck - P_pelvis)
  3. Coronal Shoulder Axis: v_shoulder = P_right_shoulder - P_left_shoulder
  4. Sagittal Depth Axis: z_body = normalize(v_shoulder x y_body)
  5. Transverse Axis: x_body = normalize(y_body x z_body)
  6. Body Rotation: R_body = [x_body, y_body, z_body]^T
  7. Normalized Joint: P_norm = R_body @ (P - P_pelvis)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
)


class MicrogravityPostureNormalizer:
    """
    Normalizes 3D skeletal joint coordinates into a standardized, body-centric
    coordinate frame independent of camera roll, pitch, yaw, or station inversion.
    """

    @staticmethod
    def _find_joint(joints: List[Joint3D], names: List[str]) -> Optional[Joint3D]:
        """Finds the first matching joint by name from a list of candidate names."""
        for name in names:
            for j in joints:
                if j.name and j.name.lower() == name.lower():
                    return j
        return None

    @classmethod
    def compute_body_basis(
        cls,
        joints: List[Joint3D],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the 3x3 orthonormal body rotation matrix and pelvis root translation.
        
        Returns:
            (R_body, p_pelvis)
            where R_body maps camera coordinates to body coordinates:
            P_body = R_body @ (P_cam - p_pelvis)
        """
        if not joints:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        # 1. Identify Pelvis Root
        pelvis = cls._find_joint(joints, ["pelvis", "left_hip", "right_hip"])
        if pelvis is None:
            pelvis = joints[0]  # Fallback to first joint if names missing
        p_pelvis = np.array([pelvis.x, pelvis.y, pelvis.z], dtype=np.float64)

        # 2. Identify Upper Torso (Neck or Spine3)
        neck = cls._find_joint(joints, ["neck", "spine3", "spine2", "nose", "head"])
        if neck is not None:
            p_neck = np.array([neck.x, neck.y, neck.z], dtype=np.float64)
        else:
            p_neck = p_pelvis + np.array([0.0, 0.5, 0.0], dtype=np.float64)

        # 3. Identify Shoulders
        r_sh = cls._find_joint(joints, ["right_shoulder", "right_collar"])
        l_sh = cls._find_joint(joints, ["left_shoulder", "left_collar"])

        if r_sh is not None and l_sh is not None:
            p_rsh = np.array([r_sh.x, r_sh.y, r_sh.z], dtype=np.float64)
            p_lsh = np.array([l_sh.x, l_sh.y, l_sh.z], dtype=np.float64)
            v_shoulder = p_rsh - p_lsh
        else:
            v_shoulder = np.array([0.4, 0.0, 0.0], dtype=np.float64)

        # Compute Orthonormal Body Frame
        v_torso = p_neck - p_pelvis
        torso_norm = float(np.linalg.norm(v_torso))
        if torso_norm > 1e-6:
            y_body = v_torso / torso_norm
        else:
            y_body = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        z_raw = np.cross(v_shoulder, y_body)
        z_norm = float(np.linalg.norm(z_raw))
        if z_norm > 1e-6:
            z_body = z_raw / z_norm
        else:
            z_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        x_body = np.cross(y_body, z_body)
        x_norm = float(np.linalg.norm(x_body))
        if x_norm > 1e-6:
            x_body = x_body / x_norm
        else:
            x_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # R_body has rows as basis vectors
        R_body = np.vstack([x_body, y_body, z_body])
        return R_body, p_pelvis

    @classmethod
    def normalize_posture(
        cls,
        joints: List[Joint3D],
        target_frame: str = "body_canonical_microgravity",
    ) -> List[Joint3D]:
        """
        Transforms all joints into the standardized body coordinate frame:
          - Pelvis positioned at origin (0, 0, 0).
          - Torso oriented along +Y.
          - Shoulders aligned along +X.
          - Anterior chest facing +Z.
        """
        if not joints:
            return []

        R_body, p_pelvis = cls.compute_body_basis(joints)
        pts = np.array([[j.x, j.y, j.z] for j in joints], dtype=np.float64)
        p_norms = (R_body @ (pts - p_pelvis).T).T

        normalized_joints: List[Joint3D] = []
        for idx, j in enumerate(joints):
            normalized_joints.append(
                Joint3D(
                    x=float(p_norms[idx, 0]),
                    y=float(p_norms[idx, 1]),
                    z=float(p_norms[idx, 2]),
                    visibility=float(j.visibility),
                    confidence=float(j.confidence),
                    name=j.name,
                )
            )

        return normalized_joints

    @classmethod
    def compute_torso_relative_wrist_vectors(
        cls,
        joints: List[Joint3D],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Computes invariant 3D reach vectors from the torso center to the left and right wrists.
        """
        if not joints:
            return {}

        R_body, p_pelvis = cls.compute_body_basis(joints)
        r_wrist = cls._find_joint(joints, ["right_wrist", "right_hand"])
        l_wrist = cls._find_joint(joints, ["left_wrist", "left_hand"])
        vectors: Dict[str, Tuple[float, float, float]] = {}

        if r_wrist is not None:
            p_r = R_body @ (np.array([r_wrist.x, r_wrist.y, r_wrist.z], dtype=np.float64) - p_pelvis)
            vectors["right_wrist"] = (round(float(p_r[0]), 5), round(float(p_r[1]), 5), round(float(p_r[2]), 5))
        if l_wrist is not None:
            p_l = R_body @ (np.array([l_wrist.x, l_wrist.y, l_wrist.z], dtype=np.float64) - p_pelvis)
            vectors["left_wrist"] = (round(float(p_l[0]), 5), round(float(p_l[1]), 5), round(float(p_l[2]), 5))

        return vectors

