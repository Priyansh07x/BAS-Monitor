"""
synthetic_hmr.py — Zero-Dependency Synthetic 3D Human Mesh Recovery Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.1)

Generates mathematically valid, deterministic 3D human skeletal joint structures
and proxy volumetric surface meshes without external models, neural weights,
or proprietary SMPL .pkl assets.

Designed for headless CI testing, offline verification, microgravity posture
simulation, and graceful fallback when real HMR backends are unavailable.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.ai.hmr.hmr_interface import HMRRecoveryEngineInterface
from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
    MeshVertex3D,
)

# Standard 24-joint kinematic skeleton topology (aligned with standard SMPL joint ordering)
SMPL_24_JOINT_NAMES: List[str] = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
    "left_hand",       # 22
    "right_hand",      # 23
]

SMPL_24_PARENT_MAP: Dict[int, Optional[int]] = {
    0: None,  # Pelvis (Root)
    1: 0,     # Left Hip -> Pelvis
    2: 0,     # Right Hip -> Pelvis
    3: 0,     # Spine1 -> Pelvis
    4: 1,     # Left Knee -> Left Hip
    5: 2,     # Right Knee -> Right Hip
    6: 3,     # Spine2 -> Spine1
    7: 4,     # Left Ankle -> Left Knee
    8: 5,     # Right Ankle -> Right Knee
    9: 6,     # Spine3 -> Spine2
    10: 7,    # Left Foot -> Left Ankle
    11: 8,    # Right Foot -> Right Ankle
    12: 9,    # Neck -> Spine3
    13: 9,    # Left Collar -> Spine3
    14: 9,    # Right Collar -> Spine3
    15: 12,   # Head -> Neck
    16: 13,   # Left Shoulder -> Left Collar
    17: 14,   # Right Shoulder -> Right Collar
    18: 16,   # Left Elbow -> Left Shoulder
    19: 17,   # Right Elbow -> Right Shoulder
    20: 18,   # Left Wrist -> Left Elbow
    21: 19,   # Right Wrist -> Right Elbow
    22: 20,   # Left Hand -> Left Wrist
    23: 21,   # Right Hand -> Right Wrist
}


def _build_static_proxy_faces() -> List[List[int]]:
    faces: List[List[int]] = []
    for box_idx in range(6):
        b = box_idx * 8
        faces.extend([
            [b + 0, b + 1, b + 2],
            [b + 0, b + 2, b + 3],
            [b + 5, b + 4, b + 7],
            [b + 5, b + 7, b + 6],
            [b + 4, b + 0, b + 3],
            [b + 4, b + 3, b + 7],
            [b + 1, b + 5, b + 6],
            [b + 1, b + 6, b + 2],
            [b + 3, b + 2, b + 6],
            [b + 3, b + 6, b + 7],
            [b + 4, b + 5, b + 1],
            [b + 4, b + 1, b + 0],
        ])
    return faces

STATIC_PROXY_FACES: List[List[int]] = _build_static_proxy_faces()


class SyntheticHMREngine(HMRRecoveryEngineInterface):
    """
    Zero-dependency deterministic 3D Human Mesh Recovery engine.
    
    Generates a 24-joint human kinematic skeleton and a 48-vertex, 72-face
    proxy surface mesh in metric camera-canonical coordinates.
    
    Coordinate System ('synthetic_camera_canonical'):
      - Metric coordinates in meters.
      - Origin: Pelvis root joint located at (0.0, 0.0, 1.5) in front of camera.
      - +X: Subject's left (viewer's right).
      - +Y: Upward / cranial direction.
      - +Z: Depth pointing away from camera.
    """

    def __init__(
        self,
        base_depth_m: float = 1.5,
        enable_mesh_generation: bool = True,
        posture_mode: str = "neutral_microgravity",
    ) -> None:
        self.base_depth_m = float(base_depth_m)
        self.enable_mesh_generation = bool(enable_mesh_generation)
        self.posture_mode = str(posture_mode)
        self._is_closed = False

    def recover_mesh(self, frame: np.ndarray) -> HMRMeshResult:
        if self._is_closed:
            return HMRMeshResult.empty(
                reason="engine_closed",
                coordinate_frame="synthetic_camera_canonical",
            )

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return HMRMeshResult.empty(
                reason="invalid_or_empty_frame",
                coordinate_frame="synthetic_camera_canonical",
            )

        try:
            h, w = frame.shape[:2]
            mean_val = float(np.mean(frame)) if frame.size > 0 else 128.0
            phase = (mean_val / 255.0) * math.pi * 2.0

            joints = self._generate_kinematic_joints(phase)

            vertices: List[MeshVertex3D] = []
            faces: List[List[int]] = []
            if self.enable_mesh_generation:
                vertices, faces = self._generate_proxy_mesh(joints)

            synthetic_beta = [0.0] * 10
            synthetic_theta = [0.0] * 72

            metadata = {
                "engine": "SyntheticHMREngine",
                "is_synthetic": True,
                "posture_mode": self.posture_mode,
                "joint_count": len(joints),
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "frame_shape": [h, w],
                "deterministic_phase": round(phase, 4),
                "timestamp": round(time.time(), 4),
            }

            return HMRMeshResult(
                joints=joints,
                vertices=vertices,
                faces=faces,
                beta=synthetic_beta,
                theta=synthetic_theta,
                mesh_present=len(vertices) > 0,
                coordinate_frame="synthetic_camera_canonical",
                representation_type="synthetic_kinematic_mesh" if len(vertices) > 0 else "synthetic_joints_only",
                metadata=metadata,
            )

        except Exception as e:
            return HMRMeshResult.empty(
                reason=f"synthesis_error: {str(e)}",
                coordinate_frame="synthetic_camera_canonical",
                metadata={"error": str(e)},
            )

    def _generate_kinematic_joints(self, phase: float) -> List[Joint3D]:
        z0 = self.base_depth_m
        dyn_y = 0.015 * math.sin(phase)
        dyn_reach = 0.025 * math.cos(phase)

        coords: Dict[int, Tuple[float, float, float]] = {
            0: (0.00, 0.00 + dyn_y, z0),
            1: (-0.09, -0.05 + dyn_y, z0),
            2: (0.09, -0.05 + dyn_y, z0),
            3: (0.00, 0.12 + dyn_y, z0 - 0.01),
            4: (-0.10, -0.42 + dyn_y, z0 + 0.05),
            5: (0.10, -0.42 + dyn_y, z0 + 0.05),
            6: (0.00, 0.26 + dyn_y, z0 - 0.02),
            7: (-0.10, -0.78 + dyn_y, z0 + 0.02),
            8: (0.10, -0.78 + dyn_y, z0 + 0.02),
            9: (0.00, 0.40 + dyn_y, z0 - 0.01),
            10: (-0.10, -0.83 + dyn_y, z0 - 0.08),
            11: (0.10, -0.83 + dyn_y, z0 - 0.08),
            12: (0.00, 0.50 + dyn_y, z0),
            13: (-0.07, 0.42 + dyn_y, z0),
            14: (0.07, 0.42 + dyn_y, z0),
            15: (0.00, 0.65 + dyn_y, z0 + 0.02),
            16: (-0.18, 0.42 + dyn_y, z0),
            17: (0.18, 0.42 + dyn_y, z0),
            18: (-0.28, 0.18 + dyn_y, z0 - 0.05),
            19: (0.28, 0.18 + dyn_y, z0 - 0.05),
            20: (-0.32, -0.02 + dyn_y + dyn_reach, z0 - 0.12),
            21: (0.32, -0.02 + dyn_y - dyn_reach, z0 - 0.12),
            22: (-0.34, -0.10 + dyn_y + dyn_reach, z0 - 0.15),
            23: (0.34, -0.10 + dyn_y - dyn_reach, z0 - 0.15),
        }

        joints: List[Joint3D] = []
        for idx in range(24):
            x, y, z = coords[idx]
            name = SMPL_24_JOINT_NAMES[idx]
            joints.append(
                Joint3D(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    visibility=1.0,
                    confidence=0.98,
                    name=name,
                )
            )
        return joints

    def _generate_proxy_mesh(
        self,
        joints: List[Joint3D],
    ) -> Tuple[List[MeshVertex3D], List[List[int]]]:
        def add_box(
            min_pt: Tuple[float, float, float],
            max_pt: Tuple[float, float, float],
            verts: List[MeshVertex3D],
        ) -> None:
            x0, y0, z0 = min_pt
            x1, y1, z1 = max_pt
            verts.extend([
                MeshVertex3D(x0, y0, z0),
                MeshVertex3D(x1, y0, z0),
                MeshVertex3D(x1, y1, z0),
                MeshVertex3D(x0, y1, z0),
                MeshVertex3D(x0, y0, z1),
                MeshVertex3D(x1, y0, z1),
                MeshVertex3D(x1, y1, z1),
                MeshVertex3D(x0, y1, z1),
            ])

        vertices: List[MeshVertex3D] = []
        p, s3 = joints[0], joints[9]
        add_box((p.x - 0.16, p.y - 0.08, p.z - 0.10), (p.x + 0.16, s3.y + 0.08, p.z + 0.10), vertices)
        h = joints[15]
        add_box((h.x - 0.10, h.y - 0.10, h.z - 0.10), (h.x + 0.10, h.y + 0.12, h.z + 0.10), vertices)
        ls, lh = joints[16], joints[22]
        add_box((min(ls.x, lh.x) - 0.05, min(ls.y, lh.y) - 0.05, min(ls.z, lh.z) - 0.05), (max(ls.x, lh.x) + 0.05, max(ls.y, lh.y) + 0.05, max(ls.z, lh.z) + 0.05), vertices)
        rs, rh = joints[17], joints[23]
        add_box((min(rs.x, rh.x) - 0.05, min(rs.y, rh.y) - 0.05, min(rs.z, rh.z) - 0.05), (max(rs.x, rh.x) + 0.05, max(rs.y, rh.y) + 0.05, max(rs.z, rh.z) + 0.05), vertices)
        lhip, lfoot = joints[1], joints[10]
        add_box((min(lhip.x, lfoot.x) - 0.06, min(lhip.y, lfoot.y) - 0.05, min(lhip.z, lfoot.z) - 0.06), (max(lhip.x, lfoot.x) + 0.06, max(lhip.y, lfoot.y) + 0.05, max(lhip.z, lfoot.z) + 0.06), vertices)
        rhip, rfoot = joints[2], joints[11]
        add_box((min(rhip.x, rfoot.x) - 0.06, min(rhip.y, rfoot.y) - 0.05, min(rhip.z, rfoot.z) - 0.06), (max(rhip.x, rfoot.x) + 0.06, max(rhip.y, rfoot.y) + 0.05, max(rhip.z, rfoot.z) + 0.06), vertices)

        return vertices, STATIC_PROXY_FACES

    def close(self) -> None:
        """Closes the synthetic engine."""
        self._is_closed = True
