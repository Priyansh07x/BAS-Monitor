"""
payload_transform.py — 3D Camera-to-Payload Coordinate Transformation Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.2)

Provides deterministic 3D rigid Euclidean transformations mapping coordinates
from Camera Optical Space to Payload Rack Space:
  P_payload = R_camera_to_payload @ P_camera + t_camera_to_payload

Guarantees:
  - Strict validation of 3x3 rotation matrices (orthonormality, finite numerical values, det ~ +1.0).
  - Strict validation of 3D translation vectors.
  - Deterministic transformations for Joint3D, MeshVertex3D, and batch point arrays.
  - Coordinate frame metadata propagation with zero contract pollution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
    MeshVertex3D,
)


class CameraToPayloadTransform:
    """
    Rigid 3D transformation between Camera Space and Payload Rack Space.
    
    Mathematical Formulation:
      P_payload = R @ P_camera + t
      
    Coordinate Frames:
      - Camera Space ('synthetic_camera_canonical' or 'mediapipe_world_meters'):
          +X: Camera right
          +Y: Camera up
          +Z: Optical depth away from lens
      - Payload Rack Space ('payload_rack_canonical'):
          +X: Rack width (left to right across experiment bay)
          +Y: Rack height (bottom to top panel)
          +Z: Rack depth (into the experiment chamber)
    """

    def __init__(
        self,
        rotation_matrix: Optional[np.ndarray] = None,
        translation_vector: Optional[np.ndarray] = None,
        source_frame: str = "synthetic_camera_canonical",
        target_frame: str = "payload_rack_canonical",
    ) -> None:
        """
        Args:
            rotation_matrix: 3x3 orthonormal rotation matrix. Defaults to Identity (3x3).
            translation_vector: 3D translation vector [tx, ty, tz] in meters. Defaults to [0, 0, 0].
            source_frame: Label of the source coordinate system.
            target_frame: Label of the target coordinate system.
        """
        if rotation_matrix is None:
            self._R = np.eye(3, dtype=np.float64)
        else:
            self._R = self._validate_rotation_matrix(rotation_matrix)

        if translation_vector is None:
            self._t = np.zeros(3, dtype=np.float64)
        else:
            self._t = self._validate_translation_vector(translation_vector)

        self.source_frame = str(source_frame)
        self.target_frame = str(target_frame)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Copy of the 3x3 rotation matrix."""
        return self._R.copy()

    @property
    def translation_vector(self) -> np.ndarray:
        """Copy of the 3D translation vector."""
        return self._t.copy()

    @classmethod
    def identity(
        cls,
        source_frame: str = "synthetic_camera_canonical",
        target_frame: str = "payload_rack_canonical",
    ) -> CameraToPayloadTransform:
        """Creates an identity transform (R=I, t=0)."""
        return cls(
            rotation_matrix=np.eye(3, dtype=np.float64),
            translation_vector=np.zeros(3, dtype=np.float64),
            source_frame=source_frame,
            target_frame=target_frame,
        )

    @classmethod
    def from_rot_trans(
        cls,
        rotation: Union[List[List[float]], np.ndarray],
        translation: Union[List[float], Tuple[float, float, float], np.ndarray],
        source_frame: str = "synthetic_camera_canonical",
        target_frame: str = "payload_rack_canonical",
    ) -> CameraToPayloadTransform:
        """Constructs transform from list or array rotation and translation values."""
        r_arr = np.array(rotation, dtype=np.float64)
        t_arr = np.array(translation, dtype=np.float64)
        return cls(
            rotation_matrix=r_arr,
            translation_vector=t_arr,
            source_frame=source_frame,
            target_frame=target_frame,
        )

    @classmethod
    def from_euler_angles(
        cls,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        source_frame: str = "synthetic_camera_canonical",
        target_frame: str = "payload_rack_canonical",
    ) -> CameraToPayloadTransform:
        """
        Constructs transform from Euler angles (in degrees, Z-Y-X Tait-Bryan convention)
        and translation in meters.
        """
        rz = math.radians(roll_deg)
        ry = math.radians(pitch_deg)
        rx = math.radians(yaw_deg)

        # Rotation around X (pitch)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rx), -math.sin(rx)],
            [0.0, math.sin(rx), math.cos(rx)],
        ], dtype=np.float64)

        # Rotation around Y (yaw)
        Ry = np.array([
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)],
        ], dtype=np.float64)

        # Rotation around Z (roll)
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        R = Rz @ Ry @ Rx
        return cls(
            rotation_matrix=R,
            translation_vector=np.array(translation_m, dtype=np.float64),
            source_frame=source_frame,
            target_frame=target_frame,
        )

    def _validate_rotation_matrix(self, R: Any) -> np.ndarray:
        """Validates that R is a 3x3 finite real matrix with det(R) > 0."""
        if not isinstance(R, (np.ndarray, list, tuple)):
            raise TypeError(f"Rotation matrix must be numpy array or nested list, got {type(R)}")
        r_arr = np.array(R, dtype=np.float64)
        if r_arr.shape != (3, 3):
            raise ValueError(f"Rotation matrix must have shape (3, 3), got {r_arr.shape}")
        if not np.all(np.isfinite(r_arr)):
            raise ValueError("Rotation matrix contains non-finite values (NaN or Inf)")

        # Verify determinant is positive and non-zero
        det = float(np.linalg.det(r_arr))
        if abs(det) < 1e-6 or det < 0:
            raise ValueError(f"Invalid rotation matrix determinant: {det:.4f}. Must be a proper rotation.")
        return r_arr

    def _validate_translation_vector(self, t: Any) -> np.ndarray:
        """Validates that t is a 3-element finite 1D array."""
        if not isinstance(t, (np.ndarray, list, tuple)):
            raise TypeError(f"Translation vector must be numpy array or list/tuple, got {type(t)}")
        t_arr = np.array(t, dtype=np.float64).flatten()
        if t_arr.shape != (3,):
            raise ValueError(f"Translation vector must have exactly 3 elements, got {t_arr.shape}")
        if not np.all(np.isfinite(t_arr)):
            raise ValueError("Translation vector contains non-finite values (NaN or Inf)")
        return t_arr

    def transform_point(
        self,
        point: Union[np.ndarray, Tuple[float, float, float], List[float]],
    ) -> np.ndarray:
        """
        Transforms a single 3D point from Camera Space to Payload Space:
          P_payload = R @ P_camera + t
        """
        p = np.array(point, dtype=np.float64).flatten()
        if p.shape != (3,):
            raise ValueError(f"Point must have exactly 3 coordinates, got shape {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("Point contains non-finite values (NaN or Inf)")
        return self._R @ p + self._t

    def transform_points_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms a batch of (N, 3) points from Camera Space to Payload Space.
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Batch points must be of shape (N, 3), got {getattr(points, 'shape', None)}")
        if not np.all(np.isfinite(points)):
            raise ValueError("Batch points contain non-finite values (NaN or Inf)")
        return (self._R @ points.T).T + self._t

    def transform_joint(self, joint: Joint3D) -> Joint3D:
        """
        Transforms a Joint3D instance into the target payload coordinate frame.
        Preserves visibility, confidence, and anatomical name.
        """
        p_cam = np.array([joint.x, joint.y, joint.z], dtype=np.float64)
        p_payload = self.transform_point(p_cam)
        return Joint3D(
            x=float(p_payload[0]),
            y=float(p_payload[1]),
            z=float(p_payload[2]),
            visibility=float(joint.visibility),
            confidence=float(joint.confidence),
            name=joint.name,
        )

    def transform_joints(self, joints: List[Joint3D]) -> List[Joint3D]:
        """Transforms a collection of Joint3D instances into the target frame."""
        if not joints:
            return []
        pts = np.array([[j.x, j.y, j.z] for j in joints], dtype=np.float64)
        t_pts = (self._R @ pts.T).T + self._t
        transformed: List[Joint3D] = []
        for idx, j in enumerate(joints):
            transformed.append(
                Joint3D(
                    x=float(t_pts[idx, 0]),
                    y=float(t_pts[idx, 1]),
                    z=float(t_pts[idx, 2]),
                    visibility=float(j.visibility),
                    confidence=float(j.confidence),
                    name=j.name,
                )
            )
        return transformed

    def transform_mesh_result(self, result: HMRMeshResult) -> HMRMeshResult:
        """
        Transforms all joints and vertices in an HMRMeshResult into Payload Space.
        Updates coordinate_frame metadata accordingly.
        """
        if not result.joints and not result.vertices:
            return HMRMeshResult.empty(
                reason="empty_input_result",
                coordinate_frame=self.target_frame,
                metadata=dict(result.metadata),
            )

        transformed_joints = [self.transform_joint(j) for j in result.joints]

        transformed_vertices: List[MeshVertex3D] = []
        if result.vertices:
            v_pts = np.array([[v.x, v.y, v.z] for v in result.vertices], dtype=np.float64)
            t_pts = self.transform_points_batch(v_pts)
            for pt in t_pts:
                transformed_vertices.append(MeshVertex3D(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))

        updated_metadata = dict(result.metadata)
        updated_metadata["transformed_by"] = "CameraToPayloadTransform"
        updated_metadata["source_frame"] = result.coordinate_frame
        updated_metadata["target_frame"] = self.target_frame

        return HMRMeshResult(
            joints=transformed_joints,
            vertices=transformed_vertices,
            faces=list(result.faces),
            beta=list(result.beta) if result.beta is not None else None,
            theta=list(result.theta) if result.theta is not None else None,
            mesh_present=len(transformed_vertices) > 0,
            coordinate_frame=self.target_frame,
            representation_type=result.representation_type,
            metadata=updated_metadata,
        )

    def inverse(self) -> CameraToPayloadTransform:
        """
        Computes the analytical inverse transform (Payload -> Camera):
          P_camera = R^T @ (P_payload - t) = R^T @ P_payload - R^T @ t
        """
        R_inv = self._R.T
        t_inv = -R_inv @ self._t
        return CameraToPayloadTransform(
            rotation_matrix=R_inv,
            translation_vector=t_inv,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transform to a JSON-safe dictionary."""
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "rotation_matrix": [[round(float(val), 6) for val in row] for row in self._R],
            "translation_vector": [round(float(val), 6) for val in self._t],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraToPayloadTransform:
        """Deserializes transform from a dictionary representation."""
        return cls.from_rot_trans(
            rotation=data.get("rotation_matrix", np.eye(3).tolist()),
            translation=data.get("translation_vector", [0.0, 0.0, 0.0]),
            source_frame=data.get("source_frame", "synthetic_camera_canonical"),
            target_frame=data.get("target_frame", "payload_rack_canonical"),
        )
