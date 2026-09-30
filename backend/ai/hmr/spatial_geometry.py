"""
spatial_geometry.py — 3D Spatial Geometry & Volumetric Reach Reasoning
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.2)

Provides geometric abstractions for 3D payload object volumes, metric reach estimation,
point-in-volume containment, and hand-to-target spatial relationships.

Strictly preserves metric/non-metric safety and canonical EXP-001 object definitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
)


@dataclass(frozen=True)
class BoundingVolume3D:
    """
    Axis-aligned 3D bounding volume for payload objects and containment racks.
    
    Fields:
      name: Canonical object identifier (e.g., 'RED_SAMPLE', 'SAMPLE_CONTAINER').
      center: (cx, cy, cz) center coordinates.
      dimensions: (dx, dy, dz) width (X), height (Y), depth (Z) bounding extents.
      coordinate_frame: Coordinate frame tag (default 'payload_rack_canonical').
      is_metric: True if dimensions and center are in meters.
    """
    name: str
    center: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    coordinate_frame: str = "payload_rack_canonical"
    is_metric: bool = True

    @property
    def min_point(self) -> Tuple[float, float, float]:
        """Calculates the minimum (x, y, z) corner of the bounding box."""
        cx, cy, cz = self.center
        dx, dy, dz = self.dimensions
        return (cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0)

    @property
    def max_point(self) -> Tuple[float, float, float]:
        """Calculates the maximum (x, y, z) corner of the bounding box."""
        cx, cy, cz = self.center
        dx, dy, dz = self.dimensions
        return (cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0)

    def contains_point(self, pt: Union[Tuple[float, float, float], List[float], np.ndarray]) -> bool:
        """Determines if a 3D point lies strictly inside or on the boundary of the volume."""
        px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
        x0, y0, z0 = self.min_point
        x1, y1, z1 = self.max_point
        return (x0 <= px <= x1) and (y0 <= py <= y1) and (z0 <= pz <= z1)

    def distance_to_point(self, pt: Union[Tuple[float, float, float], List[float], np.ndarray]) -> float:
        """
        Computes the Euclidean distance from a 3D point to the nearest surface of the box.
        Returns 0.0 if the point is inside the volume.
        """
        px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
        x0, y0, z0 = self.min_point
        x1, y1, z1 = self.max_point

        dx = max(x0 - px, 0.0, px - x1)
        dy = max(y0 - py, 0.0, py - y1)
        dz = max(z0 - pz, 0.0, pz - z1)

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def intersects_sphere(
        self,
        center: Union[Tuple[float, float, float], List[float]],
        radius: float,
    ) -> bool:
        """Checks if a sphere (e.g. hand reach radius) intersects this volume."""
        return self.distance_to_point(center) <= float(radius)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes bounding volume to dictionary."""
        return {
            "name": self.name,
            "center": [round(float(c), 5) for c in self.center],
            "dimensions": [round(float(d), 5) for d in self.dimensions],
            "coordinate_frame": self.coordinate_frame,
            "is_metric": self.is_metric,
        }


def get_canonical_exp001_volumes(
    coordinate_frame: str = "payload_rack_canonical",
) -> Dict[str, BoundingVolume3D]:
    """
    Returns standard 3D bounding volumes for the 4 canonical EXP-001 objects
    in payload rack space (units in meters):
      - RED_SAMPLE: Storage slot on left rack bay.
      - BLUE_SAMPLE: Storage slot on right rack bay.
      - SAMPLE_CONTAINER: Central containment receptacle.
      - CONTAINER_LID: Receptacle sealing lid.
    """
    return {
        "RED_SAMPLE": BoundingVolume3D(
            name="RED_SAMPLE",
            center=(-0.15, 0.00, 0.00),
            dimensions=(0.06, 0.06, 0.06),
            coordinate_frame=coordinate_frame,
            is_metric=True,
        ),
        "BLUE_SAMPLE": BoundingVolume3D(
            name="BLUE_SAMPLE",
            center=(0.15, 0.00, 0.00),
            dimensions=(0.06, 0.06, 0.06),
            coordinate_frame=coordinate_frame,
            is_metric=True,
        ),
        "SAMPLE_CONTAINER": BoundingVolume3D(
            name="SAMPLE_CONTAINER",
            center=(0.00, 0.00, 0.00),
            dimensions=(0.14, 0.12, 0.14),
            coordinate_frame=coordinate_frame,
            is_metric=True,
        ),
        "CONTAINER_LID": BoundingVolume3D(
            name="CONTAINER_LID",
            center=(0.00, 0.08, 0.00),
            dimensions=(0.15, 0.03, 0.15),
            coordinate_frame=coordinate_frame,
            is_metric=True,
        ),
    }


@dataclass
class SpatialRelationResult:
    """
    Structured geometric relationship between an anatomical joint and a payload object.
    
    Fields:
      source_entity: Name of the body joint (e.g., 'right_wrist', 'left_hand').
      target_entity: Name of the payload object (e.g., 'RED_SAMPLE', 'SAMPLE_CONTAINER').
      distance: Euclidean distance to volume surface (meters if metric, else scalar).
      is_metric: Flag indicating whether distance is a true physical meter measurement.
      is_inside: True if joint lies within the target bounding volume.
      is_within_reach: True if distance <= reach_threshold.
      coordinate_frame: Coordinate frame tag of the evaluation.
      confidence: Joint tracking confidence.
      approach_vector: Optional normalized 3D unit vector pointing from joint to object center.
    """
    source_entity: str
    target_entity: str
    distance: float
    is_metric: bool
    is_inside: bool
    is_within_reach: bool
    coordinate_frame: str
    confidence: float = 1.0
    approach_vector: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes spatial relation result to a deterministic, JSON-safe dictionary."""
        vec_out = None
        if self.approach_vector is not None:
            vec_out = [round(float(v), 5) for v in self.approach_vector]

        return {
            "source_entity": str(self.source_entity),
            "target_entity": str(self.target_entity),
            "distance": round(float(self.distance), 5),
            "is_metric": bool(self.is_metric),
            "is_inside": bool(self.is_inside),
            "is_within_reach": bool(self.is_within_reach),
            "coordinate_frame": str(self.coordinate_frame),
            "confidence": round(float(self.confidence), 4),
            "approach_vector": vec_out,
        }


def compute_euclidean_distance_3d(
    p1: Union[Tuple[float, float, float], List[float], np.ndarray],
    p2: Union[Tuple[float, float, float], List[float], np.ndarray],
) -> float:
    """Calculates 3D Euclidean distance between two points."""
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    dz = float(p1[2]) - float(p2[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def evaluate_joint_target_relation(
    joint: Joint3D,
    volume: BoundingVolume3D,
    reach_threshold: float = 0.12,
    is_metric: bool = True,
    coordinate_frame: str = "payload_rack_canonical",
) -> SpatialRelationResult:
    """
    Evaluates metric distance, reachability, and containment for a single joint
    against a 3D payload volume.
    """
    pt = (joint.x, joint.y, joint.z)
    dist = volume.distance_to_point(pt)
    inside = volume.contains_point(pt)
    within_reach = dist <= float(reach_threshold)

    # Calculate unit direction vector pointing from joint to target volume center
    cx, cy, cz = volume.center
    v_dx = cx - joint.x
    v_dy = cy - joint.y
    v_dz = cz - joint.z
    v_mag = math.sqrt(v_dx * v_dx + v_dy * v_dy + v_dz * v_dz)

    if v_mag > 1e-6:
        approach_vec = (v_dx / v_mag, v_dy / v_mag, v_dz / v_mag)
    else:
        approach_vec = (0.0, 0.0, 0.0)

    return SpatialRelationResult(
        source_entity=joint.name or "joint",
        target_entity=volume.name,
        distance=dist,
        is_metric=is_metric and volume.is_metric,
        is_inside=inside,
        is_within_reach=within_reach,
        coordinate_frame=coordinate_frame,
        confidence=joint.confidence,
        approach_vector=approach_vec,
    )


def evaluate_payload_spatial_relations(
    joints: List[Joint3D],
    volumes: Dict[str, BoundingVolume3D],
    reach_threshold: float = 0.12,
    is_metric: bool = True,
    coordinate_frame: str = "payload_rack_canonical",
) -> Dict[str, SpatialRelationResult]:
    """
    Evaluates hand/wrist interactions against all payload bounding volumes.
    Selects relevant end-effector joints (wrists, hands) and pairs them with targets.
    """
    results: Dict[str, SpatialRelationResult] = {}
    if not joints or not volumes:
        return results

    # Identify candidate manipulator joints
    effector_names = {"right_wrist", "left_wrist", "right_hand", "left_hand"}
    effector_joints = [j for j in joints if j.name in effector_names]
    if not effector_joints:
        effector_joints = joints  # Fallback to all joints if specific names absent

    for vol_name, volume in volumes.items():
        # Find the closest effector joint to this volume
        best_rel: Optional[SpatialRelationResult] = None
        for j in effector_joints:
            rel = evaluate_joint_target_relation(
                joint=j,
                volume=volume,
                reach_threshold=reach_threshold,
                is_metric=is_metric,
                coordinate_frame=coordinate_frame,
            )
            if best_rel is None or rel.distance < best_rel.distance:
                best_rel = rel

        if best_rel is not None:
            results[vol_name] = best_rel

    return results
