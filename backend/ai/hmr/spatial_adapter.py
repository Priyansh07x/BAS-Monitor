"""
spatial_adapter.py — Spatial Disambiguation & Payload Kinematics Adapter
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.2)

Provides an optional, non-blocking, opt-in side-channel adapter that connects
3D Human Mesh Recovery (HMR) outputs with 3D payload rack spatial reasoning.

CRITICAL ARCHITECTURAL BOUNDARY:
- Strictly non-blocking and observational.
- Disabled by default (enabled=False).
- Zero mutation of SequenceValidatorFSM, RecoveryManager, or frozen 8-field public AI contract.
- Safely handles non-metric, landmark-only, or empty inputs without exceptions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
)
from backend.ai.hmr.payload_transform import CameraToPayloadTransform
from backend.ai.hmr.posture_normalizer import MicrogravityPostureNormalizer
from backend.ai.hmr.spatial_geometry import (
    BoundingVolume3D,
    SpatialRelationResult,
    evaluate_payload_spatial_relations,
    get_canonical_exp001_volumes,
)


class SpatialDisambiguationAdapter:
    """
    Optional side-channel spatial disambiguation and payload-relative reasoning engine.
    
    Transforms 3D body pose observations into payload space, evaluates proximity
    and containment against canonical experiment volumes, and normalizes microgravity
    postures for downstream diagnostics.
    """

    def __init__(
        self,
        transform: Optional[CameraToPayloadTransform] = None,
        payload_volumes: Optional[Dict[str, BoundingVolume3D]] = None,
        reach_threshold_m: float = 0.12,
        enabled: bool = False,
    ) -> None:
        """
        Args:
            transform: Camera-to-Payload coordinate transform. Defaults to Identity.
            payload_volumes: Dictionary of 3D payload bounding volumes. Defaults to EXP-001 canonical volumes.
            reach_threshold_m: Maximum distance (meters) to consider an object within hand reach.
            enabled: Opt-in enable flag (defaults to False for non-blocking isolation).
        """
        self.transform = transform or CameraToPayloadTransform.identity()
        self.payload_volumes = payload_volumes or get_canonical_exp001_volumes()
        self.reach_threshold_m = float(reach_threshold_m)
        self.enabled = bool(enabled)
        self._is_closed = False

    def process_hmr_result(self, hmr_result: HMRMeshResult) -> Dict[str, Any]:
        """
        Processes an HMR result through the 3D spatial disambiguation pipeline.
        
        Returns:
            Structured diagnostic dictionary containing spatial relations,
            manipulation candidate targets, and microgravity posture vectors.
        """
        if self._is_closed or not self.enabled:
            return {
                "active": False,
                "reason": "disabled_or_closed",
                "spatial_relations": {},
                "manipulation_target": None,
                "is_metric": False,
            }

        if not hmr_result or not hmr_result.joints:
            return {
                "active": True,
                "reason": "empty_hmr_joints",
                "spatial_relations": {},
                "manipulation_target": None,
                "is_metric": False,
            }

        t0 = time.monotonic()

        try:
            # 1. Determine if input is metric
            is_metric = hmr_result.coordinate_frame in {
                "synthetic_camera_canonical",
                "mediapipe_world_meters",
                "payload_rack_canonical",
            }

            # 2. Transform joints to payload space
            transformed_joints = self.transform.transform_joints(hmr_result.joints)

            # 3. Evaluate spatial relationships against EXP-001 volumes
            relations = evaluate_payload_spatial_relations(
                joints=transformed_joints,
                volumes=self.payload_volumes,
                reach_threshold=self.reach_threshold_m,
                is_metric=is_metric,
                coordinate_frame=self.transform.target_frame,
            )

            # 4. Identify closest reachable target object
            reachable_targets = [
                rel for rel in relations.values() if rel.is_within_reach
            ]
            reachable_targets.sort(key=lambda r: r.distance)
            primary_target = reachable_targets[0].target_entity if reachable_targets else None

            # 5. Compute microgravity posture normalized wrist vectors
            wrist_vectors = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(
                hmr_result.joints
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            return {
                "active": True,
                "is_metric": is_metric,
                "coordinate_frame": self.transform.target_frame,
                "spatial_relations": {k: v.to_dict() for k, v in relations.items()},
                "manipulation_target": primary_target,
                "normalized_wrist_vectors": wrist_vectors,
                "joint_count": len(transformed_joints),
                "elapsed_ms": round(elapsed_ms, 3),
            }

        except Exception as e:
            return {
                "active": True,
                "error": str(e),
                "spatial_relations": {},
                "manipulation_target": None,
                "is_metric": False,
            }

    def close(self) -> None:
        """Closes the adapter."""
        self._is_closed = True

    def __enter__(self) -> SpatialDisambiguationAdapter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

