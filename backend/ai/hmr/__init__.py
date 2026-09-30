"""
backend/ai/hmr — 3D Human Mesh Recovery (HMR) & Kinematics Package
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phases B14.1 & B14.2)

Provides abstract interfaces, data models, synthetic kinematic generators,
representation adapters, coordinate transforms, 3D spatial geometry reasoning,
and microgravity posture normalization.
"""

from backend.ai.hmr.hmr_interface import HMRRecoveryEngineInterface
from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
    MeshVertex3D,
)
from backend.ai.hmr.mediapipe_hmr import (
    MEDIAPIPE_33_LANDMARK_NAMES,
    MediaPipeHMREngine,
)
from backend.ai.hmr.payload_transform import CameraToPayloadTransform
from backend.ai.hmr.posture_normalizer import MicrogravityPostureNormalizer
from backend.ai.hmr.spatial_adapter import SpatialDisambiguationAdapter
from backend.ai.hmr.spatial_geometry import (
    BoundingVolume3D,
    SpatialRelationResult,
    compute_euclidean_distance_3d,
    evaluate_joint_target_relation,
    evaluate_payload_spatial_relations,
    get_canonical_exp001_volumes,
)
from backend.ai.hmr.synthetic_hmr import (
    SMPL_24_JOINT_NAMES,
    SMPL_24_PARENT_MAP,
    SyntheticHMREngine,
)

__all__ = [
    # B14.1 Core
    "Joint3D",
    "MeshVertex3D",
    "HMRMeshResult",
    "HMRRecoveryEngineInterface",
    "SyntheticHMREngine",
    "MediaPipeHMREngine",
    "SMPL_24_JOINT_NAMES",
    "SMPL_24_PARENT_MAP",
    "MEDIAPIPE_33_LANDMARK_NAMES",
    # B14.2 Kinematics & Spatial Geometry
    "CameraToPayloadTransform",
    "BoundingVolume3D",
    "SpatialRelationResult",
    "MicrogravityPostureNormalizer",
    "SpatialDisambiguationAdapter",
    "get_canonical_exp001_volumes",
    "compute_euclidean_distance_3d",
    "evaluate_joint_target_relation",
    "evaluate_payload_spatial_relations",
]
