"""
mediapipe_hmr.py — MediaPipe 3D Landmark Representation Adapter
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.1)

Adapts MediaPipe 33-landmark pose detections into the standardized HMRMeshResult
interface contract.

CRITICAL ARCHITECTURAL BOUNDARY:
This module is strictly a REPRESENTATION ADAPTER, not a parametric surface mesh
estimator. It does NOT fabricate SMPL mesh vertices (6890) or SMPL shape/pose
parameters (beta/theta). It explicitly emits mesh_present=False and
representation_type='landmark_only'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from backend.ai.hmr.hmr_interface import HMRRecoveryEngineInterface
from backend.ai.hmr.hmr_models import (
    HMRMeshResult,
    Joint3D,
)
from backend.ai.pose_detector import PoseDetector

# Standard 33 MediaPipe Pose Landmark Names
MEDIAPIPE_33_LANDMARK_NAMES: List[str] = [
    "nose",                   # 0
    "left_eye_inner",         # 1
    "left_eye",               # 2
    "left_eye_outer",         # 3
    "right_eye_inner",        # 4
    "right_eye",              # 5
    "right_eye_outer",        # 6
    "left_ear",               # 7
    "right_ear",              # 8
    "mouth_left",             # 9
    "mouth_right",            # 10
    "left_shoulder",          # 11
    "right_shoulder",         # 12
    "left_elbow",             # 13
    "right_elbow",            # 14
    "left_wrist",             # 15
    "right_wrist",            # 16
    "left_pinky",             # 17
    "right_pinky",            # 18
    "left_index",             # 19
    "right_index",            # 20
    "left_thumb",             # 21
    "right_thumb",            # 22
    "left_hip",               # 23
    "right_hip",              # 24
    "left_knee",              # 25
    "right_knee",             # 26
    "left_ankle",             # 27
    "right_ankle",            # 28
    "left_heel",              # 29
    "right_heel",             # 30
    "left_foot_index",        # 31
    "right_foot_index",       # 32
]


class MediaPipeHMREngine(HMRRecoveryEngineInterface):
    """
    Representation adapter translating MediaPipe 33-landmark pose detections
    into standardized HMRMeshResult objects.
    
    Guarantees:
      - mesh_present is strictly False (landmark-only representation).
      - vertices and faces lists remain strictly empty.
      - beta (shape) and theta (pose) remain strictly None.
      - Preserves 33-landmark indexing, visibility, and confidence.
      - Handles missing MediaPipe dependency gracefully via PoseDetector heuristic fallback.
    """

    def __init__(
        self,
        pose_detector: Optional[PoseDetector] = None,
        confidence_threshold: float = 0.50,
        prefer_world_landmarks: bool = True,
    ) -> None:
        """
        Args:
            pose_detector: Optional existing PoseDetector instance. If None, one is created.
            confidence_threshold: Minimum detection confidence threshold.
            prefer_world_landmarks: If True and MediaPipe world landmarks are available,
                                   uses metric world landmarks with origin at hips.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.prefer_world_landmarks = bool(prefer_world_landmarks)
        self._owns_detector = pose_detector is None
        self._pose_detector = pose_detector or PoseDetector(confidence_threshold=confidence_threshold)
        self._is_closed = False

    def recover_mesh(self, frame: np.ndarray) -> HMRMeshResult:
        """
        Extracts 33 body landmarks and formats them into an HMRMeshResult.
        """
        if self._is_closed:
            return HMRMeshResult.empty(
                reason="engine_closed",
                coordinate_frame="mediapipe_normalized",
            )

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return HMRMeshResult.empty(
                reason="invalid_or_empty_frame",
                coordinate_frame="mediapipe_normalized",
            )

        try:
            # 1. Attempt extracting world landmarks directly from MediaPipe instance if available
            world_joints: Optional[List[Joint3D]] = None
            if self.prefer_world_landmarks and self._pose_detector._pose_instance is not None:
                world_joints = self._try_extract_world_landmarks(frame)

            if world_joints is not None and len(world_joints) == 33:
                joints = world_joints
                coord_frame = "mediapipe_world_meters"
                is_fallback = False
            else:
                # 2. Extract standard landmarks via PoseDetector detect()
                raw_lms = self._pose_detector.detect(frame)
                if not raw_lms:
                    return HMRMeshResult.empty(
                        reason="no_pose_detected",
                        coordinate_frame="mediapipe_normalized",
                    )

                joints = []
                for idx, lm in enumerate(raw_lms[:33]):
                    name = MEDIAPIPE_33_LANDMARK_NAMES[idx] if idx < len(MEDIAPIPE_33_LANDMARK_NAMES) else f"joint_{idx}"
                    joints.append(
                        Joint3D(
                            x=float(lm.get("x", 0.0)),
                            y=float(lm.get("y", 0.0)),
                            z=float(lm.get("z", 0.0)),
                            visibility=float(lm.get("visibility", 1.0)),
                            confidence=float(lm.get("visibility", 1.0)),
                            name=name,
                        )
                    )
                coord_frame = "mediapipe_normalized"
                is_fallback = self._pose_detector._pose_instance is None

            metadata = {
                "engine": "MediaPipeHMREngine",
                "is_synthetic": False,
                "is_representation_adapter": True,
                "mesh_reconstruction_available": False,
                "is_fallback": is_fallback,
                "landmark_count": len(joints),
                "coordinate_frame": coord_frame,
            }

            return HMRMeshResult(
                joints=joints,
                vertices=[],
                faces=[],
                beta=None,
                theta=None,
                mesh_present=False,
                coordinate_frame=coord_frame,
                representation_type="landmark_only",
                metadata=metadata,
            )

        except Exception as e:
            return HMRMeshResult.empty(
                reason=f"detection_error: {str(e)}",
                coordinate_frame="mediapipe_normalized",
                metadata={"error": str(e)},
            )

    def _try_extract_world_landmarks(self, frame: np.ndarray) -> Optional[List[Joint3D]]:
        """Extracts 3D metric world landmarks from active MediaPipe pose instance if present."""
        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._pose_detector._pose_instance.process(rgb)
            if res and hasattr(res, "pose_world_landmarks") and res.pose_world_landmarks:
                joints: List[Joint3D] = []
                for idx, lm in enumerate(res.pose_world_landmarks.landmark[:33]):
                    name = MEDIAPIPE_33_LANDMARK_NAMES[idx] if idx < len(MEDIAPIPE_33_LANDMARK_NAMES) else f"joint_{idx}"
                    joints.append(
                        Joint3D(
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(lm.z),
                            visibility=float(lm.visibility),
                            confidence=float(lm.visibility),
                            name=name,
                        )
                    )
                return joints
        except Exception:
            pass
        return None

    def close(self) -> None:
        """Releases underlying pose detector if owned."""
        if self._is_closed:
            return
        if self._owns_detector and self._pose_detector is not None:
            self._pose_detector.close()
        self._is_closed = True
