"""
hmr_models.py — 3D Human Mesh Recovery Data Models
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.1)

Provides structured, deterministic, JSON-serializable representations for 3D human
joints, surface mesh vertices, polygonal faces, and parametric model outputs (SMPL).

Zero external dependencies beyond standard library and NumPy.
Strictly decoupled from the frozen 8-field public AI contract (docs/architecture.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class Joint3D:
    """
    Representation of a single 3D anatomical joint.
    
    Fields:
      x: Horizontal coordinate (meters or normalized, depending on coordinate_frame).
      y: Vertical coordinate (meters or normalized).
      z: Depth coordinate (meters or pseudo-depth).
      visibility: Landmark visibility score in [0.0, 1.0].
      confidence: Detection/estimation confidence in [0.0, 1.0].
      name: Optional anatomical name (e.g., 'pelvis', 'left_wrist', 'nose').
    """
    x: float
    y: float
    z: float
    visibility: float = 1.0
    confidence: float = 1.0
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-safe dictionary representation."""
        res: Dict[str, Any] = {
            "x": round(float(self.x), 5),
            "y": round(float(self.y), 5),
            "z": round(float(self.z), 5),
            "visibility": round(float(self.visibility), 4),
            "confidence": round(float(self.confidence), 4),
        }
        if self.name is not None:
            res["name"] = str(self.name)
        return res


@dataclass(frozen=True)
class MeshVertex3D:
    """
    Representation of a single 3D mesh surface vertex.
    
    Fields:
      x: Coordinate along X-axis.
      y: Coordinate along Y-axis.
      z: Coordinate along Z-axis.
    """
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict[str, float]:
        """Returns deterministic JSON-safe dictionary representation."""
        return {
            "x": round(float(self.x), 5),
            "y": round(float(self.y), 5),
            "z": round(float(self.z), 5),
        }


@dataclass
class HMRMeshResult:
    """
    Standardized result container for 3D Human Mesh Recovery and 3D pose extraction.
    
    Provides structured representations for:
      - 3D skeletal joints (Joint3D)
      - Volumetric mesh vertices (MeshVertex3D)
      - Polygonal face indices (triangular triplets)
      - Shape parameters beta (SMPL 10-D blend weights, if available)
      - Pose parameters theta (SMPL 72-D joint rotation parameters, if available)
      - Explicit mesh availability indicator (mesh_present)
      - Explicit coordinate frame and representation type metadata
    
    Guarantees:
      - Deterministic serialization (to_dict, to_json) with JSON-safe primitives.
      - Zero NumPy scalar serialization exceptions.
      - Safe handling of empty/unavailable results without throwing errors.
      - Strict isolation from frozen public AI contract.
    """
    joints: List[Joint3D] = field(default_factory=list)
    vertices: List[MeshVertex3D] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)
    beta: Optional[List[float]] = None
    theta: Optional[List[float]] = None
    mesh_present: bool = False
    coordinate_frame: str = "unknown"
    representation_type: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        reason: str = "no_detection",
        coordinate_frame: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HMRMeshResult:
        """Constructs an explicit empty/unavailable HMR result."""
        meta = {"reason": reason, "available": False}
        if metadata:
            meta.update(metadata)
        return cls(
            joints=[],
            vertices=[],
            faces=[],
            beta=None,
            theta=None,
            mesh_present=False,
            coordinate_frame=coordinate_frame,
            representation_type="none",
            metadata=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts to a strictly typed, JSON-safe dictionary."""
        # Convert beta safely
        beta_out: Optional[List[float]] = None
        if self.beta is not None:
            beta_out = [round(float(b), 5) for b in self.beta]

        # Convert theta safely
        theta_out: Optional[List[float]] = None
        if self.theta is not None:
            theta_out = [round(float(t), 5) for t in self.theta]

        # Convert faces safely
        faces_out: List[List[int]] = []
        for face in self.faces:
            faces_out.append([int(idx) for idx in face])

        # Clean metadata of any numpy scalars
        clean_metadata: Dict[str, Any] = {}
        for k, v in self.metadata.items():
            if isinstance(v, (np.floating, float)):
                clean_metadata[k] = round(float(v), 5)
            elif isinstance(v, (np.integer, int)):
                clean_metadata[k] = int(v)
            elif isinstance(v, np.ndarray):
                clean_metadata[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                clean_metadata[k] = list(v)
            else:
                clean_metadata[k] = v

        return {
            "mesh_present": bool(self.mesh_present),
            "representation_type": str(self.representation_type),
            "coordinate_frame": str(self.coordinate_frame),
            "joint_count": len(self.joints),
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "joints": [j.to_dict() for j in self.joints],
            "vertices": [v.to_dict() for v in self.vertices],
            "faces": faces_out,
            "beta": beta_out,
            "theta": theta_out,
            "metadata": clean_metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes the result to a deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
