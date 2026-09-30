"""
hmr_interface.py — Abstract Interface for 3D Human Mesh Recovery Engines
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.1)

Defines the abstract interface contract for monocular 3D Human Mesh Recovery
and 3D body landmark extraction engines.

Ensures complete decoupling between perception callers and underlying
neural/synthetic mesh backends.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional
import numpy as np

from backend.ai.hmr.hmr_models import HMRMeshResult


class HMRRecoveryEngineInterface(abc.ABC):
    """
    Abstract Base Class defining the contract for 3D Human Mesh Recovery engines.
    
    Subclasses may implement:
      - Synthetic kinematic human generators (SyntheticHMREngine)
      - Lightweight landmark-to-mesh adapters (MediaPipeHMREngine)
      - Neural parametric mesh regressors (future HMR 2.0 / CLIFF / SPIN)
    
    Invariants:
      - Must handle null, empty, or malformed frames gracefully without throwing exceptions.
      - Must return a valid HMRMeshResult instance.
      - Must cleanly release any allocated memory or model runtimes in close().
      - Must NOT emit fields into or mutate the frozen 8-field public AI contract.
    """

    @abc.abstractmethod
    def recover_mesh(self, frame: np.ndarray) -> HMRMeshResult:
        """
        Processes a single BGR camera frame and returns the 3D mesh and/or joint structure.
        
        Args:
            frame: Input video frame as a NumPy array (H x W x C BGR uint8).
            
        Returns:
            HMRMeshResult containing 3D joints, optional surface vertices, and metadata.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Releases all engine resources, runtimes, or backend session handles."""
        pass

    def __enter__(self) -> HMRRecoveryEngineInterface:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
