"""
camera_source.py -- Abstract camera source interface.

Phase A2: Every camera type (USB, CSI, IP, Android) must implement this
interface so the rest of BAS-Monitor is decoupled from the physical source.

Phase A3: get_capabilities() returns only genuinely supported features.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ------------------------------------------------------------------ #
#  Enumerations
# ------------------------------------------------------------------ #

class CameraType(enum.Enum):
    """Identifies the physical transport of a camera source."""
    USB = "usb"
    CSI = "csi"
    IP  = "ip"
    ANDROID = "android"


class ConnectionState(enum.Enum):
    """Lifecycle state of a camera connection."""
    DISCONNECTED = "disconnected"
    CONNECTING   = "connecting"
    CONNECTED    = "connected"
    ERROR        = "error"


# ------------------------------------------------------------------ #
#  Phase A3 — Capability & Status data objects
# ------------------------------------------------------------------ #

@dataclass
class CameraCapabilities:
    """
    Reports the *genuine* capabilities of the connected camera.
    Values are probed at connect-time — nothing is invented.
    """
    # Resolution
    resolutions: list[tuple[int, int]] = field(default_factory=list)
    current_resolution: Optional[tuple[int, int]] = None

    # Frame rate
    fps_range: Optional[tuple[float, float]] = None   # (min, max)
    current_fps: float = 0.0

    # Feature support flags
    supports_autofocus: bool = False
    supports_manual_focus: bool = False
    supports_optical_zoom: bool = False
    supports_digital_zoom: bool = False
    supports_ptz: bool = False

    # Backend / codec info
    backend_name: str = ""

    def to_dict(self) -> dict:
        return {
            "resolutions": [list(r) for r in self.resolutions],
            "current_resolution": list(self.current_resolution) if self.current_resolution else None,
            "fps_range": list(self.fps_range) if self.fps_range else None,
            "current_fps": self.current_fps,
            "supports_autofocus": self.supports_autofocus,
            "supports_manual_focus": self.supports_manual_focus,
            "supports_optical_zoom": self.supports_optical_zoom,
            "supports_digital_zoom": self.supports_digital_zoom,
            "supports_ptz": self.supports_ptz,
            "backend_name": self.backend_name,
        }


@dataclass
class CameraStatus:
    """Snapshot of a camera's runtime health."""
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    camera_type: CameraType = CameraType.USB
    source_id: str = ""             # e.g. "0", "192.168.1.5:554"
    display_name: str = ""          # human-friendly label
    error_message: str = ""
    frames_delivered: int = 0       # total frames read since connect
    consecutive_failures: int = 0   # streak of failed reads

    def to_dict(self) -> dict:
        return {
            "connection_state": self.connection_state.value,
            "camera_type": self.camera_type.value,
            "source_id": self.source_id,
            "display_name": self.display_name,
            "error_message": self.error_message,
            "frames_delivered": self.frames_delivered,
            "consecutive_failures": self.consecutive_failures,
        }


# ------------------------------------------------------------------ #
#  Abstract base class
# ------------------------------------------------------------------ #

class CameraSource(ABC):
    """
    Unified camera interface.

    Lifecycle:
        connect()  →  read()  →  disconnect()

    Subclasses MUST implement the four abstract methods.
    """

    # ---- lifecycle ------------------------------------------------ #

    @abstractmethod
    def connect(self) -> bool:
        """
        Open the camera device / stream.
        Returns True on success, False on failure.
        Must be idempotent (calling twice on an open camera is safe).
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """
        Release all resources.
        Must be idempotent (calling on an already-closed camera is safe).
        """
        ...

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Return the next BGR frame, or None if unavailable.
        Must not block for longer than ~200 ms.
        """
        ...

    # ---- introspection ------------------------------------------- #

    @abstractmethod
    def get_status(self) -> CameraStatus:
        """Return a snapshot of the camera's current runtime health."""
        ...

    @abstractmethod
    def get_capabilities(self) -> CameraCapabilities:
        """
        Probe and return the camera's *genuine* capabilities.
        Must only report features that the hardware actually supports.
        """
        ...

    # ---- convenience --------------------------------------------- #

    def is_connected(self) -> bool:
        """Shorthand: is the camera in a CONNECTED state?"""
        return self.get_status().connection_state == ConnectionState.CONNECTED

    # Legacy compat aliases so existing code using .open() / .release() / .is_open() still works
    def open(self) -> bool:
        return self.connect()

    def release(self) -> None:
        return self.disconnect()

    def is_open(self) -> bool:
        return self.is_connected()
