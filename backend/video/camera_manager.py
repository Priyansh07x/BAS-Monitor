"""
camera_manager.py -- Central camera management for BAS-Monitor.

Phase A2: Enumerates available cameras, selects the active source, and
provides a single point of access for the rest of the system.

Phase A3: Exposes a readiness check and capability queries before an
experiment starts.

Lifecycle before an experiment:
    Select camera  →  Connect  →  Check capabilities  →  Readiness check  →  Start
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import cv2

from .camera_source import (
    CameraCapabilities,
    CameraSource,
    CameraStatus,
    CameraType,
    ConnectionState,
)
from .usb_camera import USBCamera
from .ip_camera import IPCamera


# ------------------------------------------------------------------ #
#  CameraManager
# ------------------------------------------------------------------ #

class CameraManager:
    """
    Single point of control for every camera in the system.

    Responsibilities:
      - Enumerate available USB cameras
      - Register IP / Android sources from config
      - Select + connect the active camera
      - Provide readiness checks (A3)
      - Expose status/capabilities to Bridge and frontend
    """

    def __init__(self):
        # Registry: source_id → CameraSource
        self._sources: dict[str, CameraSource] = {}
        self._active_id: Optional[str] = None
        self._lock = threading.Lock()

    # ================================================================== #
    #  Source registration
    # ================================================================== #

    def register(self, source_id: str, source: CameraSource) -> None:
        """Add a camera source to the registry."""
        with self._lock:
            self._sources[source_id] = source

    def unregister(self, source_id: str) -> None:
        """Remove a camera source from the registry."""
        with self._lock:
            if source_id == self._active_id:
                src = self._sources.get(source_id)
                if src:
                    src.disconnect()
                self._active_id = None
            self._sources.pop(source_id, None)

    # ================================================================== #
    #  Enumeration
    # ================================================================== #

    def enumerate_usb_cameras(self, max_index: int = 5) -> list[str]:
        """
        Probe local device indices 0..max_index and register any that
        respond.  Returns a list of source_ids that were found.
        """
        found: list[str] = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                source_id = f"usb:{idx}"
                if source_id not in self._sources:
                    self.register(source_id, USBCamera(idx, label=f"USB Camera {idx}"))
                found.append(source_id)
                cap.release()
            else:
                cap.release()
        return found

    def add_ip_camera(self, url: str, label: str | None = None) -> str:
        """Register a network camera by its stream URL. Returns its source_id."""
        source_id = f"ip:{url}"
        self.register(source_id, IPCamera(url, label=label))
        return source_id

    def load_from_config(self, config_path: str | Path | None = None) -> None:
        """
        Load camera sources from config/settings.json.
        Supports:
          - camera.source (int) → USB device index
          - camera.ip_cameras (list of {url, label}) → IP sources
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "settings.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            return

        with open(config_path, "r") as f:
            cfg = json.load(f)

        cam_cfg = cfg.get("camera", {})

        # Register the default USB source from config
        default_idx = cam_cfg.get("source", 0)
        if isinstance(default_idx, int):
            source_id = f"usb:{default_idx}"
            if source_id not in self._sources:
                self.register(source_id, USBCamera(default_idx, label=f"USB Camera {default_idx}"))

        # Register any IP cameras from config
        for entry in cam_cfg.get("ip_cameras", []):
            url = entry.get("url", "")
            label = entry.get("label", None)
            if url:
                self.add_ip_camera(url, label=label)

    # ================================================================== #
    #  Selection & connection
    # ================================================================== #

    def list_sources(self) -> list[dict]:
        """Return a JSON-serialisable list of all registered sources + status."""
        with self._lock:
            result = []
            for sid, src in self._sources.items():
                status = src.get_status()
                entry = status.to_dict()
                # Override with the registry source_id (e.g. "usb:0")
                entry["source_id"] = sid
                entry["is_active"] = sid == self._active_id
                result.append(entry)
            return result

    def select(self, source_id: str) -> bool:
        """
        Select and connect a camera source.
        Disconnects the previous active source first.
        Returns True if the new source connects successfully.
        """
        with self._lock:
            if source_id not in self._sources:
                return False

            # Disconnect current
            if self._active_id and self._active_id in self._sources:
                self._sources[self._active_id].disconnect()

            self._active_id = source_id
            return self._sources[source_id].connect()

    def get_active(self) -> Optional[CameraSource]:
        """Return the currently active camera source, or None."""
        with self._lock:
            if self._active_id and self._active_id in self._sources:
                return self._sources[self._active_id]
            return None

    def get_active_id(self) -> Optional[str]:
        """Return the source_id of the active camera, or None."""
        return self._active_id

    # ================================================================== #
    #  Phase A3 — Readiness & capability queries
    # ================================================================== #

    def get_capabilities(self, source_id: str | None = None) -> Optional[CameraCapabilities]:
        """
        Return the capabilities of a specific source (or the active one).
        Returns None if the source doesn't exist.
        """
        sid = source_id or self._active_id
        if sid and sid in self._sources:
            return self._sources[sid].get_capabilities()
        return None

    def get_status(self, source_id: str | None = None) -> Optional[CameraStatus]:
        """
        Return the status of a specific source (or the active one).
        Returns None if the source doesn't exist.
        """
        sid = source_id or self._active_id
        if sid and sid in self._sources:
            return self._sources[sid].get_status()
        return None

    def readiness_check(self, source_id: str | None = None) -> dict:
        """
        Phase A3 readiness gate:
            1. Is the source registered?
            2. Is it connected?
            3. Can it deliver a frame?
            4. What resolution / FPS is active?

        Returns a dict with 'ready' (bool) and a list of 'checks'.
        """
        sid = source_id or self._active_id
        checks: list[dict] = []

        # Check 1: source exists
        if not sid or sid not in self._sources:
            checks.append({"check": "source_registered", "passed": False, "detail": f"Source '{sid}' not found"})
            return {"ready": False, "source_id": sid, "checks": checks}

        checks.append({"check": "source_registered", "passed": True, "detail": sid})

        src = self._sources[sid]
        status = src.get_status()

        # Check 2: connected
        connected = status.connection_state == ConnectionState.CONNECTED
        checks.append({
            "check": "connected",
            "passed": connected,
            "detail": status.connection_state.value,
        })

        if not connected:
            return {"ready": False, "source_id": sid, "checks": checks}

        # Check 3: can deliver a frame
        test_frame = src.read()
        frame_ok = test_frame is not None
        checks.append({
            "check": "frame_delivery",
            "passed": frame_ok,
            "detail": f"{test_frame.shape[1]}x{test_frame.shape[0]}" if frame_ok else "No frame",
        })

        # Check 4: resolution & FPS
        caps = src.get_capabilities()
        if caps.current_resolution:
            w, h = caps.current_resolution
            checks.append({
                "check": "resolution",
                "passed": w > 0 and h > 0,
                "detail": f"{w}x{h}",
            })

        checks.append({
            "check": "fps",
            "passed": caps.current_fps > 0,
            "detail": f"{caps.current_fps:.1f} FPS",
        })

        all_passed = all(c["passed"] for c in checks)
        return {"ready": all_passed, "source_id": sid, "checks": checks}

    # ================================================================== #
    #  Frame access (convenience for Bridge)
    # ================================================================== #

    def read(self) -> Optional["np.ndarray"]:
        """Read a frame from the active camera. Returns None if no active source."""
        src = self.get_active()
        if src:
            return src.read()
        return None

    # ================================================================== #
    #  Cleanup
    # ================================================================== #

    def disconnect_all(self) -> None:
        """Disconnect every registered source."""
        with self._lock:
            for src in self._sources.values():
                src.disconnect()
            self._active_id = None


# ------------------------------------------------------------------ #
#  Module-level singleton
# ------------------------------------------------------------------ #
_manager_instance: Optional[CameraManager] = None
_manager_lock = threading.Lock()


def get_camera_manager() -> CameraManager:
    """Return or create a module-level CameraManager singleton."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = CameraManager()
        return _manager_instance
