"""
camera_manager.py -- Central camera management for BAS-Monitor.

Phase A2: Enumerates available cameras, selects the active source, and
provides a single point of access for the rest of the system.

Phase A3: Exposes a readiness check and capability queries before an
experiment starts.

Phase A4: Camera lock/unlock for experiment safety, automatic failover
with timeout detection, logged transitions, and operator notification.

Phase A5: Delegates hardware control calls to the active source.

Lifecycle before an experiment:
    Select camera  →  Connect  →  Check capabilities  →  Readiness check  →  Lock  →  Start
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .camera_source import (
    CameraCapabilities,
    CameraSource,
    CameraStatus,
    CameraType,
    ConnectionState,
)
from .usb_camera import USBCamera
from .ip_camera import IPCamera

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Failover event data
# ------------------------------------------------------------------ #

class FailoverEvent:
    """Immutable record of a camera failover transition."""
    __slots__ = ("timestamp", "from_source", "to_source", "reason", "success")

    def __init__(self, from_source: str, to_source: str, reason: str, success: bool):
        self.timestamp = time.time()
        self.from_source = from_source
        self.to_source = to_source
        self.reason = reason
        self.success = success

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "from_source": self.from_source,
            "to_source": self.to_source,
            "reason": self.reason,
            "success": self.success,
        }


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
      - Lock/unlock cameras during experiments (A4)
      - Automatic failover with operator notification (A4)
      - Delegate hardware controls (A5)
      - Expose status/capabilities to Bridge and frontend
    """

    def __init__(self):
        # Registry: source_id → CameraSource
        self._sources: dict[str, CameraSource] = {}
        self._active_id: Optional[str] = None
        self._lock = threading.Lock()

        # Phase A4 — experiment lock
        self._locked: bool = False

        # Phase A4 — failover
        self._fallback_order: list[str] = []       # ordered list of source_ids to try
        self._failure_threshold: int = 30          # consecutive failures before failover
        self._failover_history: list[FailoverEvent] = []
        self._on_failover: Optional[Callable[[FailoverEvent], None]] = None  # callback

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
          - camera.failover.threshold (int) → consecutive failures before failover
          - camera.failover.fallback_order (list of str) → fallback priority
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

        # Phase A4 — failover configuration
        failover_cfg = cam_cfg.get("failover", {})
        if "threshold" in failover_cfg:
            self._failure_threshold = int(failover_cfg["threshold"])
        if "fallback_order" in failover_cfg:
            self._fallback_order = list(failover_cfg["fallback_order"])

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

        Blocked while camera is locked (Phase A4) unless called by
        internal failover logic.
        """
        with self._lock:
            return self._select_unlocked(source_id)

    def _select_unlocked(self, source_id: str) -> bool:
        """Internal select — caller must hold self._lock."""
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
    #  Phase A4 — Camera Lock / Unlock
    # ================================================================== #

    def lock(self) -> bool:
        """
        Lock the active camera for an experiment.
        Prevents manual switching while locked.
        Returns False if no active camera is connected.
        """
        active = self.get_active()
        if active is None or not active.is_connected():
            return False
        self._locked = True
        logger.info(f"Camera locked: {self._active_id}")
        return True

    def unlock(self) -> None:
        """Unlock the camera after an experiment ends."""
        self._locked = False
        logger.info("Camera unlocked")

    @property
    def is_locked(self) -> bool:
        return self._locked

    # ================================================================== #
    #  Phase A4 — Failover
    # ================================================================== #

    def set_failover_callback(self, callback: Callable[[FailoverEvent], None]) -> None:
        """Register a callback invoked on every failover event."""
        self._on_failover = callback

    def set_failure_threshold(self, threshold: int) -> None:
        """Set the number of consecutive frame failures that trigger failover."""
        self._failure_threshold = max(1, threshold)

    def set_fallback_order(self, source_ids: list[str]) -> None:
        """Set the priority order of fallback cameras."""
        self._fallback_order = list(source_ids)

    def get_failover_history(self) -> list[dict]:
        """Return the full failover event log."""
        return [e.to_dict() for e in self._failover_history]

    def check_and_failover(self) -> Optional[FailoverEvent]:
        """
        Check the active camera's health.  If consecutive failures
        exceed the threshold, attempt controlled failover.

        Should be called periodically (e.g. from the frame-read loop).
        Returns a FailoverEvent if a transition happened, else None.
        """
        with self._lock:
            if not self._active_id or self._active_id not in self._sources:
                return None

            src = self._sources[self._active_id]
            status = src.get_status()

            if status.consecutive_failures < self._failure_threshold:
                return None

            # --- Threshold exceeded → attempt failover --- #
            failed_id = self._active_id
            reason = (
                f"Camera '{failed_id}' exceeded failure threshold "
                f"({status.consecutive_failures} consecutive failures)"
            )
            logger.warning(reason)

            # Try reconnecting the same camera first
            src.disconnect()
            if src.connect():
                logger.info(f"Reconnected to {failed_id} successfully")
                return None

            # Build candidate list: fallback_order first, then all others
            candidates = list(self._fallback_order)
            for sid in self._sources:
                if sid not in candidates and sid != failed_id:
                    candidates.append(sid)

            for candidate_id in candidates:
                if candidate_id not in self._sources:
                    continue
                if candidate_id == failed_id:
                    continue

                logger.info(f"Attempting failover to '{candidate_id}'...")
                success = self._select_unlocked(candidate_id)

                event = FailoverEvent(
                    from_source=failed_id,
                    to_source=candidate_id,
                    reason=reason,
                    success=success,
                )
                self._failover_history.append(event)

                if success:
                    logger.info(f"Failover succeeded: {failed_id} → {candidate_id}")
                    if self._on_failover:
                        self._on_failover(event)
                    return event
                else:
                    logger.warning(f"Failover to '{candidate_id}' failed, trying next...")

            # All candidates exhausted
            event = FailoverEvent(
                from_source=failed_id,
                to_source="none",
                reason=reason + " — all fallback cameras failed",
                success=False,
            )
            self._failover_history.append(event)
            if self._on_failover:
                self._on_failover(event)
            return event

    # ================================================================== #
    #  Phase A5 — Hardware control delegation
    # ================================================================== #

    def set_autofocus(self, enabled: bool) -> bool:
        """Delegate to active camera."""
        src = self.get_active()
        return src.set_autofocus(enabled) if src else False

    def set_focus(self, value: float) -> bool:
        """Delegate to active camera."""
        src = self.get_active()
        return src.set_focus(value) if src else False

    def set_zoom(self, value: float) -> bool:
        """Delegate to active camera."""
        src = self.get_active()
        return src.set_zoom(value) if src else False

    def set_resolution(self, width: int, height: int) -> bool:
        """Delegate to active camera."""
        src = self.get_active()
        return src.set_resolution(width, height) if src else False

    # ================================================================== #
    #  Frame access (convenience for Bridge)
    # ================================================================== #

    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the active camera.
        Returns None if no active source.
        Also checks failover threshold (Phase A4).
        """
        src = self.get_active()
        if src is None:
            return None

        frame = src.read()

        # Phase A4: check for sustained failure → failover
        if frame is None:
            status = src.get_status()
            if status.consecutive_failures >= self._failure_threshold:
                self.check_and_failover()

        return frame

    # ================================================================== #
    #  Cleanup
    # ================================================================== #

    def disconnect_all(self) -> None:
        """Disconnect every registered source."""
        with self._lock:
            for src in self._sources.values():
                src.disconnect()
            self._active_id = None
            self._locked = False


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
