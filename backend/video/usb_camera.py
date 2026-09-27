"""
usb_camera.py -- USB / local webcam CameraSource adapter.

Phase A2: Replaces the old monolithic Camera class with a proper
CameraSource implementation.  All capability probing (Phase A3) is
real — nothing is invented.

Phase A5: Implements capability-aware hardware controls for focus,
zoom, and resolution.  Controls are safe no-ops when the hardware
does not support them.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .camera_source import (
    CameraCapabilities,
    CameraSource,
    CameraStatus,
    CameraType,
    ConnectionState,
)


# Common resolutions to probe (w, h)
_PROBE_RESOLUTIONS: list[tuple[int, int]] = [
    (3840, 2160),  # 4K
    (2560, 1440),  # QHD
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
    (640, 480),    # VGA
    (320, 240),    # QVGA
]


class USBCamera(CameraSource):
    """
    Adapter for USB / built-in cameras accessed via OpenCV VideoCapture.
    """

    def __init__(self, device_index: int = 0, label: str | None = None):
        self._device_index = device_index
        self._label = label or f"USB Camera {device_index}"
        self._cap: Optional[cv2.VideoCapture] = None

        # Runtime counters
        self._frames_delivered: int = 0
        self._consecutive_failures: int = 0
        self._error: str = ""

        # Cached capabilities (populated on connect)
        self._capabilities = CameraCapabilities()

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True  # already connected — idempotent

        self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            self._error = f"Failed to open device index {self._device_index}"
            self._cap.release()
            self._cap = None
            return False

        self._error = ""
        self._frames_delivered = 0
        self._consecutive_failures = 0
        self._capabilities = self._probe_capabilities()
        return True

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frames_delivered = 0
        self._consecutive_failures = 0

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._consecutive_failures += 1
            return None

        self._consecutive_failures = 0
        self._frames_delivered += 1
        return frame

    # ------------------------------------------------------------------ #
    #  Status & Capabilities
    # ------------------------------------------------------------------ #

    def get_status(self) -> CameraStatus:
        if self._cap is not None and self._cap.isOpened():
            state = ConnectionState.CONNECTED
        elif self._error:
            state = ConnectionState.ERROR
        else:
            state = ConnectionState.DISCONNECTED

        return CameraStatus(
            connection_state=state,
            camera_type=CameraType.USB,
            source_id=str(self._device_index),
            display_name=self._label,
            error_message=self._error,
            frames_delivered=self._frames_delivered,
            consecutive_failures=self._consecutive_failures,
        )

    def get_capabilities(self) -> CameraCapabilities:
        return self._capabilities

    # ------------------------------------------------------------------ #
    #  Phase A5 — Hardware controls
    # ------------------------------------------------------------------ #

    def set_autofocus(self, enabled: bool) -> bool:
        """Enable or disable autofocus.  Only works if probed as supported."""
        if self._cap is None or not self._cap.isOpened():
            return False
        if not self._capabilities.supports_autofocus:
            return False
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enabled else 0)
        actual = self._cap.get(cv2.CAP_PROP_AUTOFOCUS)
        return actual == (1 if enabled else 0)

    def set_focus(self, value: float) -> bool:
        """Set manual focus.  Only works if manual focus is supported."""
        if self._cap is None or not self._cap.isOpened():
            return False
        if not self._capabilities.supports_manual_focus:
            return False
        # Disable autofocus first so manual value takes effect
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap.set(cv2.CAP_PROP_FOCUS, value)
        actual = self._cap.get(cv2.CAP_PROP_FOCUS)
        self._capabilities.current_focus = actual
        return True

    def set_zoom(self, value: float) -> bool:
        """Set optical zoom level.  Only works if optical zoom is supported."""
        if self._cap is None or not self._cap.isOpened():
            return False
        if not self._capabilities.supports_optical_zoom:
            return False
        self._cap.set(cv2.CAP_PROP_ZOOM, value)
        actual = self._cap.get(cv2.CAP_PROP_ZOOM)
        self._capabilities.current_zoom = actual
        return True

    def set_resolution(self, width: int, height: int) -> bool:
        """Change the capture resolution."""
        if self._cap is None or not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._capabilities.current_resolution = (actual_w, actual_h)
        return actual_w == width and actual_h == height

    # ------------------------------------------------------------------ #
    #  Internal: genuine capability probing
    # ------------------------------------------------------------------ #

    def _probe_capabilities(self) -> CameraCapabilities:
        """
        Probe the real camera.  Only reports what the hardware actually
        supports — nothing is fabricated.
        """
        cap = self._cap
        if cap is None or not cap.isOpened():
            return CameraCapabilities()

        # ---- backend name ---------------------------------------- #
        backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))
        backend_name = ""
        # Map the numeric backend id to a human string when available
        _backend_map = {
            cv2.CAP_AVFOUNDATION: "AVFoundation",
            cv2.CAP_V4L2:        "V4L2",
            cv2.CAP_DSHOW:       "DirectShow",
            cv2.CAP_MSMF:        "MediaFoundation",
        }
        backend_name = _backend_map.get(backend_id, f"backend-{backend_id}")

        # ---- current resolution / fps ---------------------------- #
        cur_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cur_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cur_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        # ---- probe supported resolutions ------------------------- #
        supported_res: list[tuple[int, int]] = []
        for w, h in _PROBE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            res = (actual_w, actual_h)
            if res not in supported_res:
                supported_res.append(res)

        # Restore original resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cur_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cur_h)

        # ---- probe focus ----------------------------------------- #
        supports_autofocus = False
        supports_manual_focus = False
        focus_range: Optional[tuple[float, float]] = None
        current_focus: Optional[float] = None

        # Check if autofocus property is readable / settable
        autofocus_val = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        if autofocus_val >= 0:
            # Try toggling — if it sticks, the camera supports it
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            if cap.get(cv2.CAP_PROP_AUTOFOCUS) == 1:
                supports_autofocus = True
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if cap.get(cv2.CAP_PROP_AUTOFOCUS) == 0:
                supports_manual_focus = True
            # restore
            cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus_val)

        # Probe focus range if manual focus is supported
        if supports_manual_focus:
            current_focus = cap.get(cv2.CAP_PROP_FOCUS)
            # Try to discover the range by setting extremes
            cap.set(cv2.CAP_PROP_FOCUS, 0)
            focus_min = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_FOCUS, 10000)
            focus_max = cap.get(cv2.CAP_PROP_FOCUS)
            if focus_max > focus_min:
                focus_range = (focus_min, focus_max)
            # Restore
            if current_focus is not None:
                cap.set(cv2.CAP_PROP_FOCUS, current_focus)

        # ---- probe zoom ----------------------------------------- #
        supports_optical_zoom = False
        zoom_range: Optional[tuple[float, float]] = None
        current_zoom: Optional[float] = None
        zoom_val = cap.get(cv2.CAP_PROP_ZOOM)
        if zoom_val > 0:
            current_zoom = zoom_val
            # Try to change zoom; if it sticks the camera supports it
            cap.set(cv2.CAP_PROP_ZOOM, zoom_val + 1)
            if cap.get(cv2.CAP_PROP_ZOOM) != zoom_val:
                supports_optical_zoom = True
                # Probe range
                cap.set(cv2.CAP_PROP_ZOOM, 0)
                zoom_min = cap.get(cv2.CAP_PROP_ZOOM)
                cap.set(cv2.CAP_PROP_ZOOM, 10000)
                zoom_max = cap.get(cv2.CAP_PROP_ZOOM)
                if zoom_max > zoom_min:
                    zoom_range = (zoom_min, zoom_max)
            cap.set(cv2.CAP_PROP_ZOOM, zoom_val)

        # FPS range: we can only test what OpenCV reports
        fps_range = (cur_fps, cur_fps) if cur_fps > 0 else None

        return CameraCapabilities(
            resolutions=supported_res,
            current_resolution=(cur_w, cur_h),
            fps_range=fps_range,
            current_fps=cur_fps,
            supports_autofocus=supports_autofocus,
            supports_manual_focus=supports_manual_focus,
            supports_optical_zoom=supports_optical_zoom,
            supports_digital_zoom=False,  # OpenCV has no digital zoom API
            supports_ptz=False,           # USB cams generally don't support PTZ
            focus_range=focus_range,
            current_focus=current_focus,
            zoom_range=zoom_range,
            current_zoom=current_zoom,
            backend_name=backend_name,
        )
