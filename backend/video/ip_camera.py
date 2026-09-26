"""
ip_camera.py -- IP / network camera CameraSource adapter.

Phase A2: Supports RTSP, HTTP MJPEG, and any stream URL that OpenCV's
VideoCapture can decode.  Capability probing is real — nothing is invented.
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


class IPCamera(CameraSource):
    """
    Adapter for network cameras accessed via a stream URL
    (RTSP, HTTP/MJPEG, etc.).
    """

    def __init__(self, url: str, label: str | None = None):
        self._url = url
        self._label = label or f"IP Camera ({url})"
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
            return True  # already open — idempotent

        self._cap = cv2.VideoCapture(self._url)

        if not self._cap.isOpened():
            self._error = f"Failed to connect to stream: {self._url}"
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
            camera_type=CameraType.IP,
            source_id=self._url,
            display_name=self._label,
            error_message=self._error,
            frames_delivered=self._frames_delivered,
            consecutive_failures=self._consecutive_failures,
        )

    def get_capabilities(self) -> CameraCapabilities:
        return self._capabilities

    # ------------------------------------------------------------------ #
    #  Internal: genuine capability probing
    # ------------------------------------------------------------------ #

    def _probe_capabilities(self) -> CameraCapabilities:
        """
        Probe whatever the IP stream reports.  Network cameras rarely
        expose focus or zoom through OpenCV, so those flags stay False
        unless the hardware actually responds.
        """
        cap = self._cap
        if cap is None or not cap.isOpened():
            return CameraCapabilities()

        cur_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cur_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cur_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        # IP cameras typically only expose one resolution via the stream
        supported_res = [(cur_w, cur_h)] if cur_w > 0 and cur_h > 0 else []

        fps_range = (cur_fps, cur_fps) if cur_fps > 0 else None

        # Backend id
        backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))
        backend_name = f"backend-{backend_id}"

        return CameraCapabilities(
            resolutions=supported_res,
            current_resolution=(cur_w, cur_h) if cur_w > 0 else None,
            fps_range=fps_range,
            current_fps=cur_fps,
            supports_autofocus=False,      # not controllable via stream
            supports_manual_focus=False,
            supports_optical_zoom=False,
            supports_digital_zoom=False,
            supports_ptz=False,
            backend_name=backend_name,
        )
