"""
test_camera_manager.py -- Tests for Phase A2 (camera abstraction) and
Phase A3 (capability detection / readiness checks).
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from backend.video.camera_source import (
    CameraCapabilities,
    CameraSource,
    CameraStatus,
    CameraType,
    ConnectionState,
)
from backend.video.usb_camera import USBCamera
from backend.video.ip_camera import IPCamera
from backend.video.camera_manager import CameraManager


# ================================================================== #
#  CameraSource ABC contract
# ================================================================== #

class TestCameraSourceContract:
    """Verify that the ABC forces subclasses to implement all methods."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CameraSource()

    def test_usb_camera_is_camera_source(self):
        cam = USBCamera(device_index=99)
        assert isinstance(cam, CameraSource)

    def test_ip_camera_is_camera_source(self):
        cam = IPCamera(url="rtsp://fake")
        assert isinstance(cam, CameraSource)


# ================================================================== #
#  USBCamera
# ================================================================== #

class TestUSBCamera:

    def test_default_state_is_disconnected(self):
        cam = USBCamera(device_index=99)
        status = cam.get_status()
        assert status.connection_state == ConnectionState.DISCONNECTED
        assert status.camera_type == CameraType.USB
        assert status.source_id == "99"
        assert not cam.is_connected()

    def test_read_returns_none_when_disconnected(self):
        cam = USBCamera(device_index=99)
        assert cam.read() is None

    def test_disconnect_is_idempotent(self):
        cam = USBCamera(device_index=99)
        cam.disconnect()  # should not raise
        cam.disconnect()  # still should not raise

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_connect_success(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_vc_cls.return_value = mock_cap

        cam = USBCamera(device_index=0)
        assert cam.connect() is True
        assert cam.is_connected()

        status = cam.get_status()
        assert status.connection_state == ConnectionState.CONNECTED

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_connect_failure(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc_cls.return_value = mock_cap

        cam = USBCamera(device_index=99)
        assert cam.connect() is False
        assert not cam.is_connected()

        status = cam.get_status()
        assert status.connection_state == ConnectionState.ERROR
        assert "Failed to open" in status.error_message

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_read_frame(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_vc_cls.return_value = mock_cap

        cam = USBCamera(device_index=0)
        cam.connect()

        frame = cam.read()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        status = cam.get_status()
        assert status.frames_delivered == 1
        assert status.consecutive_failures == 0

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_read_failure_increments_counter(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_cap.read.return_value = (False, None)
        mock_vc_cls.return_value = mock_cap

        cam = USBCamera(device_index=0)
        cam.connect()

        assert cam.read() is None
        assert cam.read() is None

        status = cam.get_status()
        assert status.consecutive_failures == 2

    def test_legacy_aliases(self):
        """open() / release() / is_open() must work as aliases."""
        cam = USBCamera(device_index=99)
        assert cam.is_open() is False
        cam.release()  # should not raise


# ================================================================== #
#  IPCamera
# ================================================================== #

class TestIPCamera:

    def test_default_state_is_disconnected(self):
        cam = IPCamera(url="rtsp://fake:554/stream")
        status = cam.get_status()
        assert status.connection_state == ConnectionState.DISCONNECTED
        assert status.camera_type == CameraType.IP
        assert "rtsp://fake:554/stream" in status.source_id

    @patch("backend.video.ip_camera.cv2.VideoCapture")
    def test_connect_failure(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc_cls.return_value = mock_cap

        cam = IPCamera(url="rtsp://fake")
        assert cam.connect() is False
        assert cam.get_status().connection_state == ConnectionState.ERROR


# ================================================================== #
#  CameraManager
# ================================================================== #

class TestCameraManager:

    def test_empty_registry(self):
        mgr = CameraManager()
        assert mgr.list_sources() == []
        assert mgr.get_active() is None
        assert mgr.read() is None

    def test_register_and_list(self):
        mgr = CameraManager()
        cam = USBCamera(device_index=0)
        mgr.register("usb:0", cam)

        sources = mgr.list_sources()
        assert len(sources) == 1
        assert sources[0]["source_id"] == "usb:0"
        assert sources[0]["is_active"] is False

    def test_unregister(self):
        mgr = CameraManager()
        cam = USBCamera(device_index=0)
        mgr.register("usb:0", cam)
        mgr.unregister("usb:0")
        assert mgr.list_sources() == []

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_select_and_connect(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_vc_cls.return_value = mock_cap

        mgr = CameraManager()
        mgr.register("usb:0", USBCamera(device_index=0))

        assert mgr.select("usb:0") is True
        assert mgr.get_active_id() == "usb:0"
        assert mgr.get_active().is_connected()

    def test_select_nonexistent_returns_false(self):
        mgr = CameraManager()
        assert mgr.select("usb:99") is False

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_readiness_check_passes(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # FPS
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_vc_cls.return_value = mock_cap

        mgr = CameraManager()
        mgr.register("usb:0", USBCamera(device_index=0))
        mgr.select("usb:0")

        result = mgr.readiness_check()
        assert result["ready"] is True
        assert result["source_id"] == "usb:0"
        assert all(c["passed"] for c in result["checks"])

    def test_readiness_check_fails_when_no_source(self):
        mgr = CameraManager()
        result = mgr.readiness_check()
        assert result["ready"] is False

    @patch("backend.video.usb_camera.cv2.VideoCapture")
    def test_capabilities_probed(self, mock_vc_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 1280,   # FRAME_WIDTH
            4: 720,    # FRAME_HEIGHT
            5: 30.0,   # FPS
            42: -1,    # BACKEND
            39: -1,    # AUTOFOCUS
            27: 0,     # ZOOM
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_vc_cls.return_value = mock_cap

        mgr = CameraManager()
        mgr.register("usb:0", USBCamera(device_index=0))
        mgr.select("usb:0")

        caps = mgr.get_capabilities()
        assert caps is not None
        assert caps.current_fps > 0

        d = caps.to_dict()
        assert "current_resolution" in d
        assert "supports_autofocus" in d

    def test_disconnect_all(self):
        mgr = CameraManager()
        cam = USBCamera(device_index=99)
        mgr.register("usb:99", cam)
        mgr.disconnect_all()
        assert mgr.get_active() is None

    def test_add_ip_camera(self):
        mgr = CameraManager()
        sid = mgr.add_ip_camera("rtsp://10.0.0.5:554/stream", label="Lab Cam")
        assert sid == "ip:rtsp://10.0.0.5:554/stream"

        sources = mgr.list_sources()
        assert len(sources) == 1
        assert sources[0]["display_name"] == "Lab Cam"
        assert sources[0]["camera_type"] == "ip"


# ================================================================== #
#  Serialization
# ================================================================== #

class TestSerialization:

    def test_capabilities_to_dict(self):
        caps = CameraCapabilities(
            resolutions=[(1920, 1080), (1280, 720)],
            current_resolution=(1280, 720),
            fps_range=(15.0, 30.0),
            current_fps=30.0,
            supports_autofocus=True,
            backend_name="AVFoundation",
        )
        d = caps.to_dict()
        assert d["resolutions"] == [[1920, 1080], [1280, 720]]
        assert d["current_fps"] == 30.0
        assert d["supports_autofocus"] is True
        assert d["backend_name"] == "AVFoundation"

    def test_status_to_dict(self):
        status = CameraStatus(
            connection_state=ConnectionState.CONNECTED,
            camera_type=CameraType.USB,
            source_id="0",
            display_name="USB Camera 0",
            frames_delivered=42,
        )
        d = status.to_dict()
        assert d["connection_state"] == "connected"
        assert d["camera_type"] == "usb"
        assert d["frames_delivered"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
