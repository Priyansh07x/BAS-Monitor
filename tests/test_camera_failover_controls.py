"""
test_camera_failover_controls.py -- Tests for Phase A4 (camera switching &
failover) and Phase A5 (focus / zoom / PTZ controls).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
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
from backend.video.camera_manager import CameraManager, FailoverEvent


# ================================================================== #
#  Helpers
# ================================================================== #

def _make_mock_camera(source_id: str, connect_ok=True, read_ok=True):
    """Create a USBCamera with a fully mocked cv2.VideoCapture."""
    cam = USBCamera(int(source_id.split(":")[-1]) if ":" in source_id else 0)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = connect_ok
    mock_cap.get.return_value = 0
    if read_ok:
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    else:
        mock_cap.read.return_value = (False, None)
    cam._cap = mock_cap if connect_ok else None
    cam._capabilities = CameraCapabilities()
    cam._error = "" if connect_ok else "simulated"
    return cam


def _make_failing_primary(failures=50):
    """Create a primary camera that has exceeded the failure threshold
    and whose connect() will also fail (simulating a dead device)."""
    cam = USBCamera(0)
    cam._consecutive_failures = failures
    cam._cap = None
    cam._error = "dead device"
    # Patch connect so reconnect also fails
    cam.connect = MagicMock(return_value=False)
    return cam


def _make_healthy_fallback(device_index=1):
    """Create a fallback camera that connects and reads successfully."""
    cam = USBCamera(device_index)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cam._cap = mock_cap
    cam._capabilities = CameraCapabilities()
    # Patch connect to succeed
    cam.connect = MagicMock(return_value=True)
    return cam


# ================================================================== #
#  Phase A4 — Lock / Unlock
# ================================================================== #

class TestCameraLock:

    def test_lock_succeeds_when_connected(self):
        mgr = CameraManager()
        cam = _make_mock_camera("usb:0", connect_ok=True)
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"

        assert mgr.lock() is True
        assert mgr.is_locked is True

    def test_lock_fails_when_no_camera(self):
        mgr = CameraManager()
        assert mgr.lock() is False
        assert mgr.is_locked is False

    def test_lock_fails_when_disconnected(self):
        mgr = CameraManager()
        cam = _make_mock_camera("usb:0", connect_ok=False)
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"

        assert mgr.lock() is False

    def test_unlock(self):
        mgr = CameraManager()
        cam = _make_mock_camera("usb:0", connect_ok=True)
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"
        mgr.lock()

        mgr.unlock()
        assert mgr.is_locked is False

    def test_disconnect_all_clears_lock(self):
        mgr = CameraManager()
        cam = _make_mock_camera("usb:0", connect_ok=True)
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"
        mgr.lock()

        mgr.disconnect_all()
        assert mgr.is_locked is False


# ================================================================== #
#  Phase A4 — Failover
# ================================================================== #

class TestFailover:

    def test_no_failover_below_threshold(self):
        """No failover if failures are below threshold."""
        cam = _make_mock_camera("usb:0")
        cam._consecutive_failures = 5

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"

        assert mgr.check_and_failover() is None

    def test_no_failover_when_no_active(self):
        mgr = CameraManager()
        assert mgr.check_and_failover() is None

    def test_failover_triggers_on_threshold(self):
        """When consecutive failures exceed threshold and reconnect fails,
        failover should switch to the fallback camera."""
        primary = _make_failing_primary(failures=50)
        fallback = _make_healthy_fallback(device_index=1)

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr.register("usb:1", fallback)
        mgr._active_id = "usb:0"

        event = mgr.check_and_failover()
        assert event is not None
        assert event.success is True
        assert event.from_source == "usb:0"
        assert event.to_source == "usb:1"
        assert mgr.get_active_id() == "usb:1"

    def test_failover_callback_invoked(self):
        """Failover callback should be called with the event."""
        primary = _make_failing_primary(failures=50)
        fallback = _make_healthy_fallback(device_index=1)

        callback = MagicMock()

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr.register("usb:1", fallback)
        mgr._active_id = "usb:0"
        mgr.set_failover_callback(callback)

        mgr.check_and_failover()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, FailoverEvent)
        assert event.success is True

    def test_failover_all_candidates_fail(self):
        """When all fallback cameras also fail, event.success should be False."""
        primary = _make_failing_primary(failures=50)
        # Fallback also can't connect
        fallback = USBCamera(1)
        fallback.connect = MagicMock(return_value=False)
        fallback._cap = None

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr.register("usb:1", fallback)
        mgr._active_id = "usb:0"

        event = mgr.check_and_failover()
        assert event is not None
        assert event.success is False
        assert event.to_source == "none"

    def test_failover_respects_fallback_order(self):
        """Fallback should be attempted in the configured order."""
        primary = _make_failing_primary(failures=50)

        # cam1 will fail
        cam1 = USBCamera(1)
        cam1.connect = MagicMock(return_value=False)
        cam1._cap = None

        # cam2 will succeed
        cam2 = _make_healthy_fallback(device_index=2)

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr.register("usb:1", cam1)
        mgr.register("usb:2", cam2)
        mgr._active_id = "usb:0"
        mgr.set_fallback_order(["usb:2", "usb:1"])  # cam2 first

        event = mgr.check_and_failover()
        assert event is not None
        assert event.success is True
        assert event.to_source == "usb:2"

    def test_failover_history_accumulates(self):
        """Failover history should accumulate events."""
        primary = _make_failing_primary(failures=50)
        fallback = _make_healthy_fallback(device_index=1)

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr.register("usb:1", fallback)
        mgr._active_id = "usb:0"

        mgr.check_and_failover()

        history = mgr.get_failover_history()
        assert len(history) >= 1
        assert "from_source" in history[0]
        assert "timestamp" in history[0]

    def test_read_triggers_failover(self):
        """CameraManager.read() should auto-trigger failover on sustained failure."""
        primary = USBCamera(0)
        # Set up a mock cap that returns failed reads
        mock_primary_cap = MagicMock()
        mock_primary_cap.isOpened.return_value = True
        mock_primary_cap.read.return_value = (False, None)
        primary._cap = mock_primary_cap
        primary._consecutive_failures = 29  # one more will hit threshold of 30
        # Patch connect so reconnect also fails
        primary.connect = MagicMock(return_value=False)

        fallback = _make_healthy_fallback(device_index=1)

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr.register("usb:1", fallback)
        mgr._active_id = "usb:0"

        # This read fails, bumps consecutive_failures to 30, triggers failover
        frame = mgr.read()
        assert mgr.get_active_id() == "usb:1"

    def test_reconnect_succeeds_avoids_failover(self):
        """If the primary reconnects successfully, no failover event occurs."""
        primary = USBCamera(0)
        primary._consecutive_failures = 50
        primary._cap = None
        # Patch connect to succeed on reconnect
        primary.connect = MagicMock(return_value=True)

        mgr = CameraManager()
        mgr._failure_threshold = 30
        mgr.register("usb:0", primary)
        mgr._active_id = "usb:0"

        event = mgr.check_and_failover()
        assert event is None  # reconnect worked, no failover
        assert mgr.get_active_id() == "usb:0"


# ================================================================== #
#  Phase A4 — FailoverEvent
# ================================================================== #

class TestFailoverEvent:

    def test_to_dict(self):
        event = FailoverEvent(
            from_source="usb:0",
            to_source="usb:1",
            reason="test reason",
            success=True,
        )
        d = event.to_dict()
        assert d["from_source"] == "usb:0"
        assert d["to_source"] == "usb:1"
        assert d["reason"] == "test reason"
        assert d["success"] is True
        assert "timestamp" in d


# ================================================================== #
#  Phase A5 — Hardware Controls (USBCamera)
# ================================================================== #

class TestUSBCameraControls:

    def test_controls_return_false_when_disconnected(self):
        cam = USBCamera(device_index=99)
        assert cam.set_autofocus(True) is False
        assert cam.set_focus(50.0) is False
        assert cam.set_zoom(2.0) is False
        assert cam.set_resolution(1920, 1080) is False

    def test_set_autofocus_requires_support(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_autofocus = False
        assert cam.set_autofocus(True) is False

    def test_set_autofocus_with_support(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_autofocus = True
        cam._cap.get.return_value = 1  # autofocus readback = enabled
        result = cam.set_autofocus(True)
        assert result is True

    def test_set_focus_requires_manual_support(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_manual_focus = False
        assert cam.set_focus(50.0) is False

    def test_set_focus_with_support(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_manual_focus = True
        cam._cap.get.return_value = 50.0
        result = cam.set_focus(100.0)
        assert result is True
        assert cam._capabilities.current_focus == 50.0  # from mock get()

    def test_set_zoom_requires_support(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_optical_zoom = False
        assert cam.set_zoom(2.0) is False

    def test_set_zoom_with_support(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_optical_zoom = True
        cam._cap.get.return_value = 2.0
        result = cam.set_zoom(3.0)
        assert result is True
        assert cam._capabilities.current_zoom == 2.0

    def test_set_resolution_accepted(self):
        cam = _make_mock_camera("usb:0")

        def mock_get(prop):
            if prop == 3:  # FRAME_WIDTH
                return 1920
            if prop == 4:  # FRAME_HEIGHT
                return 1080
            return 0

        cam._cap.get.side_effect = mock_get

        result = cam.set_resolution(1920, 1080)
        assert result is True
        assert cam._capabilities.current_resolution == (1920, 1080)

    def test_set_resolution_rejected(self):
        """Camera may clamp to a different resolution."""
        cam = _make_mock_camera("usb:0")

        def mock_get(prop):
            if prop == 3:
                return 1280  # camera clamped
            if prop == 4:
                return 720
            return 0

        cam._cap.get.side_effect = mock_get

        result = cam.set_resolution(3840, 2160)
        assert result is False
        assert cam._capabilities.current_resolution == (1280, 720)


# ================================================================== #
#  Phase A5 — Controls via CameraManager delegation
# ================================================================== #

class TestManagerControlDelegation:

    def test_controls_return_false_with_no_active(self):
        mgr = CameraManager()
        assert mgr.set_autofocus(True) is False
        assert mgr.set_focus(50.0) is False
        assert mgr.set_zoom(2.0) is False
        assert mgr.set_resolution(1920, 1080) is False

    def test_controls_delegate_to_active(self):
        cam = _make_mock_camera("usb:0")
        cam._capabilities.supports_autofocus = False
        cam._capabilities.supports_manual_focus = False
        cam._capabilities.supports_optical_zoom = False

        mgr = CameraManager()
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"

        # These return False because capabilities aren't set,
        # but they should not raise exceptions
        assert mgr.set_autofocus(True) is False
        assert mgr.set_focus(50.0) is False
        assert mgr.set_zoom(2.0) is False

    def test_resolution_delegates(self):
        cam = _make_mock_camera("usb:0")

        def mock_get(prop):
            if prop == 3:
                return 1280
            if prop == 4:
                return 720
            return 0

        cam._cap.get.side_effect = mock_get

        mgr = CameraManager()
        mgr.register("usb:0", cam)
        mgr._active_id = "usb:0"

        result = mgr.set_resolution(1280, 720)
        assert result is True


# ================================================================== #
#  Phase A5 — IPCamera controls (safe no-ops)
# ================================================================== #

class TestIPCameraControls:

    def test_ip_camera_controls_are_noop(self):
        """IP cameras don't support focus/zoom via OpenCV, so all return False."""
        cam = IPCamera(url="rtsp://fake")
        assert cam.set_autofocus(True) is False
        assert cam.set_focus(50.0) is False
        assert cam.set_zoom(2.0) is False
        assert cam.set_resolution(1920, 1080) is False


# ================================================================== #
#  Phase A5 — Capabilities serialization with new fields
# ================================================================== #

class TestCapabilitiesSerialization:

    def test_capabilities_include_ranges(self):
        caps = CameraCapabilities(
            resolutions=[(1920, 1080)],
            current_resolution=(1920, 1080),
            current_fps=30.0,
            supports_manual_focus=True,
            focus_range=(0.0, 255.0),
            current_focus=128.0,
            supports_optical_zoom=True,
            zoom_range=(1.0, 10.0),
            current_zoom=1.0,
        )
        d = caps.to_dict()
        assert d["focus_range"] == [0.0, 255.0]
        assert d["current_focus"] == 128.0
        assert d["zoom_range"] == [1.0, 10.0]
        assert d["current_zoom"] == 1.0

    def test_capabilities_none_ranges(self):
        caps = CameraCapabilities()
        d = caps.to_dict()
        assert d["focus_range"] is None
        assert d["current_focus"] is None
        assert d["zoom_range"] is None
        assert d["current_zoom"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
