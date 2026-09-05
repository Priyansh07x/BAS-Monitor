"""
BackendBridge — QWebChannel IPC bridge between the PySide6 shell and the
frontend JavaScript application.

This bridge exposes Python slots that JavaScript can call via
`window.backend.<method>()`, and emits Qt signals that JavaScript
can listen to for real-time updates.

Architecture (from GUIDE.md §63):
    JS Frontend ←→ QWebChannel ←→ Python Backend
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication

from backend.video.recorder import VideoRecorder
from backend.logging.system_logger import get_system_logger


class BackendBridge(QObject):
    """
    Python ↔ JavaScript bridge object exposed to the frontend via QWebChannel.

    JS calls:   window.backend.startRecording()
    Python emits: recordingStarted, recordingStopped, frameCaptured, logMessage
    """

    # ----- Signals (Python → JS) -----
    recordingStarted = Signal(str)       # filename
    recordingStopped = Signal(str)       # JSON summary string
    logMessage = Signal(str, str)        # message, type (SYS/AI/WARN/ERR)
    cameraStatusChanged = Signal(str)    # status string
    streamingStatusChanged = Signal(str) # status string
    recTimerTick = Signal(str)           # "MM:SS" formatted string

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slog = get_system_logger()
        self._recorder = VideoRecorder()
        self._camera: Optional[cv2.VideoCapture] = None
        self._capture_timer: Optional[QTimer] = None
        self._rec_timer: Optional[QTimer] = None
        self._rec_seconds: int = 0
        self._is_camera_active: bool = False

    # ================================================================== #
    #  RECORDING CONTROLS (called from JS)
    # ================================================================== #

    @Slot()
    def startRecording(self):
        """Start recording the live camera feed to a local MP4 file."""
        if self._recorder.is_recording:
            self.logMessage.emit("Recording already in progress.", "WARN")
            return

        if not self._is_camera_active:
            # Auto-start the Python camera pipeline for recording
            self._start_camera_silent()
            if not self._is_camera_active:
                self.logMessage.emit("Cannot record: Failed to open system webcam.", "ERR")
                return

        try:
            # Force standard 720p broadcast resolution for QuickTime compatibility
            fps = 30.0
            w, h = 1280, 720

            filepath = self._recorder.start_recording(
                fps=fps,
                frame_size=(w, h),
            )

            # Start the recording elapsed timer
            self._rec_seconds = 0
            self._rec_timer = QTimer(self)
            self._rec_timer.timeout.connect(self._tick_rec_timer)
            self._rec_timer.start(1000)

            self._slog.info(f"Recording started: {filepath.name}")
            self.logMessage.emit(f"Recording started -> {filepath.name}", "SYS")
            self.recordingStarted.emit(filepath.name)

        except RuntimeError as e:
            self._slog.error(f"Recording failed: {e}")
            self.logMessage.emit(f"Recording error: {e}", "ERR")

    @Slot()
    def stopRecording(self):
        """Stop the current recording and finalize the MP4 file."""
        if not self._recorder.is_recording:
            self.logMessage.emit("No active recording to stop.", "WARN")
            return

        # Stop rec timer
        if self._rec_timer:
            self._rec_timer.stop()
            self._rec_timer = None

        summary = self._recorder.stop_recording()

        if summary["status"] == "saved":
            filepath = summary["filepath"]
            size_mb = summary["file_size_bytes"] / (1024 * 1024)
            duration = summary["duration_seconds"]
            frames = summary["frame_count"]

            msg = (
                f"Recording saved: {filepath.name} "
                f"({size_mb:.1f} MB, {frames} frames, {duration}s)"
            )
            self._slog.info(msg)
            self.logMessage.emit(msg, "SYS")
            self.recordingStopped.emit(str(filepath))

    @Slot(result=bool)
    def isRecording(self) -> bool:
        """Check if a recording is currently in progress."""
        return self._recorder.is_recording

    def _tick_rec_timer(self):
        """Called every second while recording to update the UI timer."""
        self._rec_seconds += 1
        mins = self._rec_seconds // 60
        secs = self._rec_seconds % 60
        self.recTimerTick.emit(f"{mins:02d}:{secs:02d}")

    # ================================================================== #
    #  CAMERA CONTROLS (called from JS)
    # ================================================================== #

    @Slot()
    def startPythonCamera(self):
        """
        Open the system webcam via OpenCV and start feeding frames.
        In this mode, Python owns the camera — not the browser.
        Frames are fed to the recorder if recording is active.
        """
        if self._is_camera_active:
            self.logMessage.emit("Camera already active.", "WARN")
            return

        self._camera = cv2.VideoCapture(0)
        if not self._camera.isOpened():
            self.logMessage.emit("Failed to open system webcam (index 0).", "ERR")
            self._slog.error("cv2.VideoCapture(0) failed to open.")
            self._camera = None
            return

        self._is_camera_active = True

        # Start a QTimer to grab frames at ~30fps
        self._capture_timer = QTimer(self)
        self._capture_timer.timeout.connect(self._grab_frame)
        self._capture_timer.start(33)  # ~30fps

        self._slog.info("Python camera pipeline started (OpenCV).")
        self.logMessage.emit("Python camera pipeline active (OpenCV).", "SYS")
        self.cameraStatusChanged.emit("ACTIVE")

    @Slot()
    def stopPythonCamera(self):
        """Stop the Python camera capture pipeline."""
        if self._capture_timer:
            self._capture_timer.stop()
            self._capture_timer = None

        if self._camera:
            self._camera.release()
            self._camera = None

        self._is_camera_active = False
        self._slog.info("Python camera pipeline stopped.")
        self.logMessage.emit("Python camera pipeline stopped.", "SYS")
        self.cameraStatusChanged.emit("STANDBY")

    @Slot(result=bool)
    def isCameraActive(self) -> bool:
        """Check if the Python camera is currently active."""
        return self._is_camera_active

    def _start_camera_silent(self):
        """
        Open the OpenCV camera without emitting log messages to the frontend.
        Used internally when recording auto-starts the camera.
        """
        if self._is_camera_active:
            return

        self._camera = cv2.VideoCapture(0)
        if not self._camera.isOpened():
            self._slog.error("cv2.VideoCapture(0) failed to open (silent start).")
            self._camera = None
            return

        self._is_camera_active = True

        self._capture_timer = QTimer(self)
        self._capture_timer.timeout.connect(self._grab_frame)
        self._capture_timer.start(33)

        self._slog.info("Python camera pipeline auto-started for recording.")

    def _grab_frame(self):
        """
        Called by QTimer ~30 times/sec. Grabs one frame from OpenCV
        and feeds it to the recorder if recording is active.
        """
        if not self._camera or not self._is_camera_active:
            return

        ret, frame = self._camera.read()
        if not ret:
            return

        # Feed frame to recorder (non-blocking enqueue)
        if self._recorder.is_recording:
            self._recorder.enqueue_frame(frame)

    # ================================================================== #
    #  CLEANUP
    # ================================================================== #

    def shutdown(self):
        """Release all resources on app exit."""
        if self._recorder.is_recording:
            self.stopRecording()
        self.stopPythonCamera()
        self._recorder.shutdown()
