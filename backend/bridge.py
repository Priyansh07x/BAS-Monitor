"""
bridge.py -- QWebChannel bridge exposed to JavaScript.

Every method here should be a thin wrapper: parse/serialize data and
delegate to experiment_manager / app_state. No business logic here.
"""

import json
import cv2

from PySide6.QtCore import QObject, Slot, Signal, QTimer

from . import experiment_manager
from .app_state import AppState
from .video.camera import Camera
from .video.recorder import VideoRecorder
from .logging.system_logger import get_system_logger


class Bridge(QObject):
    stateChanged = Signal(str)
    # Recording signals
    recordingStarted = Signal(str)
    recordingStopped = Signal(str)
    logMessage = Signal(str, str)
    recTimerTick = Signal(str)

    def __init__(self, app_state: AppState):
        super().__init__()
        self.state = app_state
        self.camera = self.state.camera
        # Recording support
        self._slog = get_system_logger()
        self._recorder = VideoRecorder()
        self._rec_seconds = 0
        self._rec_timer = None

    @Slot(result=str)
    def getExperiments(self):
        experiments = experiment_manager.get_experiments()
        return json.dumps(experiments)

    @Slot(str, result=str)
    def getExperiment(self, experiment_id):
        exp = experiment_manager.get_experiment(experiment_id)
        return json.dumps(exp) if exp else "null"

    @Slot(str, result=str)
    def createExperiment(self, experiment_json):
        try:
            payload = json.loads(experiment_json)
            new_exp = experiment_manager.create_experiment(
                name=payload.get("name", ""),
                description=payload.get("description", ""),
                steps=payload.get("steps", []),
            )
            return json.dumps({"success": True, "experiment": new_exp})
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)})

    @Slot(str, str, result=str)
    def saveExperiment(self, experiment_id, experiment_json):
        payload = json.loads(experiment_json)
        updated = experiment_manager.update_experiment(
            experiment_id,
            name=payload.get("name"),
            description=payload.get("description"),
            steps=payload.get("steps"),
        )
        if updated:
            return json.dumps({"success": True, "experiment": updated})
        return json.dumps({"success": False, "error": "Experiment not found"})

    @Slot(str, result=bool)
    def deleteExperiment(self, experiment_id):
        return experiment_manager.delete_experiment(experiment_id)

    @Slot(str, result=str)
    def loadExperiment(self, experiment_id):
        exp = experiment_manager.get_experiment(experiment_id)
        if not exp:
            return json.dumps({"success": False, "error": "Not found"})
        self.state.set_experiment(experiment_id)
        self._emit_state()
        return json.dumps({"success": True, "experiment": exp})

    @Slot(result=bool)
    def startCamera(self):
        success = self.camera.open()

        if success:
            self.state.set_video_source("camera")
            self._emit_state()

        return success

    @Slot(result=bool)
    def cameraIsOpen(self):
        return self.camera.is_open()


    @Slot()
    def stopCamera(self):
        self.camera.release()
        self.state.set_video_source(None)
        self._emit_state()
    @Slot(str)
    def selectVideoSource(self, source):
        self.state.set_video_source(source)
        self._emit_state()


    @Slot(result=str)
    def getState(self):
        return json.dumps(self.state.to_dict())
    
    @Slot()
    def startMonitoring(self):
        self.state.start_monitoring()
        self._emit_state()

    @Slot()
    def pauseMonitoring(self):
        self.state.pause_monitoring()
        self._emit_state()

    @Slot()
    def stopMonitoring(self):
        self.state.stop_monitoring()
        self._emit_state()

    def _emit_state(self):
        self.stateChanged.emit(json.dumps(self.state.to_dict()))

    # ================================================================== #
    #  VIDEO RECORDING
    # ================================================================== #

    @Slot()
    def startRecording(self):
        """Start recording the live camera feed to a local MP4 file."""
        if self._recorder.is_recording:
            self.logMessage.emit("Recording already in progress.", "WARN")
            return

        if not self.camera.is_open():
            self.startCamera()
            if not self.camera.is_open():
                self.logMessage.emit("Cannot record: camera not available.", "ERR")
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
    #  FRAME FEED (hook into upstream getCameraFrame)
    # ================================================================== #

    # Override getCameraFrame to also feed frames to the recorder
    @Slot(result=str)
    def getCameraFrame(self):
        frame = self.camera.read()

        if frame is None:
            return ""

        # Feed frame to recorder if recording is active
        if self._recorder.is_recording:
            self._recorder.enqueue_frame(frame)

        import base64
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    # ================================================================== #
    #  CLEANUP
    # ================================================================== #

    def shutdown(self):
        """Release all resources on app exit."""
        if self._recorder.is_recording:
            self.stopRecording()
        self.stopCamera()
        self._recorder.shutdown()