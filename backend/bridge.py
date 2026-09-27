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
from .video.camera_manager import get_camera_manager
from .video.recorder import VideoRecorder
from .network.streamer import get_ip_streamer
from .logging.system_logger import get_system_logger
from .voice.voice_alert import get_voice_service


class Bridge(QObject):
    stateChanged = Signal(str)
    # Recording signals
    recordingStarted = Signal(str)
    recordingStopped = Signal(str)
    
    # Streaming signals
    streamStarted = Signal(str)
    streamStopped = Signal()

    # Phase A4 — failover notification
    cameraFailover = Signal(str)     # emits JSON FailoverEvent
    
    logMessage = Signal(str, str)
    recTimerTick = Signal(str)

    def __init__(self, app_state: AppState):
        super().__init__()
        self.state = app_state
        self._cam_mgr = self.state.camera_manager
        # Recording & Streaming support
        self._slog = get_system_logger()
        self._recorder = VideoRecorder()
        self._streamer = get_ip_streamer()
        self._rec_seconds = 0
        self._rec_timer = None
        self._voice = get_voice_service()

        # Phase A4 — wire failover callback for operator notification
        self._cam_mgr.set_failover_callback(self._on_camera_failover)

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
        """Start the default (or previously selected) camera."""
        active = self._cam_mgr.get_active()
        if active and active.is_connected():
            self.state.set_video_source("camera")
            self._emit_state()
            return True

        # Enumerate USB cameras if registry is empty
        sources = self._cam_mgr.list_sources()
        if not sources:
            self._cam_mgr.enumerate_usb_cameras(max_index=3)
            sources = self._cam_mgr.list_sources()

        if not sources:
            return False

        success = self._cam_mgr.select(sources[0]["source_id"])
        if success:
            self.state.set_video_source("camera")
            self._emit_state()
        return success

    @Slot(result=bool)
    def cameraIsOpen(self):
        active = self._cam_mgr.get_active()
        return active is not None and active.is_connected()

    @Slot()
    def stopCamera(self):
        active = self._cam_mgr.get_active()
        if active:
            active.disconnect()
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

        if not self.cameraIsOpen():
            self.startCamera()
            if not self.cameraIsOpen():
                self.logMessage.emit("Cannot record: camera not available.", "ERR")
                return

        try:
            # Match 10fps polling rate from frontend (100ms interval)
            fps = 10.0
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
    #  IP STREAMING
    # ================================================================== #

    @Slot(result=str)
    def startStreaming(self):
        """Start the MJPEG HTTP IP stream."""
        if self._streamer.is_running:
            url = f"http://{self._streamer.get_local_ip()}:{self._streamer.port}/stream"
            self.logMessage.emit("Streaming already active.", "WARN")
            return url

        if not self.cameraIsOpen():
            self.startCamera()
            if not self.cameraIsOpen():
                self.logMessage.emit("Cannot stream: camera not available.", "ERR")
                return ""

        url = self._streamer.start()
        self._slog.info(f"IP Streaming started at {url}")
        self.logMessage.emit(f"HTTP MJPEG Stream active at: {url}", "STREAM")
        self.streamStarted.emit(url)
        return url

    @Slot()
    def stopStreaming(self):
        """Stop the IP stream."""
        if not self._streamer.is_running:
            return
            
        self._streamer.stop()
        self._slog.info("IP Streaming stopped.")
        self.logMessage.emit("HTTP MJPEG Stream stopped.", "STREAM")
        self.streamStopped.emit()

    @Slot(result=bool)
    def isStreaming(self) -> bool:
        return self._streamer.is_running

    # ================================================================== #
    #  FRAME FEED (hook into upstream getCameraFrame)
    # ================================================================== #

    # Override getCameraFrame to also feed frames to the recorder/streamer
    @Slot(result=str)
    def getCameraFrame(self):
        frame = self._cam_mgr.read()

        if frame is None:
            return ""

        # Feed frame to recorder if recording is active
        if self._recorder.is_recording:
            self._recorder.enqueue_frame(frame)
            
        # Feed frame to IP streamer if active
        if self._streamer.is_running:
            self._streamer.update_frame(frame)

        import base64
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    # ================================================================== #
    #  VOICE TTS
    # ================================================================== #

    @Slot(str)
    def speak(self, text):
        """Send text to the offline TTS engine for spoken output."""
        self._voice.speak(text, priority=False)

    @Slot(str)
    def speakPriority(self, text):
        """Send high-priority text to the TTS engine (flushes queue)."""
        self._voice.speak(text, priority=True)

    @Slot(bool)
    def setVoiceEnabled(self, enabled):
        """Toggle the offline TTS engine on or off."""
        self._voice.set_enabled(enabled)
        self._slog.info(f"Voice TTS {'enabled' if enabled else 'disabled'}")

    # ================================================================== #
    #  CLEANUP
    # ================================================================== #
    @Slot(str, str)
    def logFromFrontend(self, message: str, level: str) -> None:
        """Receive a log message from the JS frontend and write to Python system logger."""
        if level == "ERR":
            self._slog.error(f"[UI] {message}")
        elif level == "WARN":
            self._slog.warn(f"[UI] {message}")
        elif level == "AI":
            self._slog.ai(f"[UI] {message}")
        elif level == "STREAM":
            self._slog.stream(f"[UI] {message}")
        else:
            self._slog.info(f"[UI] {message}")

    # ================================================================== #
    #  PHASE A2 — Camera Source Management
    # ================================================================== #

    @Slot(result=str)
    def listCameraSources(self):
        """Return JSON array of all registered camera sources with status."""
        return json.dumps(self._cam_mgr.list_sources())

    @Slot(result=str)
    def enumerateUSBCameras(self):
        """Probe USB indices 0-4 and return the source_ids found."""
        found = self._cam_mgr.enumerate_usb_cameras(max_index=5)
        return json.dumps(found)

    @Slot(str, str, result=str)
    def addIPCamera(self, url, label):
        """Register an IP camera by its stream URL. Returns its source_id."""
        source_id = self._cam_mgr.add_ip_camera(url, label=label or None)
        self._slog.info(f"Registered IP camera: {source_id}")
        return source_id

    @Slot(str, result=bool)
    def selectCameraSource(self, source_id):
        """Select and connect a specific camera source by its source_id."""
        success = self._cam_mgr.select(source_id)
        if success:
            self.state.set_video_source("camera")
            self._slog.info(f"Camera selected: {source_id}")
            self._emit_state()
        else:
            self._slog.error(f"Failed to select camera: {source_id}")
        return success

    # ================================================================== #
    #  PHASE A3 — Camera Capabilities & Readiness
    # ================================================================== #

    @Slot(result=str)
    def getCameraCapabilities(self):
        """Return JSON capabilities of the active camera source."""
        caps = self._cam_mgr.get_capabilities()
        if caps:
            return json.dumps(caps.to_dict())
        return json.dumps(None)

    @Slot(result=str)
    def getCameraStatus(self):
        """Return JSON status of the active camera source."""
        status = self._cam_mgr.get_status()
        if status:
            return json.dumps(status.to_dict())
        return json.dumps(None)

    @Slot(result=str)
    def cameraReadinessCheck(self):
        """
        Phase A3 readiness gate.
        Returns JSON: { ready: bool, source_id: str, checks: [...] }
        """
        result = self._cam_mgr.readiness_check()
        return json.dumps(result)

    # ================================================================== #
    #  PHASE A4 — Camera Lock & Failover
    # ================================================================== #

    @Slot(result=bool)
    def lockCamera(self):
        """Lock the active camera for experiment use. Prevents manual switching."""
        success = self._cam_mgr.lock()
        if success:
            self._slog.info(f"Camera locked: {self._cam_mgr.get_active_id()}")
        else:
            self._slog.error("Cannot lock camera: no active camera connected")
        return success

    @Slot()
    def unlockCamera(self):
        """Unlock the camera after experiment ends."""
        self._cam_mgr.unlock()
        self._slog.info("Camera unlocked")

    @Slot(result=bool)
    def isCameraLocked(self):
        """Check if the camera is currently locked for an experiment."""
        return self._cam_mgr.is_locked

    @Slot(result=str)
    def getFailoverHistory(self):
        """Return JSON array of all failover events."""
        return json.dumps(self._cam_mgr.get_failover_history())

    def _on_camera_failover(self, event):
        """
        Phase A4 callback — invoked by CameraManager on failover.
        Never silently switches viewpoints: logs, emits signal, and
        speaks an alert to the operator.
        """
        event_json = json.dumps(event.to_dict())
        self.cameraFailover.emit(event_json)

        if event.success:
            msg = f"Camera failover: switched from {event.from_source} to {event.to_source}"
            self._slog.warn(msg)
            self.logMessage.emit(msg, "WARN")
            self._voice.speak(
                f"Warning: camera switched from {event.from_source} to {event.to_source}",
                priority=True,
            )
        else:
            msg = f"Camera failover FAILED: {event.reason}"
            self._slog.error(msg)
            self.logMessage.emit(msg, "ERR")
            self._voice.speak("Critical: all cameras have failed", priority=True)

    # ================================================================== #
    #  PHASE A5 — Camera Hardware Controls
    # ================================================================== #

    @Slot(bool, result=bool)
    def setCameraAutofocus(self, enabled):
        """Enable or disable autofocus on the active camera."""
        success = self._cam_mgr.set_autofocus(enabled)
        if success:
            self._slog.info(f"Autofocus {'enabled' if enabled else 'disabled'}")
        return success

    @Slot(float, result=bool)
    def setCameraFocus(self, value):
        """Set manual focus value on the active camera."""
        success = self._cam_mgr.set_focus(value)
        if success:
            self._slog.info(f"Manual focus set to {value}")
        return success

    @Slot(float, result=bool)
    def setCameraZoom(self, value):
        """Set optical zoom level on the active camera."""
        success = self._cam_mgr.set_zoom(value)
        if success:
            self._slog.info(f"Zoom set to {value}")
        return success

    @Slot(int, int, result=bool)
    def setCameraResolution(self, width, height):
        """Change capture resolution on the active camera."""
        success = self._cam_mgr.set_resolution(width, height)
        if success:
            self._slog.info(f"Resolution set to {width}x{height}")
        else:
            self._slog.warn(f"Resolution {width}x{height} not accepted by camera")
        return success

    # --- Cleanup ---
    def shutdown(self):
        """Release all resources on app exit."""
        if self._recorder.is_recording:
            self.stopRecording()
        if self._streamer.is_running:
            self.stopStreaming()
        self._cam_mgr.disconnect_all()
        self._recorder.shutdown()
        self._voice.shutdown()