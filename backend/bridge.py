"""
bridge.py -- QWebChannel bridge exposed to JavaScript.

Every method here should be a thin wrapper: parse/serialize data and
delegate to experiment_manager / app_state. No business logic here.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PySide6.QtCore import QObject, Slot, Signal, QTimer
except ImportError:
    # Lightweight pure-Python fallback for headless/CI test environments without PySide6
    class Signal:
        def __init__(self, *types):
            self._types = types
            self._slots = []

        def connect(self, slot):
            if slot not in self._slots:
                self._slots.append(slot)

        def disconnect(self, slot=None):
            if slot is None:
                self._slots.clear()
            elif slot in self._slots:
                self._slots.remove(slot)

        def emit(self, *args):
            for slot in list(self._slots):
                try:
                    slot(*args)
                except Exception:
                    pass

    def Slot(*types, result=None):
        def decorator(func):
            return func
        return decorator

    class QObject:
        def __init__(self, parent=None):
            self._parent = parent

    class QTimer:
        def __init__(self, parent=None):
            self._parent = parent
            self.timeout = Signal()
            self._is_active = False

        def start(self, msec=1000):
            self._is_active = True

        def stop(self):
            self._is_active = False

from . import experiment_manager
from .app_state import AppState
from .video.camera_manager import get_camera_manager
from .video.recorder import VideoRecorder
from .network.streamer import get_ip_streamer
from .logging.system_logger import get_system_logger
from .voice.voice_alert import get_voice_service
from .ai.inference_worker import InferenceWorker


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
    
    # AI Perception signals (frozen 8-field public AI contract)
    aiResultReady = Signal(dict)
    aiResultJsonReady = Signal(str)

    # Dedicated Procedural Recovery Channel (Gate B12.2)
    recoveryAlertReady = Signal(dict)
    recoveryAlertJsonReady = Signal(str)

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
        # Inference Worker (owned via AppState)
        self.inference_worker: Optional[InferenceWorker] = getattr(self.state, "inference_worker", None)
        if self.inference_worker is not None:
            self.inference_worker.result_callback = self._on_ai_result_from_worker
            self.inference_worker.recovery_callback = self._on_recovery_alert_from_worker

        # Recovery Debounce & Cache State
        self._recovery_debounce_lock = threading.Lock()
        self._last_recovery_key: Optional[tuple] = None
        self._last_recovery_time: float = 0.0

        # Camera Ingestion Telemetry (Monotonic)
        self._camera_telemetry_lock = threading.Lock()
        self._camera_frame_count: int = 0
        self._camera_ingest_times: collections.deque[float] = collections.deque(maxlen=200)

    # ================================================================== #
    #  EXPERIMENT MANAGEMENT
    # ================================================================== #


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

    # ================================================================== #
    #  CAMERA CONTROLS
    # ================================================================== #

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

    # ================================================================== #
    #  STATE & MONITORING / FSM LIFECYCLE (B9.3)
    # ================================================================== #

    @Slot(result=str)
    def getState(self):
        return json.dumps(self.state.to_dict())
    
    def _reset_recovery_debounce(self) -> None:
        """Resets the bridge and voice alert recovery debounce cache."""
        with self._recovery_debounce_lock:
            self._last_recovery_key = None
            self._last_recovery_time = 0.0
        if hasattr(self, "_voice") and self._voice is not None:
            self._voice.reset_debounce()

    @Slot(result=bool)
    def startMonitoring(self) -> bool:
        """Starts the authoritative FSM procedure and background AI inference worker."""
        self._reset_recovery_debounce()
        self.state.start_monitoring()
        if self.inference_worker is not None and not self.inference_worker.is_running:
            self.inference_worker.start()
        self._emit_state()
        self._slog.info("Procedure monitoring started via Bridge.")
        return True

    @Slot(str, str, result=str)
    def startProcedure(self, operator_id: str = "Astronaut-01", notes: str = "") -> str:
        """Starts the authoritative FSM procedure session and background AI worker."""
        self._reset_recovery_debounce()
        session = self.state.start_monitoring(operator_id=operator_id, notes=notes)
        if self.inference_worker is not None and not self.inference_worker.is_running:
            self.inference_worker.start()
        self._emit_state()
        self._slog.info(f"Procedure session started for operator {operator_id} via Bridge.")
        return json.dumps(session) if isinstance(session, dict) else json.dumps({"status": "RUNNING"})

    @Slot()
    def pauseMonitoring(self) -> None:
        """Pauses procedure monitoring on FSM and pauses background AI worker."""
        self.state.pause_monitoring()
        if self.inference_worker is not None and self.inference_worker.is_running:
            self.inference_worker.pause()
        self._emit_state()
        self._slog.info("Procedure monitoring paused via Bridge.")

    @Slot()
    def pauseProcedure(self) -> None:
        """Alias for pauseMonitoring."""
        self.pauseMonitoring()

    @Slot()
    def resumeMonitoring(self) -> None:
        """Resumes procedure monitoring on FSM and resumes background AI worker."""
        self.state.resume_monitoring()
        if self.inference_worker is not None and self.inference_worker.is_running:
            self.inference_worker.resume()
        self._emit_state()
        self._slog.info("Procedure monitoring resumed via Bridge.")

    @Slot()
    def resumeProcedure(self) -> None:
        """Alias for resumeMonitoring."""
        self.resumeMonitoring()

    @Slot()
    def stopMonitoring(self) -> None:
        """Stops procedure monitoring on FSM and stops AI worker."""
        self._reset_recovery_debounce()
        self.state.stop_monitoring()
        if self.inference_worker is not None:
            self.inference_worker.stop()
        self._emit_state()
        self._slog.info("Procedure monitoring stopped via Bridge.")

    @Slot()
    def stopProcedure(self) -> None:
        """Alias for stopMonitoring."""
        self.stopMonitoring()

    @Slot()
    def resetMonitoring(self) -> None:
        """Resets FSM procedure sequence, clears anomalies, recovery state, and resets AI worker."""
        self._reset_recovery_debounce()
        self.state.reset_monitoring()
        if self.inference_worker is not None:
            self.inference_worker.reset()
        self._emit_state()
        self._slog.info("Procedure monitoring reset via Bridge.")

    @Slot()
    def resetProcedure(self) -> None:
        """Alias for resetMonitoring."""
        self.resetMonitoring()

    @Slot(result=str)
    def getProcedureState(self) -> str:
        """Returns JSON containing authoritative FSM state and progress metrics."""
        return json.dumps(self.state.sequence_validator.get_progress())

    @Slot(result=str)
    def getProcedureProgress(self) -> str:
        """Alias for getProcedureState."""
        return self.getProcedureState()

    @Slot(result=str)
    def getFSMState(self) -> str:
        """Returns current FSM lifecycle state string ('IDLE', 'RUNNING', 'PAUSED', 'COMPLETED')."""
        return self.state.sequence_validator.state

    @Slot(result=str)
    def getActiveRecoveryGuidance(self) -> str:
        """
        Returns active procedural recovery guidance as JSON, or 'null' if no recovery is active.
        """
        rm = getattr(self.state, "recovery_manager", None)
        if rm is not None and rm.state == "RECOVERY_ACTIVE" and rm.last_recovery_event is not None:
            return json.dumps(rm.last_recovery_event.to_dict())
        return "null"

    def _emit_state(self):
        self.stateChanged.emit(json.dumps(self.state.to_dict()))

    # ================================================================== #
    #  AI INFERENCE WORKER INTEGRATION (B7.2 / B12.2)
    # ================================================================== #

    def _on_ai_result_from_worker(self, result: Dict[str, Any]) -> None:
        """
        Thread-safe callback invoked by background AIInferenceWorker thread.
        Forwards the frozen 8-field public AI contract payload to Qt queued signals.
        """
        try:
            self.aiResultReady.emit(result)
            self.aiResultJsonReady.emit(json.dumps(result))
        except Exception as err:
            self._slog.error(f"InferenceWorker callback signal error: {err}")

    def _on_recovery_alert_from_worker(self, rec_dict: Dict[str, Any]) -> None:
        """
        Thread-safe callback invoked by background AIInferenceWorker thread.
        Forwards structured procedural recovery events to dedicated out-of-band signals.
        """
        self.emit_recovery_alert(rec_dict)

    def emit_recovery_alert(
        self,
        event: Union[Any, Dict[str, Any]],
        debounce_seconds: float = 3.0,
    ) -> None:
        """
        Emits out-of-band recovery alert signals, plays audible guidance, and logs recovery events.
        Debounces repeated identical recovery conditions across consecutive frames.
        """
        if event is None:
            return

        event_dict = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        step_id = event_dict.get("expected_step", "")
        rec_instruction = event_dict.get("recovery_instruction", "")
        status = event_dict.get("procedural_status", "")

        # 1. Emit dedicated Qt recovery signals
        try:
            self.recoveryAlertReady.emit(event_dict)
            self.recoveryAlertJsonReady.emit(json.dumps(event_dict))
        except Exception as err:
            self._slog.error(f"Bridge recoveryAlert signal error: {err}")

        # 2. Debounce voice guidance and operational logging
        key = (str(step_id), str(rec_instruction).strip(), str(status))
        now = time.monotonic()
        should_voice_and_log = False

        with self._recovery_debounce_lock:
            if self._last_recovery_key != key or (now - self._last_recovery_time) >= debounce_seconds:
                self._last_recovery_key = key
                self._last_recovery_time = now
                should_voice_and_log = True

        if should_voice_and_log:
            if rec_instruction:
                self._voice.alert_recovery_guidance(step_id, rec_instruction, debounce_seconds=debounce_seconds)
                self._slog.log_recovery(str(step_id), rec_instruction, str(status))

            exp_logger = getattr(self.state.sequence_validator, "_exp_logger", None)
            if exp_logger and getattr(exp_logger, "active_session", None):
                try:
                    exp_logger.log_recovery(event_dict)
                except Exception as log_err:
                    self._slog.error(f"ExperimentLogger log_recovery error: {log_err}")

    @Slot(result=bool)
    def startAI(self) -> bool:
        """Starts the dedicated background AI inference worker thread."""
        if self.inference_worker is not None:
            ok = self.inference_worker.start()
            if ok:
                self._slog.info("AI Inference Worker started via Bridge.")
            return ok
        return False

    @Slot(result=bool)
    def startInference(self) -> bool:
        """Alias for startAI."""
        return self.startAI()

    @Slot()
    def stopAI(self) -> None:
        """Stops the dedicated background AI inference worker."""
        if self.inference_worker is not None:
            self.inference_worker.stop()
            self._slog.info("AI Inference Worker stopped via Bridge.")

    @Slot()
    def stopInference(self) -> None:
        """Alias for stopAI."""
        self.stopAI()

    @Slot()
    def pauseAI(self) -> None:
        """Pauses AI processing in the background worker."""
        if self.inference_worker is not None:
            self.inference_worker.pause()
            self._slog.info("AI Inference Worker paused via Bridge.")

    @Slot()
    def pauseInference(self) -> None:
        """Alias for pauseAI."""
        self.pauseAI()

    @Slot()
    def resumeAI(self) -> None:
        """Resumes AI processing in the background worker."""
        if self.inference_worker is not None:
            self.inference_worker.resume()
            self._slog.info("AI Inference Worker resumed via Bridge.")

    @Slot()
    def resumeInference(self) -> None:
        """Alias for resumeAI."""
        self.resumeAI()

    @Slot()
    def resetAI(self) -> None:
        """Flushes worker buffer and resets pipeline temporal state."""
        if self.inference_worker is not None:
            self.inference_worker.reset()
            self._slog.info("AI Inference Worker reset via Bridge.")

    @Slot()
    def resetInference(self) -> None:
        """Alias for resetAI."""
        self.resetAI()

    @Slot(result=str)
    def getLatestAIResult(self) -> str:
        """Returns the most recent frozen 8-field public AI contract result as JSON."""
        if self.inference_worker is not None:
            latest = self.inference_worker.get_latest_result()
            if latest is not None:
                return json.dumps(latest)
        return "null"

    @Slot(result=bool)
    def isAIRunning(self) -> bool:
        """Returns True if the background AI inference worker is active."""
        if self.inference_worker is not None:
            return self.inference_worker.is_running
        return False

    @Slot(result=bool)
    def isAIPaused(self) -> bool:
        """Returns True if the background AI inference worker is paused."""
        if self.inference_worker is not None:
            return self.inference_worker.is_paused
        return False

    @Slot(result=str)
    def getAITelemetry(self) -> str:
        """Returns full AI inference worker telemetry and rolling stats as JSON."""
        if self.inference_worker is not None:
            return json.dumps(self.inference_worker.get_telemetry())
        return "{}"

    @Slot(result=str)
    def getTelemetrySnapshot(self) -> str:
        """Returns unified telemetry and diagnostic snapshot as JSON string (Gate B13.1)."""
        if self.inference_worker is not None and getattr(self.inference_worker, "telemetry_aggregator", None) is not None:
            return self.inference_worker.telemetry_aggregator.get_snapshot_json()
        return "{}"

    @Slot(result=str)
    def getDiagnosticSnapshot(self) -> str:
        """Alias for getTelemetrySnapshot."""
        return self.getTelemetrySnapshot()

    @Slot(result=str)
    def getCameraTelemetry(self) -> str:
        """Returns camera acquisition telemetry and rolling FPS as JSON."""
        return json.dumps(self.get_camera_telemetry())

    def get_camera_telemetry(self) -> Dict[str, Any]:
        """Returns dictionary containing camera frame count and rolling ingest FPS."""
        with self._camera_telemetry_lock:
            cnt = self._camera_frame_count
            times = list(self._camera_ingest_times)
        fps = self._compute_rolling_fps(times)
        return {
            "camera_frames_acquired": cnt,
            "camera_ingest_fps": round(fps, 2),
            "target_fps": 30.0,
        }

    def reset_camera_telemetry(self) -> None:
        """Resets camera acquisition frame counters and timestamp history."""
        with self._camera_telemetry_lock:
            self._camera_frame_count = 0
            self._camera_ingest_times.clear()

    @staticmethod
    def _compute_rolling_fps(timestamps: Sequence[float], window_sec: float = 2.0) -> float:
        """Computes rolling frequency (FPS) from a sequence of monotonic timestamps."""
        if not timestamps or len(timestamps) < 2:
            return 0.0
        t_latest = timestamps[-1]
        recent = [t for t in timestamps if (t_latest - t) <= window_sec]
        if len(recent) < 2:
            return 0.0
        duration = recent[-1] - recent[0]
        if duration <= 0.0:
            return 0.0
        return float((len(recent) - 1) / duration)

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

        if summary.get("status") == "saved" and summary.get("filepath") is not None:
            filepath = summary["filepath"]
            size_mb = summary.get("file_size_bytes", 0) / (1024 * 1024)
            duration = summary.get("duration_seconds", 0)
            frames = summary.get("frame_count", 0)

            fp_name = getattr(filepath, "name", str(filepath))
            msg = (
                f"Recording saved: {fp_name} "
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

    # Override getCameraFrame to also feed frames to the recorder/streamer and AI worker
    @Slot(result=str)
    def getCameraFrame(self):
        import time
        t_capture = time.monotonic()
        frame = self._cam_mgr.read()

        if frame is None:
            return ""

        with self._camera_telemetry_lock:
            self._camera_frame_count += 1
            self._camera_ingest_times.append(t_capture)

        # 1. Feed frame to recorder if recording is active (non-blocking)
        if self._recorder.is_recording:
            self._recorder.enqueue_frame(frame)
            
        # 2. Feed frame to IP streamer if active (non-blocking)
        if self._streamer.is_running:
            self._streamer.update_frame(frame)

        # 3. Submit latest frame to InferenceWorker if running (non-blocking)
        if self.inference_worker is not None and self.inference_worker.is_running:
            self.inference_worker.submit_frame(
                frame=frame,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "source": self.state.selected_video_source or "camera",
                    "capture_monotonic": t_capture,
                },
            )

        if cv2 is not None:
            import base64
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                return ""
            return base64.b64encode(buffer).decode("utf-8")
        return ""

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
    #  LOGGING & CLEANUP
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
        if self.inference_worker is not None:
            self.inference_worker.shutdown(timeout=2.0)
        self.stopCamera()
        self._recorder.shutdown()
        self._voice.shutdown()