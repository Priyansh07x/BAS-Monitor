# flow.md — BAS-Monitor Execution Flow

> This document maps the **complete execution flow** of the BAS-Monitor application.
> Updated: 2026-09-16

---

## 1. Project Structure

```
BAS-Monitor/                          ← PROJECT_ROOT (repo root)
├── main.py                           ← **PRIMARY APPLICATION ENTRY POINT**
├── backend/                          ← ACTIVE backend
│   ├── __init__.py
│   ├── app_state.py
│   ├── bridge.py
│   ├── experiment_manager.py
│   ├── storage.py
│   ├── ai/
│   ├── experiment/
│   ├── video/
│   ├── voice/
│   ├── network/
│   └── logging/
├── config/                           ← Configuration files
├── data/                             ← Data, logs, and video assets
├── docs/                             ← Documentation and Architecture logs
│   ├── decision.md                   ← Architectural decision log
│   ├── flow.md                       ← Execution flow mapping
│   ├── DESIGN.md
│   └── GUIDE.md
├── frontend/                         ← Web/GUI assets (HTML, JS, CSS)
├── tests/                            ← Automated tests
│   ├── test_*.py                     ← Unit test files
│   └── manual/                       ← Manual test scripts (ignored by pytest)
│       ├── test_functionalities.py
│       ├── test_backend.py
│       └── ...
├── package.json                      ← Frontend dependency config
└── requirements.txt                  ← Python dependencies
```

---

## 2. Application Entry Point & Startup Sequence

**Entry command:** `python BAS-Monitor/main.py`

```
                    ┌─────────────────────────────────────┐
                    │   python BAS-Monitor/main.py        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ sys.path.insert(0, PROJECT_ROOT)    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ QApplication(sys.argv)              │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ QMainWindow() — Status Display      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ AppState() instantiated              │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ Bridge(app_state) instantiated       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ GUIServer(bridge, FRONTEND_DIR)      │
                    │ → HTTP Server (8000)                 │
                    │ → QWebSocketServer (8001)            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ WebRTCServer(port=8555)              │
                    │ → aiohttp (sdp offer/answer)         │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ window.show()                        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ aboutToQuit → cleanup()              │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ app.exec() — Qt event loop starts    │
                    └─────────────────────────────────────┘
```

### Detailed Startup Chain

| Step | File | Function/Line | What Happens |
|------|------|--------------|--------------|
| 1 | `main.py:87` | `if __name__ == "__main__"` | Entry gate |
| 2 | `main.py:12-15` | `BASE_DIR / PROJECT_ROOT` | Computes paths, inserts repo root into `sys.path` |
| 3 | `main.py:53` | `QApplication(sys.argv)` | Creates Qt application |
| 4 | `main.py:58-65` | `QMainWindow()` | Creates status window showing LAN IP |
| 5 | `main.py:68` | `AppState()` | → `app_state.py:10` — creates Camera(), sets IDLE state |
| 6 | `main.py:69` | `Bridge(app_state)` | → `bridge.py:35` — inits recorder, streamer, voice singletons |
| 7 | `main.py:72` | `GUIServer()` | Starts HTTP (8000) for GUI & WebSocket (8001) for QWebChannel |
| 8 | `main.py:75-77` | `WebRTCServer()` | Starts `aiortc` + `aiohttp` signaling server (8555) |
| 9 | `main.py:82` | `aboutToQuit.connect(cleanup)` | Registers cleanup handler for servers and bridge |
| 10 | `main.py:85` | `app.exec()` | Starts Qt event loop (blocks here) |

---

## 3. Bridge.__init__ — Singleton Instantiation Chain

When `Bridge(app_state)` is called at `bridge.py:35`:

```
Bridge.__init__(app_state)
    │
    ├── self.state = app_state
    ├── self.camera = app_state.camera
    │
    ├── get_system_logger()               ← singleton
    │   └── SystemLogger()
    │       └── Creates data/logs/system_YYYYMMDD.log
    │
    ├── VideoRecorder()
    │   └── Creates data/videos/ directory
    │
    ├── get_ip_streamer()                 ← singleton
    │   └── IPStreamer(port=8554)
    │
    └── get_voice_service()               ← singleton
        └── VoiceAlertService()
            ├── _init_tts_engine()
            │   ├── macOS → engine_type = "macos_say"
            │   ├── else  → try pyttsx3
            │   └── fallback → engine_type = "mock"
            └── Start VoiceAlertWorker daemon thread
```

---

## 4. Frontend Initialization (JavaScript)

After `index.html` loads, `app.js` executes:

```
DOMContentLoaded fires
    │
    ├── new QWebChannel(qt.webChannelTransport, callback)
    │   │
    │   ├── backend = channel.objects.backend
    │   ├── Log 4 startup messages via log()
    │   ├── Connect backend.logMessage signal → log()
    │   ├── Connect backend.recTimerTick signal → basOnRecTimerTick()
    │   │
    │   ├── loadExperiments()
    │   │   └── backend.getExperiments() → JSON.parse → renderExperiments()
    │   │
    │   └── backend.startCamera(callback)
    │       ├── [Success] → startPythonCameraFeed()
    │       │   └── setInterval(100ms): backend.getCameraFrame() → img.src = base64
    │       └── [Failure] → console.error
    │
    ├── renderProcedureSteps()   ← Initial render (hardcoded 5 steps)
    ├── updateStatusUI()         ← Set initial status display
    ├── setSourceStandby()       ← Set input to standby
    ├── Attach event listeners to all control buttons
    └── setInterval(1000ms): updateClock()
```

### Frontend Key State Object
Defined at `app.js:556`:
```javascript
state = {
    isAnalyzing, isPaused, isRecording, isStreaming, isVoiceEnabled,
    inputSource: 'standby' | 'camera' | 'file',
    currentStepIndex, confidence, detectedAction, validationState,
    ipStreamUrl, analysisInterval, boxes, speechPending
}
```

---

## 5. Live Camera Frame Flow

This is the primary real-time loop. JS polls Python every 100ms:

```
    JS (setInterval 100ms)          Bridge.getCameraFrame()         Camera.read()
    ────────────────────            ───────────────────────         ─────────────
         │                                  │                           │
         │── backend.getCameraFrame() ─────►│                           │
         │                                  │── self.camera.read() ────►│
         │                                  │                           │
         │                                  │◄── frame (numpy) ────────│
         │                                  │                           │
         │                                  │── [if recording] ────────►  VideoRecorder.enqueue_frame()
         │                                  │── [if streaming] ────────►  IPStreamer.update_frame()
         │                                  │                           │
         │                                  │── cv2.imencode(".jpg") ───►
         │                                  │── base64.b64encode() ────►
         │                                  │                           │
         │◄── base64 JPEG string ──────────│                           │
         │                                  │                           │
         │── img.src = "data:image/         │                           │
         │   jpeg;base64," + data           │                           │
```

### Function Call Chain:
1. **JS**: `startPythonCameraFeed()` → `setInterval(100ms)` → `backend.getCameraFrame()`
2. **Python**: `Bridge.getCameraFrame()` — `bridge.py:269-288`
3. **Python**: → `Camera.read()` — `camera.py:22-31` → `cv2.VideoCapture.read()`
4. **Python**: → `VideoRecorder.enqueue_frame()` — `recorder.py:176-195` (if recording)
5. **Python**: → `IPStreamer.update_frame()` — `streamer.py:96-103` (if streaming)
6. **Python**: → `cv2.imencode()` → `base64.b64encode()` → return string
7. **JS**: Updates `<img>` element src

---

## 6. Experiment CRUD Flow

### Load Experiments
```
JS: loadExperiments()
  → backend.getExperiments()
    → Bridge.getExperiments()                 [bridge.py:48]
      → experiment_manager.get_experiments()   [experiment_manager.py:14]
        → storage.load_experiments()           [storage.py:25]
          → open("data/experiments.json")      [storage.py:28]
  → JSON.parse(result)
  → renderExperiments(experiments)             [app.js:240]
```

### Create Experiment
```
JS: btnSaveExp click
  → backend.createExperiment(JSON)
    → Bridge.createExperiment()                   [bridge.py:58]
      → experiment_manager.create_experiment()     [experiment_manager.py:26]
        → uuid.uuid4() for ID
        → _normalize_steps(steps)                  [experiment_manager.py:68]
        → storage.save_experiments()               [storage.py:35]
          → json.dump to data/experiments.json
  → closeExpModal()
  → loadExperiments()                             ← refresh list
```

### Load/Select Experiment
```
JS: loadExperiment(experimentId)                  [app.js:345]
  → backend.loadExperiment(experimentId)
    → Bridge.loadExperiment()                     [bridge.py:88]
      → experiment_manager.get_experiment()
      → app_state.set_experiment(id)              [app_state.py:34]
        → reset_monitoring()                      [app_state.py:18]
      → _emit_state()                            [bridge.py:141]
        → stateChanged.emit(JSON)
```

### Delete Experiment
```
JS: deleteExperiment(experimentId)                [app.js:521]
  → backend.deleteExperiment(experimentId)
    → Bridge.deleteExperiment()                   [bridge.py:84]
      → experiment_manager.delete_experiment()     [experiment_manager.py:59]
        → storage.save_experiments(filtered)
  → loadExperiments()                             ← refresh list
```

---

## 7. Recording Flow

```
    JS (app.js)              Bridge                VideoRecorder            WriterThread
    ───────────              ──────                ─────────────            ────────────
         │                      │                       │                       │
         │── startRecording() ─►│                       │                       │
         │                      │── check camera ──────►│                       │
         │                      │── start_recording() ─►│                       │
         │                      │                       │── VideoWriter(avc1) ─►│
         │                      │                       │── Start daemon ──────►│
         │                      │── QTimer(1s) ────────►│                       │
         │◄── recordingStarted ─│                       │                       │
         │                      │                       │                       │
         │    [Each getCameraFrame() call:]             │                       │
         │                      │── enqueue_frame() ───►│                       │
         │                      │                       │── queue.put() ───────►│
         │                      │                       │                       │── write()
         │                      │                       │                       │
         │── stopRecording() ──►│                       │                       │
         │                      │── stop QTimer ───────►│                       │
         │                      │── stop_recording() ──►│                       │
         │                      │                       │── recording.clear() ─►│
         │                      │                       │── sentinel None ─────►│
         │                      │                       │                       │── exit
         │                      │                       │── thread.join() ─────►│
         │                      │                       │── writer.release() ──►│
         │                      │◄── summary dict ─────│                       │
         │◄── recordingStopped ─│                       │                       │
```

### Key Functions:
| Step | Location | Function |
|------|----------|----------|
| Start | `bridge.py:149` | `Bridge.startRecording()` |
| Init Writer | `recorder.py:105` | `VideoRecorder.start_recording()` |
| Frame Queue | `recorder.py:176` | `VideoRecorder.enqueue_frame()` |
| BG Writer | `recorder.py:255` | `VideoRecorder._writer_loop()` |
| Stop | `bridge.py:186` | `Bridge.stopRecording()` |
| Finalize | `recorder.py:197` | `VideoRecorder.stop_recording()` |
| Timer Tick | `bridge.py:218` | `Bridge._tick_rec_timer()` → JS `basOnRecTimerTick()` |

---

## 8. IP Streaming Flow

```
    JS (app.js)          Bridge             IPStreamer           HTTPServer         Client
    ───────────          ──────             ─────────           ──────────         ──────
         │                  │                   │                   │                 │
         │── startStream ──►│                   │                   │                 │
         │                  │── start() ───────►│                   │                 │
         │                  │                   │── HTTPServer() ──►│                 │
         │                  │                   │── serve_forever ─►│                 │
         │                  │◄── URL ──────────│                   │                 │
         │◄── streamStarted │                   │                   │                 │
         │                  │                   │                   │                 │
         │  [getCameraFrame calls update_frame]  │                   │                 │
         │                  │── update_frame ──►│                   │                 │
         │                  │                   │── resize(854x480)►│                 │
         │                  │                   │                   │                 │
         │                  │                   │                   │◄── POST /offer ─│
         │                  │                   │◄── get_latest ───│                 │
         │                  │                   │── frame.copy() ──►│                 │
         │                  │                   │                   │── WebRTC RTP ──►│
         │                  │                   │                   │                 │
         │── stopStream ───►│                   │                   │                 │
         │                  │── stop() ────────►│                   │                 │
         │                  │                   │── shutdown() ────►│                 │
         │◄── streamStopped │                   │                   │                 │
```

### Key Functions:
| Step | Location | Function |
|------|----------|----------|
| Start | `bridge.py:230` | `Bridge.startStreaming()` |
| Server | `webrtc_server.py:68` | `WebRTCServer.start()` |
| Frame Update | `streamer.py:96` | `IPStreamer.update_frame()` |
| WebRTC Handler| `webrtc_server.py:40` | `offer(request) -> RTCPeerConnection` |
| Stop | `streamer.py:124` | `IPStreamer.stop()` |

---

## 9. Voice TTS Flow

```
    JS (app.js)          Bridge            VoiceAlertService      Worker Thread
    ───────────          ──────            ─────────────────      ─────────────
         │                  │                     │                     │
         │── speakVoice() ─►│                     │                     │
         │  (estimate dur,  │                     │                     │
         │   speechPending) │                     │                     │
         │                  │                     │                     │
         │── speak(text) ──►│                     │                     │
         │                  │── speak(text) ─────►│                     │
         │                  │                     │── queue.put() ─────►│
         │                  │                     │                     │
         │                  │                     │                     │── _dispatch_speech()
         │                  │                     │                     │   ├── macOS: say cmd
         │                  │                     │                     │   ├── pyttsx3: engine
         │                  │                     │                     │   └── mock: sleep
         │                  │                     │                     │
    [speechPending cleared  │                     │                     │
     after estimated time]  │                     │                     │
```

### Key Functions:
| Step | Location | Function |
|------|----------|----------|
| JS Speak | `app.js:1129` | `speakVoice()` |
| Bridge Speak | `bridge.py:295` | `Bridge.speak()` |
| Bridge Priority | `bridge.py:300` | `Bridge.speakPriority()` |
| Enqueue | `voice_alert.py:97` | `VoiceAlertService.speak()` |
| Worker | `voice_alert.py:132` | `VoiceAlertService._speech_worker()` |
| Dispatch | `voice_alert.py:152` | `VoiceAlertService._dispatch_speech()` |

---

## 10. Analysis Simulation Loop (Currently Hardcoded)

> **WARNING:** The analysis engine currently runs a **simulation loop** with hardcoded steps and random confidence values. There is NO real AI inference pipeline connected.

```
startAnalysis()                           [app.js:1167]
    │
    ├── Check inputSource != 'standby'
    │   └── [No] → Log warning, speak "select source", return
    │
    ├── state.isAnalyzing = true
    ├── Show AI detection overlay
    ├── startCanvasDrawing()              [app.js:1224]
    │   └── requestAnimationFrame loop:
    │       └── Draw animated bounding box + label on canvas
    │
    └── startSimulationLoop()             [app.js:1294]
        └── setInterval(3000ms):
            │
            ├── Skip if !isAnalyzing || isPaused || speechPending
            │
            ├── Random confidence = 88-99
            │
            ├── [confidence >= 95] → VALID
            │   ├── Log "step validated"
            │   ├── speakVoice("step complete")
            │   ├── [More steps?]
            │   │   ├── [Yes] → currentStepIndex++
            │   │   │          → renderProcedureSteps()
            │   │   │          → speakVoice("next step...")
            │   │   └── [No]  → "All steps done" → stopAnalysis()
            │   └── updateStatusUI()
            │
            ├── [confidence 90-94] → WARNING
            │   ├── Log "uncertain"
            │   └── updateStatusUI()
            │
            └── [confidence < 90] → ERROR
                ├── Log "failed"
                ├── speakVoice("retry step", priority=true)
                └── updateStatusUI()
```

### Key Functions:
| Function | Location | Purpose |
|----------|----------|---------|
| `startAnalysis()` | `app.js:1167` | Enable analysis, show overlays, start loops |
| `startCanvasDrawing()` | `app.js:1224` | Animated bounding box on canvas (simulated) |
| `startSimulationLoop()` | `app.js:1294` | 3-second interval: random confidence → validate → advance |
| `pauseAnalysis()` | `app.js:1189` | Toggle pause state |
| `stopAnalysis()` | `app.js:1201` | Clear interval, hide overlays |
| `resetSequence()` | `app.js:1213` | Reset to step 0 |

---

## 11. Monitoring State Flow (Python Backend)

When JS buttons trigger monitoring:

```
JS: btnStartAnalysis click
  → startAnalysis()                        [app.js:1167]
  → backend.startMonitoring()
    → Bridge.startMonitoring()             [bridge.py:127]
      → app_state.start_monitoring()       [app_state.py:22]
        → monitoring_status = "RUNNING"
      → _emit_state()                     [bridge.py:141]
        → stateChanged.emit(JSON)

JS: btnPauseAnalysis click
  → pauseAnalysis()                        [app.js:1189]
  → backend.pauseMonitoring()
    → Bridge.pauseMonitoring()             [bridge.py:132]
      → app_state.pause_monitoring()       [app_state.py:25]
        → monitoring_status = "PAUSED"

JS: btnStopAnalysis click
  → stopAnalysis()                         [app.js:1201]
  → backend.stopMonitoring()
    → Bridge.stopMonitoring()              [bridge.py:137]
      → app_state.stop_monitoring()        [app_state.py:28]
        → monitoring_status = "STOPPED"
```

---

## 12. Shutdown Sequence

```
app.aboutToQuit signal
    │
    └── Bridge.shutdown()                  [bridge.py:328]
        │
        ├── [Recording active?]
        │   └── Yes → stopRecording()
        │       └── VideoRecorder.stop_recording()
        │           ├── Writer thread joins
        │           └── VideoWriter.release()
        │
        ├── [Streaming active?]
        │   └── Yes → stopStreaming()
        │       └── IPStreamer.stop()
        │           ├── HTTPServer.shutdown()
        │           └── server_close()
        │
        ├── stopCamera()
        │   └── Camera.release()
        │       └── cv2.VideoCapture.release()
        │
        ├── VideoRecorder.shutdown()
        │
        └── VoiceAlertService.shutdown()
            ├── _stop_event.set()
            ├── stop_current_speech()
            └── worker_thread.join()
```

---

## 13. Thread Map (Runtime)

| Thread | Name | Owner | Purpose |
|--------|------|-------|---------|
| Main Thread | `MainThread` | Qt Event Loop | GUI rendering, QWebChannel IPC, QTimer callbacks |
| Voice Worker | `VoiceAlertWorker` | VoiceAlertService | Pulls speech from queue, dispatches to TTS engine |
| Video Writer | `VideoRecorderWorker` | VideoRecorder | Pops frames from queue, writes to MP4 (only while recording) |
| HTTP Server | daemon thread | GUIServer | Serves GUI over LAN port 8000 |
| WebRTC Server | daemon thread | WebRTCServer | Runs `aiohttp` event loop for signaling and RTP transport |

---

## 14. Data Flow Summary

```
    ┌──────────────────┐
    │  OpenCV Camera    │
    │ (cv2.VideoCapture)│
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Raw Frame       │
    │  (numpy BGR)      │
    └──┬─────┬──────┬──┘
       │     │      │
       ▼     │      ▼
  ┌───────────┐ │  ┌──────────────┐
  │WebRTC Track│ │  │ GUIServer    │
  │→ aiortc    │ │  │→ HTTP 8000   │
  └────┬──────┘ │  └──────┬───────┘
       │        │         │
       ▼        │         ▼
  ┌───────────┐ │  ┌──────────────┐
  │ JS UI     │ │  │HTTP 8000     │
  │video.src  │ │  │(ext clients) │
  └───────────┘ │  └──────────────┘
             │
             ▼
     ┌──────────────┐
     │VideoRecorder  │
     │queue → writer │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │data/videos/  │
     │REC_*.mp4     │
     └──────────────┘

     ┌──────────────────┐        ┌─────────────────────┐
     │Experiment CRUD   │───────►│data/experiments.json │
     └──────────────────┘        └─────────────────────┘

     ┌──────────────────┐        ┌─────────────────────┐
     │SystemLogger      │───────►│data/logs/system_*.log│
     └──────────────────┘        └─────────────────────┘

     ┌──────────────────┐
     │Simulation Loop   │───────► VoiceAlertService → TTS
     │(hardcoded, no AI)│
     └──────────────────┘
```

---

## 15. Active Modules Status

All core modules have been created and unit-tested:

| Category | Modules | Status |
|---|---|---|
| Input & Ingestion | `camera.py`, `video_loader.py`, `frame_processor.py` | ✅ Fully implemented & passing |
| AI Perception | `object_detector.py`, `pose_detector.py`, `hand_detector.py`, `hailo_inference.py`, `action_classifier.py`, `action_recognizer.py`, `inference_pipeline.py` | ✅ Fully implemented & passing |
| Sequence Validation | `procedure_manager.py`, `sequence_validator.py`, `interaction_logic.py` | ✅ Fully implemented & passing |
| Video Recording & IP Streaming | `recorder.py`, `streamer.py` | ✅ Fully implemented & passing |
| Speech Alerts & Logs | `voice_alert.py`, `system_logger.py`, `experiment_logger.py` | ✅ Fully implemented & passing |
| Desktop UI & App State | `main.py`, `app_state.py`, `bridge.py`, `frontend/index.html`, `frontend/assets/js/app.js` | ✅ Active |

---

## 16. Multimodal Perception & FSM Execution Flow

The real-time edge processing graph operates across visual perception, spatial geometry, temporal classification, and deterministic FSM validation:

```
                  ┌───────────────────────────────┐
                  │    Video Source (Live / File) │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   InferencePipeline.process   │
                  └───────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  ObjectDetector  │    │   PoseDetector   │    │   HandDetector   │
│ (YOLOv8 INT8)    │    │ (3D Body Mesh)   │    │(Hand Landmarks)  │
│  Outputs Boxes   │    │  Outputs 33 XYZ  │    │  Outputs Boxes   │
└─────────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │       InteractionEngine       │
                  │ (Hand-Object IoU & Proximity) │
                  │ State: APPROACHING / HOLDING  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   FrameProcessor: Keypoints   │
                  │ 33x3 Joint XYZ + Box Centroids│
                  │   ➔ Flattened 1D Feature Vector│
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    ActionClassifier (CPU)     │
                  │ 1D-TCN over 30-Frame Window   │
                  │ Outputs Action + Confidence   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │     SequenceValidatorFSM      │
                  │   Deterministic State Machine │
                  └───────┬───────────────┬───────┘
                          │               │
            [Step Valid]  │               │  [Step Skipped / Out of Order]
                          ▼               ▼
                  ┌───────────────┐ ┌───────────────┐
                  │ VoiceAlert    │ │ VoiceAlert    │
                  │ "Step X Valid"│ │ "Warning: OOO"│
                  └───────┬───────┘ └───────┬───────┘
                          │                 │
                          └────────┬────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────┐
                  │       ExperimentLogger        │
                  │  Writes Structured JSON & TXT │
                  │  Telemetry Logs to data/logs/ │
                  └───────────────────────────────┘
```

---

## Changelog

| Date | Change | Affected Files |
|------|--------|----------------|
| 2026-09-16 | Initial flow documentation created | flow.md (new) |
| 2026-09-16 | Added root `main.py` entry point, `config/settings.json`, and `config/network.json` | main.py, config/settings.json, config/network.json |
| 2026-09-16 | Added `video_loader.py` (offline video file reader) and `frame_processor.py` (letterboxing, keypoint extraction, annotations) | backend/video/video_loader.py, backend/video/frame_processor.py |
| 2026-09-16 | Added `procedure_manager.py` (step loading/indexing), `sequence_validator.py` (FSM sequence verification engine), and `interaction_logic.py` (hand-object proximity/IoU) | backend/experiment/procedure_manager.py, backend/experiment/sequence_validator.py, backend/experiment/interaction_logic.py |
| 2026-09-16 | Added full AI Perception suite (`object_detector.py`, `pose_detector.py`, `hand_detector.py`, `action_classifier.py`, `action_recognizer.py`, `hailo_inference.py`, `inference_pipeline.py`) | backend/ai/* |
| 2026-09-16 | Verified complete test suite: 18/18 tests passing in `test_functionalities.py`, 10/10 passing in `BAS-Monitor/tests` | test_functionalities.py, tests |
| 2026-09-30 | Migrated GUI to LAN HTTP Server and upgraded MJPEG stream to WebRTC/UDP | main.py, frontend/assets/js/app.js, backend/network/gui_server.py, backend/network/webrtc_server.py |
