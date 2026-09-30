# BAS Monitor --- Part 2 Implementation Workstream A

## Ownership

**Workstream A owns the runtime/system side of Part 2.**

This explicitly includes:

-   `file01.md`
-   `local-streaming-plan.md`
-   camera architecture and camera switching
-   Android wireless camera
-   local wireless networking without Internet
-   voice/TTS
-   recording
-   structured logging
-   GUI/runtime status
-   diagnostics
-   packaging and deployment plumbing

The objective is to build the reliable system around the AI pipeline
being developed by Workstream B.

------------------------------------------------------------------------

# 1. What Workstream A owns

``` text
Camera Sources
     ↓
Camera Manager
     ↓
Frame Buffer
     ├──────────────→ Recorder
     ├──────────────→ Local Streaming
     └──────────────→ AI input contract
                           ↑
                    Workstream B
                           ↓
                    AI Result Contract
                           ↓
                 GUI / TTS / Logging
```

Workstream A should **not redesign the AI model internally**. It
consumes the stable interfaces supplied by Workstream B.

------------------------------------------------------------------------

# 2. Shared engineering rules

1.  Core operation must work without Internet.
2.  No fake AI confidence, camera status, network status, FPS, or
    latency.
3.  Camera capture, streaming, recording, GUI, and TTS must not block AI
    inference.
4.  Every important runtime event must be timestamped.
5.  Camera source must be abstracted from the AI pipeline.
6.  Local wireless networking must be tested with Internet disconnected.
7.  The final application must expose real runtime state.
8.  Changes must preserve existing tests and be committed incrementally.

------------------------------------------------------------------------

# 3. Phase A1 --- Freeze and clean the runtime baseline

### Tasks

-   Create a development branch.
-   Run the application from a clean checkout.
-   Verify GUI startup.
-   Verify current camera operation.
-   Verify experiment loading.
-   Verify recording.
-   Verify logging.
-   Verify model-loading status display.
-   Identify placeholders and fallback states relevant to runtime.
-   Confirm root `backend/` is the active implementation.
-   Mark legacy code for later cleanup.

### Deliverable

Create/update:

``` text
docs/baseline.md
```

Record:

-   OS
-   Python version
-   package versions
-   current camera behavior
-   current streaming behavior
-   current recording behavior
-   current TTS behavior
-   current FPS
-   current latency
-   RAM/CPU observations

### Gate

Runtime baseline is reproducible before continuing.

------------------------------------------------------------------------

# 4. Phase A2 --- Camera abstraction

Create a common camera interface:

``` python
class CameraSource:
    connect()
    disconnect()
    read()
    get_status()
    get_capabilities()
```

Implement adapters for:

``` text
USB/local camera
CSI camera
IP/network camera
Android wireless camera
```

The rest of BAS Monitor must consume frames through this abstraction.

The AI pipeline must not need to know whether the frame came from USB,
CSI, IP, or Android.

------------------------------------------------------------------------

# 5. Phase A3 --- Camera readiness and capability detection

Expose actual capabilities:

``` text
resolution
FPS
focus
optical zoom
digital zoom
PTZ
connection state
```

Only show controls that are genuinely supported by the selected camera.

Do not invent camera capabilities.

Before an experiment:

``` text
Select camera
 ↓
Connect
 ↓
Check capabilities
 ↓
Readiness check
 ↓
Start
```

------------------------------------------------------------------------

# 6. Phase A4 --- Camera switching and failover

## Before experiment

Allow:

``` text
Select
Check
Readiness
Lock
Start
```

## During experiment

Keep the selected camera locked unless controlled failover is
implemented.

## Controlled failover

``` text
Primary camera failure
 ↓
Timeout detection
 ↓
Fallback camera
 ↓
Reinitialize perception
 ↓
Log transition
 ↓
Notify operator
```

Never silently switch viewpoints.

Test:

-   USB → IP
-   IP → Android
-   reconnect after disconnect
-   camera rotation
-   camera unavailable at startup

------------------------------------------------------------------------

# 7. Phase A5 --- Focus / zoom / PTZ

Implement capability-aware controls.

Expose:

``` text
Focus: supported / unsupported
Optical zoom: supported / unsupported
Digital zoom: supported / unsupported
PTZ: supported / unsupported
```

For cameras that do not support a capability, disable or hide the
control.

------------------------------------------------------------------------

# 8. Phase A6 --- Local wireless network

Target architecture:

``` text
                 LOCAL Wi-Fi / LAN
                       │
       ┌───────────────┼────────────────┐
       │               │                │
 Edge computer    Monitor PC       Camera/Phone
```

The network does **not** require Internet access.

### Critical test

Disconnect Internet.

The following must continue:

``` text
Camera       → WORKING
AI           → WORKING
GUI          → WORKING
Recording    → WORKING
FSM          → WORKING
Logs         → WORKING
Streaming    → WORKING
TTS          → WORKING
```

Only optional Internet-dependent features may fail.

Do not describe this as Android ad-hoc networking unless that exact
networking method is implemented.

------------------------------------------------------------------------

# 9. Phase A7 --- Android phone as a wireless camera

Requirement:

> Use an Android phone's camera wirelessly as a camera source for BAS
> Monitor.

This is a **camera-source requirement**, not an Internet-streaming
requirement.

### Procedure

1.  Connect the phone and BAS Monitor system to the same local wireless
    network.
2.  Establish the selected phone-camera transport.
3.  Discover/connect the Android camera.
4.  Acquire frames through `CameraSource`.
5.  Display camera status in the GUI.
6.  Measure actual FPS and latency.
7.  Test reconnect.
8.  Test phone rotation.
9.  Test focus/zoom availability.
10. Run the same AI input path used by other camera sources.

The phone must appear to BAS Monitor as another camera source:

``` text
Camera Source
├── USB Camera
├── IP Camera
└── Android Camera
```

------------------------------------------------------------------------

# 10. Phase A8 --- Local video streaming upgrade

### Mandatory source document

Use:

``` text
local-streaming-plan.md
```

as the detailed implementation specification for the streaming
subsystem.

The plan identifies the current MJPEG/HTTP implementation and proposes a
tiered upgrade toward RTP/RTSP/H.264, including pacing, buffering,
encode-once distribution, and edge deployment considerations.

### Workstream A implementation responsibility

Implement and test:

``` text
Tier 1
Improved MJPEG

Tier 2
FFmpeg + H.264 + RTSP

Tier 3
GStreamer / edge hardware
```

Tier 3 remains hardware-dependent and can be completed when the
Raspberry Pi target is available.

------------------------------------------------------------------------

# 11. Phase A9 --- Apply `file01.md` networking design

### Mandatory source document

Use:

``` text
file01.md
```

as the detailed protocol/algorithm specification.

It defines the intended move toward:

``` text
RTP over UDP
RTSP control
Monotonic-clock pacing
Encode-once / publish-subscribe
Receiver-side jitter buffering
```

Do not duplicate or rewrite that document as a separate design.
Implement it as the networking sub-plan.

------------------------------------------------------------------------

# 12. Phase A10 --- Streaming architecture

Target:

``` text
Camera
 ↓
Frame Buffer
 ↓
Stream Manager
 ├── RTSP/H.264 primary
 └── MJPEG fallback
       ↓
Local LAN clients
```

The stream manager should hide the specific streaming backend.

Record:

``` text
backend
resolution
FPS
bitrate
frames written
frames dropped
connection status
```

------------------------------------------------------------------------

# 13. Phase A11 --- Streaming verification

Run:

``` text
python -m pytest tests/ -v
python -m pytest tests/test_streaming.py -v
```

Verify:

-   JPEG validity
-   frame pacing
-   sequence numbers
-   stale-frame handling
-   RTSP startup
-   MJPEG fallback
-   multiple clients
-   bandwidth
-   latency
-   jitter
-   reconnect/failure behavior

Use the targets from `local-streaming-plan.md` as engineering targets,
not guaranteed results.

------------------------------------------------------------------------

# 14. Phase A12 --- Recording

Keep recording independent from display and AI.

``` text
Camera
 ├── Display
 ├── AI
 ├── Stream
 └── Recorder
```

Record metadata:

``` text
experiment ID
procedure version
camera ID
start/end time
model version
software version
result
```

Recording must continue even if the GUI display refresh rate changes.

------------------------------------------------------------------------

# 15. Phase A13 --- Structured logging

Log:

-   experiment start/stop
-   current step
-   expected action
-   detected action
-   confidence
-   validation result
-   camera connection
-   camera change
-   model loading
-   inference errors
-   recording
-   network state
-   warnings
-   TTS events
-   streaming events

Every event should have a timestamp.

------------------------------------------------------------------------

# 16. Phase A14 --- GUI runtime integration

Replace prototype/hard-coded indicators with real values.

Examples:

``` text
AI Model: LOADED
Camera: CONNECTED
Recording: ACTIVE
Voice: READY
Network: OFFLINE LAN
FPS: 12.7
Latency: 74 ms
```

The GUI must consume runtime events/results rather than simulate them.

------------------------------------------------------------------------

# 17. Phase A15 --- Live preview

Provide two display modes:

``` text
Clean View
AI Evidence
```

AI Evidence may display:

``` text
Current action
Confidence
Expected action
Step N / total
Step timer
Pose
Hands
Objects
Interaction
FPS
Latency
```

Do not permanently clutter the main preview.

------------------------------------------------------------------------

# 18. Phase A16 --- TTS / voice system

Implement:

``` text
FSM event
 ↓
Priority + debounce
 ↓
Voice queue
 ↓
Background worker
 ↓
Speaker
```

Required behavior:

-   no GUI blocking;
-   no inference blocking;
-   duplicate-warning cooldown;
-   critical-alert priority;
-   enable/disable configuration;
-   queue interruption/clearing for critical alerts.

Normal successful steps should primarily use GUI feedback.

Voice should be reserved for important status, warnings, errors, and
recovery guidance.

------------------------------------------------------------------------

# 19. Phase A17 --- Diagnostics

Create a diagnostics panel.

Where supported, show:

``` text
CPU %
RAM %
Temperature
NPU %
Storage
Camera FPS
AI FPS
Inference latency
End-to-end latency
Network latency
Model memory
```

All displayed values must come from real measurements.

------------------------------------------------------------------------

# 20. Phase A18 --- Package the runtime

Final package should contain all required local resources:

``` text
BAS-Monitor/
├── application
├── models/
├── config/
├── frontend/
├── backend/
├── data/
└── docs/
```

Remove mandatory external CDN dependencies.

Test the package on a clean machine.

------------------------------------------------------------------------

# 21. Phase A19 --- Workstream A acceptance tests

Workstream A is complete when:

-   [ ] Camera abstraction works.
-   [ ] USB/local camera works.
-   [ ] Network/IP camera works.
-   [ ] Android camera source works if included in final scope.
-   [ ] Camera capability detection works.
-   [ ] Camera selection works before experiments.
-   [ ] Camera switching/failover is controlled and logged.
-   [ ] Local wireless network works without Internet.
-   [ ] `file01.md` networking design is implemented/tested.
-   [ ] `local-streaming-plan.md` streaming implementation is
    integrated/tested.
-   [ ] RTSP/H.264 path works where dependencies are available.
-   [ ] MJPEG fallback works.
-   [ ] Recording works independently.
-   [ ] Structured logging works.
-   [ ] TTS is non-blocking.
-   [ ] GUI values are real runtime values.
-   [ ] Live preview works.
-   [ ] Diagnostics work.
-   [ ] Clean-machine package works.

------------------------------------------------------------------------

# 22. Interface contract with Workstream B

Workstream A provides to B:

``` text
FrameSource
CameraStatus
FrameTimestamp
CameraCapabilities
```

Workstream B provides to A:

``` text
DetectedAction
Confidence
ObjectEvidence
InteractionEvidence
Timestamp
ModelStatus
```

The shared procedural result should be:

``` text
VALID
UNCERTAIN
OUT_OF_ORDER
SKIPPED
UNRECOGNIZED
```

Workstream A then presents/logs these results through:

``` text
GUI
TTS
Recording metadata
Logs
```

------------------------------------------------------------------------

# 23. Parallel-work rule

Workstream A can proceed without waiting for the final AI model by using
a **strict interface/mock contract** during development.

However:

> Mock AI results must never be presented as final AI performance.

The final integration must use Workstream B's real model output.

------------------------------------------------------------------------

# 24. Workstream A → Workstream B handoff

Before final integration, A must provide:

``` text
1. Stable camera frame interface
2. Frame timestamps
3. Camera status
4. Stream status
5. Recording API
6. Event/log API
7. TTS API
8. GUI result API
9. AI result schema
10. Diagnostics API
```

Once these are stable, B can plug the real model into the runtime
without rewriting the camera/network/UI architecture.

------------------------------------------------------------------------

# 25. Workstream A high-level sequence

``` text
Baseline
 ↓
Camera abstraction
 ↓
Camera capabilities
 ↓
Local wireless network
 ↓
Android camera
 ↓
Streaming upgrade
 ↓
file01 networking implementation
 ↓
Recording
 ↓
Logging
 ↓
GUI runtime integration
 ↓
Live preview
 ↓
TTS
 ↓
Diagnostics
 ↓
Packaging
 ↓
Real AI integration
 ↓
End-to-end testing
```
