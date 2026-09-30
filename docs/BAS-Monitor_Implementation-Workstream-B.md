# BAS Monitor --- Part 2 Implementation Workstream B (40%)

## Ownership

**Workstream B owns the AI/procedure-intelligence side of Part 2.**

This is the **40% portion**.

It includes:

-   demonstration experiment definition
-   action/object vocabulary
-   experiment-specific dataset
-   orientation/microgravity robustness
-   action-model architecture
-   real inference pipeline
-   temporal confirmation
-   FSM integration
-   uncertainty handling
-   recovery logic
-   HMR/3D extension
-   payload-relative reasoning
-   AI/procedure evaluation

Workstream B supplies stable AI/procedure result interfaces to
Workstream A.

------------------------------------------------------------------------

# 1. Workstream B architecture

``` text
Camera frames
     ↓
Object Detection
     ↓
Pose
     ↓
Hands
     ↓
Tracking
     ↓
Interaction
     ↓
Feature Extraction
     ↓
Temporal Buffer
     ↓
Action Model
     ↓
Action + Confidence
     ↓
Procedure FSM
     ├── VALID
     ├── UNCERTAIN
     ├── OUT OF ORDER
     ├── SKIPPED
     └── UNRECOGNIZED
     ↓
Recovery / Next Step
```

The camera source itself is owned by Workstream A.

------------------------------------------------------------------------

# 2. Shared engineering rules

1.  Start with one realistic demonstration experiment.
2.  Do not add action classes merely to make the model appear more
    advanced.
3.  Real model output must drive the final demo.
4.  `catch/not-catch` remains a baseline, not automatically the final
    experiment model.
5.  Measure accuracy, latency, memory, and robustness.
6.  Do not infer edge performance from a VM.
7.  The neural model recognizes activity; the deterministic FSM
    validates procedure order.
8.  Do not treat low confidence as automatically equivalent to operator
    error.
9.  HMR/3D is an extension and must not block the core system.

------------------------------------------------------------------------

# 3. Phase B1 --- Define the demonstration experiment

Replace the placeholder experiment with one realistic demonstration
procedure.

Define:

``` text
Experiment ID
Version
Purpose
Objects
Actions
Step sequence
Expected duration
Preconditions
Completion conditions
Failure conditions
Recovery instructions
```

Use a machine-readable versioned configuration.

Example:

``` json
{
  "id": "EXP-001",
  "version": "1.0",
  "steps": [
    {
      "id": 1,
      "action": "PICK_OBJECT",
      "target": "container",
      "timeout_s": 10,
      "confidence_threshold": 0.75,
      "recovery": "Return the container to the starting position."
    }
  ]
}
```

This is only a schema example. It must be replaced by the team's
selected experiment.

------------------------------------------------------------------------

# 4. Phase B2 --- Define action/object vocabulary

For each action define:

``` text
Action name
Target object
Start condition
Completion condition
Minimum duration
Maximum duration
Confidence threshold
Allowed next states
Recovery
```

The final action vocabulary must come from the experiment.

Possible examples:

``` text
PICK
HOLD
MOVE
PLACE
RELEASE
INSERT
REMOVE
POUR
```

Do not assume every action is required.

------------------------------------------------------------------------

# 5. Phase B3 --- Build experiment-specific dataset

Collect:

-   multiple people;
-   different speeds;
-   different starting positions;
-   left/right hand usage;
-   partial occlusion;
-   lighting variation;
-   camera-distance variation;
-   viewpoint variation;
-   object-position variation;
-   pauses;
-   incomplete actions;
-   wrong actions;
-   out-of-order actions.

Negative examples must represent **actual experiment failures**, not
merely generic action-recognition negatives.

Use:

``` text
Train
Validation
Test
```

Avoid leakage between clips from the same performance/session.

------------------------------------------------------------------------

# 6. Phase B4 --- Orientation and microgravity robustness

Evaluate:

``` text
0°
45°
90°
135°
180°
225°
270°
```

Do not distort rotated videos to force incorrect geometry.

Prefer training-time augmentation:

``` text
Rotation
Scale
Translation
Brightness
Blur
Occlusion
Perspective
```

Create separate test subsets:

``` text
Normal orientation
Moderate rotation
Extreme rotation
Occlusion
Lighting changes
```

Report metrics separately.

The objective is to reduce dependence on image-axis assumptions.

------------------------------------------------------------------------

# 7. Phase B5 --- Resolve action-model architecture

Compare:

## Model A --- Existing video baseline

``` text
Raw video
 ↓
R(2+1)D-18
 ↓
Action
```

## Model B --- Compact feature-based temporal model

``` text
Objects + Pose + Hands + Interaction
 ↓
Feature vector
 ↓
Temporal buffer
 ↓
1D-TCN or comparable lightweight model
 ↓
Action
```

Measure:

``` text
Precision
Recall
F1
Model size
RAM
Inference latency
FPS
CPU
Robustness
```

Do not choose using accuracy alone.

Document the decision in `decision.md`.

------------------------------------------------------------------------

# 8. Phase B6 --- Build real inference pipeline

The runtime should eventually receive frames from Workstream A and
perform:

``` text
Frame
 ↓
Object Detector
 ↓
Pose
 ↓
Hands
 ↓
Tracking
 ↓
Interaction
 ↓
Feature Extraction
 ↓
Temporal Buffer
 ↓
Action Model
 ↓
Action + Confidence
```

Do not keep the current heuristic action recognition as the final
mechanism.

------------------------------------------------------------------------

# 9. Phase B7 --- Non-blocking AI execution

The runtime must support:

``` text
Camera Thread
      ↓
Latest-frame buffer
      ↓
Inference Worker
      ↓
Result Queue
      ↓
Workstream A GUI/TTS/Logging
```

Do not place camera capture, inference, recording, and GUI updates
inside one blocking loop.

------------------------------------------------------------------------

# 10. Phase B8 --- Frame-rate strategy

A possible architecture is:

``` text
Camera capture: 30 FPS
Display:        30 FPS
AI inference:   10–15 FPS
```

These are targets to test, not guaranteed values.

Measure whether reduced AI frequency changes recognition quality.

------------------------------------------------------------------------

# 11. Phase B9 --- Integrate action recognition with FSM

AI returns:

``` text
Detected action
Confidence
Target/object evidence
Timestamp
```

FSM decides:

``` text
VALID
UNCERTAIN
OUT OF ORDER
SKIPPED
UNRECOGNIZED
```

Example:

``` text
AI:
PICK_CONTAINER
0.88

FSM:
Expected PICK_CONTAINER

→ VALID
```

The neural model answers:

> What happened?

The FSM answers:

> Was that the correct action at this point in the procedure?

------------------------------------------------------------------------

# 12. Phase B10 --- Temporal confirmation

Do not switch actions from a single noisy frame.

Use appropriate combinations of:

``` text
Temporal window
Majority voting
Minimum duration
Hysteresis
Temporal model output
```

Tune these on the validation set.

------------------------------------------------------------------------

# 13. Phase B11 --- Uncertainty handling

Use:

``` text
VALID
UNCERTAIN
ERROR
```

Example:

``` text
Expected:
PICK SAMPLE

Detected:
PICK SAMPLE

Confidence:
0.48

Result:
UNCERTAIN
```

The system should observe additional evidence before declaring a
procedural error where appropriate.

------------------------------------------------------------------------

# 14. Phase B12 --- Recovery handling

For every procedural violation define:

``` text
Expected
Observed
Explanation
Recovery
Timeout
```

Example:

``` text
Expected: PLACE SAMPLE
Observed: CLOSE LID

Status:
OUT OF ORDER

Recovery:
Return to PLACE SAMPLE.
```

The resulting event must be sent to Workstream A for GUI, TTS, and
logging.

------------------------------------------------------------------------

# 15. Phase B13 --- AI result contract

Workstream B must provide a stable result structure such as:

``` python
{
    "action": "PICK_CONTAINER",
    "confidence": 0.88,
    "target": "container",
    "evidence": {...},
    "timestamp": "...",
    "model": "action_v1"
}
```

The exact schema may evolve, but it must be stable before final
integration.

Workstream A should not need to know internal model architecture.

------------------------------------------------------------------------

# 16. Phase B14 --- HMR / 3D extension

Only begin after the core AI/FSM loop is stable.

First collect real failure cases:

``` text
rotated astronaut
partial occlusion
viewpoint change
unusual posture
hand/object ambiguity
```

If the current pose pipeline fails materially, evaluate lightweight HMR.

Possible architecture:

``` text
Person Detection
 ↓
Lightweight HMR
 ↓
3D joints / mesh
 ↓
Payload-relative features
 ↓
Interaction
```

Benchmark computational cost before making HMR the default.

------------------------------------------------------------------------

# 17. Phase B15 --- Payload-relative coordinate system

If 3D pose is adopted:

``` text
Camera coordinates
 ↓
Calibration
 ↓
Payload coordinate system
 ↓
3D astronaut/object relationships
```

Test:

``` text
Upright
Sideways
Inverted
Rotated
Different camera viewpoints
```

The purpose is to reduce dependence on the camera's image axes.

------------------------------------------------------------------------

# 18. Phase B16 --- Failure testing

Create dedicated AI/procedure failure sets.

## AI failures

-   wrong action;
-   missed action;
-   low confidence;
-   occlusion;
-   poor lighting;
-   blur;
-   fast motion;
-   multiple objects.

## Procedure failures

-   skipped step;
-   out-of-order step;
-   repeated step;
-   premature action;
-   incomplete action;
-   timeout.

For every failure record:

``` text
Input
Expected
Observed
System response
Failure cause
Fix
```

------------------------------------------------------------------------

# 19. Phase B17 --- Evaluation

## Action level

Report:

``` text
Precision
Recall
F1
Confusion matrix
```

## Procedure level

Report:

``` text
Complete-sequence success
Skipped-step detection
Out-of-order detection
False-alarm rate
Missed-violation rate
```

## AI runtime level

Report:

``` text
AI FPS
Inference latency
Model memory
CPU usage
```

------------------------------------------------------------------------

# 20. Phase B18 --- Workstream B acceptance tests

Workstream B is complete when:

-   [ ] One real demonstration experiment is fully defined.
-   [ ] Action/object vocabulary is finalized.
-   [ ] Experiment-specific dataset is prepared.
-   [ ] Dataset leakage is controlled.
-   [ ] Orientation/robustness subsets are evaluated.
-   [ ] R(2+1)D-18 baseline is documented.
-   [ ] Compact temporal candidate is evaluated.
-   [ ] Final action model is selected using measured evidence.
-   [ ] Real inference pipeline works.
-   [ ] AI runs independently of GUI refresh.
-   [ ] AI returns real action/confidence results.
-   [ ] FSM receives real AI predictions.
-   [ ] Temporal confirmation works.
-   [ ] VALID / UNCERTAIN / ERROR works.
-   [ ] Skipped/out-of-order detection works.
-   [ ] Recovery guidance works.
-   [ ] AI/procedure failure testing is documented.
-   [ ] Action-level and procedure-level metrics are reported.
-   [ ] HMR/3D is either validated/integrated or documented as a future
    extension.

------------------------------------------------------------------------

# 21. Parallel-work strategy

Workstream B can train and test models independently from the final
camera/network implementation.

Use:

``` text
Recorded videos
+
synthetic/test frame sources
+
defined input schema
```

while Workstream A develops camera and networking.

B should not wait for Android camera or RTSP implementation to begin
model development.

A should not wait for the final model to begin camera/network
development.

------------------------------------------------------------------------

# 22. Integration contract with Workstream A

Workstream B consumes:

``` text
Frame
Timestamp
Camera ID
Resolution
FPS
```

Workstream B produces:

``` text
Action
Confidence
Object/interaction evidence
Timestamp
Model status
```

Workstream B's procedural output is:

``` text
VALID
UNCERTAIN
OUT_OF_ORDER
SKIPPED
UNRECOGNIZED
```

Workstream A presents those results through:

``` text
GUI
TTS
Logs
Recording metadata
```

------------------------------------------------------------------------

# 23. Workstream B high-level sequence

``` text
One real experiment
 ↓
Action/object definition
 ↓
Dataset
 ↓
Orientation robustness
 ↓
Model comparison
 ↓
Real inference
 ↓
Non-blocking AI
 ↓
FSM integration
 ↓
Temporal confirmation
 ↓
Uncertainty
 ↓
Recovery
 ↓
AI/procedure evaluation
 ↓
HMR / 3D extension
 ↓
Final integration
```

------------------------------------------------------------------------

# 24. Joint integration gate

The two workstreams are ready to merge when:

``` text
A:
Camera → stable frames
             +
B:
AI → stable predictions
```

Then:

``` text
Camera
 ↓
Frame Buffer
 ↓
AI
 ↓
Action + Confidence
 ↓
FSM
 ↓
GUI / TTS / Logging / Recording
```

Run the following jointly:

1.  Correct sequence.
2.  Skipped step.
3.  Out-of-order step.
4.  Low-confidence action.
5.  Camera disconnect.
6.  Network disconnect.
7.  Android camera.
8.  Offline Internet-disconnected test.
9.  Recording.
10. Uploaded-video analysis.

------------------------------------------------------------------------

# 25. Important boundary

Do not duplicate ownership.

### Workstream A owns

``` text
Camera
Networking
Streaming
Android camera
Recording
TTS
GUI runtime
Diagnostics
Packaging
```

### Workstream B owns

``` text
Dataset
AI model
Inference
Action recognition
FSM
Uncertainty
Recovery
HMR/3D
AI/procedure evaluation
```

### Shared

``` text
Experiment configuration
AI result schema
Procedure result schema
End-to-end testing
Final documentation
```

------------------------------------------------------------------------

# 26. Final 60/40 responsibility map

  Area                         A --- 60%   B --- 40%
  --------------------------- ----------- -----------
  Baseline/runtime                 ✓      
  Experiment definition                        ✓
  Action vocabulary                            ✓
  Dataset                                      ✓
  Orientation robustness                       ✓
  Model architecture                           ✓
  Real inference                               ✓
  FSM                                          ✓
  Uncertainty/recovery                         ✓
  Camera abstraction               ✓      
  Camera switching/failover        ✓      
  Focus/zoom/PTZ                   ✓      
  Android camera                   ✓      
  Local wireless network           ✓      
  `file01.md`                      ✓      
  `local-streaming-plan.md`        ✓      
  Recording                        ✓      
  Logging                          ✓      
  GUI/runtime status               ✓      
  Live preview                     ✓      
  Voice/TTS                        ✓      
  Diagnostics                      ✓      
  Packaging                        ✓      
  HMR/3D                                       ✓
  AI evaluation                                ✓
  End-to-end integration           ✓           ✓

------------------------------------------------------------------------

# 27. Recommended simultaneous starting point

Both people can start immediately.

### Person A

Start with:

``` text
A1 Baseline
 ↓
A2 Camera abstraction
 ↓
A3 Camera capability detection
 ↓
A6 Local wireless network
```

Then proceed into:

``` text
A7 Android camera
A8/A9 Streaming
A16 TTS
A14 GUI
```

### Person B

Start with:

``` text
B1 Experiment definition
 ↓
B2 Action/object vocabulary
 ↓
B3 Dataset
 ↓
B4 Orientation robustness
```

Then:

``` text
B5 Model comparison
 ↓
B6 Real inference
 ↓
B9 FSM
```

The two tracks converge only at the defined interface contract.

------------------------------------------------------------------------

# 28. Definition of successful parallel development

The split is successful if:

``` text
Person A can build/test the system infrastructure
without waiting for the final AI model.

Person B can build/test the AI/procedure system
without waiting for the final camera/network implementation.
```

At integration:

``` text
A's camera frames
       ↓
B's AI
       ↓
B's FSM
       ↓
A's GUI + TTS + Logging + Recording
```

This is the intended 60/40 division.
