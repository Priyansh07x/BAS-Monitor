# GUIDE.md — SIH26174 Complete Universal Guide

> **Problem Statement:** SIH26174 — **AI Human Activity Recognition for On-board BAS Experiments**
>
> **Organization:** Indian Space Research Organisation (ISRO)
>
> **Track:** Software
>
> **Purpose of this guide:** Make the problem understandable to someone who has no prior knowledge of ISRO, BAS, computer vision, AI, networking, GUI development, or the technologies used in this project.

---

# 0. Read This First

This guide is the common understanding of the SIH26174 project that the team should use before writing code.

The project has **two major parts**:

```text
PART 1 — AI MODEL DEVELOPMENT

Custom / external datasets
        ↓
Google Colab
        ↓
Train / fine-tune models
        ↓
Evaluate
        ↓
Export trained model files
(.pt / .pth / possibly .onnx)


PART 2 — OFFLINE EDGE APPLICATION

Camera / Video File
        ↓
Local Python application
        ↓
AI models
        ↓
Experiment sequence validation
        ↓
GUI + Voice + Logs + Local Recording + IP Streaming
```

The final application is intended to behave like an **onboard/edge experiment assistant**: a fixed-payload camera observes an astronaut performing a predefined experiment, the local system analyzes the video, determines which experiment step is happening, checks whether it is the correct step, suggests the next step, warns about skipped/out-of-order actions, logs the experiment, stores the video locally, and streams the video to a specified IP.

The official PS requires local video processing, next-step suggestion, voice alerts, timestamped structured logging, IP video streaming, local video storage, a GUI, and an offline standalone trained AI model. fileciteturn1file1L21-L23

---

# 1. The Problem Statement in Simple Language

## 1.1 One-sentence explanation

> **ISRO wants an AI system that can watch an astronaut perform a predefined scientific experiment and automatically check whether the astronaut is following the correct sequence of steps.**

The PS calls this **Human Activity Recognition (HAR)** and asks the system to recognize and validate the sequence of a predefined experiment. fileciteturn1file1L21-L23

---

# 2. What Is ISRO?

**ISRO** stands for **Indian Space Research Organisation**.

It is India's national space organisation. It develops and operates Indian space missions, launch vehicles, satellites, human-spaceflight programmes, and related technologies.

For this problem statement, ISRO is the organization that wants a technology concept for future onboard space-station experiments.

---

# 3. What Is BAS?

**BAS** stands for **Bharatiya Antariksh Station**.

It is India's planned space station.

ISRO's current public roadmap describes:

- **first BAS module around 2028** in the current national space-vision roadmap;
- **full BAS establishment by 2035**.

ISRO's recent official material describes BAS as being in a conceptualization/development phase and says the station is planned to be assembled in phases. citeturn725179search38turn725179search37

So do **not** think:

> "BAS is an already-operational Indian space station and SIH wants us to install something there next week."

Instead think:

```text
Today
  ↓
We build a prototype on Earth
  ↓
Future onboard deployment concept
  ↓
BAS / future Indian human-spaceflight missions
```

ISRO has also explicitly discussed microgravity experiments as part of its human-space programme and BAS vision. citeturn725179search1turn725179search4

---

# 4. What Does “On-board” Mean?

In this PS, **on-board** means **on the spacecraft / space-station side**, rather than relying on an Earth-based server for the critical AI decision.

Think of:

```text
EARTH-BASED DEVELOPMENT

Laptop / PC
    ↓
Our AI software
```

versus the eventual concept:

```text
BAS / SPACE STATION

Fixed camera
     ↓
Local onboard computer
     ↓
Our AI software
     ↓
Immediate result
```

The PS specifically says that data should be processed locally at the **edge** instead of depending on continuous raw-video streaming to ground control. fileciteturn1file1L21-L23

---

# 5. What Does “Edge” Mean?

**Edge computing** means processing data close to where the data is generated.

For this PS:

```text
Camera
  ↓
LOCAL / ONBOARD COMPUTER
  ↓
AI
```

instead of:

```text
Camera
  ↓
Raw video to Earth
  ↓
Earth server
  ↓
AI
  ↓
Result back to space
```

The reason is that space communication can have limited bandwidth and delay.

### Important distinction

**Offline does not mean “no network.”**

A computer can be offline from the Internet and still communicate over a local IP network.

For example:

```text
PC A  ←→  Local Switch  ←→  PC B
```

can work without Internet access.

---

# 6. The Real Scenario

This is the safest interpretation of the physical scenario based on the PS:

```text
                      FUTURE BAS
┌──────────────────────────────────────────────────────┐
│                                                      │
│        Experiment / Payload Area                    │
│                                                      │
│             Fixed-Payload Camera                    │
│                     │                                │
│                     │ observes                       │
│                     ▼                                │
│                Astronaut                             │
│                     │                                │
│                     │ performs                       │
│                     ▼                                │
│              Scientific Experiment                  │
│                     │                                │
│                     ▼                                │
│         Local / Onboard Edge Computer                │
│                     │                                │
│                     ▼                                │
│              SIH26174 AI System                      │
│                                                      │
│  AI analysis → sequence validation → feedback       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Important camera clarification

The PS says the input is from **fixed-payload cameras**. fileciteturn1file1L21-L23

Therefore:

- We should **not** assume the camera is mounted on the astronaut's suit.
- We should **not** assume the camera is specifically mounted on the outer surface of BAS.
- The exact physical mounting location is **not specified by the PS**.
- The safest statement is: **a fixed camera in the experiment/payload environment observes the astronaut and experiment.**

Our Earth-based demo laptop/webcam is a prototype representation of the future local/onboard computer + fixed-camera environment.

---

# 7. Who Is the System Helping?

The most important immediate user is the **astronaut / experiment performer**, because the system must provide next-step guidance and voice alerts.

There may also be an operator/monitoring user because the PS requires a GUI and IP video streaming.

So conceptually:

```text
PRIMARY USER
Astronaut
    ↑
Voice / local guidance
    ↑
AI system

SECONDARY USER
Operator / monitoring system
    ↑
GUI + IP video stream
```

The PS does **not** require us to build an Earth-to-space command/control system or an astronaut-worn device.

---

# 8. What Problem Exists That Requires This System?

## Problem 1 — Continuous Earth support is difficult

In future long-duration space missions, continuous dependence on Earth for every experimental action is undesirable because communication links have delay and bandwidth limitations.

The PS explicitly motivates local edge processing rather than sending raw video continuously to ground control. fileciteturn1file1L21-L23

---

## Problem 2 — Humans must follow exact scientific procedures

A scientific experiment is normally not:

```text
Do anything you want.
```

It is more like:

```text
Step 1
  ↓
Step 2
  ↓
Step 3
  ↓
Step 4
  ↓
Step 5
```

If Step 3 is skipped, the experiment could produce a wrong result or an invalid protocol execution.

The PS therefore focuses on **sequence validation**. fileciteturn1file1L21-L23

---

## Problem 3 — The camera sees motion, but raw video does not automatically explain the action

A camera gives us pixels.

The AI has to infer:

```text
What is the astronaut doing?
Which object is involved?
What experiment step does this action represent?
```

That is the computer-vision problem.

---

## Problem 4 — An action can be visually similar to another action

Example:

```text
Hand near object
```

does not necessarily mean:

```text
Object picked up
```

We need to reason about motion, hand position, object movement, and time.

---

## Problem 5 — Step order matters

Suppose the correct sequence is:

```text
A → B → C → D
```

but the astronaut does:

```text
A → B → D
```

The system must recognize:

> **C was skipped.**

If the astronaut does:

```text
A → C → B → D
```

then the system must recognize:

> **C happened out of order.**

The PS explicitly requires skipped/out-of-sequence detection. fileciteturn1file1L21-L23

---

## Problem 6 — The astronaut can work in unusual orientations

In microgravity there is no permanent Earth-like “floor” relationship.

An astronaut may work:

```text
upright
sideways
upside down
```

The PS therefore mentions an **optional orientation-agnostic 3D Human Mesh Recovery (HMR)** challenge, where body posture is tracked relative to the payload rack rather than assuming the floor defines “up.” fileciteturn1file1L21-L23

This is **optional**, not part of the minimum prototype.

---

## Problem 7 — The experiment needs a record

At the end, it is useful to know:

```text
time
step
observed action
status
errors
```

The PS explicitly requires a timestamped, structured lightweight text file with conducted steps and outcomes/status. fileciteturn1file1L21-L23

---

## Problem 8 — Video must also be retained and streamable

The PS requires:

- local video storage;
- video streaming to a specified IP.

These are separate functions. fileciteturn1file1L21-L23

---

# 9. What the PS Is NOT Asking Us to Build

We are **not** primarily building:

- a spacecraft;
- a space station;
- a satellite communication system;
- a radio/antenna system;
- an astronaut suit;
- a wearable computer;
- a general AI assistant that understands everything an astronaut does;
- a cloud-only AI service;
- a generic website;
- a huge Earth-to-space communications network.

The core ask is:

> **An onboard/offline AI experiment-monitoring system that recognizes and validates the sequence of a predefined experiment from fixed-camera video.**

---

# 10. The Sample Experiment vs the Real Future Use Case

This distinction is critical.

The PS uses a simplified sample experiment involving a box and smaller colored boxes. The available PS text gives the beginning of the sample description, but the provided extracted content is truncated. fileciteturn1file1L21-L23

Therefore:

```text
Red box / blue box example
        ↓
Prototype example
        ↓
Demonstrates the recognition/sequence problem
```

It should **not** be presented as:

> “This exact red/blue-box operation is ISRO's actual BAS scientific experiment.”

The real future experiment could involve samples, containers, instruments, tools, racks, sensors, or other payload components.

The architecture is intended to stay the same while the experiment-specific objects and actions change.

---

# 11. The Problem Reduced to One Question

The entire PS can be reduced to:

> **“Is the experiment being performed correctly and in the correct sequence?”**

The AI must answer that question from video.

---

# 12. What Is Human Activity Recognition (HAR)?

**Human Activity Recognition** means using sensors/cameras and AI to determine what a human is doing.

For this project:

```text
Video
  ↓
Person detected
  ↓
Body/hand information
  ↓
Object information
  ↓
Movement over time
  ↓
Action
```

Example:

```text
Hand approaches red object
        ↓
Hand contacts red object
        ↓
Red object moves with hand
        ↓
Hand releases object
        ↓
Action = PICK / MOVE RED OBJECT
```

The PS is not asking for universal HAR. It asks for HAR targeted at the predefined experiment. fileciteturn1file1L21-L23

---

# 13. The Five Important Technical Functions

```text
1. Object Detection
2. Pose Estimation
3. Hand Tracking / Landmarks
4. Hand–Object Interaction + Action Recognition
5. Sequence Validation
```

These are **not five sequential CNN layers**.

They are functional components.

---

# 14. Object Detection

### Meaning

Object detection answers:

> **“What objects are present and where are they?”**

Example:

```text
Red sample → bounding box
Blue sample → bounding box
Container  → bounding box
Tool       → bounding box
```

### Technology choice

**YOLO** (Ultralytics)

YOLO is a deep-learning computer-vision model family that can perform object detection and other vision tasks.

Official documentation:

- https://docs.ultralytics.com/

---

# 15. Pose Estimation

### Meaning

Pose estimation answers:

> **“Where are the important points of the human body?”**

For example:

```text
head
shoulder
elbow
wrist
hip
knee
ankle
```

### Technology choice

Initial recommendation:

- YOLO Pose for body keypoints, or
- another mature pretrained pose model if testing shows it is better.

We should use pretrained pose models first instead of training body-pose recognition from scratch.

---

# 16. Hand Tracking / Hand Landmarks

A hand bounding box tells us:

> “The hand is here.”

A hand-landmark model gives more detail:

```text
wrist
thumb
index
middle
ring
pinky
```

### Technology choice

**MediaPipe Hand Landmarker** is a practical candidate for detailed hand landmarks.

Official resources:

- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

We may use YOLO for hand detection and/or MediaPipe for detailed landmarks depending on the experiment.

---

# 17. Object Tracking

Detection answers:

> “What is this frame showing?”

Tracking answers:

> “Is this the same object I saw in the previous frame?”

Example:

```text
Frame 1 → Red Box = ID 1
Frame 2 → Red Box = ID 1
Frame 3 → Red Box = ID 1
```

This lets us estimate motion/trajectory.

YOLO-based tracking can be used as part of the pipeline.

---

# 18. Hand–Object Interaction

This is the relationship between a person's hand and an experiment object.

Conceptually:

```text
hand position
+
object position
+
distance
+
relative motion
+
time
        ↓
interaction state
```

Possible states:

```text
near object
contacting object
holding object
moving object
releasing object
```

For the first prototype, this does not necessarily need another neural network. Geometry + tracking + temporal logic may be sufficient.

---

# 19. Action Recognition

Action recognition answers:

> **“What action is happening over a sequence of frames?”**

This is where a video model such as **SlowFast** can be considered.

---

# 20. What Is a 3D CNN?

A normal 2D convolution mainly operates over:

```text
Height × Width
```

A video model with 3D convolution operates conceptually over:

```text
Time × Height × Width
```

So a 3D CNN can learn both:

- appearance;
- temporal change.

That is why it is suitable for video/action recognition.

---

# 21. SlowFast

**SlowFast** is a video action-recognition architecture with two temporal pathways:

- a **Slow** pathway for semantic/spatial information at a lower frame rate;
- a **Fast** pathway for rapidly changing motion information at a higher frame rate.

It is a candidate for our action-recognition stage.

Important:

> **SlowFast is an action-recognition model, not the final sequence validator.**

We should not blindly do:

```text
LSTM → SlowFast → validation
```

because LSTM and SlowFast are alternative temporal-action modeling approaches.

For the final project, we should benchmark a practical baseline and then choose the model that gives the best balance of accuracy, speed, complexity, and offline deployment.

### SlowFast references

- PyTorchVideo / SlowFast ecosystem: https://pytorchvideo.org/
- Meta/FacebookResearch SlowFast repository: https://github.com/facebookresearch/SlowFast

---

# 22. Sequence Validation Is NOT a CNN

This is one of the most important architectural distinctions.

The AI may say:

```text
Detected action = PICK_RED
```

The sequence engine knows:

```text
Expected action = PICK_RED
```

Therefore:

```text
MATCH
  ↓
VALID
  ↓
Advance to next step
```

If:

```text
Expected = PLACE_RED
Detected = PICK_BLUE
```

then:

```text
MISMATCH
  ↓
OUT OF SEQUENCE
  ↓
Voice alert
  ↓
Log
```

This can be implemented as a **finite-state machine (FSM)** / rules engine.

No model training is required for this part.

---

# 23. Final AI Architecture

```text
                         VIDEO
                           │
                           ▼
                        OpenCV
                           │
                           ▼
              ┌────────────────────────┐
              │   VISUAL PERCEPTION   │
              │                        │
              │ YOLO Object Detection │
              │ YOLO Pose              │
              │ Tracking               │
              │ Hand Model             │
              └───────────┬────────────┘
                          │
                          ▼
                Hand–Object Features
                          │
                          ▼
                 ACTION RECOGNITION
                    SlowFast / other
                        video model
                          │
                          ▼
                   Detected Action
                          │
                          ▼
                EXPERIMENT STATE MACHINE
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
               VALID            INVALID
                 │                 │
                 ▼                 ▼
            Next Step         Voice Alert
                 │                 │
                 └────────┬────────┘
                          ▼
                          GUI
```

---

# 24. Part 1 — Google Colab / Model Development

Part 1 is where the AI models are prepared.

## 24.1 What Google Colab is

Google Colab is a cloud-hosted notebook environment used for Python programming and ML experimentation.

We use it because training deep-learning models can be computationally expensive.

### Important

**Google Colab is NOT the final BAS system.**

It is the training environment.

---

# 25. Part 1 Training Flow

```text
Dataset
   ↓
Google Colab
   ↓
Preprocessing
   ↓
Object detection model training/fine-tuning
   ↓
Pose/hand model testing (usually pretrained first)
   ↓
Action-recognition model training/fine-tuning
   ↓
Evaluation
   ↓
Export
   ↓
.pt / .pth / .onnx
   ↓
Part 2 application
```

---

# 26. What We Actually Train vs Reuse

| Component | Initial approach |
|---|---|
| Object detection | Pretrained YOLO → fine-tune on experiment objects |
| Pose estimation | Use pretrained model |
| Hand landmarks | Use pretrained hand model |
| Object tracking | Existing tracking mechanism |
| Hand-object interaction | Geometry/features/rules first |
| Action recognition | Custom fine-tuning / training; SlowFast is a candidate |
| Sequence validation | Normal Python state machine; no ML training |
| GUI | Normal application code; no training |
| Voice | Normal offline TTS; no training |
| Recording | Normal video I/O; no training |
| IP streaming | Normal network/video streaming; no training |

This is intentional. We do **not** need to train every component from scratch.

---

# 27. What Is a `.pt`, `.pth`, `.onnx` File?

### `.pt` / `.pth`

Common PyTorch model checkpoint formats.

Example:

```text
object_detector.pt
action_model.pth
```

### `.onnx`

A deployment/interchange format that can be used with ONNX Runtime and other runtimes.

The final application can use `.pt/.pth` initially. ONNX can be considered later if deployment testing shows it is useful.

---

# 28. What Is Inference?

**Training** means teaching/fitting a model.

**Inference** means using a trained model to make a prediction on new data.

```text
TRAINING
Dataset → Model → learned weights

INFERENCE
New video → Model → Prediction
```

The final application performs **inference**, not training.

---

# 29. Training Dataset Strategy

The PS explicitly asks for dataset generation around the steps of the experiment, including object detection, pose estimation, and hand-object interaction. fileciteturn1file1L21-L23

The PS also says the dataset can be locally generated, including using a webcam.

### Fast project strategy

Because the team has limited time:

```text
External datasets
      +
Pretrained vision models
      +
Small custom experiment dataset
      ↓
Practical prototype
```

Do **not** spend the entire project trying to record thousands of videos.

---

# 30. Dataset Decision Tree

```text
Do we already have a dataset for our exact experiment?
                    │
                  Usually no
                    │
                    ▼
Use external action datasets for generic action learning
                    +
Create a small custom dataset for our exact objects/actions
                    │
                    ▼
Fine-tune / adapt
                    │
                    ▼
Test on unseen videos/people
```

---

# 31. Dataset 1 — MicroG-4M

### Why it is interesting

MicroG-4M is specifically designed for understanding human actions/scenes in microgravity environments.

The published work describes:

- **4,759 video clips**
- **50 actions**
- **1,238 captions**
- **7,000+ question-answer pairs**
- real space footage + cinematic simulations
- action recognition, captioning, and visual question answering tasks. citeturn567338academia38turn667739search4

The dataset repository exposes action annotations and bounding-box-related annotations. citeturn667739search4

### Links

- GitHub: https://github.com/LEI-QI-233/HAR-in-Space
- Hugging Face dataset: https://huggingface.co/datasets/LEI-QI-233/MicroG-4M
- Paper: https://arxiv.org/abs/2506.02845
- Fine-tuned models: https://huggingface.co/lei-qi-233/MicroG-4M-models

### License

The Hugging Face dataset card lists **CC BY 4.0**. Always read the current dataset terms before redistribution or publication. citeturn667739search2turn667739search4

### How it helps

```text
Space-domain video
       ↓
Microgravity action examples
       ↓
Better starting point for space-related action recognition
```

### Limitation

It does **not** replace the exact SIH experiment dataset.

It is a general microgravity human-action dataset, whereas SIH26174 is about validating the sequence of a **specific predefined experiment**.

### Recommended use

**High relevance, but not the first dataset to depend on if setup time is limited.**

---

# 32. Dataset 2 — HMDB51

### What it is

HMDB51 is a standard human-action video dataset.

The official/commonly mirrored dataset contains **6,849 clips across 51 action categories** and predefined train/test splits. citeturn540262search9

### Link

- Official UCF/CRCV data index: https://www.crcv.ucf.edu/data/
- Dataset page/mirror: https://huggingface.co/datasets/Serrelab/hmdb51

### Useful actions

Depending on the chosen experiment, actions such as:

- pick
- pour
- push
- catch
- throw
- etc.

can provide generic action-recognition training material.

### How it helps

```text
Generic human action
        ↓
Pretrain / fine-tune temporal action model
        ↓
Adapt to our experiment
```

### Limitation

It is not a microgravity dataset and is not an exact experiment dataset.

### Recommendation

**Best practical fallback for a quick action-recognition baseline.**

---

# 33. Dataset 3 — UCF50

UCF50 contains **50 action categories and 6,676 videos** collected from realistic YouTube videos. The official UCF page emphasizes large variation in camera motion, object appearance, pose, scale, viewpoint, clutter and illumination. citeturn540262search1turn540262search3

### Link

- https://www.crcv.ucf.edu/data/UCF50.php

### Useful idea

It provides diverse human actions that can help test whether a temporal video pipeline is functioning.

### Limitation

Many classes are sports/actions that are unrelated to laboratory/space experiments.

### Recommendation

**Secondary option.** Use a subset rather than the whole dataset.

---

# 34. Dataset 4 — UCF101

UCF101 contains **101 action classes and 13,320 videos**. The official documentation reports an average clip length of about 7.21 seconds, 25 FPS, and 320×240 resolution. citeturn725179search40

### Link

- https://www.crcv.ucf.edu/data/UCF101.php
- Paper: https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101_CRCV-TR-12-01.pdf

### Why useful

It is a standard action-recognition benchmark and can provide generic temporal-action examples.

### Limitation

It is broad and includes many sports actions that are irrelevant to our target experiment.

### Recommendation

Use only relevant classes if needed.

---

# 35. Dataset 5 — Something-Something V2

This is conceptually one of the **most relevant generic datasets** for our problem because it contains humans performing predefined basic actions with everyday objects.

It contains **220,847 labeled video clips**. Its action labels include examples such as:

- putting something onto something;
- putting something next to something;
- pushing something;
- holding something;
- closing something;
- pouring something;
- attaching something. citeturn667739search0turn667739search5turn567338search37

### Links

- Qualcomm dataset instructions: https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/20bn-something-something_download_instructions_-_091622-v2.pdf
- Dataset documentation: https://developer.qualcomm.com/software/ai-datasets/something-something
- Hugging Face dataset metadata: https://huggingface.co/datasets/HuggingFaceM4/something_something_v2

### Limitation

The official archive is large: roughly **19.4 GB** of video data according to the download instructions, and preparation is non-trivial. citeturn725179search39

### Recommendation

**Excellent conceptual match, but not the first choice under an 8–10 day hackathon deadline.**

---

# 36. Dataset 6 — JHMDB

JHMDB is a smaller, pose-oriented human-action dataset.

Commonly reported characteristics:

- **21 action classes**
- **928 video sequences**
- pose annotations / skeleton-related information
- temporal action localization utility. citeturn567338search1turn567338search2

### Why it can help

It is useful when the team wants to test action recognition with human-motion information rather than purely raw visual appearance.

### Limitation

It is not a space experiment dataset.

### Recommendation

**Optional research/benchmark dataset.**

---

# 37. NASA / Space-Station Video Resources

NASA and other space agencies publish extensive ISS experiment and research material.

A useful official starting point is:

- NASA Space Station Research Explorer: https://www.nasa.gov/mission/station/research-explorer/
- NASA Space Station Research Resources: https://www.nasa.gov/international-space-station/space-station-research-and-technology/space-station-research-resources/

### Important

Raw NASA video is **not automatically a ready-to-train supervised dataset**.

If we use a raw video, we may still have to annotate:

- action boundaries;
- objects;
- steps;
- interaction states.

Therefore NASA material is best treated as:

```text
reference / supplementary data
```

rather than our main training dataset.

---

# 38. Dataset Priority for This Project

| Priority | Dataset | Best use |
|---|---|---|
| 1 | **HMDB51** | Fast generic action-recognition baseline |
| 2 | **MicroG-4M** | Space/microgravity action understanding |
| 3 | **UCF101 subset** | Generic video-action pretraining/testing |
| 4 | **UCF50 subset** | Generic action baseline |
| 5 | **Something-Something V2** | Fine-grained object interaction; excellent but large |
| 6 | **JHMDB** | Pose/action research |
| 7 | **NASA/ISS videos** | Domain reference/supplementary material |

### Critical rule

**None of these automatically replace the small custom dataset for the exact experiment.**

---

# 39. Custom Dataset — What We Actually Need

The final custom dataset should be based on the chosen prototype experiment.

For each experiment step, we want videos showing:

- correct execution;
- different people;
- different speeds;
- small viewpoint/lighting variations;
- skipped-step examples;
- out-of-order examples;
- repeated actions;
- ambiguous cases.

We do not need huge amounts of data if the experiment is narrowly defined.

---

# 40. How to Collect a Small Custom Dataset Quickly

If team members are geographically separated:

```text
Same experiment specification
        ↓
Each member records locally
        ↓
Webcam / phone camera
        ↓
Upload clips to shared storage
        ↓
Standardize video format
        ↓
Annotate / split
        ↓
Google Colab
```

The videos should maintain approximately consistent camera framing where possible.

### Do not split train/test by random frames from the same video

Bad:

```text
Same video
Frame 1 → Train
Frame 2 → Test
Frame 3 → Train
```

Better:

```text
Training videos → people A/B/C
Testing videos  → person D
```

This gives a more meaningful generalization test.

---

# 41. What Should Be Annotated?

Depending on the model:

### Action-level annotation

```text
00:00–00:04 → STEP_1
00:04–00:09 → STEP_2
00:09–00:15 → STEP_3
```

### Object detection annotation

```text
red_sample → bounding box
blue_sample → bounding box
container  → bounding box
```

### Interaction information

Examples:

```text
hand near object
hand contacts object
holding object
moving object
release
```

We do not necessarily need to manually annotate every body keypoint if a pretrained pose model is providing keypoints.

---

# 42. Part 2 — GUI / Offline Application

Part 2 is the software that runs after the models have been trained.

It should run locally and offline.

The PS explicitly calls for a trained AI model running on an offline standalone system. fileciteturn1file1L21-L23

---

# 43. Final Application Inputs

The application should have two principal modes:

```text
INPUT
│
├── Live Camera
│
└── Uploaded Video
```

### Live Camera

```text
Camera
  ↓
OpenCV
  ↓
Live frames
  ↓
AI pipeline
```

### Uploaded Video

```text
MP4 / AVI / etc.
  ↓
OpenCV / video decoder
  ↓
Frames/clips
  ↓
Same AI pipeline
```

We should not build two separate AI systems.

---

# 44. Final Application Outputs

The application should provide:

- current step;
- expected step;
- detected action;
- AI confidence;
- valid/invalid status;
- next-step suggestion;
- skipped/out-of-sequence alert;
- voice instruction/alert;
- structured experiment log;
- local live-video recording;
- IP stream status;
- GUI monitoring.

These map to the PS requirements. fileciteturn1file1L21-L23

---

# 45. Live Video Flow

```text
                     LIVE CAMERA
                          │
                          ▼
                     OpenCV
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
           AI          Recorder       IP Stream
            │             │              │
            ▼             ▼              ▼
      AI decision      Local MP4     Network/IP
            │
            ▼
    Sequence Validator
            │
       ┌────┴─────┐
       ▼          ▼
     VALID      INVALID
       │          │
       ▼          ▼
    Next step   Voice alert
       │          │
       └────┬─────┘
            ▼
           GUI
```

---

# 46. Uploaded Video Flow

```text
User selects video
        ↓
Application reads local file
        ↓
OpenCV / decoder
        ↓
AI pipeline
        ↓
Action recognition
        ↓
Sequence validation
        ↓
GUI output
        ↓
Experiment log
```

The original uploaded video does not need to be copied merely because it is analyzed.

A processed/annotated output can be generated if useful.

---

# 47. Local Recording vs IP Streaming

These are different things.

## Local recording

```text
Camera
  ↓
Local storage
  ↓
experiment.mp4
```

## IP streaming

```text
Camera / processed video
        ↓
Network stream
        ↓
Specified IP
        ↓
Receiving system
```

The PS explicitly says to **stream the video to a specific IP and also store it locally**. fileciteturn1file1L21-L23

It does **not** explicitly say that a completed MP4 file must be copied to the destination IP.

Streaming normally means the video is transmitted as it is being produced.

---

# 48. How IP Networking Fits In

A basic local demonstration can look like:

```text
             ┌──────────────┐
             │    SWITCH    │
             └──────┬───────┘
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
   IP Camera      AI PC       Monitor PC
                    │
                    └──── IP Stream ───→ Monitor
```

One switch can handle both:

- camera → AI computer;
- AI computer → monitoring computer(s).

A separate switch is **not inherently required for each direction**.

---

# 49. What Is an IP Address?

An IP address identifies a device on an IP network.

Example:

```text
AI computer       192.168.1.10
Monitor computer  192.168.1.20
Second monitor    192.168.1.30
```

The AI system can be configured to stream to:

```text
192.168.1.20
```

An IP network can operate without Internet access.

---

# 50. Physical Network Hardware for Our Prototype

Possible components:

| Hardware | Purpose |
|---|---|
| Fixed/USB/IP camera | Captures experiment video |
| AI computer | Runs our software |
| Ethernet switch | Connects devices on LAN |
| Ethernet cables | Physical network link |
| NIC / Ethernet port | Network interface |
| Monitor/receiver PC | Receives video stream |
| Local storage | Saves recordings/logs |
| Power | Powers hardware |

The exact BAS hardware is not specified by the PS.

Our SIH prototype only needs enough local hardware to demonstrate the software architecture.

---

# 51. Do We Need a Router?

Not necessarily.

For a simple local network:

```text
AI PC
  ↕
Switch
  ↕
Monitor PC
```

can be enough.

A router becomes relevant when different networks need to communicate.

---

# 52. Do We Need Internet During Final Inference?

No.

The final application should be designed to run without:

- cloud AI APIs;
- Google Colab;
- remote model inference;
- mandatory Internet services.

The PS explicitly requires offline standalone model operation. fileciteturn1file1L21-L23

A local IP network is still allowed because:

> **LAN/IP networking is not the same thing as Internet access.**

---

# 53. GUI Design — Final Version

The finalized GUI direction is a **normal desktop application**, optimized for a maximized/full-screen monitoring window rather than a small fixed box.

The current design uses a mission-control style dashboard with:

```text
BAS Monitor
│
├── Overview
├── Procedures
├── Logs
└── Settings
```

and the Overview contains:

```text
Procedure Sequence
        │
        ├── Video Feed
        │
        ├── Experiment Status
        │
        ├── System Readiness
        │
        ├── Live / Upload Input
        │
        ├── IP Stream Status
        │
        ├── Recording Status
        │
        ├── Start / Pause / Stop
        │
        └── Terminal Log
```

The current HTML implements this layout. fileciteturn4file0L211-L234 fileciteturn4file0L235-L330

---

# 54. Final GUI Technologies

## Recommended

**Desktop shell:** PySide6 / Qt 6

**Frontend rendering:** Qt WebEngine

**Frontend:** the existing HTML/CSS/JavaScript design, packaged locally

**Backend:** Python

### Why this combination?

The team already has a designed HTML interface. Rather than rebuilding it entirely as native GUI widgets, PySide6 + Qt WebEngine can embed local HTML/CSS/JS inside a desktop application. Qt WebEngine supports HTML/CSS/JavaScript rendering inside Qt applications. citeturn429092search0turn429092search4turn429092search12

Qt for Python provides the official Python bindings through PySide6. citeturn429092search0

---

# 55. Why Not Keep the Current HTML as a Website?

Because the PS requires an **offline standalone system**.

The current prototype HTML loads external resources such as:

```html
https://cdn.tailwindcss.com
https://fonts.googleapis.com
```

The existing HTML confirms these external dependencies. fileciteturn4file0L3-L9

For the final application:

```text
No CDN
No mandatory Internet
No cloud rendering dependency
```

All resources should be packaged locally.

---

# 56. GUI Screens

## 56.1 Overview

Main monitoring screen.

Contains:

- live/upload video;
- procedure timeline;
- current/expected step;
- status;
- confidence;
- next step;
- system readiness;
- alert/status;
- recording;
- IP stream;
- controls;
- live system log.

## 56.2 Procedures

Defines/selects the predefined experiment and its ordered steps.

## 56.3 Logs

Shows complete experiment records.

## 56.4 Settings

Configures:

- camera;
- models;
- thresholds;
- storage;
- voice;
- network/IP streaming;
- theme.

## 56.5 Diagnostics

Checks:

- models;
- camera;
- storage;
- voice;
- network/IP stream;
- GPU/CPU/memory as useful diagnostics.

---

# 57. Login / Account System?

## No.

The PS does not require:

- login;
- account creation;
- password;
- user profile;
- authentication backend.

We should keep:

```text
Operator: Alpha-1
Session: EXP-2026-001
```

as experiment/session metadata rather than a login system.

Do not add unnecessary authentication scope.

---

# 58. Dark / Light Theme

Include a button to switch:

```text
Dark Mode ↔ Light Mode
```

### Default

Dark Mode.

### Meaning colors stay consistent

- Green = valid/success/ready
- Amber = warning/pending
- Red = error/critical
- Cyan = active/running

The design system already establishes this semantic color scheme. fileciteturn4file1L101-L115

---

# 59. Final GUI Layout

```text
┌─────────────────────────────────────────────────────────────────┐
│ BAS Experiment Monitor                 Time   Operator  ☼  ⚙  │
├───────────────┬──────────────────────────────┬──────────────────┤
│               │                              │                  │
│ Procedure     │         VIDEO                │ Experiment       │
│ Sequence      │                              │ Status           │
│               │    Astronaut + Experiment    │                  │
│ Step 1 ✓      │                              │ Current Step     │
│ Step 2 ✓      │     AI overlays              │ Expected Step    │
│ Step 3 ●      │                              │ Status           │
│ Step 4 ○      │                              │ Confidence       │
│ Step 5 ○      │                              │ Next Step        │
│               │                              │                  │
├───────────────┴──────────────────────────────┤                  │
│                                               │                  │
│ Live Camera | Upload Video    IP Stream       │ System Readiness │
│                                               │                  │
│ Start   Pause   Stop      Recording           │                  │
│                                               │                  │
├───────────────────────────────────────────────┼──────────────────┤
│ Terminal Log                                  │                 │
└───────────────────────────────────────────────┴─────────────────┘
```

---

# 60. Current Design System

The finalized design document uses:

### Typography

- Inter for normal UI text
- JetBrains Mono for data/status/timestamps. fileciteturn4file1L51-L84

### Colors

- Neon cyan = active/running
- Amber = warning/caution
- Emerald green = success/stable
- dark neutral surfaces = background. fileciteturn4file1L101-L115

### Layout

- 12-column fixed desktop grid;
- functional zones;
- panel-based layout;
- data-dense control-room style. fileciteturn4file1L123-L129

### Shapes

- soft-industrial;
- approximately 4 px rounded corners;
- circular LEDs for status indicators. fileciteturn4file1L141-L147

### Components

- low-profile buttons;
- status LEDs;
- data cards;
- gauges;
- terminal/log panel. fileciteturn4file1L149-L156

---

# 61. Existing GUI Files

The current UI prototype files are:

```text
code(2).html
DESIGN(2).md
screen(2).png
```

### `code(2).html`

Visual/UI implementation of the current dashboard.

The current HTML contains the final sidebar, header, procedure sequence, video area, experiment status, system readiness, input controls, IP stream display, recording status, and terminal log. fileciteturn4file0L164-L234 fileciteturn4file0L235-L330 fileciteturn4file0L331-L486

### `DESIGN(2).md`

Visual design system:

- colors;
- typography;
- spacing;
- layout;
- component conventions;
- status semantics. fileciteturn4file1L101-L156

### `screen(2).png`

Reference screenshot showing the target appearance.

---

# 62. What Must Be Changed from the HTML Prototype for Final Offline App

The prototype currently contains external CDN/font links. fileciteturn4file0L3-L9

Final application:

```text
External assets
   ↓
LOCAL packaged assets
```

Also remove/deactivate unnecessary elements:

- mobile bottom navigation;
- account/profile icon;
- logout;
- unnecessary telemetry page;
- anything not required by the PS.

The final application should remain focused.

---

# 63. Backend Architecture

```text
                 PySide6 Desktop Shell
                          │
                    Qt WebEngine UI
                          │
                     JS ↔ Python
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Input Manager     AI Inference      Experiment Engine
        │                 │                 │
        │           ┌─────┼─────┐           │
        │           ▼     ▼     ▼           │
        │          YOLO  Pose  Hands        │
        │                         │         │
        │                         ▼         │
        │                    Action Model   │
        │                                   │
        └──────────────────┬────────────────┘
                           ▼
                     Result Manager
                           │
            ┌──────────────┼─────────────┐
            ▼              ▼             ▼
        GUI Updates     Logger       Voice Alerts
                           │
                           ▼
                       Storage
                           │
                           ▼
                     IP Streaming
```

---

# 64. Recommended Part-2 Modules

```text
backend/
│
├── video/
│   ├── camera.py
│   ├── video_loader.py
│   ├── frame_processor.py
│   └── recorder.py
│
├── ai/
│   ├── object_detector.py
│   ├── pose_detector.py
│   ├── hand_detector.py
│   ├── action_recognizer.py
│   └── inference_pipeline.py
│
├── experiment/
│   ├── procedure_manager.py
│   ├── interaction_logic.py
│   └── sequence_validator.py
│
├── network/
│   ├── network_manager.py
│   └── ip_stream.py
│
├── voice/
│   └── voice_alert.py
│
├── logging/
│   ├── experiment_logger.py
│   └── system_logger.py
│
└── storage/
    └── storage_manager.py
```

---

# 65. What Each File Does

## `main.py`

Starts the application.

## `camera.py`

Connects to webcam/camera source.

## `video_loader.py`

Loads uploaded video files.

## `frame_processor.py`

Normalizes/prepares frames and clips for inference.

## `recorder.py`

Saves live experiment video locally.

## `object_detector.py`

Loads YOLO object model and returns detected experiment objects.

## `pose_detector.py`

Gets body keypoints.

## `hand_detector.py`

Gets hand landmarks/hand information.

## `action_recognizer.py`

Runs the trained action-recognition model, e.g. SlowFast.

## `inference_pipeline.py`

Coordinates the complete visual inference chain.

## `interaction_logic.py`

Combines hand/object positions and motion to infer interactions.

## `procedure_manager.py`

Loads the experiment definition.

## `sequence_validator.py`

Checks detected actions against expected step order.

## `ip_stream.py`

Starts/maintains video streaming to the configured destination.

## `network_manager.py`

Checks network/connection configuration.

## `voice_alert.py`

Provides local speech alerts/instructions.

## `experiment_logger.py`

Writes structured experiment records.

## `system_logger.py`

Writes internal debug/status logs.

## `storage_manager.py`

Manages directories and output filenames.

---

# 66. Configuration Files

## `config/experiment.json`

Contains:

```json
{
  "name": "Sample Experiment",
  "steps": [
    {"id": "S1", "action": "ACTION_1"},
    {"id": "S2", "action": "ACTION_2"},
    {"id": "S3", "action": "ACTION_3"}
  ]
}
```

The exact action names will change after we select the experiment.

## `config/settings.json`

Contains:

- camera source;
- FPS/resolution;
- confidence thresholds;
- storage directories;
- theme;
- voice settings.

## `config/network.json`

Contains:

- destination IP;
- streaming port;
- streaming protocol;
- local network settings if necessary.

---

# 67. Recommended Technology Stack — Part 1

## A. Environment

- Google Colab
- Google Drive for dataset/model storage if useful

## B. Python / ML

- Python
- PyTorch
- Ultralytics YOLO
- MediaPipe / pose/hand libraries
- torchvision / PyTorch-related utilities
- scikit-learn for conventional metrics and baselines

## C. Video/Data

- OpenCV
- NumPy
- Pandas
- PyAV/FFmpeg where needed for video loading

## D. Dataset Annotation

- CVAT or equivalent

## E. Model Deployment Export

- PyTorch `.pt/.pth`
- optional ONNX

---

# 68. Recommended Technology Stack — Part 2

## A. Desktop Application

- Python
- PySide6
- Qt 6
- Qt WebEngine

## B. Video

- OpenCV
- FFmpeg where appropriate

## C. AI inference

- PyTorch
- YOLO
- pose/hand model
- SlowFast/action model

## D. Experiment logic

- pure Python
- state machine / rules

## E. Voice

- offline TTS such as pyttsx3 / OS speech interface / another packaged offline engine

## F. Storage

- local filesystem
- JSON/CSV/TXT experiment logs
- SQLite only if needed

## G. Network

- IP networking
- Ethernet/Wi-Fi LAN
- RTSP or another appropriate local video streaming protocol

## H. Packaging

- PyInstaller or equivalent

---

# 69. Key Technology Links

## Core Python / GUI

- PySide6 / Qt for Python: https://doc.qt.io/qtforpython-6/
- Qt WebEngine: https://doc.qt.io/qtforpython-6/QtWebEngine.html
- PySide6 installation: https://doc.qt.io/qtforpython-6/gettingstarted.html

## AI / CV

- PyTorch: https://pytorch.org/
- Ultralytics: https://docs.ultralytics.com/
- MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/guide
- OpenCV: https://opencv.org/
- FFmpeg: https://ffmpeg.org/

## Action Recognition

- SlowFast repository: https://github.com/facebookresearch/SlowFast
- PyTorchVideo: https://pytorchvideo.org/

## Annotation

- CVAT: https://www.cvat.ai/
- CVAT documentation: https://docs.cvat.ai/

## Packaging

- PyInstaller: https://pyinstaller.org/

## Training environment

- Google Colab: https://colab.google/

## Space context

- ISRO: https://www.isro.gov.in/
- ISRO Space Vision / BAS context: https://www.isro.gov.in/
- NASA ISS Research Explorer: https://www.nasa.gov/mission/station/research-explorer/

---

# 70. Dataset / Model Training File Structure

```text
training/
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_yolo_training.ipynb
│   ├── 04_action_recognition_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_export_models.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── splits/
│
├── checkpoints/
│   ├── yolo/
│   └── action_model/
│
├── exports/
│   ├── object_detector.pt
│   ├── action_model.pt
│   └── optional.onnx
│
└── reports/
    ├── metrics.json
    ├── confusion_matrix.png
    └── evaluation.md
```

---

# 71. Suggested Colab Notebook Responsibilities

## `01_dataset_inspection.ipynb`

- mount Google Drive;
- inspect videos;
- inspect metadata;
- visualize samples;
- verify labels.

## `02_preprocessing.ipynb`

- frame extraction;
- resizing;
- temporal clip generation;
- data splitting.

## `03_yolo_training.ipynb`

- load experiment object annotations;
- train/fine-tune object detector;
- evaluate precision/recall/mAP;
- export weights.

## `04_action_recognition_training.ipynb`

- prepare clips;
- map labels;
- train/fine-tune SlowFast or alternative action model;
- measure accuracy/F1/confusion matrix.

## `05_evaluation.ipynb`

- test new videos;
- test unseen people;
- test wrong sequences;
- evaluate temporal recognition.

## `06_export_models.ipynb`

- save final checkpoints;
- optionally export ONNX;
- copy model files into application `models/` directory.

---

# 72. Complete Model Development Flow

```text
                    DATASET
                       │
              ┌────────┴─────────┐
              ▼                  ▼
      External datasets      Custom data
              │                  │
              └────────┬─────────┘
                       ▼
                Google Colab
                       │
                 Preprocessing
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Objects          Pose          Actions
        │              │              │
      YOLO          pretrained      SlowFast
        │              │           / alternative
        └──────────────┼──────────────┘
                       ▼
                   Evaluation
                       │
                       ▼
                  Model export
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
             .pt      .pth     .onnx
                       │
                       ▼
                Part 2 Application
```

---

# 73. Complete Final Runtime Flow

```text
                 FIXED CAMERA
                      │
                      ▼
                Input Manager
                      │
                      ▼
                  OpenCV
                      │
                      ▼
               Video Frame(s)
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Visual Perception   Video Recorder
             │                 │
       ┌─────┼─────┐           ▼
       ▼     ▼     ▼       Local MP4
      YOLO  Pose  Hands
       │     │     │
       └─────┼─────┘
             ▼
      Interaction Logic
             ▼
       Action Recognition
             ▼
        Detected Action
             ▼
      Experiment Mapping
             ▼
      Sequence Validator
             │
       ┌─────┴─────┐
       ▼           ▼
     VALID       INVALID
       │           │
       ▼           ├────→ Voice Alert
   Next Step       │
       │           ├────→ Experiment Log
       │           │
       └─────┬─────┘
             ▼
            GUI
             │
             ├────→ Status
             ├────→ Procedure timeline
             ├────→ Alert panel
             └────→ Action/confidence

             +

       IP Stream Manager
             ↓
       Local Network / IP
```

---

# 74. GUI ↔ Backend Data Flow

The GUI should not contain AI logic.

Instead:

```text
GUI
 ↓
command
 ↓
Python backend
 ↓
operation
 ↓
result
 ↓
GUI update
```

Example:

```text
User clicks START ANALYSIS
        ↓
GUI sends command
        ↓
Backend starts camera
        ↓
AI starts inference
        ↓
Results arrive
        ↓
GUI updates status
```

---

# 75. Example of One AI Result

A backend event can conceptually look like:

```json
{
  "timestamp": "2026-08-26T19:20:31",
  "action": "PICK",
  "object": "RED_SAMPLE",
  "confidence": 0.94,
  "expected_step": "STEP_3",
  "detected_step": "STEP_3",
  "status": "VALID",
  "next_step": "STEP_4"
}
```

The GUI then transforms that into:

```text
Current Step: STEP 3
Expected: STEP 3
Status: VALID
Confidence: 94%
Next Step: STEP 4
```

---

# 76. Example of an Error Result

```json
{
  "timestamp": "2026-08-26T19:22:11",
  "action": "PLACE_BLUE",
  "confidence": 0.91,
  "expected_step": "STEP_3",
  "detected_step": "STEP_4",
  "status": "OUT_OF_SEQUENCE",
  "alert": true
}
```

GUI:

```text
STATUS: OUT OF SEQUENCE

Expected:
Step 3

Detected:
Step 4

Voice alert:
ACTIVE
```

---

# 77. What Is a State Machine?

A state machine is simply a program that remembers **where we currently are in a sequence**.

Example:

```text
START
  ↓
S1
  ↓
S2
  ↓
S3
  ↓
S4
  ↓
END
```

If we are at S3, the next normal state is S4.

If the AI detects S5 instead:

```text
S3
 ↓
S5
```

then the software can flag an error.

This is why sequence validation does not need another neural network.

---

# 78. What Is a Bounding Box?

A bounding box is a rectangle around a detected object or person.

```text
┌───────────────┐
│    OBJECT     │
│               │
└───────────────┘
```

The AI can report:

```text
x1, y1, x2, y2
class
confidence
```

---

# 79. What Is a Keypoint?

A keypoint is a specific body landmark.

Example:

```text
head
shoulder
elbow
wrist
```

A pose model can produce many of these points.

---

# 80. What Is Confidence?

Confidence is the model's estimate of how strongly it believes a prediction.

Example:

```text
PICK_RED = 0.94
```

means approximately:


> “The model is strongly confident in this classification according to its scoring mechanism.”

It is **not automatically equivalent to true 94% real-world probability**.

The GUI can display confidence as a useful diagnostic.

---

# 81. What Is Fine-Tuning?

Instead of training a model from zero:

```text
Random weights
   ↓
Huge training requirement
```

we start from an existing trained model:

```text
Pretrained model
   ↓
Our dataset
   ↓
Fine-tuning
   ↓
Domain-adapted model
```

For this project, fine-tuning is preferred where practical.

---

# 82. Do We Need Blender?

## No.

Blender is **not required by the PS**.

It could optionally be used later for synthetic data generation:

```text
3D experiment
   ↓
Virtual cameras
   ↓
Lighting variation
   ↓
Synthetic frames
```

But synthetic data can introduce a **domain gap** between renders and real camera footage.

Given the short deadline:

> **Real/near-real data should be the primary source. Blender is optional and should not block the project.**

---

# 83. Do We Need a Local AI Model?

## Yes, for final inference.

But:

> “Local” means the trained model is present on the inference machine; it does not mean we must train the model on that same machine.

So:

```text
Google Colab
   ↓
train
   ↓
export model
   ↓
local PC
   ↓
inference
```

That is compatible with the PS's offline standalone requirement. fileciteturn1file1L21-L23

---

# 84. Can Google Colab Be Used for Training?

## Yes.

Colab is our development/training environment.

The final application must not depend on Colab during inference.

---

# 85. Do We Need to Build Earth-to-Space Communication?

## No.

The communication/bandwidth limitation is part of the **motivation for edge processing**.

The PS does not ask us to design:

- satellite communications;
- antennas;
- ground stations;
- radio modems;
- orbital networking.

Do not scope-creep into spacecraft communications.

---

# 86. Do We Need the AI to Run on an Astronaut's Wearable Device?

## Not required by the PS.

The PS says fixed-payload cameras are the input.

It does not specify a wearable device.

The best prototype assumption is:

```text
Fixed camera
   ↓
Onboard/local computer
   ↓
AI
```

---

# 87. What Does the “Specific IP” Requirement Mean?

The PS says:

> Stream the experiment video to a specific IP and also store the video locally. fileciteturn1file1L21-L23

The safest interpretation is:

```text
Central AI system
      ↓
Local/network video stream
      ↓
Configured destination IP
```

The PS itself does **not clearly specify where that IP is geographically located**.

Therefore, do not claim:

> “The PS requires streaming the video from BAS all the way to Earth.”

It does not say that.

For the prototype, a local network destination is sufficient to demonstrate the concept without recreating a spacecraft-to-Earth communications link.

---

# 88. IP Streaming vs File Transfer

### Streaming

```text
Frame 1 → receiver
Frame 2 → receiver
Frame 3 → receiver
Frame 4 → receiver
```

The receiver can display video while the experiment is still happening.

### File transfer

```text
Record entire experiment
        ↓
Complete MP4
        ↓
Copy file
```

The PS says **stream**, not explicitly “copy the final MP4 file to the destination.”

---

# 89. Final Project Folder Structure

```text
SIH26174-BAS-Monitor/
│
├── GUIDE.md
├── README.md
├── requirements.txt
├── main.py
├── LICENSE
│
├── frontend/
│   ├── index.html
│   ├── css/
│   ├── js/
│   ├── assets/
│   └── pages/
│
├── backend/
│   ├── video/
│   ├── ai/
│   ├── experiment/
│   ├── network/
│   ├── voice/
│   ├── logging/
│   └── storage/
│
├── models/
│   ├── object_detector.pt
│   ├── action_model.pt
│   └── optional/
│
├── config/
│   ├── experiment.json
│   ├── settings.json
│   └── network.json
│
├── data/
│   ├── recordings/
│   ├── logs/
│   └── snapshots/
│
├── training/
│   ├── notebooks/
│   ├── data/
│   ├── checkpoints/
│   ├── exports/
│   └── reports/
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    └── demo.md
```

---

# 90. Which Side Owns Which File?

```text
PART 1 — MODEL DEVELOPMENT

training/
models/

        ↓ exports model weights

PART 2 — APPLICATION

frontend/
backend/
config/
data/
main.py
```

---

# 91. What the Final Demonstration Should Show

The strongest SIH demonstration should not be just:

> “Here is an AI model that predicts an action.”

It should show the whole loop.

### Demo 1 — Correct sequence

```text
Live Camera
   ↓
AI sees experiment
   ↓
Step 1
   ↓
Step 2
   ↓
Step 3
   ↓
VALID
   ↓
Next step suggestion
```

### Demo 2 — Skipped step

```text
Expected: Step 3
Observed: Step 4
        ↓
SKIPPED / OUT OF SEQUENCE
        ↓
Voice alert
        ↓
Log event
```

### Demo 3 — Uploaded video

```text
Upload pre-recorded experiment
   ↓
Analyze
   ↓
Show final sequence result
```

### Demo 4 — Local recording

```text
Live run
   ↓
Save MP4
```

### Demo 5 — IP stream

```text
Live experiment
   ↓
Stream
   ↓
Second computer receives video
```

This demonstrates far more of the actual PS than a simple action-classification screen.

---

# 92. What the GUI Should NOT Falsely Claim

Because the UI is a prototype, we must not hard-code claims that aren't actually true.

For example, this prototype currently displays:

```text
AI Confidence: 94%
IP Stream: CONNECTED
Storage: READY
Voice: READY
```

Those should become **real runtime values** in the final product.

The current HTML is visually correct as a prototype, but the final system must bind those indicators to actual system health and inference results.

---

# 93. Development Order

The team should build in this order:

```text
1. Finalize experiment definition
        ↓
2. Prepare datasets
        ↓
3. Establish action-recognition baseline
        ↓
4. Fine-tune object detector
        ↓
5. Validate pose/hand pipeline
        ↓
6. Implement interaction logic
        ↓
7. Implement sequence validator
        ↓
8. Test the complete AI pipeline offline
        ↓
9. Convert GUI prototype into PySide6 app
        ↓
10. Connect GUI ↔ backend
        ↓
11. Add recording
        ↓
12. Add voice alerts
        ↓
13. Add structured logging
        ↓
14. Add local IP streaming
        ↓
15. Add diagnostics
        ↓
16. Package the application
        ↓
17. Final integrated demo
```

---

# 94. What NOT to Build First

Avoid these until the core pipeline works:

- 3D HMR;
- Blender synthetic environment;
- login system;
- cloud backend;
- mobile app;
- remote database;
- large-scale user management;
- complicated telemetry;
- Earth-space communication simulation;
- LLM chatbot.

These are not core requirements.

---

# 95. Optional 3D HMR Feature

If time remains, the optional orientation-agnostic 3D Human Mesh Recovery feature can be explored.

Conceptually:

```text
Camera
  ↓
3D body reconstruction
  ↓
Body orientation relative to payload rack
  ↓
Robust pose understanding in microgravity
```

This is genuinely difficult and research-heavy.

Therefore:

> **Treat it as a stretch goal, not the foundation of the 8–10 day prototype.**

The PS itself marks this challenge as optional. fileciteturn1file1L21-L23

---

# 96. Important Terminology Cheat Sheet

| Term | Simple meaning |
|---|---|
| ISRO | India's space organisation |
| BAS | Bharatiya Antariksh Station, India's planned space station |
| LEO | Low Earth Orbit |
| Microgravity | Very low effective gravity / near-weightless environment |
| HAR | Human Activity Recognition |
| Edge AI | AI running near where data is generated |
| Offline | No dependency on Internet/cloud for core operation |
| Fixed-payload camera | Camera fixed in/around the experiment payload environment |
| Object Detection | Finds objects and their positions |
| Pose Estimation | Finds human body keypoints |
| Hand Landmark | Detailed hand/finger points |
| Tracking | Following the same object/person through time |
| Interaction | Relationship between a hand/person and an object |
| Action Recognition | Determines what action is occurring across time |
| CNN | Convolutional neural network family for visual feature learning |
| 3D CNN | Convolution over time + spatial dimensions for video |
| SlowFast | A video action-recognition architecture using slow/fast pathways |
| State Machine | Software that tracks the current step/state in an ordered process |
| Inference | Using a trained model to make predictions |
| Fine-tuning | Adapting a pretrained model to a specific dataset/domain |
| `.pt/.pth` | Common PyTorch model checkpoint files |
| ONNX | Model interchange/deployment format |
| GUI | Graphical User Interface |
| LAN | Local Area Network |
| IP | Network address used by IP devices |
| Switch | Device that connects multiple network devices on a LAN |
| RTSP | A common network protocol for media streaming/control |
| TTS | Text-to-Speech |
| Logging | Recording system/experiment events |
| Dataset annotation | Attaching labels/boxes/timestamps to data |
| Bounding box | Rectangle around an object/person |
| Keypoint | Important coordinate on a body/hand |
| Confidence | Model score indicating prediction strength |
| FPS | Frames per second |
| Latency | Time delay between input and response |
| mAP | Common object-detection evaluation metric |
| F1 | Combined precision/recall metric |

---

# 97. Important Architectural Truths

These rules should be preserved even if the implementation changes.

### Truth 1

```text
Google Colab ≠ final application
```

Colab is for training.

### Truth 2

```text
GUI ≠ AI model
```

GUI displays/controls the AI system.

### Truth 3

```text
Sequence validation ≠ CNN
```

It is primarily deterministic experiment logic.

### Truth 4

```text
SlowFast ≠ sequence validator
```

SlowFast is an action-recognition candidate.

### Truth 5

```text
Offline ≠ no LAN
```

A local IP network can work without Internet.

### Truth 6

```text
Streaming ≠ sending a completed MP4 file
```

Streaming is continuous media transmission.

### Truth 7

```text
No login is required by the PS
```

Do not build one unless the team later finds a concrete reason.

### Truth 8

```text
BAS is future deployment context
```

Our Earth-based computer is a prototype for the onboard edge computer.

---

# 98. What the Final Product Means

When everything is finished, the system should conceptually look like:

```text
                       BAS EXPERIMENT
                              │
                              ▼
                     Fixed-Payload Camera
                              │
                              ▼
                      Local Edge Computer
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
            Video Processing          Video Storage
                 │                         │
                 ▼                         ▼
            AI Perception               Local MP4
                 │
                 ▼
          Hand–Object Interaction
                 │
                 ▼
          Action Recognition
                 │
                 ▼
          Sequence Validation
                 │
         ┌───────┴────────┐
         ▼                ▼
       VALID            INVALID
         │                │
         ▼                ▼
     Next Step        Voice Alert
         │                │
         └───────┬────────┘
                 ▼
                GUI
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
     Status    Logs      IP Stream
```

---

# 99. How the Prototype Maps to the Future BAS Concept

```text
OUR EARTH PROTOTYPE

Webcam
  ↓
PC/Laptop
  ↓
Our application
  ↓
Local inference
  ↓
GUI / speaker / network

                 ↓ conceptual deployment ↓

FUTURE BAS

Fixed-payload camera
  ↓
Onboard/edge computer
  ↓
Our application
  ↓
Local inference
  ↓
Astronaut feedback / local monitoring
```

The hardware details of the final BAS deployment are not specified by the SIH PS. Therefore, the prototype should demonstrate the **software architecture and edge-processing concept**, not claim that we have reproduced the final spacecraft hardware.

---

# 100. Final Recommended Stack at a Glance

## PART 1 — GOOGLE COLAB / MODEL DEVELOPMENT

```text
Python
│
├── PyTorch
├── Ultralytics YOLO
├── MediaPipe / pose libraries
├── OpenCV
├── NumPy / Pandas
├── scikit-learn
├── PyTorchVideo / SlowFast ecosystem
├── CVAT
└── Google Colab
```

### Responsibilities

```text
Dataset preparation
Object detection training
Action recognition training
Evaluation
Model export
```

---

## PART 2 — GUI / OFFLINE EDGE APPLICATION

```text
Python
│
├── PySide6
├── Qt WebEngine
├── HTML/CSS/JS (local packaged frontend)
├── OpenCV
├── PyTorch
├── YOLO inference
├── Pose/Hand inference
├── SlowFast/action model
├── Python state machine
├── Offline TTS
├── JSON/CSV/TXT logging
├── Local filesystem / optional SQLite
├── FFmpeg / RTSP or suitable stream
└── PyInstaller
```

### Responsibilities

```text
Live video
Uploaded video
AI inference
Sequence validation
GUI
Voice
Recording
Logs
IP stream
Diagnostics
```

---

# 101. External Resources / Links

## SIH / PS

- SIH Problem Statement portal: https://sih.gov.in/sih2026PS
- Problem mirror/reference (use only as a convenience): https://sih-fit.vercel.app/problem/SIH26174

## ISRO / BAS

- ISRO main site: https://www.isro.gov.in/
- ISRO Human Spaceflight / microgravity context: https://www.isro.gov.in/Indian_microgravity_research_Axiom4_mission.html
- ISRO IMEx-2026: https://www.isro.gov.in/IndianMicrogravityExperiments_IMEx2026.html

## GUI

- Qt for Python: https://doc.qt.io/qtforpython-6/
- Qt WebEngine: https://doc.qt.io/qtforpython-6/QtWebEngine.html

## AI / Vision

- PyTorch: https://pytorch.org/
- Ultralytics YOLO: https://docs.ultralytics.com/
- MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/guide
- OpenCV: https://opencv.org/
- FFmpeg: https://ffmpeg.org/
- SlowFast repository: https://github.com/facebookresearch/SlowFast
- PyTorchVideo: https://pytorchvideo.org/

## Datasets

- MicroG-4M GitHub: https://github.com/LEI-QI-233/HAR-in-Space
- MicroG-4M Hugging Face: https://huggingface.co/datasets/LEI-QI-233/MicroG-4M
- HMDB51: https://www.crcv.ucf.edu/data/
- HMDB51 Hugging Face: https://huggingface.co/datasets/Serrelab/hmdb51
- UCF50: https://www.crcv.ucf.edu/data/UCF50.php
- UCF101: https://www.crcv.ucf.edu/data/UCF101.php
- UCF101 paper/report: https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101_CRCV-TR-12-01.pdf
- Something-Something V2: https://developer.qualcomm.com/software/ai-datasets/something-something
- Something-Something V2 instructions PDF: https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/20bn-something-something_download_instructions_-_091622-v2.pdf
- JHMDB: use the dataset/research references for the Joint-annotated Human Motion Database; verify access/licensing before use.

## Annotation / Packaging

- CVAT: https://www.cvat.ai/
- CVAT docs: https://docs.cvat.ai/
- PyInstaller: https://pyinstaller.org/
- Google Colab: https://colab.google/

## Space video/research references

- NASA ISS Research Explorer: https://www.nasa.gov/mission/station/research-explorer/
- NASA ISS Research Resources: https://www.nasa.gov/international-space-station/space-station-research-and-technology/space-station-research-resources/

---

# 102. What We Can Safely Claim in the SIH Presentation

### Good claim

> “We designed an offline edge-AI system that processes fixed-camera experiment video locally, recognizes experiment activities, validates the predefined procedure sequence, provides next-step guidance and voice alerts, records the experiment, and supports IP video streaming.”

### Avoid this claim

> “ISRO has already deployed this exact architecture inside BAS.”

We do not have evidence for that.

### Good claim

> “Our prototype demonstrates the software architecture intended for future onboard deployment.”

### Avoid this claim

> “Our laptop is the actual BAS onboard computer.”

It is a prototype representation.

### Good claim

> “The system is designed for offline local inference.”

### Avoid this claim

> “The system has solved spacecraft communication.”

It has not.

---

# 103. One-Paragraph Explanation for a Completely New Team Member

> SIH26174 is a software problem from ISRO about future experiments on the Bharatiya Antariksh Station (BAS). Imagine an astronaut performing a scientific experiment in space. The experiment has a fixed sequence of steps. A fixed camera watches the astronaut and the experiment area. Instead of sending all raw video to Earth for processing, a local computer on the station processes the video itself. AI models detect objects, estimate the astronaut's pose and hands, understand the astronaut's interaction with the objects, and recognize the current action. A separate software state machine checks whether the recognized action is the correct next step. If a step is skipped or performed out of order, the system gives a voice alert. It also suggests what should happen next, creates a timestamped experiment log, stores the live video locally, streams the video to a specified IP, and shows everything in an offline desktop GUI. We train the AI models in Google Colab, export the trained model files, and then deploy those files inside our local application. Our Earth-based computer is a prototype for the future onboard/edge computer; we are not building the spacecraft or Earth-space communication infrastructure.

---

# 104. Final Mental Model

Remember this single diagram:

```text
                         BAS / FUTURE SPACE STATION
                                      │
                              Fixed-Payload Camera
                                      │
                                      ▼
                               Local Edge Computer
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                         Video Input      Local Recording
                              │
                              ▼
                         AI PERCEPTION
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
                Objects      Pose       Hands
                  └───────────┼───────────┘
                              ▼
                    Hand–Object Interaction
                              ▼
                      Action Recognition
                              ▼
                    Sequence Validation
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
                   VALID             ERROR
                     │                 │
                     ▼                 ▼
                NEXT STEP          VOICE ALERT
                     │                 │
                     └────────┬────────┘
                              ▼
                             GUI
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                 Status     Logs     IP Stream


TRAINING SIDE:

Dataset → Google Colab → Train/Fine-tune → Export model → Local application
```

---

# 105. Final Project Definition

## Project Name

**BAS Experiment Monitor**

## Project Goal

Build an offline edge-AI prototype for monitoring and validating a predefined astronaut experiment sequence from fixed-camera video.

## Core AI

- Object detection
- Pose estimation
- Hand tracking/landmarks
- Hand-object interaction
- Temporal action recognition
- Sequence validation

## Core application

- Live camera input
- Video upload input
- Real-time analysis
- Next-step guidance
- Voice alerts
- Local recording
- Timestamped experiment logs
- IP video streaming
- Desktop GUI
- Diagnostics

## Deployment concept

```text
Earth prototype computer
          ↓
Demonstrates
          ↓
Future onboard/edge deployment concept
```

## Core principle

> **Train in Google Colab. Run inference locally. Validate the experiment sequence deterministically. Keep the system offline during inference.**

---

# 106. Source and Scope Notes

This GUIDE intentionally distinguishes three categories of information:

### A. PS-supported requirements

Items explicitly present in the supplied SIH26174 material, such as fixed-payload camera input, local edge processing, custom experiment-oriented dataset generation, next-step suggestion, voice alerts, structured logging, IP streaming, local video storage, GUI, and offline standalone execution. fileciteturn1file1L21-L23

### B. Official current context

BAS roadmap information such as the current 2028 first-module / 2035 full-station targets comes from current ISRO publications. citeturn725179search38turn725179search37

### C. Team implementation choices

Technologies such as PySide6, Qt WebEngine, YOLO, MediaPipe, SlowFast, JSON, SQLite, FFmpeg, RTSP, and PyInstaller are **recommended implementation choices**, not explicit vendor-mandated technologies from the PS.

---

# 107. Final Rule for Future AI Assistants Using This Guide

Any AI tool receiving this `GUIDE.md` should follow these assumptions unless the team explicitly changes them:

1. SIH26174 is an **offline edge-AI experiment-monitoring system**.
2. The camera input is from **fixed-payload cameras**.
3. The exact camera mounting location is **not specified** by the PS.
4. The system is conceptually intended for **onboard/local processing**.
5. Earth-space communication hardware is **outside the core project scope**.
6. Google Colab is a **training environment**, not the final inference environment.
7. The final application runs on a **local/offline computer**.
8. External datasets can help, but a **small custom experiment-specific dataset is still needed**.
9. Object detection, pose, hands, action recognition, and sequence validation are **different functions**, not one giant “CNN layer.”
10. SlowFast is an **action-recognition candidate**, not the final sequence validator.
11. Sequence validation should initially be a **deterministic state machine/rule engine**.
12. The GUI is a **desktop monitoring application**, not a normal public website.
13. A login/account system is **not required** by the PS.
14. The current HTML/CSS design is a **frontend prototype** that should be embedded/converted into the final offline desktop application.
15. The GUI should support both **live camera** and **uploaded video** using the same backend processing pipeline.
16. Live experiment video should be **stored locally** and can also be **streamed to a configured IP endpoint**.
17. The exact destination represented by “specific IP” is **not defined by the supplied PS**, so no claim should be made that it necessarily means “Earth.”
18. Optional 3D orientation-agnostic HMR is a **stretch goal**, not the minimum viable implementation.

---

# END OF GUIDE.md
