# Workstream B B5.2 — Deterministic Architecture Benchmark Report

**Date:** 2026-09-24  
**Gate:** B5.2 — Deterministic Architecture Benchmark Harness  
**Author:** Workstream B (AI / Procedure Intelligence)  
**Status:** PARTIALLY PASSED — PIPELINE BENCHMARKS EMPIRICALLY MEASURED / MODEL INFERENCE BLOCKED (NO WEIGHTS / NO DATASET)

---

## 1. Purpose & Scope

Gate B5.2 establishes a deterministic, reproducible benchmark harness ([`evaluation/architecture_benchmark.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/architecture_benchmark.py)) to evaluate the computational and data-preparation pipelines of the two Gate B5 candidates:

- **Candidate A:** End-to-End Spatio-Temporal Video Model (**R(2+1)D-18**)
- **Candidate B:** Compact Feature-Based Temporal Model (**1D-TCN**)

### Core Operational Principles
1. **Zero Fabrication:** Only pipeline stages with real, executable code in the repository are measured. Absent model weights and uncollected datasets are explicitly reported as **`BLOCKED`**.
2. **Epistemic Segregation:** Verified empirical measurements, published theoretical estimates, and blocked experimental metrics are strictly separated.
3. **Architectural Neutrality:** Gate B5.2 evaluates preparation pipelines and computational footprints; it does **not** declare a winning architecture.

---

## 2. Summary Comparison: Verified Measurements vs. Estimates vs. Blocked Metrics

| Evaluation Dimension | Candidate A: R(2+1)D-18 | Candidate B: 1D-TCN | Epistemic Status |
|---|---|---|:---:|
| **Input Modality** | 5D RGB Video Tensor: `(1, 3, T, H, W)` | 3D Spatial Feature Tensor: `(1, 30, 111)` | **VERIFIED REALITY** |
| **Spatial Feature Extraction Time** | N/A (Direct pixel consumer) | $\mathbf{0.043\text{ ms}}$ per frame ($>23,000\text{ FPS}$) | **VERIFIED MEASUREMENT** |
| **Temporal Buffer Stacking Time** | N/A (Video buffer sliding) | $\mathbf{0.103\text{ ms}}$ per window ($>9,500\text{ WPS}$) | **VERIFIED MEASUREMENT** |
| **Total Input Prep Latency** | $94.1\text{ ms}$ ($16\times 112$) / $545.9\text{ ms}$ ($32\times 224$) | $\mathbf{0.146\text{ ms}}$ total preparation time | **VERIFIED MEASUREMENT** |
| **Data Memory Footprint** | $2.35\text{ MB}$ ($16\times 112$) / $18.82\text{ MB}$ ($32\times 224$) | $\mathbf{13.3\text{ KB}}$ (30-frame window buffer) | **VERIFIED MEASUREMENT** |
| **Theoretical Parameter Count** | $\approx 31.5\text{M}$ parameters | $\approx 50\text{K}$ parameters | **THEORETICAL ESTIMATE** |
| **Theoretical Compute Scale** | $\approx 7.5\text{ GFLOPs}$ ($16\times 112$) | $\approx 2.5\text{ MFLOPs}$ ($30\times 111$) | **THEORETICAL ESTIMATE** |
| **Model Checkpoint on Disk** | **`ABSENT`** (`models/r2plus1d_18.pt` not found) | **`ABSENT`** (`data/temporal_action.tflite` not found) | **VERIFIED REALITY** |
| **Model Inference Latency** | **`BLOCKED`** (No PyTorch runtime / no checkpoint) | **`BLOCKED`** (No TFLite runtime / no checkpoint) | **BLOCKED** |
| **Model Classification Accuracy / F1** | **`BLOCKED`** (No EXP-001 dataset / no checkpoint) | **`BLOCKED`** (No EXP-001 dataset / no checkpoint) | **BLOCKED** |
| **Orientation Robustness Drop ($\Delta F1$)** | **`BLOCKED`** (Requires trained 5-action model) | **`BLOCKED`** (Requires trained 5-action model) | **BLOCKED** |

---

## 3. Candidate B (1D-TCN) Verified Pipeline Benchmarks

Candidate B utilizes a decoupled 2-stage architecture where upstream detectors extract spatial landmarks and object bounding boxes, and the temporal classifier evaluates sliding windows of these geometric coordinates.

### 3.1 Measured Pipeline Stages (Pure NumPy on Windows CPU)
- **Input Vector Breakdown ($D = 111$):**
  - 33 3D Pose Landmarks $\times 3\ (x, y, z) = 99\text{ features}$
  - 4 Primary Object Centroids $\times 3\ (c_x, c_y, \text{conf}) = 12\text{ features}$
- **Spatial Feature Vector Extraction Latency ([`FrameProcessor.extract_keypoint_vector`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/frame_processor.py#L305-L341)):**
  - Mean: $0.043\text{ ms}$
  - Median: $0.042\text{ ms}$
  - 95th Percentile ($p_{95}$): $0.051\text{ ms}$
  - Throughput: $> 23,000\text{ frames/second}$
- **Sliding Window Buffer Accumulation & Stacking ([`ActionClassifier._vector_buffer`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/action_classifier.py#L47)):**
  - Tensor Shape: `(1, 30, 111)` float32
  - Mean Stacking Latency: $0.103\text{ ms}$
  - 95th Percentile ($p_{95}$): $0.124\text{ ms}$
  - Throughput: $> 9,500\text{ windows/second}$
- **Total End-to-End Preparation Time:** $\mathbf{0.146\text{ ms}}$ per evaluation step.
- **Memory Consumption:**
  - Single Frame Vector: $444\text{ bytes}$
  - 30-Frame Deque Buffer: $13,320\text{ bytes}$ ($\approx 13.0\text{ KB}$)
  - Stacked Tensor `(1, 30, 111)`: $13,320\text{ bytes}$

---

## 4. Candidate A (R(2+1)D-18) Verified Pipeline Benchmarks

Candidate A processes raw video voxels directly, requiring video frame decoding, RGB conversion, spatial letterbox/resizing, channel transposition (HWC $\to$ CHW), and temporal stacking into a 5-dimensional tensor.

### 4.1 Measured Pipeline Stages (Pure NumPy on Windows CPU)
- **Standard Clip Tensor Preparation — `(1, 3, 16, 112, 112)`:**
  - Mean Latency: $94.111\text{ ms}$
  - 95th Percentile ($p_{95}$): $106.840\text{ ms}$
  - Throughput: $10.6\text{ clips/second}$
  - Tensor Memory Footprint: $2,408,448\text{ bytes}$ ($\approx 2.35\text{ MB}$)
- **High-Resolution Clip Tensor Preparation — `(1, 3, 32, 224, 224)`:**
  - Mean Latency: $545.909\text{ ms}$
  - 95th Percentile ($p_{95}$): $582.110\text{ ms}$
  - Throughput: $1.8\text{ clips/second}$
  - Tensor Memory Footprint: $19,267,584\text{ bytes}$ ($\approx 18.82\text{ MB}$)

> [!NOTE]
> Video tensor preparation in Candidate A on host CPU is computationally expensive when executed in software (NumPy bilinear interpolation). On target edge deployment, this step would be offloaded to hardware decoders (e.g., V4L2 / FFmpeg HWaccel / OpenCV CUDA/Vulkan).

---

## 5. Explicitly Blocked Metrics Register

The following metrics cannot be measured in the current repository and are recorded as **`BLOCKED`** in the structured JSON report ([`evaluation/results/b5_2_architecture_benchmark_results.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/results/b5_2_architecture_benchmark_results.json)):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              BLOCKED METRICS REGISTER                                  │
├───────────────────────────────────┬────────┬───────────────────────────────────────────┤
│ Metric Name                       │ Status │ Blocker Reason                            │
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ candidate_a_inference_latency     │ BLOCKED│ PyTorch is not installed in the active    │
│                                   │        │ environment and no R(2+1)D checkpoint exists│
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ candidate_b_inference_latency     │ BLOCKED│ tflite_runtime is not installed and       │
│                                   │        │ data/temporal_action.tflite is absent     │
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ candidate_a_accuracy_f1           │ BLOCKED│ No EXP-001 dataset; no R(2+1)D checkpoint │
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ candidate_b_accuracy_f1           │ BLOCKED│ No EXP-001 dataset; no 1D-TCN checkpoint  │
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ orientation_robustness_drop_delta │ BLOCKED│ Requires real per-angle F1 scores across  │
│                                   │        │ all 7 canonical orientation angles        │
└───────────────────────────────────┴────────┴───────────────────────────────────────────┘
```

---

## 6. Controlled Benchmark Invariants & Reproducibility

### 6.1 Reproducibility Command
To reproduce all empirical pipeline measurements and update the structured results file:

```bash
# Default benchmark (50 iterations, seed 42)
python evaluation/architecture_benchmark.py

# Custom iterations and quiet mode
python evaluation/architecture_benchmark.py --num-iterations 100 --seed 42 --quiet
```

### 6.2 Execution Context
- **Random Seed:** `42`
- **Output Artifact:** [`evaluation/results/b5_2_architecture_benchmark_results.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/results/b5_2_architecture_benchmark_results.json)
- **Host Platform:** Windows (Intel/AMD x64), Python 3.14.0, NumPy 2.5.3

---

## 7. Conclusion & Next Sub-Gate

Sub-gate **B5.2** successfully implements and validates the deterministic architecture benchmark harness:
1. Validated the extreme efficiency of the **Candidate B** 111-dimensional spatial vector pipeline ($0.146\text{ ms}$ prep latency, $13\text{ KB}$ buffer memory).
2. Quantified the computational profile of **Candidate A** 5D video tensor preparation ($94.1\text{ ms}$ for $16\times 112\times 112$, $2.35\text{ MB}$ tensor size).
3. Maintained strict epistemic integrity by reporting absent checkpoints and uncollected datasets as **`BLOCKED`** rather than fabricating numbers.

**Handoff to B5.3:** Gate **B5.3 (Benchmark Available Components)** will formalize the component-level benchmark scorecards across spatial perception, interaction logic, and temporal buffers.
