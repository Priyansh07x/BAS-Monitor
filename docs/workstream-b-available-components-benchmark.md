# Workstream B B5.3 — Benchmark Available Components Report

**Date:** 2026-09-24  
**Gate:** B5.3 — Benchmark Available Components (Sub-gate of B5: Model Architecture Evaluation)  
**Author:** Workstream B (AI / Procedure Intelligence)  
**Status:** PARTIALLY PASSED — PIPELINE COMPONENTS EMPIRICALLY BENCHMARKED / MODEL INFERENCE & ACCURACY BLOCKED

---

## 1. Purpose & Scope

Gate B5.3 executes multi-iteration, deterministic computational benchmarks across all pipeline components currently available in the repository. 

### Operational Scope:
- **Measured Available Components:** Spatial vector extraction, temporal sliding window accumulation, frame letterboxing/normalization, camera mounting rectification, 7-angle orientation preprocessing, and Candidate A vs Candidate B tensor construction.
- **Unmeasured Model-Level Components:** Neural network inference latency, parameter counts on disk, and procedure action classification accuracy (Precision, Recall, F1) remain explicitly **`BLOCKED`** due to the absence of physical model checkpoints and the uncollected EXP-001 dataset.
- **Architectural Neutrality:** This gate evaluates data and tensor pipelines; it does **not** declare an architectural winner.

---

## 2. Dynamic Runtime Environment

The benchmark harness dynamically detected the following local execution context:

| System Attribute | Detected Environment Value |
|---|---|
| **Operating System** | Windows 10 (Build 10.0.19045) |
| **Python Version** | Python 3.14.0 |
| **CPU Architecture** | Intel/AMD x64 (8 logical cores) |
| **Total System RAM** | 17.95 GB |
| **NumPy Version** | 2.5.3 (Pure NumPy execution path active) |
| **OpenCV (`cv2`)** | `Not Installed` (Pure NumPy fallback active) |
| **PyTorch (`torch`)** | `Not Installed` (Candidate A model inference blocked) |
| **TFLite (`tflite_runtime`)** | `Not Installed` (Candidate B model inference blocked) |

---

## 3. Verified Empirical Component Measurements

All benchmarks below were executed with fixed seed $42$, $5$ warm-up iterations, and $30\text{--}50$ measurement iterations on $640\times 480$ input frames.

### 3.1 Summary Scorecard

| Component Pipeline | Mean Latency | 95th Percentile ($p_{95}$) | Throughput | Memory Footprint |
|---|---:|---:|---:|---:|
| **Spatial Vector Extraction ($D=111$)** | $\mathbf{0.043\text{ ms}}$ | $0.048\text{ ms}$ | $>23,000\text{ FPS}$ | $444\text{ B}$ / frame |
| **Temporal Buffering ($T=30$)** | $\mathbf{0.069\text{ ms}}$ | $0.096\text{ ms}$ | $>14,500\text{ WPS}$ | $13.0\text{ KB}$ / buffer |
| **Combined Feature + Buffer Prep** | $\mathbf{0.112\text{ ms}}$ | $0.144\text{ ms}$ | $>8,900\text{ FPS}$ | $13.0\text{ KB}$ |
| **Detector Letterbox ($640\times 640$)** | $\mathbf{4.672\text{ ms}}$ | $5.210\text{ ms}$ | $214.0\text{ FPS}$ | $1.23\text{ MB}$ |
| **Pose Normalization ($256\times 256$)** | $\mathbf{29.932\text{ ms}}$ | $33.410\text{ ms}$ | $33.4\text{ FPS}$ | $0.20\text{ MB}$ |
| **Rectification Overhead ($90^\circ$)** | $\mathbf{136.493\text{ ms}}$ | $152.660\text{ ms}$ | $7.3\text{ FPS}$ | Pure NumPy CPU |
| **Candidate A Tensor ($16\times 112$)** | $\mathbf{98.281\text{ ms}}$ | $108.400\text{ ms}$ | $10.2\text{ clips/s}$ | $2.35\text{ MB}$ |
| **Candidate A Tensor ($32\times 224$)** | $\mathbf{540.429\text{ ms}}$ | $578.100\text{ ms}$ | $1.8\text{ clips/s}$ | $18.82\text{ MB}$ |
| **Candidate B Tensor ($30\times 111$)** | $\mathbf{0.069\text{ ms}}$ | $0.096\text{ ms}$ | $>14,500\text{ WPS}$ | $13.0\text{ KB}$ |

---

## 4. Multi-Angle Orientation Preprocessing Latency

Preprocessing rotation latency was measured across all 7 canonical microgravity orientation angles on $640\times 480$ frames:

> [!IMPORTANT]
> **Scope Clarification:** This benchmark measures the computational latency of rotating image frames in the preprocessing layer. It does **not** measure or imply model action classification robustness.

| Angle | Category | Mean Latency | 95th Percentile ($p_{95}$) | Throughput |
|:---:|:---:|---:|---:|---:|
| **0°** | `NORMAL` | $0.090\text{ ms}$ | $0.110\text{ ms}$ | $11,066.4\text{ FPS}$ (Identity Bypass) |
| **45°** | `MODERATE_ROTATION` | $43.313\text{ ms}$ | $48.200\text{ ms}$ | $23.1\text{ FPS}$ |
| **90°** | `EXTREME_ROTATION` | $44.714\text{ ms}$ | $49.500\text{ ms}$ | $22.4\text{ FPS}$ |
| **135°** | `MODERATE_ROTATION` | $41.001\text{ ms}$ | $45.800\text{ ms}$ | $24.4\text{ FPS}$ |
| **180°** | `EXTREME_ROTATION` | $38.464\text{ ms}$ | $42.600\text{ ms}$ | $26.0\text{ FPS}$ |
| **225°** | `MODERATE_ROTATION` | $50.864\text{ ms}$ | $56.200\text{ ms}$ | $19.7\text{ FPS}$ |
| **270°** | `EXTREME_ROTATION` | $40.545\text{ ms}$ | $44.900\text{ ms}$ | $24.7\text{ FPS}$ |

*Mean non-zero rotation latency:* $\approx 43.1\text{ ms}$ on Windows CPU using pure NumPy inverse-mapping interpolation.

---

## 5. Candidate Architecture Input Preparation Comparison

### 5.1 Candidate A: Spatio-Temporal Video Tensor Pipeline
- **Process:** Video frame decoding $\to$ RGB conversion $\to$ Spatial resize/pad $\to$ Channel transpose (HWC $\to$ CHW) $\to$ Temporal concatenation $\to$ 5D Tensor `(1, 3, T, H, W)`.
- **Standard Tensor `(1, 3, 16, 112, 112)`:**
  - Latency: $98.281\text{ ms}$ per clip
  - Memory: $2,408,448\text{ bytes}$ ($\approx 2.35\text{ MB}$)
  - Throughput: $10.2\text{ clips/second}$
- **High-Resolution Tensor `(1, 3, 32, 224, 224)`:**
  - Latency: $540.429\text{ ms}$ per clip
  - Memory: $19,267,584\text{ bytes}$ ($\approx 18.82\text{ MB}$)
  - Throughput: $1.8\text{ clips/second}$

### 5.2 Candidate B: Compact Spatial Feature Tensor Pipeline
- **Process:** Upstream Pose & Object centroid extraction $\to$ Flattening to $111$-dim vector $\to$ Sliding window queue $\to$ 3D Tensor `(1, 30, 111)`.
- **Standard Tensor `(1, 30, 111)`:**
  - Feature Extraction Latency: $0.0431\text{ ms}$
  - Buffer Stacking Latency: $0.0688\text{ ms}$
  - Total Preparation Latency: $\mathbf{0.1119\text{ ms}}$ per window
  - Memory: $13,320\text{ bytes}$ ($\approx 13.01\text{ KB}$)
  - Throughput: $> 8,900\text{ windows/second}$

---

## 6. Explicitly Blocked Metrics Register

The following metrics are blocked and recorded as `null` in [`evaluation/results/b5_3_available_components.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/results/b5_3_available_components.json):

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
│ model_classification_accuracy     │ BLOCKED│ EXP-001 procedure dataset is NOT COLLECTED│
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ model_precision_recall_f1         │ BLOCKED│ Requires real predictions on held-out data│
├───────────────────────────────────┼────────┼───────────────────────────────────────────┤
│ orientation_f1_degradation        │ BLOCKED│ Requires real per-angle F1 scores across  │
│                                   │        │ all 7 canonical orientation angles        │
└───────────────────────────────────┴────────┴───────────────────────────────────────────┘
```

---

## 7. Theoretical Reference Estimates

| Architecture Candidate | Model Parameter Scale | Compute Complexity | Reference Citation |
|---|---|---|---|
| **Candidate A: R(2+1)D-18** | $\approx 31.5\text{M}$ parameters | $\approx 7.5\text{ GFLOPs}$ ($16\times 112$) | Tran et al., CVPR 2018 |
| **Candidate B: 1D-TCN** | $\approx 50\text{K}$ parameters | $\approx 2.5\text{ MFLOPs}$ ($30\times 111$) | Lea et al., CVPR 2017 |

---

## 8. Reproducibility Guide

To reproduce these empirical measurements and generate the structured JSON artifact:

```bash
# Run full component benchmark (30 iterations, seed 42)
python evaluation/available_components_benchmark.py --iterations 30

# Run with custom iterations and quiet console output
python evaluation/available_components_benchmark.py --iterations 50 --seed 42 --quiet
```

**Output Artifact:** [`evaluation/results/b5_3_available_components.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/results/b5_3_available_components.json)

---

## 9. Conclusion & Handoff to Gate B5.4

Gate B5.3 has comprehensively profiled all operational components in the repository:
1. Empirically demonstrated that Candidate B's spatial vector extraction and buffer accumulation pipeline introduces negligible overhead ($0.11\text{ ms}$, $13\text{ KB}$ RAM), leaving $>98\%$ of the per-frame compute budget for upstream spatial detectors.
2. Characterized the raw CPU cost of 5D video tensor preparation in Candidate A ($98.3\text{ ms}$ for $16\times 112$, $2.35\text{ MB}$ memory), confirming the necessity of hardware-accelerated video decoding if Candidate A is deployed on edge hardware.
3. Formally preserved epistemic separation by recording unexecutable model inference and classification metrics as **`BLOCKED`**.

**Handoff to B5.4:** Gate **B5.4 (Architecture Decision)** will synthesize the trade-off specification (B5.1), benchmark framework (B5.2), and component profiles (B5.3) into the formal architectural decision record.
