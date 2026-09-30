# Workstream B — Sub-Gate B8.2: Rate-Matching Strategy Benchmark Report

> **Document Status:** IMPLEMENTED & VERIFIED  
> **Gate:** B8.2 — Rate-Matching Strategy Benchmark (Sub-gate of B8: Frame-Rate Strategy)  
> **Timestamp:** 2026-09-26  
> **Source Results:** [`evaluation/results/b8_2_rate_matching_benchmark.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/results/b8_2_rate_matching_benchmark.json)  
> **Benchmark Harness:** [`evaluation/rate_matching_benchmark.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/rate_matching_benchmark.py)  
> **Test Suite:** [`tests/test_b8_2_rate_matching_benchmark.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b8_2_rate_matching_benchmark.py) (16/16 PASS)  

---

## 1. Executive Summary & Epistemic Boundaries

This document provides empirical benchmarking and comparative analysis of four frame-rate consumption strategies operating on the Workstream B non-blocking perception architecture (Gate B7) instrumented with monotonic timing telemetry (Gate B8.1).

> [!IMPORTANT]
> **Synthetic Scheduling Benchmark Disclaimer:**  
> All neural model processing times in this benchmark ($10\text{ ms}, 33\text{ ms}, 66\text{ ms}, 100\text{ ms}, 150\text{ ms}$) are **controlled deterministic synthetic workloads**. They measure scheduling, rate pacing, frame replacement, frame age, and queueing behavior. They do **NOT** represent real neural model weights or physical NPU inference execution.

> [!WARNING]
> **Real-Model Metrics are Strictly BLOCKED:**  
> This benchmark cannot determine:
> 1. Minimum AI FPS required for `EXP-001` procedure recognition.
> 2. Action miss rate / temporal boundary degradation under downsampled frame rates.
> 3. Macro-F1 / precision / recall trade-offs across action classes.
> 4. Physical Hailo-8L / Raspberry Pi 5 NPU hardware inference latency.
>
> Determination of these metrics requires the future `EXP-001` physical dataset and trained checkpoints.

> [!NOTE]
> **Architectural Neutrality:**  
> In accordance with Gate B8.2 governance, this document reports objective trade-off data without selecting a winner. The final rate-control strategy decision is formally **DEFERRED** to Gate B8.3. The production runtime strategy remains `OPPORTUNISTIC_LATEST` (**UNCHANGED**).

---

## 2. Strategy Definitions

Four strategy modes were implemented and benchmarked within [`backend/ai/inference_worker.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_worker.py):

| Strategy Mode | Target AI FPS | Minimum Interval ($\Delta t_{\min}$) | Semantics & Execution Mechanics |
|---|:---:|:---:|---|
| `OPPORTUNISTIC_LATEST` | Unbounded | $0.0\text{ ms}$ | **Current B7 baseline.** Worker pops the newest available frame from `LatestFrameBuffer` as soon as it completes the previous frame. Maximum possible throughput for given compute. |
| `FIXED_10FPS` | $10.0\text{ FPS}$ | $100.0\text{ ms}$ | Target one processing opportunity every $100\text{ ms}$. Worker waits for $\Delta t_{\min} - \Delta t_{\text{elapsed}}$ on OS synchronization before popping the freshest pending frame. |
| `FIXED_15FPS` | $15.0\text{ FPS}$ | $66.67\text{ ms}$ | Target one processing opportunity every $\approx 66.67\text{ ms}$ ($1/15\text{ s}$). Retains single-slot replacement semantics while waiting. |
| `TIME_DECIMATED` | Configurable | Configurable (e.g. $100.0\text{ ms}$) | Monotonic time-based rate decimation using `time.monotonic()`. Does not use frame counters or modulo arithmetic. Configured via `min_ai_interval_sec`. |

### Architectural Guarantees Across All Strategies:
1. **Single-Slot Replacement:** `LatestFrameBuffer` strictly contains $\le 1$ frame at all times.
2. **Zero Queue Accumulation:** Stale frames are never queued; intermediate arrivals atomically replace unconsumed frames.
3. **Non-Blocking Producers:** Video ingestion in `getCameraFrame()` or producer threads completes in $<0.05\text{ ms}$ ($O(1)$).
4. **OS Synchronization:** Pacing uses `threading.Event.wait(timeout=...)` on monotonic time differences, consuming 0% CPU while idle and waking instantaneously on worker pause/shutdown.

---

## 3. Benchmark Methodology & Parameter Matrix

The benchmark systematically evaluated the full Cartesian product across 3 independent experimental axes:

$$\text{Producer Rates (3)} \times \text{Strategies (4)} \times \text{Workload Latencies (5)} = 60\text{ Configurations}$$

```
                                ┌── 10 FPS (Low Ingest)
         ┌── Producer Rates ────┼── 15 FPS (Target AI Rate)
         │                      └── 30 FPS (Full Camera Preview)
         │
         │                      ┌── OPPORTUNISTIC_LATEST (B7 Baseline)
         ├── Strategies ────────┼── FIXED_10FPS (100 ms Interval)
Matrix ──┤                      ├── FIXED_15FPS (66.67 ms Interval)
         │                      └── TIME_DECIMATED (Configurable Interval)
         │
         │                      ┌── 10 ms (Light Feature Extraction / NPU Offload)
         │                      ├── 33 ms (Fast 30 FPS Detector)
         └── Workload Durations ┼── 66 ms (Target 15 FPS Budget)
                                ├── 100 ms (10 FPS Budget / CPU Fallback)
                                └── 150 ms (Heavy Multimodal Pipeline)
```

- **Run Duration:** Deterministic $0.25\text{ s}$ per configuration run ($23.71\text{ s}$ total suite).
- **Frame Ingestion:** Paced submission using monotonic timestamps attached to metadata (`capture_monotonic`).
- **Telemetry Collection:** Extracted via `InferenceWorker.get_telemetry()`.
- **Reproducibility Seed:** `BENCHMARK_SEED = 42`.

---

## 4. Measured Results (60-Configuration Matrix)

### Table 1: Standard 30 FPS Camera Ingestion (Key Comparison Set)

Below is the empirical behavior of all 4 strategies under standard 30 FPS camera capture across synthetic workloads:

| Strategy | Workload | Target AI FPS | Effective AI FPS | Submitted | Processed | Replaced | Drop Rate | Mean Frame Age | Worker Util. | Scheduling Overhead |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `OPPORTUNISTIC` | $10\text{ ms}$ | Unbounded | $28.1\text{ FPS}$ | 8 | 8 | 0 | $0.0\%$ | $0.1\text{ ms}$ | $28.1\%$ | $0.1\text{ ms}$ |
| `FIXED_10FPS` | $10\text{ ms}$ | $10.0\text{ FPS}$ | $11.4\text{ FPS}$ | 8 | 4 | 4 | $50.0\%$ | $32.1\text{ ms}$ | $11.4\%$ | $32.1\text{ ms}$ |
| `FIXED_15FPS` | $10\text{ ms}$ | $15.0\text{ FPS}$ | $15.4\text{ FPS}$ | 8 | 5 | 3 | $37.5\%$ | $25.5\text{ ms}$ | $15.4\%$ | $25.5\text{ ms}$ |
| `TIME_DECIMATED` | $10\text{ ms}$ | $10.0\text{ FPS}$ | $11.4\text{ FPS}$ | 8 | 4 | 4 | $50.0\%$ | $27.3\text{ ms}$ | $11.4\%$ | $27.3\text{ ms}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `OPPORTUNISTIC` | $33\text{ ms}$ | Unbounded | $28.2\text{ FPS}$ | 8 | 8 | 0 | $0.0\%$ | $3.7\text{ ms}$ | $96.1\%$ | $3.7\text{ ms}$ |
| `FIXED_10FPS` | $33\text{ ms}$ | $10.0\text{ FPS}$ | $10.4\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $32.2\text{ ms}$ | $35.1\%$ | $32.2\text{ ms}$ |
| `FIXED_15FPS` | $33\text{ ms}$ | $15.0\text{ FPS}$ | $14.3\text{ FPS}$ | 8 | 4 | 3 | $37.5\%$ | $22.3\text{ ms}$ | $48.2\%$ | $22.3\text{ ms}$ |
| `TIME_DECIMATED` | $33\text{ ms}$ | $10.0\text{ FPS}$ | $10.4\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $30.2\text{ ms}$ | $35.1\%$ | $30.2\text{ ms}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `OPPORTUNISTIC` | $66\text{ ms}$ | Unbounded | $14.7\text{ FPS}$ | 8 | 5 | 3 | $37.5\%$ | $18.5\text{ ms}$ | $98.0\%$ | $18.5\text{ ms}$ |
| `FIXED_10FPS` | $66\text{ ms}$ | $10.0\text{ FPS}$ | $9.2\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $31.3\text{ ms}$ | $61.3\%$ | $31.3\text{ ms}$ |
| `FIXED_15FPS` | $66\text{ ms}$ | $15.0\text{ FPS}$ | $12.5\text{ FPS}$ | 8 | 4 | 3 | $37.5\%$ | $14.4\text{ ms}$ | $83.4\%$ | $14.4\text{ ms}$ |
| `TIME_DECIMATED` | $66\text{ ms}$ | $10.0\text{ FPS}$ | $9.2\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $31.5\text{ ms}$ | $61.3\%$ | $31.5\text{ ms}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `OPPORTUNISTIC` | $100\text{ ms}$ | Unbounded | $7.8\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $0.3\text{ ms}$ | $78.4\%$ | $0.3\text{ ms}$ |
| `FIXED_10FPS` | $100\text{ ms}$ | $10.0\text{ FPS}$ | $8.3\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $17.2\text{ ms}$ | $84.0\%$ | $17.2\text{ ms}$ |
| `FIXED_15FPS` | $100\text{ ms}$ | $15.0\text{ FPS}$ | $8.9\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $18.3\text{ ms}$ | $89.5\%$ | $18.3\text{ ms}$ |
| `TIME_DECIMATED` | $100\text{ ms}$ | $10.0\text{ FPS}$ | $8.2\text{ FPS}$ | 8 | 3 | 4 | $50.0\%$ | $19.6\text{ ms}$ | $82.7\%$ | $19.6\text{ ms}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `OPPORTUNISTIC` | $150\text{ ms}$ | Unbounded | $6.5\text{ FPS}$ | 8 | 2 | 5 | $62.5\%$ | $27.9\text{ ms}$ | $97.7\%$ | $27.9\text{ ms}$ |
| `FIXED_10FPS` | $150\text{ ms}$ | $10.0\text{ FPS}$ | $5.3\text{ FPS}$ | 8 | 2 | 5 | $62.5\%$ | $28.5\text{ ms}$ | $80.2\%$ | $28.5\text{ ms}$ |
| `FIXED_15FPS` | $150\text{ ms}$ | $15.0\text{ FPS}$ | $5.7\text{ FPS}$ | 8 | 2 | 5 | $62.5\%$ | $27.9\text{ ms}$ | $86.5\%$ | $27.9\text{ ms}$ |
| `TIME_DECIMATED` | $150\text{ ms}$ | $10.0\text{ FPS}$ | $5.4\text{ FPS}$ | 8 | 2 | 5 | $62.5\%$ | $31.6\text{ ms}$ | $82.3\%$ | $31.6\text{ ms}$ |

---

## 5. Objective Empirical Observations

### 5.1 Effective AI FPS & Rate Adherence
1. **Opportunistic Strategy:**
   - Under fast workloads ($10\text{ ms}, 33\text{ ms}$), opportunistic processing operates at the producer rate ($28.1\text{--}28.2\text{ FPS}$ at $30\text{ FPS}$ ingest), consuming all available frames with $0\%$ drops.
   - When workload latency increases beyond the frame ingestion interval ($66\text{ ms}, 100\text{ ms}, 150\text{ ms}$), opportunistic processing naturally throttles to $1/\text{latency}$ ($14.7\text{ FPS}, 7.8\text{ FPS}, 6.5\text{ FPS}$), dropping intermediate frames in the single-slot buffer.
2. **Fixed 10 FPS & Time-Decimated:**
   - Strictly throttles processing to approximately $10\text{ FPS}$ regardless of how fast inference executes ($10.4\text{--}11.4\text{ FPS}$ at $10\text{ ms}$ and $33\text{ ms}$ workloads).
   - When workload duration exceeds $100\text{ ms}$ (e.g. $150\text{ ms}$), effective FPS drops below target to $5.3\text{--}5.4\text{ FPS}$ without queueing lag.
3. **Fixed 15 FPS:**
   - Maintains $\approx 14.3\text{--}15.4\text{ FPS}$ under fast workloads ($10\text{ ms}, 33\text{ ms}$), matching the system's target 15 FPS perception goal.

### 5.2 Frame Replacement & Drop Behavior
- Under 30 FPS camera ingestion, `FIXED_10FPS` drops exactly $\approx 50\text{--}62.5\%$ of frames (dropping 2 out of every 3 frames to maintain 10 FPS).
- `FIXED_15FPS` drops exactly $\approx 37.5\%$ of frames (dropping 1 out of every 2 frames to maintain 15 FPS).
- `OPPORTUNISTIC` drops $0\%$ of frames when workload $\le 33\text{ ms}$, and drops $37.5\%\text{--}62.5\%$ only when compute-bound.

### 5.3 Frame Freshness & Frame Age at Inference
- Because `LatestFrameBuffer` replaces unconsumed frames atomically, frame age at pickup remains low across all strategies ($0.1\text{--}32.2\text{ ms}$).
- Under paced strategies (`FIXED_10FPS`, `FIXED_15FPS`), mean frame age is bounded by the camera frame arrival period ($\le 33.3\text{ ms}$ at 30 FPS ingest) because the worker picks up whichever frame arrived closest to the timer expiration.

### 5.4 Worker Utilization & CPU Conservation
- Under `OPPORTUNISTIC` with fast inference ($33\text{ ms}$), worker utilization reaches $96.1\%$ because the worker processes every frame continuously.
- Under `FIXED_10FPS` with fast inference ($33\text{ ms}$), worker utilization drops to $35.1\%$ ($64.9\%$ idle sleep time), conserving host CPU and thermal headroom for Workstream A video encoding, streaming, and UI rendering.

---

## 6. Trade-off Matrix (Neutral Comparison)

| Dimension | `OPPORTUNISTIC_LATEST` | `FIXED_10FPS` | `FIXED_15FPS` | `TIME_DECIMATED` |
|---|---|---|---|---|
| **Rate Predictability** | Variable (tied to pipeline speed) | High ($\approx 10\text{ FPS}$) | High ($\approx 15\text{ FPS}$) | High (configurable) |
| **CPU / Thermal Overhead** | High under fast workloads | Minimal (65% idle at 33ms) | Moderate (52% idle at 33ms) | Configurable |
| **Max Frame Sampling** | Highest (processes every frame if fast) | Fixed downsample | Fixed downsample | Fixed downsample |
| **Edge Hardware Suitability** | Squeezes maximum work from NPU | Ideal for constrained CPU | Ideal balance for 15 FPS budget | Adaptable |
| **Temporal Stability for FSM** | Fluctuating sample intervals | Constant 100 ms step | Constant 66.7 ms step | Constant $\Delta t$ step |

---

## 7. Limitations & Real-Model Separation

```
Verified Benchmark Reality (B8.2)           Blocked Future Reality (Part 1 & B17)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│  Synthetic Workloads (10-150 ms)     │     │  Trained Neural Checkpoints (.pt)    │
│  Monotonic OS Synchronization        │ ──► │  EXP-001 Labeled Video Dataset       │
│  Rate Pacing & Replacement Counters  │     │  Macro-F1 & Procedure Recognition    │
│  Frame Age & Overhead Profile        │     │  Hailo-8L NPU Physical Latency       │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

1. **Synthetic Workloads:** Simulated latencies do not reflect dynamic jitter from video decoding, memory bus contention, or GPU/NPU kernel dispatch.
2. **Recognition Quality:** Whether 10 FPS or 15 FPS is sufficient to capture astronaut hand movements during `PICK_RED` or `CLOSE_LID` cannot be known until real data is evaluated in Part 1 / Gate B17.
3. **No Winner Declared:** Selection between opportunistic execution and fixed-rate throttling depends on system-wide thermal budgets and recognition accuracy, which will be formally evaluated in Gate B8.3.

---

## 8. Test Verification

The benchmark harness and strategy implementations were validated by [`tests/test_b8_2_rate_matching_benchmark.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b8_2_rate_matching_benchmark.py) across 16 unit and integration test criteria:

- [x] `test_strategy_configuration_validation`: Validates enum and string configuration.
- [x] `test_invalid_strategy_handling`: Verifies rejection of unknown strategies.
- [x] `test_opportunistic_strategy_behavior`: Verifies unthrottled execution.
- [x] `test_fixed_10fps_configuration_and_pacing`: Verifies 100 ms pacing.
- [x] `test_fixed_15fps_configuration`: Verifies 66.67 ms pacing.
- [x] `test_time_decimation_configuration`: Verifies custom interval decimation.
- [x] `test_latest_frame_semantics_preserved_under_pacing`: Proves latest frame replacement during wait.
- [x] `test_no_unbounded_queue_buffer_remains_single_slot`: Proves single-slot bound.
- [x] `test_producer_remains_non_blocking_under_all_strategies`: Proves producer non-blocking submission ($<1\text{ ms}$).
- [x] `test_deterministic_synthetic_workload_simulation`: Verifies synthetic workload precision.
- [x] `test_drop_and_replacement_calculation`: Verifies drop rate formula.
- [x] `test_frame_age_calculation`: Verifies frame age calculation at pickup.
- [x] `test_mean_and_p95_percentiles_calculation`: Verifies NumPy percentiles.
- [x] `test_single_benchmark_run_execution`: Verifies single benchmark result dataclass.
- [x] `test_benchmark_result_schema_and_epistemic_notice`: Verifies output JSON schema and blocked notices.
- [x] `test_synthetic_vs_real_labeling_invariance`: Verifies synthetic workload tagging.
