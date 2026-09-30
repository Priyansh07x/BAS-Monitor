# Workstream B — Sub-Gate B8.3: Rate-Matching Strategy Decision Record

> **Document Status:** IMPLEMENTED & VERIFIED  
> **Gate:** B8.3 — Rate-Matching Strategy Decision (Sub-gate of B8: Frame-Rate Strategy)  
> **Timestamp:** 2026-09-26  
> **Selected Runtime Strategy:** `RateStrategy.OPPORTUNISTIC_LATEST` (with Single-Slot Latest-Frame Buffer)  
> **Recognition-Quality Minimum FPS:** `DEFERRED` (Pending EXP-001 dataset & trained model)  
> **Real-Model Metrics:** `BLOCKED`  
> **Decision Reference:** [`docs/decision.md#decision-041`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/decision.md)  
> **Test Suite:** [`tests/test_b8_3_rate_matching_decision.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b8_3_rate_matching_decision.py)  

---

## 1. Objective & Scope

Gate B8.3 establishes the formal engineering decision for the runtime frame-rate consumption strategy in Workstream B. This decision governs how [`backend/ai/inference_worker.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_worker.py) consumes frames from the single-slot [`LatestFrameBuffer`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_worker.py#L45) supplied by the Workstream A camera acquisition pipeline.

### Fundamental Separation of Concerns:
1. **Runtime Scheduling Decision (RESOLVED in B8.3):** How the inference thread dispatches frame execution, handles worker idle states, and interfaces with the single-slot buffer.
2. **Recognition-Quality Minimum AI FPS (DEFERRED):** The minimum perception frame rate necessary to reliably recognize `EXP-001` procedural micro-actions (e.g., `PICK_RED`, `PLACE_RED`, `CLOSE_LID`) without temporal aliasing or missed state transitions. This remains strictly **DEFERRED** until the physical `EXP-001` dataset and trained neural model checkpoints exist.

---

## 2. Candidate Strategies Evaluated

The four candidate strategies benchmarked in Gate B8.2 were:

1. **`OPPORTUNISTIC_LATEST`:**  
   Worker unconstrained mode. Whenever the worker completes an inference cycle and is idle, it immediately pops and processes the newest available frame from `LatestFrameBuffer`.
2. **`FIXED_10FPS`:**  
   Worker is paced using monotonic OS timers (`threading.Event.wait`) to initiate inference no more frequently than once every $100.0\text{ ms}$.
3. **`FIXED_15FPS`:**  
   Worker is paced using monotonic OS timers to initiate inference no more frequently than once every $\approx 66.67\text{ ms}$.
4. **`TIME_DECIMATED`:**  
   Generalized monotonic time-based interval throttling ($\Delta t_{\min}$) without frame counting or modulo arithmetic.

---

## 3. Summary of B8.2 Empirical Evidence

From the 60-configuration deterministic benchmark suite recorded in [`evaluation/results/b8_2_rate_matching_benchmark.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/evaluation/results/b8_2_rate_matching_benchmark.json) under 30 FPS camera ingestion:

| Dimension | `OPPORTUNISTIC_LATEST` | `FIXED_10FPS` | `FIXED_15FPS` |
|---|:---:|:---:|:---:|
| **Throughput under Fast Compute (10 ms)** | $28.1\text{ FPS}$ ($0.0\%$ drops) | $11.4\text{ FPS}$ ($50.0\%$ drops) | $15.4\text{ FPS}$ ($37.5\%$ drops) |
| **Throughput under 33 ms Workload** | $28.2\text{ FPS}$ ($0.0\%$ drops) | $10.4\text{ FPS}$ ($50.0\%$ drops) | $14.3\text{ FPS}$ ($37.5\%$ drops) |
| **Throughput under Compute-Bound (66 ms)** | $14.7\text{ FPS}$ ($37.5\%$ drops) | $9.2\text{ FPS}$ ($50.0\%$ drops) | $12.5\text{ FPS}$ ($37.5\%$ drops) |
| **Throughput under Heavy Workload (150 ms)** | $6.5\text{ FPS}$ ($62.5\%$ drops) | $5.3\text{ FPS}$ ($62.5\%$ drops) | $5.7\text{ FPS}$ ($62.5\%$ drops) |
| **Mean Frame Age at Pickup (All Workloads)** | $0.1\text{--}27.9\text{ ms}$ | $2.6\text{--}32.2\text{ ms}$ | $1.8\text{--}25.5\text{ ms}$ |
| **Worker Utilization (33 ms Workload)** | $96.1\%$ | $35.1\%$ ($64.9\%$ idle) | $48.2\%$ ($51.8\%$ idle) |
| **Scheduler Synchronization Overhead** | $< 0.05\text{ ms}$ | $0.05\text{--}0.15\text{ ms}$ | $0.05\text{--}0.15\text{ ms}$ |
| **Queue Depth Guarantee** | $\le 1$ frame | $\le 1$ frame | $\le 1$ frame |
| **Producer Non-Blocking Duration** | $< 0.05\text{ ms}$ ($O(1)$) | $< 0.05\text{ ms}$ ($O(1)$) | $< 0.05\text{ ms}$ ($O(1)$) |

---

## 4. Formal Decision: `OPPORTUNISTIC_LATEST`

### Selected Production Runtime Strategy:
**`RateStrategy.OPPORTUNISTIC_LATEST` with Single-Slot `LatestFrameBuffer`**

### Evidence-Based Rationale:
1. **Dynamic Self-Scaling Across Heterogeneous Compute:**  
   In microgravity edge deployments (ranging from Raspberry Pi 5 CPU fallback to Hailo-8L NPU acceleration), neural model execution latency will vary widely depending on scene complexity and hardware availability. `OPPORTUNISTIC_LATEST` automatically processes frames at the natural maximum rate supported by the hardware ($FPS_{\text{effective}} = \min(FPS_{\text{camera}}, 1/\text{latency})$) without requiring manual configuration or artificial throttling.
2. **Preservation of Frame Freshness Under All Workloads:**  
   Because `LatestFrameBuffer` enforces atomic single-slot replacement, `OPPORTUNISTIC_LATEST` maintains low frame pickup age ($0.1\text{--}27.9\text{ ms}$ at 30 FPS ingest) even when compute-bound. Unprocessed intermediate frames are dropped at the buffer level with zero queue buildup.
3. **Prevention of Unnecessary Temporal Starvation:**  
   When hardware compute is plentiful (e.g. $10\text{--}33\text{ ms}$ inference on NPU), `FIXED_10FPS` and `FIXED_15FPS` artificially discard $37.5\%\text{--}50.0\%$ of camera frames. In rapid astronaut manual interactions (such as container lid closing or sample vial insertion), discarding high-frequency visual evidence before model quality requirements are known introduces unnecessary temporal aliasing risk.
4. **Minimal Scheduling Complexity and Zero Synchronization Drift:**  
   `OPPORTUNISTIC_LATEST` relies entirely on condition-variable wakeups without timer event scheduling, eliminating timer jitter, clock-drift adjustments, and scheduling overhead.
5. **Configurability Retained for Edge Thermal Throttling:**  
   While `OPPORTUNISTIC_LATEST` is the production default, the pacing infrastructure implemented in Gate B8.2 (`set_rate_strategy()`, `FIXED_10FPS`, `FIXED_15FPS`, `TIME_DECIMATED`) is preserved in `InferenceWorker`. If future flight qualifications impose strict thermal dissipation limits on the edge CPU/NPU, fixed-rate throttling can be activated via configuration without code changes.

---

## 5. Runtime Architecture & Semantics

```
Workstream A Camera Ingest (30 FPS)
               │
               ▼ (non-blocking put, <0.05ms)
┌──────────────────────────────────────────────┐
│  LatestFrameBuffer (Single-Slot Replacement) │
│  - Max queue depth: exactly 1 frame          │
│  - Stale frames replaced atomically          │
└──────────────────────────────────────────────┘
               │
               ▼ (immediate pop on worker idle)
┌──────────────────────────────────────────────┐
│  InferenceWorker Thread                      │
│  - Strategy: OPPORTUNISTIC_LATEST (Default)   │
│  - Continuous execution on dedicated thread  │
│  - Yields to condition variable when empty   │
└──────────────────────────────────────────────┘
               │
               ▼ (callback / Qt Queued Signal)
Workstream A Bridge (`aiResultReady` / `aiResultJsonReady`)
```

- **Buffer Bound:** Single-slot ($\le 1$ pending item).
- **Producer Impact:** Zero blocking ($O(1)$ lock release).
- **Public Interface:** Frozen 8-field public AI contract (`timestamp`, `action`, `object`, `confidence`, `expected_step`, `detected_step`, `status`, `next_step`).

---

## 6. Limitations & Deferred Validation

> [!WARNING]
> **Recognition-Quality Minimum FPS Remains DEFERRED:**  
> This decision selects the **runtime thread scheduling mode**. It does **NOT** certify that any specific FPS number is sufficient for action recognition.
> 
> The following parameters cannot be determined until physical dataset collection and model training:
> 1. Minimum temporal sampling rate required to detect sub-second micro-actions.
> 2. Temporal feature buffer length ($N=30$ frames at 10 FPS vs 15 FPS vs 30 FPS).
> 3. False-positive and false-negative trade-offs under frame decimation.
> 
> These questions will be resolved empirically in Part 1 model evaluation and Phase B17 validation.

---

## 7. Verification Invariants

The decision is verified by [`tests/test_b8_3_rate_matching_decision.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b8_3_rate_matching_decision.py):

1. `test_default_strategy_is_opportunistic_latest`: `InferenceWorker` defaults to `RateStrategy.OPPORTUNISTIC_LATEST`.
2. `test_runtime_configuration_and_switching`: Validates dynamic switching between `OPPORTUNISTIC_LATEST`, `FIXED_10FPS`, `FIXED_15FPS`, and `TIME_DECIMATED`.
3. `test_buffer_strict_single_slot_replacement`: Proves single-slot capacity bound ($\le 1$ frame).
4. `test_producer_non_blocking_invariance`: Proves producer frame submission is non-blocking.
5. `test_frozen_public_contract_invariance`: Proves public AI contract remains 8 frozen fields.
6. `test_worker_lifecycle_invariance`: Proves start, stop, pause, resume, reset, and shutdown operate cleanly.
7. `test_epistemic_recognition_fps_deferred_status`: Proves recognition-quality minimum FPS is marked as deferred and not fabricated.
