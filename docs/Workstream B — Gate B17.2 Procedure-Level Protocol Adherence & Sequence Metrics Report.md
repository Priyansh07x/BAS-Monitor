# Workstream B — Gate B17.2: Procedure-Level Protocol Adherence & Sequence Metric Suite Report
**ISRO SIH26174 BAS Experiment Monitor**  
**Date:** 2026-09-30  
**Gate Status:** PASSED (CLOSED)  
**Author:** Antigravity (AI Assistant)

---

## 1. Executive Summary

Sub-Gate **B17.2** implements and verifies the **Procedure-Level Protocol Adherence & Sequence Metric Suite** for Workstream B. This suite deterministically evaluates:
1. **Complete-Sequence Success Rate (CSSR)** across nominal EXP-001 protocol execution cycles ($S1 \to S2 \to S3 \to S4 \to S5$).
2. **Skipped-Step Detection Rate (SSDR / Recall on Forward Skips)** across single-step and multi-step forward skip injections.
3. **Out-of-Order Detection Rate (OODR / Recall on Backward / Repeated Past Steps)** across out-of-order execution scenarios.
4. **False Alarm Rate (FAR / Procedural False Positives)** across nominal execution sequences and perception noise streams.
5. **Missed-Violation Rate (MVR / False Negative Rate)** across all injected procedural anomaly classes.
6. **Multi-Cycle Scenario Breakdown & Ground-Truth Event Mapping** capturing detailed performance across nominal, anomaly injection, and noise rejection categories.

All evaluations execute headlessly on CPU with zero physical camera or GPU dependencies, preserve the authoritative procedural sovereignty of `SequenceValidatorFSM`, maintain `RecoveryManager` isolation, and uphold the frozen 8-field public AI contract (`docs/architecture.md` §2).

---

## 2. Formal Metric Definitions & Mathematical Formulations

$$\text{Complete-Sequence Success Rate (CSSR)} = \frac{N_{\text{nominal\_sequences\_completed}}}{N_{\text{nominal\_sequences\_total}}}$$

$$\text{Skipped-Step Detection Rate (SSDR)} = \frac{\text{True Detected Skips}}{\text{Total Injected Skips}}$$

$$\text{Out-of-Order Detection Rate (OODR)} = \frac{\text{True Detected Out-of-Order}}{\text{Total Injected Out-of-Order}}$$

$$\text{False Alarm Rate (FAR)} = \frac{\text{False Positive Anomalies}}{\text{Total Nominal / Noise Steps Evaluated}}$$

$$\text{Missed-Violation Rate (MVR)} = \frac{\text{Total Injected Violations} - \text{Total Detected Violations}}{\text{Total Injected Violations}} = 1 - \text{Recall}$$

$$\text{Procedural Accuracy Ratio} = \frac{\text{Nominal Steps Passed} + \text{Total Detected Violations} + (\text{Noise Steps} - \text{Noise False Alarms})}{\text{Total Steps Evaluated}}$$

*Zero-Division Protection:* All denominator terms are bounded with $\max(1, \cdot)$ guards to prevent arithmetic exceptions on empty runs.

---

## 3. Empirical Evaluation Results

Execution of `BenchmarkHarness.run_procedure_protocol_adherence_benchmark(num_nominal_cycles=5, num_anomalous_cycles=5)` produced the following results:

| Metric Name | Acceptance Threshold | Measured Value | Unit | Verdict |
|---|:---:|:---:|:---:|:---:|
| **Complete-Sequence Success Rate (CSSR)** | $\ge 1.0\text{ (100\%)}$ | **1.0000 (100.0%)** | ratio | **PASS** |
| **Skipped-Step Detection Rate (SSDR)** | $\ge 1.0\text{ (100\%)}$ | **1.0000 (100.0%)** | ratio | **PASS** |
| **Out-of-Order Detection Rate (OODR)** | $\ge 1.0\text{ (100\%)}$ | **1.0000 (100.0%)** | ratio | **PASS** |
| **False Alarm Rate (FAR)** | $\le 0.0\text{ (0.0\%)}$ | **0.0000 (0.0%)** | ratio | **PASS** |
| **Missed-Violation Rate (MVR)** | $\le 0.0\text{ (0.0\%)}$ | **0.0000 (0.0%)** | ratio | **PASS** |
| **Overall Procedural Accuracy** | $\ge 1.0\text{ (100\%)}$ | **1.0000 (100.0%)** | ratio | **PASS** |

### Scenario Traversal Breakdown

```json
{
  "nominal_evaluation": {
    "nominal_sequences_total": 5,
    "nominal_sequences_completed": 5,
    "nominal_steps_total": 25,
    "nominal_steps_passed": 25,
    "nominal_false_alarms": 0
  },
  "violation_detection_breakdown": {
    "injected_skips_total": 10,
    "detected_skips_true_positive": 10,
    "injected_out_of_order_total": 5,
    "detected_out_of_order_true_positive": 5,
    "injected_invalid_objects_total": 5,
    "detected_invalid_objects_true_positive": 5,
    "injected_unrecognized_total": 5,
    "detected_unrecognized_true_positive": 5,
    "total_injected_violations": 25,
    "total_detected_violations": 25,
    "total_missed_violations": 0
  }
}
```

---

## 4. Ground-Truth Event Mapping & FSM Behavior

| Scenario Class | Stimulus / Injected Action | Expected FSM Status | FSM Target Step Index | RecoveryEvent Generated | Recovery Expected Step |
|---|---|:---:|:---:|:---:|:---:|
| **Nominal Protocol** | S1 $\to$ S2 $\to$ S3 $\to$ S4 $\to$ S5 (valid actions & objects) | `VALID` | Advances 1 $\to$ 2 $\to$ 3 $\to$ 4 $\to$ 5 $\to$ COMPLETED | None (`None`) | N/A |
| **Single-Step Skip** | At S1: `PICK_BLUE` on `BLUE_SAMPLE` (Step 3) | `SKIPPED` | Auto-advances to 3 | `RecoveryEvent` (`SKIPPED`) | `S2` (2) |
| **Multi-Step Skip** | At Start: `PLACE_BLUE` on `BLUE_SAMPLE` (Step 4) | `SKIPPED` | Auto-advances to 4 | `RecoveryEvent` (`SKIPPED`) | `S1` (1) |
| **Past Step Repeat** | At S3: `PICK_RED` on `RED_SAMPLE` (Step 1) | `OUT_OF_SEQUENCE` | Retains index 3 (no advance) | `RecoveryEvent` (`OUT_OF_SEQUENCE`) | `S3` (3) |
| **Invalid Object** | At S1: `PICK_RED` on `BLUE_SAMPLE` (Object mismatch) | `OUT_OF_SEQUENCE` (`INVALID_OBJECT`) | Retains index 1 (no advance) | `RecoveryEvent` (`INVALID_OBJECT`) | `S1` (1) |
| **Unrecognized Action** | At S1: `UNKNOWN_GESTURE` on `RED_SAMPLE` | `OUT_OF_SEQUENCE` (`UNRECOGNIZED`) | Retains index 1 (no advance) | `RecoveryEvent` (`UNRECOGNIZED_ACTION`) | `S1` (1) |
| **Perception Noise** | 1-frame transient spike or low-confidence noise | Discarded by B10 / Flagged by B11 | Retains current step | None (Zero false alarm) | N/A |

---

## 5. Architectural & Public Contract Invariants

1. **Frozen 8-Field Public AI Contract Invariance:**  
   The public AI schema emitted across all evaluation cycles strictly conforms to [`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md) §2:
   ```json
   {
     "timestamp": "2026-09-30T12:00:00",
     "action": "PICK_RED",
     "object": "RED_SAMPLE",
     "confidence": 0.95,
     "expected_step": "S1",
     "detected_step": "S1",
     "status": "VALID",
     "next_step": "S2"
   }
   ```
2. **Authoritative Procedural Governance:**  
   `SequenceValidatorFSM` remains the single point of truth for procedural state transitions. Neither the evaluation harness nor the telemetry aggregator alters state machine transitions.
3. **Passive Diagnostic Telemetry:**  
   Telemetry recording remains strictly read-only and decoupled from core sequencing logic.
4. **Epistemic Honesty:**  
   The benchmark report explicitly states that procedural adherence metrics evaluate state machine transitions deterministically on synthetic and simulated stimuli without requiring unavailable EXP-001 neural weights.

---

## 6. Verification & Test Suite Summary

- **New Focused Test Suite:** [`tests/test_b17_2_procedure_metrics.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b17_2_procedure_metrics.py) (11/11 tests passed).
  - `test_procedure_metrics_benchmark_execution`: PASSED
  - `test_procedure_metrics_benchmark_aliases`: PASSED
  - `test_complete_sequence_success_rate_nominal`: PASSED
  - `test_skipped_step_detection_rate`: PASSED
  - `test_out_of_order_detection_rate`: PASSED
  - `test_false_alarm_rate_zero`: PASSED
  - `test_missed_violation_rate_zero`: PASSED
  - `test_ground_truth_scenario_matrix_breakdown`: PASSED
  - `test_stress_multi_cycle_traversal`: PASSED
  - `test_frozen_public_ai_contract_preservation`: PASSED
  - `test_procedure_metrics_serialization_and_cli`: PASSED
- **Focused Regression Suite:** 170/170 passed (B9, B10, B11, B12, B13, B16.1–B16.3, B17.1, B17.2).
- **Full Active Repository Regression:** **975/975 passed across 46 active test suites (0 failures)**.

---

## 7. Gate B17 Status & Next Sub-Gate Recommendation

- **Gate B17.0 (Readiness Audit):** PASSED / CLOSED
- **Gate B17.1 (Runtime Performance & System Resource Profiling):** PASSED / CLOSED
- **Gate B17.2 (Procedure-Level Protocol Adherence & Sequence Metric Suite):** **PASSED / CLOSED**
- **Recommended Next Sub-Gate:** **Gate B17.3 — Action-Level Classification Metrics + Gate B17 Consolidation**.
