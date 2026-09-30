# Workstream B — Gate B17.3 Action-Level Classification Metrics & Gate B17 Consolidation Report

**ISRO SIH26174 BAS Experiment Monitor — Workstream B (AI & Procedure Intelligence)**  
**Gate:** B17.3 / B17 (Action-Level Classification Metrics & Gate B17 Consolidation)  
**Status:** `PASSED` / `CLOSED`  
**Date:** 2026-09-30  
**Author:** Antigravity (AI Assistant)  

---

## 1. Executive Summary

Gate B17.3 concludes **Gate B17 ("Evaluation & Metrics Reporting")** of Workstream B by implementing a rigorous, mathematically verified action-level classification metric engine and consolidating all Gate B17 evaluation dimensions into a unified, deterministic benchmark suite.

### Core Deliverables Completed:
1. **Multi-Class Action Classification Engine:** Computes per-class True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), True Negatives ($TN$), Support, Precision, Recall, and F1-score alongside Macro Precision, Macro Recall, Macro F1, Global Accuracy, and a canonical $5 \times 5$ multi-class Confusion Matrix $C \in \mathbb{Z}^{5 \times 5}$.
2. **Deterministic Mathematical Verification:** Validated via deterministic labeled test vectors including a perfect nominal sequence ($N=5$), hand-calculated mixed substitution vector ($N=6$, 1 error, Accuracy $= 0.8333$, Macro $F1 = 0.8667$), zero-support and zero-prediction boundary vectors, empty inputs ($N=0$), and unmapped label containment.
3. **Consolidated Gate B17 Benchmark Harness (`consolidated_b17_evaluation`):** Aggregates all four evaluation dimensions:
   - **Dimension A:** Runtime Performance & System Resource Profiling (B17.1 — AI FPS, Latency Mean/P50/P95/P99, Stage A–F breakdowns, RAM RSS footprint, CPU %).
   - **Dimension B:** Procedure-Level Protocol Adherence & Sequence Metrics (B17.2 — Complete-Sequence Success Rate, Skipped-Step Detection Rate, Out-of-Order Detection Rate, False Alarm Rate, Missed-Violation Rate).
   - **Dimension C:** Action-Level Classification Metric Engine (B17.3 — Precision, Recall, F1, $5 \times 5$ Confusion Matrix).
   - **Dimension D:** Real Model & Dataset Availability Status Disclosures (`UNAVAILABLE / DEFERRED`).
4. **Epistemic Integrity Guaranteed:** Transparently discloses that action classification metrics are validated on deterministic synthetic fixtures; physical EXP-001 labeled video and Part-1 neural checkpoints remain explicitly deferred. Zero synthetic neural accuracy is fabricated.
5. **Frozen Public AI Contract Preservation:** Strictly enforces the 8-field public AI contract ([`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md) §2) across all benchmark and execution pathways.
6. **Full Active Repository Regression:** **987 / 987 tests passed across 47 active test suites** (100% PASS rate, 0 failures).

---

## 2. Mathematical Metric Formulation

The metric engine evaluates multi-class predictions against canonical EXP-001 action classes in strict fixed order:
$$\mathcal{C} = [\text{PICK\_RED}, \text{PLACE\_RED}, \text{PICK\_BLUE}, \text{PLACE\_BLUE}, \text{CLOSE\_LID}], \quad K = 5$$

### 2.1 Multi-Class Confusion Matrix
Given true labels $y_{\text{true}} \in \mathcal{C}^N$ and predicted labels $y_{\text{pred}} \in \mathcal{C}^N$, the confusion matrix $C \in \mathbb{Z}^{K \times K}$ is defined as:
$$C_{i, j} = \sum_{k=1}^N \mathbb{I}(y_{\text{true}}^{(k)} = \mathcal{C}_i \land y_{\text{pred}}^{(k)} = \mathcal{C}_j)$$
where rows correspond to the ground-truth classes and columns correspond to the predicted classes.

### 2.2 Per-Class Confusion Counts & Metrics
For each class $c \in \mathcal{C}$ with row/column index $i$:
- **True Positives ($TP_c$):** $C_{i, i}$
- **False Positives ($FP_c$):** $\sum_{j \neq i} C_{j, i}$
- **False Negatives ($FN_c$):** $\sum_{j \neq i} C_{i, j}$
- **True Negatives ($TN_c$):** $N - (TP_c + FP_c + FN_c)$
- **Support ($Support_c$):** $\sum_{j=1}^K C_{i, j} = TP_c + FN_c$
- **Precision ($P_c$):** $\frac{TP_c}{TP_c + FP_c} \quad (\text{if } TP_c + FP_c = 0 \implies 0.0)$
- **Recall ($R_c$):** $\frac{TP_c}{TP_c + FN_c} \quad (\text{if } TP_c + FN_c = 0 \implies 0.0)$
- **F1-Score ($F1_c$):** $\frac{2 \cdot P_c \cdot R_c}{P_c + R_c} \quad (\text{if } P_c + R_c = 0 \implies 0.0)$

### 2.3 Macro-Averaged & Global Metrics
- **Macro Precision:** $\text{Macro } P = \frac{1}{K} \sum_{c \in \mathcal{C}} P_c$
- **Macro Recall:** $\text{Macro } R = \frac{1}{K} \sum_{c \in \mathcal{C}} R_c$
- **Macro F1-Score:** $\text{Macro } F1 = \frac{1}{K} \sum_{c \in \mathcal{C}} F1_c$
- **Global Accuracy:** $\text{Accuracy} = \frac{\sum_{c \in \mathcal{C}} TP_c}{N} \quad (\text{if } N = 0 \implies 0.0)$

---

## 3. Test Vector Verification

### Vector 1: Perfect Nominal Sequence ($N=5$)
- $y_{\text{true}} = [\text{"PICK\_RED"}, \text{"PLACE\_RED"}, \text{"PICK\_BLUE"}, \text{"PLACE\_BLUE"}, \text{"CLOSE\_LID"}]$
- $y_{\text{pred}} = [\text{"PICK\_RED"}, \text{"PLACE\_RED"}, \text{"PICK\_BLUE"}, \text{"PLACE\_BLUE"}, \text{"CLOSE\_LID"}]$
- **Results:**
  - Accuracy: $1.0000$ (100%)
  - Macro Precision: $1.0000$, Macro Recall: $1.0000$, Macro F1: $1.0000$
  - Confusion Matrix: Identity matrix $I_5$
  - Per-class: $TP=1, FP=0, FN=0, TN=4, Support=1, P=1.0, R=1.0, F1=1.0$ for all 5 classes.

### Vector 2: Hand-Calculated Mixed Substitution Vector ($N=6$)
- $y_{\text{true}} = [\text{"PICK\_RED"}, \text{"PICK\_RED"}, \text{"PLACE\_RED"}, \text{"PICK\_BLUE"}, \text{"PLACE\_BLUE"}, \text{"CLOSE\_LID"}]$
- $y_{\text{pred}} = [\text{"PICK\_RED"}, \text{"PICK\_BLUE"}, \text{"PLACE\_RED"}, \text{"PICK\_BLUE"}, \text{"PLACE\_BLUE"}, \text{"CLOSE\_LID"}]$
- **Breakdown & Exact Hand-Calculations:**
  - `PICK_RED`: $TP=1, FP=0, FN=1, TN=4, \text{Support}=2 \implies P=1.0, R=0.5, F1=0.6667$
  - `PLACE_RED`: $TP=1, FP=0, FN=0, TN=5, \text{Support}=1 \implies P=1.0, R=1.0, F1=1.0$
  - `PICK_BLUE`: $TP=1, FP=1, FN=0, TN=4, \text{Support}=1 \implies P=0.5, R=1.0, F1=0.6667$
  - `PLACE_BLUE`: $TP=1, FP=0, FN=0, TN=5, \text{Support}=1 \implies P=1.0, R=1.0, F1=1.0$
  - `CLOSE_LID`: $TP=1, FP=0, FN=0, TN=5, \text{Support}=1 \implies P=1.0, R=1.0, F1=1.0$
  - **Averages:**
    - Accuracy: $\frac{5}{6} \approx 0.8333$
    - Macro Precision: $\frac{1.0 + 1.0 + 0.5 + 1.0 + 1.0}{5} = \frac{4.5}{5} = 0.9000$
    - Macro Recall: $\frac{0.5 + 1.0 + 1.0 + 1.0 + 1.0}{5} = \frac{4.5}{5} = 0.9000$
    - Macro F1: $\frac{0.6667 + 1.0 + 0.6667 + 1.0 + 1.0}{5} = \frac{4.3333}{5} = 0.8667$
  - **Confusion Matrix:**
    $$\begin{pmatrix}
    1 & 0 & 1 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 1
    \end{pmatrix}$$

### Vector 3: Edge Case Zero Support & Zero Predictions ($N=4$)
- $y_{\text{true}} = [\text{"PICK\_RED"}, \text{"PLACE\_RED"}, \text{"PICK\_BLUE"}, \text{"PICK\_BLUE"}]$
- $y_{\text{pred}} = [\text{"PICK\_RED"}, \text{"PLACE\_RED"}, \text{"PICK\_BLUE"}, \text{"PICK\_RED"}]$
- **Results:**
  - `CLOSE_LID`: $\text{Support}=0 \implies R=0.0, P=0.0, F1=0.0$ (no ZeroDivisionError)
  - `PLACE_BLUE`: $TP=0, FP=0, FN=0, \text{Support}=0 \implies P=0.0, R=0.0, F1=0.0$
  - Accuracy: $0.7500$, Macro Precision: $0.5000$, Macro Recall: $0.5000$, Macro F1: $0.4667$.

---

## 4. Gate B17 Consolidated Evaluation Matrix

The consolidated benchmark (`consolidated_b17_evaluation`) integrates all four sub-gate evaluation dimensions:

| Dimension | Evaluation Focus | Key Metrics Reported | Threshold / Requirement | Empirical Verdict |
|---|---|---|---|:---:|
| **Dimension A** | Runtime Performance & Resource Profiling (B17.1) | AI FPS: $\ge 20.0$<br>Mean Latency: $< 50\text{ ms}$<br>P50/P95/P99 latencies<br>Stage A–F breakdowns<br>RAM RSS & CPU % | Pipeline FPS $\ge 20.0$<br>Mean Latency $\le 50.0\text{ ms}$<br>RAM growth $\le 50\text{ MB}$ | **PASSED** |
| **Dimension B** | Procedure-Level Protocol Adherence (B17.2) | CSSR: $1.0$ ($100\%$ nominal)<br>SSDR: $1.0$ ($100\%$ skips detected)<br>OODR: $1.0$ ($100\%$ OOO detected)<br>FAR: $0.0$ ($0\%$ false alarms)<br>MVR: $0.0$ ($0\%$ missed violations) | $\text{CSSR} = 1.0$<br>$\text{SSDR} = 1.0$<br>$\text{OODR} = 1.0$<br>$\text{FAR} = 0.0$<br>$\text{MVR} = 0.0$ | **PASSED** |
| **Dimension C** | Action-Level Classification Metrics (B17.3) | Precision, Recall, F1 per class<br>Macro Precision, Macro Recall, Macro F1<br>Global Accuracy<br>$5 \times 5$ Multi-class Confusion Matrix | Mathematical exactness<br>Zero-division safety<br>Canonical class ordering | **PASSED** |
| **Dimension D** | Epistemic Status & Dataset Availability Disclosures | EXP-001 Video Dataset: `UNAVAILABLE / DEFERRED`<br>Part-1 Neural Checkpoint: `UNAVAILABLE / DEFERRED`<br>Physical Hailo NPU: `DEFERRED (CPU Fallback Verified)`<br>Integrity: Zero synthetic accuracy fabricated | Full transparency<br>No fabricated neural metrics | **DECLARED / VERIFIED** |

---

## 5. Architectural Invariants & Boundary Compliance

1. **FSM Authority:** `SequenceValidatorFSM` remains the exclusive source of truth for step state transitions, anomaly classification (`SKIPPED`, `OUT_OF_SEQUENCE`, `INVALID_OBJECT`, `UNRECOGNIZED`), and procedure completion. Metric computation is purely passive observation.
2. **Recovery Decoupling:** `RecoveryManager` generates structured `RecoveryEvent` objects only on confirmed FSM violations; noise rejection in B10/B11 prevents false recovery events ($FAR = 0.0$).
3. **Public Contract Preservation:** `AIResultAdapter.adapt()` strictly outputs the frozen 8-field public AI schema (`timestamp`, `action`, `object`, `confidence`, `expected_step`, `detected_step`, `status`, `next_step`).
4. **Epistemic Honesty:** All benchmark results explicitly state the synthetic test nature of classification metric validation and the deferred status of physical video datasets and neural checkpoints.

---

## 6. Test Suite & Regression Summary

| Test Suite | File | Tests | Verdict |
|---|---|:---:|:---:|
| B17.1 Runtime Performance Profiling | [`tests/test_b17_1_runtime_profiling.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b17_1_runtime_profiling.py) | 9 | **PASS** |
| B17.2 Procedure Protocol Adherence | [`tests/test_b17_2_procedure_metrics.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b17_2_procedure_metrics.py) | 11 | **PASS** |
| B17.3 Action Classification & Consolidation | [`tests/test_b17_3_classification_and_consolidation.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b17_3_classification_and_consolidation.py) | 12 | **PASS** |
| **Gate B17 Subsystem Total** | | **32 / 32** | **100% PASS** |
| **Full Repository Regression** | Active 47 Test Suites | **987 / 987** | **100% PASS** |

---

## 7. Gate Completion Verdict & Next Steps

Gate B17.3 and Gate B17 ("Evaluation & Metrics Reporting") are formally **`PASSED`** and **`CLOSED`**.

All requirements of Gate B17 are satisfied:
- Runtime FPS, latency percentiles (Mean, P50, P95, P99), Stage A–F breakdowns, and RAM/CPU utilization verified.
- Procedure complete sequence success (CSSR), skipped step detection (SSDR), out-of-order detection (OODR), false alarm rate (FAR), and missed violation rate (MVR) verified.
- Action-level Precision, Recall, F1, and multi-class Confusion Matrix mathematically verified on canonical EXP-001 classes.
- Unified Gate B17 benchmark harness operational and exportable to JSON/text summaries.
- Next scheduled gate: **Gate B18 — Joint Acceptance Testing**.
