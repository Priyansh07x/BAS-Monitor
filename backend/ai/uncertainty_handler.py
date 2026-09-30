"""
uncertainty_handler.py — Deterministic Perception Uncertainty Handler & Evidence Accumulator
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B11.1)

Evaluates perception reliability, accumulates marginal-confidence evidence (0.50 <= conf < 0.70),
and prevents premature procedural anomaly escalation.

Architectural Rule:
Perception: "What physical action and object appear to be happening in this frame?"
TemporalConfirmationEngine (B10): "Is this composite observation temporally stable across frames?"
UncertaintyHandler (B11): "Is the available evidence reliable enough to trust?"
SequenceValidatorFSM (B9): "Is this reliable observation procedurally valid at current state?"
"""

from __future__ import annotations

import collections
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from .result_adapter import (
    CANONICAL_ACTIONS,
    CANONICAL_OBJECTS,
    normalize_action,
    normalize_confidence,
    normalize_object,
    normalize_timestamp,
)


class UncertaintyHandler:
    """
    Deterministic, bounded perception uncertainty handler and evidence accumulator.

    Guarantees:
    1. Operates on normalized composite perception tuples (action, object).
    2. Categorizes observations into discrete internal uncertainty states:
       - 'CONFIDENT': High-confidence observation (>= high_confidence_threshold, default 0.70).
       - 'UNCERTAIN_LOW_CONFIDENCE': Sub-marginal observation (< marginal_confidence_threshold, default 0.50).
       - 'UNCERTAIN_EVIDENCE_ACCUMULATING': Marginal observation (0.50 <= conf < 0.70) undergoing accumulation.
       - 'RESOLVED': Marginal observation that accumulated sufficient consistent evidence.
       - 'CONFIRMED_ANOMALY': Persistent anomalous/contradictory evidence exceeding anomaly budget.
       - 'IDLE': Neutral/IDLE state.
       - 'PAUSED' / 'COMPLETED': Lifecycle states.
    3. Accumulates consistent composite evidence over a bounded sliding window without cross-object mixing.
    4. Distinguishes unresolved uncertainty from reliable evidence (is_reliable flag).
    5. Does not maintain procedural sequence state (FSM remains sole procedural authority).
    6. Memory-bounded with zero background threads or queues.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.70,
        marginal_confidence_threshold: float = 0.50,
        required_evidence_count: int = 3,
        anomaly_persistence_threshold: int = 5,
        max_evidence_history: int = 10,
    ):
        """
        Args:
            high_confidence_threshold: Minimum confidence to immediately classify as CONFIDENT (default: 0.70).
            marginal_confidence_threshold: Lower bound for marginal evidence accumulation (default: 0.50).
            required_evidence_count: Number of consistent marginal frames required to resolve uncertainty (default: 3).
            anomaly_persistence_threshold: Number of persistent conflicting frames before marking CONFIRMED_ANOMALY (default: 5).
            max_evidence_history: Maximum capacity of the bounded evidence deque (default: 10).
        """
        if high_confidence_threshold <= marginal_confidence_threshold:
            raise ValueError(
                f"high_confidence_threshold ({high_confidence_threshold}) must be > "
                f"marginal_confidence_threshold ({marginal_confidence_threshold})"
            )
        if required_evidence_count < 1:
            raise ValueError(f"required_evidence_count must be >= 1, got {required_evidence_count}")
        if anomaly_persistence_threshold < 1:
            raise ValueError(f"anomaly_persistence_threshold must be >= 1, got {anomaly_persistence_threshold}")
        if max_evidence_history < 1:
            raise ValueError(f"max_evidence_history must be >= 1, got {max_evidence_history}")

        self._high_conf_thresh: float = float(high_confidence_threshold)
        self._marginal_conf_thresh: float = float(marginal_confidence_threshold)
        self._required_evidence: int = int(required_evidence_count)
        self._anomaly_persistence: int = int(anomaly_persistence_threshold)
        self._max_history: int = int(max_evidence_history)

        # Bounded evidence deque: Tuple[Tuple[str, str], float, str]
        # Format: ((norm_action, norm_object), confidence, timestamp_str)
        self._evidence_window: Deque[Tuple[Tuple[str, str], float, str]] = collections.deque(
            maxlen=self._max_history
        )

        # Active tracking candidate and counter
        self._active_candidate: Optional[Tuple[str, str]] = None
        self._consecutive_marginal_count: int = 0
        self._persistent_anomaly_count: int = 0

        # Lifecycle & Telemetry
        self._state: str = "RUNNING"
        self._is_paused: bool = False
        self._total_evaluations: int = 0
        self._total_resolved: int = 0
        self._total_anomalies: int = 0

    # -------------------------------------------------------------------------
    # Configuration Properties
    # -------------------------------------------------------------------------

    @property
    def high_confidence_threshold(self) -> float:
        """Threshold above which evidence is immediately considered confident."""
        return self._high_conf_thresh

    @property
    def marginal_confidence_threshold(self) -> float:
        """Lower threshold below which evidence is considered insufficient."""
        return self._marginal_conf_thresh

    @property
    def required_evidence_count(self) -> int:
        """Required consistent marginal frames to resolve uncertainty."""
        return self._required_evidence

    @property
    def anomaly_persistence_threshold(self) -> int:
        """Persistence threshold before escalating to confirmed anomaly."""
        return self._anomaly_persistence

    @property
    def max_evidence_history(self) -> int:
        """Capacity of bounded evidence deque."""
        return self._max_history

    @property
    def is_paused(self) -> bool:
        """Whether uncertainty handling is currently paused."""
        return self._is_paused

    @property
    def state(self) -> str:
        """Current handler lifecycle state."""
        return self._state

    @property
    def total_evaluations(self) -> int:
        """Total evaluations performed."""
        return self._total_evaluations

    @property
    def total_resolved(self) -> int:
        """Total marginal candidates successfully resolved."""
        return self._total_resolved

    @property
    def total_anomalies(self) -> int:
        """Total anomalies confirmed through persistent evidence."""
        return self._total_anomalies

    # -------------------------------------------------------------------------
    # Core Evaluation Logic
    # -------------------------------------------------------------------------

    def process_observation(
        self,
        action: Optional[str],
        object_name: Optional[str] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[Union[str, float]] = None,
        expected_step_action: Optional[str] = None,
        expected_step_object: Optional[str] = None,
        is_conflict: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates perception observation for uncertainty and accumulates marginal evidence.

        Returns an internal evaluation dictionary:
        {
            "state": str,        # 'CONFIDENT', 'UNCERTAIN_LOW_CONFIDENCE', 'UNCERTAIN_EVIDENCE_ACCUMULATING',
                                 # 'RESOLVED', 'CONFIRMED_ANOMALY', 'IDLE', 'PAUSED', 'COMPLETED'
            "is_reliable": bool, # True if CONFIDENT or RESOLVED
            "action": str,
            "object": str,
            "confidence": float,
            "evidence_count": int,
            "timestamp": str,
            "candidate": Optional[Tuple[str, str]],
            "message": str,
        }
        """
        self._total_evaluations += 1
        ts_str = normalize_timestamp(timestamp)
        norm_act = normalize_action(action)
        norm_obj = normalize_object(object_name, action=norm_act)
        conf = normalize_confidence(confidence)
        current_candidate = (norm_act, norm_obj)

        # 1. Lifecycle Checks
        if self._state == "COMPLETED":
            return {
                "state": "COMPLETED",
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "evidence_count": 0,
                "timestamp": ts_str,
                "candidate": None,
                "message": "Procedure completed; uncertainty handler frozen.",
            }

        if self._is_paused:
            return {
                "state": "PAUSED",
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "evidence_count": 0,
                "timestamp": ts_str,
                "candidate": None,
                "message": "Uncertainty handler is paused.",
            }

        # 2. Neutral / IDLE Observation
        if norm_act == "IDLE":
            self._flush_marginal_accumulation()
            self._persistent_anomaly_count = 0
            return {
                "state": "IDLE",
                "is_reliable": False,
                "action": "IDLE",
                "object": "NONE",
                "confidence": conf,
                "evidence_count": 0,
                "timestamp": ts_str,
                "candidate": None,
                "message": "Neutral IDLE observation; accumulation cleared.",
            }

        # 3. Persistent Perception Anomaly Escalation
        if is_conflict or (expected_step_action and norm_act != expected_step_action and norm_act != "IDLE"):
            self._persistent_anomaly_count += 1
            if self._persistent_anomaly_count >= self._anomaly_persistence:
                self._total_anomalies += 1
                return {
                    "state": "CONFIRMED_ANOMALY",
                    "is_reliable": False,
                    "action": norm_act,
                    "object": norm_obj,
                    "confidence": conf,
                    "evidence_count": self._persistent_anomaly_count,
                    "timestamp": ts_str,
                    "candidate": current_candidate,
                    "message": f"Persistent perception anomaly confirmed ({self._persistent_anomaly_count} frames).",
                }
            return {
                "state": "UNCERTAIN_CONFLICT",
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "evidence_count": self._persistent_anomaly_count,
                "timestamp": ts_str,
                "candidate": current_candidate,
                "message": f"Perception conflict accumulating ({self._persistent_anomaly_count}/{self._anomaly_persistence} frames).",
            }

        # 4. High Confidence Observation (Case A: conf >= high_confidence_threshold)
        if conf >= self._high_conf_thresh:
            self._flush_marginal_accumulation()
            self._persistent_anomaly_count = 0
            self._evidence_window.append((current_candidate, conf, ts_str))
            return {
                "state": "CONFIDENT",
                "is_reliable": True,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "evidence_count": 1,
                "timestamp": ts_str,
                "candidate": current_candidate,
                "message": f"High-confidence perception (conf={conf:.2f} >= {self._high_conf_thresh:.2f}).",
            }

        # 4. Very Low Confidence (Case C: conf < marginal_confidence_threshold)
        if conf < self._marginal_conf_thresh:
            self._flush_marginal_accumulation()
            self._persistent_anomaly_count = 0
            self._evidence_window.append((current_candidate, conf, ts_str))
            return {
                "state": "UNCERTAIN_LOW_CONFIDENCE",
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "evidence_count": 0,
                "timestamp": ts_str,
                "candidate": current_candidate,
                "message": f"Insufficient confidence (conf={conf:.2f} < {self._marginal_conf_thresh:.2f}); rejected.",
            }

        # 5. Marginal Confidence (Case B / D: marginal_confidence_threshold <= conf < high_confidence_threshold)
        # Check if this continues the same active candidate
        if self._active_candidate is not None and current_candidate != self._active_candidate:
            # Candidate switched during marginal observation -> reset accumulation for new candidate
            self._flush_marginal_accumulation()

        self._active_candidate = current_candidate
        self._consecutive_marginal_count += 1
        self._evidence_window.append((current_candidate, conf, ts_str))

        # Check if consistent marginal evidence reaches resolution threshold (Case C)
        if self._consecutive_marginal_count >= self._required_evidence:
            self._total_resolved += 1
            # Compute aggregate mean confidence over accumulated marginal evidence for this candidate
            matching_confs = [
                c_val for c_cand, c_val, _ in self._evidence_window if c_cand == current_candidate
            ]
            agg_conf = (
                round(float(sum(matching_confs) / len(matching_confs)), 2)
                if matching_confs
                else conf
            )

            # Reset accumulation after resolution
            resolved_count = self._consecutive_marginal_count
            self._flush_marginal_accumulation()

            return {
                "state": "RESOLVED",
                "is_reliable": True,
                "action": norm_act,
                "object": norm_obj,
                "confidence": agg_conf,
                "evidence_count": resolved_count,
                "timestamp": ts_str,
                "candidate": current_candidate,
                "message": f"Marginal evidence resolved ({resolved_count} consistent frames; agg_conf={agg_conf:.2f}).",
            }

        # Still accumulating evidence
        return {
            "state": "UNCERTAIN_EVIDENCE_ACCUMULATING",
            "is_reliable": False,
            "action": norm_act,
            "object": norm_obj,
            "confidence": conf,
            "evidence_count": self._consecutive_marginal_count,
            "timestamp": ts_str,
            "candidate": current_candidate,
            "message": f"Accumulating evidence for {current_candidate} ({self._consecutive_marginal_count}/{self._required_evidence} frames).",
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _flush_marginal_accumulation(self) -> None:
        """Flushes active candidate marginal accumulator."""
        self._active_candidate = None
        self._consecutive_marginal_count = 0

    # -------------------------------------------------------------------------
    # Lifecycle Management API
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Flushes all accumulated evidence, resets counters, and returns to RUNNING."""
        self._evidence_window.clear()
        self._flush_marginal_accumulation()
        self._persistent_anomaly_count = 0
        self._state = "RUNNING"
        self._is_paused = False
        self._total_evaluations = 0
        self._total_resolved = 0
        self._total_anomalies = 0

    def pause(self) -> None:
        """Suspends uncertainty evaluation and discards transient un-resolved evidence."""
        self._is_paused = True
        self._state = "PAUSED"
        self._evidence_window.clear()
        self._flush_marginal_accumulation()

    def resume(self) -> None:
        """Resumes uncertainty evaluation with a clean evidence buffer."""
        self._is_paused = False
        self._state = "RUNNING"
        self._evidence_window.clear()
        self._flush_marginal_accumulation()

    def complete(self) -> None:
        """Freezes uncertainty handler upon procedure completion."""
        self._state = "COMPLETED"
        self._evidence_window.clear()
        self._flush_marginal_accumulation()
