"""
temporal_filter.py — Deterministic Temporal Confirmation Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B10.1)

Implements rolling-window M-of-N majority confirmation and post-commit cooldown/debouncing
on composite (action, object) perception pairs between AI perception and SequenceValidatorFSM.

Architectural Rule:
Perception answers: "What physical action and object appear to be happening in this frame?"
TemporalConfirmationEngine answers: "Is this composite observation temporally stable across frames?"
SequenceValidatorFSM answers: "Is this confirmed action/object procedurally valid at current state?"
"""

from __future__ import annotations

import collections
import time
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from .result_adapter import (
    CANONICAL_ACTIONS,
    CANONICAL_OBJECTS,
    normalize_action,
    normalize_confidence,
    normalize_object,
    normalize_timestamp,
)


class TemporalConfirmationEngine:
    """
    Deterministic M-of-N rolling window temporal confirmation engine.
    
    Guarantees:
    1. Operates on the composite perception tuple (action, object).
    2. Only observations meeting minimum confidence threshold count towards confirmation.
    3. Requires at least M matching occurrences in the last N frames to trigger confirmation.
    4. Enforces post-commit cooldown/debounce to prevent a sustained physical action from
       repeatedly triggering procedure transitions.
    5. Releases cooldown upon observing a neutral state (IDLE/NONE) or a distinct action candidate.
    6. Does not maintain procedure state (FSM remains sole procedural authority).
    """

    def __init__(
        self,
        window_size: int = 5,
        confirmation_threshold: int = 3,
        min_confidence: float = 0.70,
        cooldown_frames: int = 5,
        cooldown_sec: Optional[float] = None,
    ):
        """
        Args:
            window_size (N): Number of recent frames tracked in the rolling window (default: 5).
            confirmation_threshold (M): Minimum matching occurrences required for confirmation (default: 3).
            min_confidence (tau): Minimum per-frame confidence required to count as a vote (default: 0.70).
            cooldown_frames: Number of consecutive frames to suppress re-committing the same candidate (default: 5).
            cooldown_sec: Optional monotonic time-based cooldown in seconds.
        """
        if window_size < 1:
            raise ValueError(f"window_size (N) must be >= 1, got {window_size}")
        if confirmation_threshold < 1:
            raise ValueError(f"confirmation_threshold (M) must be >= 1, got {confirmation_threshold}")
        if confirmation_threshold > window_size:
            raise ValueError(
                f"confirmation_threshold (M={confirmation_threshold}) cannot exceed window_size (N={window_size})"
            )

        self._window_size: int = int(window_size)
        self._confirmation_threshold: int = int(confirmation_threshold)
        self._min_confidence: float = float(min_confidence)
        self._cooldown_frames: int = int(cooldown_frames)
        self._cooldown_sec: Optional[float] = float(cooldown_sec) if cooldown_sec is not None else None

        # Rolling window storing: Tuple[Tuple[str, str], float, str, float]
        # Format: ((norm_action, norm_object), confidence, timestamp_str, monotonic_time)
        self._window: Deque[Tuple[Tuple[str, str], float, str, float]] = collections.deque(maxlen=self._window_size)

        # Cooldown tracking
        self._last_committed_candidate: Optional[Tuple[str, str]] = None
        self._last_committed_monotonic: float = 0.0
        self._cooldown_remaining: int = 0

        # Lifecycle & Telemetry
        self._state: str = "RUNNING"
        self._is_paused: bool = False
        self._total_commits: int = 0
        self._total_observations: int = 0

    # -------------------------------------------------------------------------
    # Configuration Properties
    # -------------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Rolling window capacity N."""
        return self._window_size

    @property
    def confirmation_threshold(self) -> int:
        """Required vote threshold M."""
        return self._confirmation_threshold

    @property
    def min_confidence(self) -> float:
        """Minimum confidence threshold."""
        return self._min_confidence

    @property
    def cooldown_frames(self) -> int:
        """Configured post-commit cooldown frames."""
        return self._cooldown_frames

    @property
    def is_paused(self) -> bool:
        """Whether observation accumulation is paused."""
        return self._is_paused

    @property
    def state(self) -> str:
        """Current engine state ('RUNNING', 'PAUSED', 'COMPLETED')."""
        return self._state

    @property
    def total_commits(self) -> int:
        """Total number of confirmed action emissions."""
        return self._total_commits

    @property
    def total_observations(self) -> int:
        """Total raw observations processed."""
        return self._total_observations

    @property
    def last_committed_candidate(self) -> Optional[Tuple[str, str]]:
        """Most recent (action, object) candidate committed to FSM."""
        return self._last_committed_candidate

    # -------------------------------------------------------------------------
    # Observation Processing (Core Logic)
    # -------------------------------------------------------------------------

    def process_observation(
        self,
        action: Optional[str],
        object_name: Optional[str] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[Union[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single-frame composite perception observation through the M-of-N filter.

        Returns an internal evaluation dictionary:
        {
            "confirmed": bool,
            "action": str,
            "object": str,
            "confidence": float,
            "timestamp": str,
            "vote_count": int,
            "window_size": int,
            "state": str,  # 'CONFIRMED', 'UNCONFIRMED', 'COOLDOWN', 'LOW_CONFIDENCE', 'IDLE', 'PAUSED', 'COMPLETED'
            "candidate": Optional[Tuple[str, str]],
        }
        """
        self._total_observations += 1
        mono_now = time.monotonic()
        ts_str = normalize_timestamp(timestamp)
        norm_act = normalize_action(action)
        norm_obj = normalize_object(object_name, action=norm_act)
        try:
            raw_c = float(confidence) if confidence is not None else 0.0
            conf = max(0.0, min(1.0, raw_c))
        except (ValueError, TypeError):
            conf = 0.0

        # 1. Lifecycle: Paused or Completed state checks
        if self._state == "COMPLETED":
            return {
                "confirmed": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": 0,
                "window_size": len(self._window),
                "state": "COMPLETED",
                "candidate": None,
            }

        if self._is_paused:
            return {
                "confirmed": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": 0,
                "window_size": len(self._window),
                "state": "PAUSED",
                "candidate": None,
            }

        # 2. Neutral / IDLE Observation
        if norm_act == "IDLE":
            # Observing IDLE releases post-commit cooldown on previous actions
            self._release_cooldown()
            self._window.append((("IDLE", "NONE"), conf, ts_str, mono_now))
            return {
                "confirmed": False,
                "action": "IDLE",
                "object": "NONE",
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": 0,
                "window_size": len(self._window),
                "state": "IDLE",
                "candidate": None,
            }

        # 3. Low-Confidence Filtering
        if conf < self._min_confidence:
            # Low confidence observations do not count towards votes, but age out older votes
            self._window.append(((norm_act, norm_obj), conf, ts_str, mono_now))
            return {
                "confirmed": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": 0,
                "window_size": len(self._window),
                "state": "LOW_CONFIDENCE",
                "candidate": (norm_act, norm_obj),
            }

        # 4. Post-Commit Cooldown / Debounce Check for Sustained Action
        current_candidate = (norm_act, norm_obj)
        if self._is_in_cooldown(current_candidate, mono_now):
            self._decrement_cooldown()
            return {
                "confirmed": False,
                "action": current_candidate[0],
                "object": current_candidate[1],
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": 0,
                "window_size": len(self._window),
                "state": "COOLDOWN",
                "candidate": current_candidate,
            }

        # 5. If a new different candidate arrives, release previous candidate lock
        if self._last_committed_candidate is not None and current_candidate != self._last_committed_candidate:
            self._release_cooldown()

        # 6. Valid High-Confidence Candidate - Append to rolling window
        self._window.append((current_candidate, conf, ts_str, mono_now))

        # 7. Evaluate M-of-N Majority Vote over Rolling Window
        # Group valid votes by exact composite key (action, object)
        candidate_votes: Dict[Tuple[str, str], List[float]] = collections.defaultdict(list)
        encounter_order: List[Tuple[str, str]] = []

        for cand, c_val, _, _ in self._window:
            if cand[0] != "IDLE" and c_val >= self._min_confidence:
                if cand not in candidate_votes:
                    encounter_order.append(cand)
                candidate_votes[cand].append(c_val)

        if not candidate_votes:
            return {
                "confirmed": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": 0,
                "window_size": len(self._window),
                "state": "UNCONFIRMED",
                "candidate": current_candidate,
            }

        # Determine winning composite candidate
        def candidate_sort_key(k: Tuple[str, str]) -> Tuple[int, float, int]:
            v_list = candidate_votes[k]
            last_idx = max(i for i, (c, _, _, _) in enumerate(self._window) if c == k)
            return (len(v_list), sum(v_list), last_idx)

        winning_candidate = max(candidate_votes.keys(), key=candidate_sort_key)
        winning_confs = candidate_votes[winning_candidate]
        winning_count = len(winning_confs)

        # 8. Check Confirmation Threshold (M)
        if winning_count < self._confirmation_threshold:
            return {
                "confirmed": False,
                "action": norm_act,
                "object": norm_obj,
                "confidence": conf,
                "timestamp": ts_str,
                "vote_count": winning_count,
                "window_size": len(self._window),
                "state": "UNCONFIRMED",
                "candidate": winning_candidate,
            }

        # 9. Confirmation Reached!
        mean_confirmed_conf = round(float(sum(winning_confs) / winning_count), 2)
        self._last_committed_candidate = winning_candidate
        self._last_committed_monotonic = mono_now
        self._cooldown_remaining = self._cooldown_frames
        self._total_commits += 1

        # Clear window after commit to ensure fresh temporal accumulation for subsequent step
        self._window.clear()

        return {
            "confirmed": True,
            "action": winning_candidate[0],
            "object": winning_candidate[1],
            "confidence": mean_confirmed_conf,
            "timestamp": ts_str,
            "vote_count": winning_count,
            "window_size": winning_count,
            "state": "CONFIRMED",
            "candidate": winning_candidate,
        }

    # -------------------------------------------------------------------------
    # Cooldown Helpers
    # -------------------------------------------------------------------------

    def _is_in_cooldown(self, candidate: Tuple[str, str], mono_now: float) -> bool:
        """Determines if a candidate is actively locked by post-commit cooldown."""
        if self._last_committed_candidate is None or candidate != self._last_committed_candidate:
            return False

        # Candidate is identical to last committed candidate.
        # Enforce sustained action lockout until released by neutral IDLE, action change, or reset.
        return True

    def _decrement_cooldown(self) -> None:
        """Decrements the remaining frame cooldown count if tracked."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def _release_cooldown(self) -> None:
        """Releases any active post-commit cooldown lock."""
        self._last_committed_candidate = None
        self._cooldown_remaining = 0

    # -------------------------------------------------------------------------
    # Lifecycle Management API
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Flushes rolling window, clears cooldown lock, and resets commit counters."""
        self._window.clear()
        self._last_committed_candidate = None
        self._last_committed_monotonic = 0.0
        self._cooldown_remaining = 0
        self._state = "RUNNING"
        self._is_paused = False
        self._total_commits = 0
        self._total_observations = 0

    def pause(self) -> None:
        """Suspends confirmation voting and discards pre-pause observations."""
        self._is_paused = True
        self._state = "PAUSED"
        self._window.clear()

    def resume(self) -> None:
        """Resumes confirmation voting with a clean temporal window."""
        self._is_paused = False
        self._state = "RUNNING"
        self._window.clear()

    def complete(self) -> None:
        """Freezes confirmation engine upon procedure completion until explicit reset."""
        self._state = "COMPLETED"
        self._window.clear()
