"""
inference_worker.py — Non-Blocking Edge AI Inference Worker & Latest-Frame Buffer
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B7.1)

Executes continuous AI perception on a dedicated background Python thread.
Consumes frames via a thread-safe single-slot latest-frame replacement buffer,
preventing stale frame accumulation and guaranteeing that the worker always
evaluates the freshest available camera frame without blocking upstream producers.
"""

from __future__ import annotations

import collections
import enum
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from .inference_pipeline import InferencePipeline
from .result_adapter import AIResultAdapter


logger = logging.getLogger("BAS_InferenceWorker")


class RateStrategy(str, enum.Enum):
    """
    Frame-rate consumption strategy modes for InferenceWorker (Gate B8.2).

    Modes:
      OPPORTUNISTIC_LATEST: Unthrottled consumption. Worker processes newest available frame whenever idle.
      FIXED_10FPS: Paced consumption targeting 10 FPS (100 ms minimum interval between frame processing starts).
      FIXED_15FPS: Paced consumption targeting 15 FPS (~66.67 ms minimum interval between frame processing starts).
      TIME_DECIMATED: Monotonic time interval throttling using a configurable minimum AI interval.
    """
    OPPORTUNISTIC_LATEST = "OPPORTUNISTIC_LATEST"
    FIXED_10FPS = "FIXED_10FPS"
    FIXED_15FPS = "FIXED_15FPS"
    TIME_DECIMATED = "TIME_DECIMATED"


class LatestFrameBuffer:
    """
    Thread-safe single-slot latest-frame replacement buffer with telemetry counters.
    
    Guarantees:
    - Exactly 0 or 1 pending frame in storage at any time.
    - Submitting a new frame atomically replaces any existing unconsumed pending frame.
    - Never blocks the submitting thread.
    - Wakes the consumer thread immediately upon new frame arrival.
    - Tracks total submissions and drop/replacement occurrences.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending_data: Optional[Tuple[np.ndarray, Optional[Union[float, str]], Optional[Dict[str, Any]], Optional[str]]] = None
        self._pending_capture_mono: Optional[float] = None
        self._last_pop_capture_mono: Optional[float] = None
        self._is_cleared = False
        self._submission_count = 0
        self._replacement_count = 0

    @property
    def submission_count(self) -> int:
        """Total number of frame submissions received."""
        with self._lock:
            return self._submission_count

    @property
    def replacement_count(self) -> int:
        """Total number of unconsumed frames replaced/dropped by newer frames."""
        with self._lock:
            return self._replacement_count

    @property
    def last_capture_monotonic(self) -> Optional[float]:
        """Monotonic capture timestamp of the most recently popped frame."""
        with self._lock:
            return self._last_pop_capture_mono

    def put(
        self,
        frame: np.ndarray,
        timestamp: Optional[Union[float, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_step: Optional[str] = None,
        copy_frame: bool = False,
        capture_monotonic: Optional[float] = None,
    ) -> bool:
        """
        Submits a frame to the single-slot buffer (non-blocking).
        Replaces any unconsumed pending frame and records replacement count.
        """
        if frame is None or frame.size == 0:
            return False

        # Store frame reference or copy
        stored_frame = frame.copy() if copy_frame else frame

        if capture_monotonic is None:
            if metadata and isinstance(metadata, dict) and "capture_monotonic" in metadata:
                capture_monotonic = metadata["capture_monotonic"]
            else:
                capture_monotonic = time.monotonic()

        with self._condition:
            if self._pending_data is not None:
                self._replacement_count += 1
            self._submission_count += 1
            self._pending_data = (stored_frame, timestamp, metadata, expected_step)
            self._pending_capture_mono = capture_monotonic
            self._condition.notify()
            return True

    def get(self, timeout: Optional[float] = 0.5) -> Optional[Tuple[np.ndarray, Optional[Union[float, str]], Optional[Dict[str, Any]], Optional[str]]]:
        """
        Pops and returns the latest available frame data, blocking up to timeout.
        Returns None if no frame is available within timeout.
        """
        with self._condition:
            if self._pending_data is None:
                self._condition.wait(timeout=timeout)
            
            data = self._pending_data
            self._last_pop_capture_mono = self._pending_capture_mono
            self._pending_data = None
            self._pending_capture_mono = None
            return data

    def clear(self) -> None:
        """Clears any unconsumed pending frame from the buffer."""
        with self._condition:
            self._pending_data = None
            self._pending_capture_mono = None
            self._last_pop_capture_mono = None

    def reset_counts(self) -> None:
        """Resets submission and replacement counters to zero."""
        with self._lock:
            self._submission_count = 0
            self._replacement_count = 0

    def has_pending(self) -> bool:
        """Checks if a frame is currently waiting in the buffer."""
        with self._lock:
            return self._pending_data is not None

    def notify_all(self) -> None:
        """Wakes any sleeping threads (used on shutdown/stop)."""
        with self._condition:
            self._condition.notify_all()


class InferenceWorker:
    """
    Dedicated background inference worker executing the multimodal perception graph
    with monotonic telemetry, timing stats, and rate measurement.
    
    Architecture:
      Producer ➔ LatestFrameBuffer (single-slot) ➔ InferenceWorker Thread ➔ InferencePipeline ➔ Result Callback
    """

    def __init__(
        self,
        pipeline: Optional[InferencePipeline] = None,
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        recovery_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_result_queue_size: int = 10,
        copy_input_frames: bool = False,
        rate_strategy: Union[RateStrategy, str] = RateStrategy.OPPORTUNISTIC_LATEST,
        min_ai_interval_sec: Optional[float] = None,
        validator: Optional[Any] = None,
        recovery_manager: Optional[Any] = None,
        telemetry_aggregator: Optional[Any] = None,
    ):
        """
        Args:
            pipeline: Owned or injected InferencePipeline instance.
            result_callback: Optional thread-safe callback invoked on each public result.
            recovery_callback: Optional thread-safe callback invoked on structured recovery events.
            max_result_queue_size: Capacity of internal optional result queue.
            copy_input_frames: Whether to copy frames on submission.
            rate_strategy: Rate consumption strategy mode (default: OPPORTUNISTIC_LATEST).
            min_ai_interval_sec: Minimum processing interval for TIME_DECIMATED mode.
            validator: Optional authoritative SequenceValidatorFSM instance.
            recovery_manager: Optional procedural RecoveryManager instance.
            telemetry_aggregator: Optional TelemetryDiagnosticAggregator instance.
        """
        self.pipeline = pipeline or InferencePipeline(
            validator=validator,
            recovery_manager=recovery_manager,
            telemetry_aggregator=telemetry_aggregator,
        )
        self.validator = validator or getattr(self.pipeline, "validator", None)
        if self.validator is not None and getattr(self.pipeline, "validator", None) is None:
            self.pipeline.validator = self.validator
        if recovery_manager is not None and getattr(self.pipeline, "recovery_manager", None) is None:
            self.pipeline.recovery_manager = recovery_manager

        self.telemetry_aggregator = telemetry_aggregator or getattr(self.pipeline, "telemetry_aggregator", None)
        if self.telemetry_aggregator is not None and getattr(self.pipeline, "telemetry_aggregator", None) is None:
            self.pipeline.telemetry_aggregator = self.telemetry_aggregator

        self.result_callback = result_callback
        self.recovery_callback = recovery_callback
        self.copy_input_frames = copy_input_frames

        self._buffer = LatestFrameBuffer()
        self._result_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_result_queue_size)

        self._running = threading.Event()
        self._paused = threading.Event()
        self._pacing_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Strategy & Rate Matching State
        self._rate_strategy: RateStrategy = RateStrategy.OPPORTUNISTIC_LATEST
        self._min_ai_interval_sec: Optional[float] = None
        self._last_process_start: float = 0.0
        self.set_rate_strategy(rate_strategy, min_ai_interval_sec=min_ai_interval_sec)

        # Telemetry & Timing Tracking (Monotonic)
        self._telemetry_lock = threading.Lock()
        self._processed_count = 0
        self._dropped_pending_count = 0
        self._error_count = 0
        self._emitted_count = 0
        self._last_result: Optional[Dict[str, Any]] = None

        self._submission_times: collections.deque[float] = collections.deque(maxlen=200)
        self._processing_times: collections.deque[float] = collections.deque(maxlen=200)
        self._inference_durations_ms: collections.deque[float] = collections.deque(maxlen=200)
        self._frame_ages_ms: collections.deque[float] = collections.deque(maxlen=200)
        self._dispatch_latencies_ms: collections.deque[float] = collections.deque(maxlen=200)
        self._end_to_end_latencies_ms: collections.deque[float] = collections.deque(maxlen=200)

        self._latest_inference_duration_ms: Optional[float] = None
        self._latest_frame_age_ms: Optional[float] = None
        self._latest_end_to_end_ms: Optional[float] = None

    # -------------------------------------------------------------------------
    # Rate Strategy & Configuration API
    # -------------------------------------------------------------------------

    @property
    def rate_strategy(self) -> RateStrategy:
        """Current rate matching strategy mode."""
        with self._lock:
            return self._rate_strategy

    @property
    def min_ai_interval_sec(self) -> float:
        """Configured minimum AI processing interval in seconds (0.0 for opportunistic)."""
        with self._lock:
            if self._rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST:
                return 0.0
            elif self._rate_strategy == RateStrategy.FIXED_10FPS:
                return 0.100
            elif self._rate_strategy == RateStrategy.FIXED_15FPS:
                return 1.0 / 15.0
            elif self._rate_strategy == RateStrategy.TIME_DECIMATED:
                return self._min_ai_interval_sec if self._min_ai_interval_sec is not None else 0.100
            return 0.0

    @property
    def target_ai_fps(self) -> Optional[float]:
        """Target AI FPS based on strategy mode (None for opportunistic)."""
        with self._lock:
            if self._rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST:
                return None
            elif self._rate_strategy == RateStrategy.FIXED_10FPS:
                return 10.0
            elif self._rate_strategy == RateStrategy.FIXED_15FPS:
                return 15.0
            elif self._rate_strategy == RateStrategy.TIME_DECIMATED:
                interval = self._min_ai_interval_sec if self._min_ai_interval_sec is not None else 0.100
                return (1.0 / interval) if interval > 0.0 else None
            return None

    def set_rate_strategy(
        self,
        strategy: Union[RateStrategy, str],
        min_ai_interval_sec: Optional[float] = None,
        min_interval_sec: Optional[float] = None,
    ) -> None:
        """
        Configures the rate matching strategy mode and interval.

        Args:
            strategy: RateStrategy enum or string name ('OPPORTUNISTIC_LATEST', 'FIXED_10FPS', 'FIXED_15FPS', 'TIME_DECIMATED').
            min_ai_interval_sec: Custom interval in seconds for TIME_DECIMATED mode (must be > 0.0).
            min_interval_sec: Alias for min_ai_interval_sec.
        """
        interval = min_ai_interval_sec if min_ai_interval_sec is not None else min_interval_sec

        if isinstance(strategy, str):
            try:
                resolved_strategy = RateStrategy[strategy.upper()]
            except KeyError:
                valid_names = [s.value for s in RateStrategy]
                raise ValueError(
                    f"Invalid rate strategy '{strategy}'. Valid options are: {valid_names}"
                )
        elif isinstance(strategy, RateStrategy):
            resolved_strategy = strategy
        else:
            valid_names = [s.value for s in RateStrategy]
            raise ValueError(
                f"Invalid rate strategy type '{type(strategy)}'. Valid options are: {valid_names}"
            )

        if resolved_strategy == RateStrategy.TIME_DECIMATED:
            if interval is not None:
                if not isinstance(interval, (int, float)) or interval <= 0.0:
                    raise ValueError(f"min_ai_interval_sec must be a positive float, got {interval}")
            else:
                interval = 0.100

        with self._lock:
            self._rate_strategy = resolved_strategy
            self._min_ai_interval_sec = float(interval) if interval is not None else None
            self._pacing_event.set()
            self._pacing_event.clear()

        if getattr(self, "telemetry_aggregator", None) is not None:
            self.telemetry_aggregator.record_rate_config(
                strategy=self.rate_strategy.value,
                min_ai_interval_sec=self.min_ai_interval_sec,
                target_ai_fps=self.target_ai_fps,
            )

    # -------------------------------------------------------------------------
    # Lifecycle Management API
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the worker thread is active."""
        return self._running.is_set()

    @property
    def is_paused(self) -> bool:
        """True while processing is suspended."""
        return self._paused.is_set()

    @property
    def processed_count(self) -> int:
        """Total number of frames processed in this worker lifecycle."""
        with self._telemetry_lock:
            return self._processed_count

    @property
    def frames_processed(self) -> int:
        """Alias for processed_count."""
        return self.processed_count

    @property
    def frames_submitted(self) -> int:
        """Total number of frames submitted to this worker."""
        return self._buffer.submission_count

    @property
    def frames_replaced(self) -> int:
        """Total number of frames dropped/replaced in buffer prior to processing."""
        return self._buffer.replacement_count

    @property
    def results_emitted(self) -> int:
        """Total number of AI public results dispatched via callback."""
        with self._telemetry_lock:
            return self._emitted_count

    @property
    def error_count(self) -> int:
        """Total number of caught inference errors."""
        with self._telemetry_lock:
            return self._error_count

    def start(self) -> bool:
        """
        Starts the background worker thread. Idempotent.
        """
        with self._lock:
            if self._running.is_set() and self._thread is not None and self._thread.is_alive():
                return True

            self._running.set()
            self._paused.clear()
            self._pacing_event.clear()
            self._last_process_start = 0.0
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="AIInferenceWorker",
                daemon=True,
            )
            self._thread.start()
            if self.telemetry_aggregator is not None:
                self.telemetry_aggregator.start_session()
            logger.info("InferenceWorker started.")
            return True

    def stop(self, timeout: float = 5.0) -> bool:
        """
        Signals worker to terminate and cleanly joins the background thread.
        """
        with self._lock:
            if not self._running.is_set():
                return True

            self._running.clear()
            self._paused.clear()
            self._pacing_event.set()
            self._buffer.notify_all()

        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.stop_session()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            is_stopped = not self._thread.is_alive()
            self._thread = None
            logger.info("InferenceWorker stopped.")
            return is_stopped
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Performs idempotent complete resource shutdown.
        """
        self.stop(timeout=timeout)
        self.reset()
        with self._lock:
            self.pipeline.release()

    def pause(self) -> None:
        """
        Suspends inference execution while retaining worker resources.
        New incoming frames update the latest pending slot.
        """
        self._paused.set()
        self._pacing_event.set()
        self._buffer.notify_all()
        if hasattr(self.pipeline, "pause"):
            self.pipeline.pause()
        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.pause_session()
        logger.info("InferenceWorker paused.")

    def resume(self) -> None:
        """
        Resumes inference execution on the newest pending frame.
        """
        self._paused.clear()
        self._pacing_event.clear()
        self._buffer.notify_all()
        if hasattr(self.pipeline, "resume"):
            self.pipeline.resume()
        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.resume_session()
        logger.info("InferenceWorker resumed.")

    def reset(self) -> None:
        """
        Clears pending input frames, result queues, resets owned InferencePipeline state,
        and clears telemetry metrics.
        """
        self._buffer.clear()
        self._buffer.reset_counts()
        with self._lock:
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break
            self.pipeline.reset()
            self._last_result = None
            self._last_process_start = 0.0
            self._pacing_event.clear()

        with self._telemetry_lock:
            self._processed_count = 0
            self._dropped_pending_count = 0
            self._error_count = 0
            self._emitted_count = 0
            self._submission_times.clear()
            self._processing_times.clear()
            self._inference_durations_ms.clear()
            self._frame_ages_ms.clear()
            self._dispatch_latencies_ms.clear()
            self._end_to_end_latencies_ms.clear()
            self._latest_inference_duration_ms = None
            self._latest_frame_age_ms = None
            self._latest_end_to_end_ms = None

        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.reset()

        logger.info("InferenceWorker reset.")

    def reset_telemetry(self) -> None:
        """Resets telemetry counters and latency tracking deques."""
        self._buffer.reset_counts()
        with self._telemetry_lock:
            self._processed_count = 0
            self._dropped_pending_count = 0
            self._error_count = 0
            self._emitted_count = 0
            self._submission_times.clear()
            self._processing_times.clear()
            self._inference_durations_ms.clear()
            self._frame_ages_ms.clear()
            self._dispatch_latencies_ms.clear()
            self._end_to_end_latencies_ms.clear()
            self._latest_inference_duration_ms = None
            self._latest_frame_age_ms = None
            self._latest_end_to_end_ms = None

        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.reset()

    # -------------------------------------------------------------------------
    # Producer Submission API
    # -------------------------------------------------------------------------

    def submit_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[Union[float, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_step: Optional[str] = None,
    ) -> bool:
        """
        Non-blocking frame submission for upstream producers (e.g. Camera thread / VideoLoader).
        Attaches monotonic capture/submission timestamp if absent. Replaces any unconsumed pending frame.
        """
        if not self._running.is_set():
            return False

        t_submit = time.monotonic()
        cap_mono = None
        if metadata and isinstance(metadata, dict) and "capture_monotonic" in metadata:
            cap_mono = metadata["capture_monotonic"]
        else:
            cap_mono = t_submit

        with self._telemetry_lock:
            self._submission_times.append(t_submit)

        put_res = self._buffer.put(
            frame=frame,
            timestamp=timestamp,
            metadata=metadata,
            expected_step=expected_step,
            copy_frame=self.copy_input_frames,
            capture_monotonic=cap_mono,
        )

        if self.telemetry_aggregator is not None:
            self.telemetry_aggregator.record_frame_submission(t_submit)
            self.telemetry_aggregator.record_buffer_status(
                submission_count=self._buffer.submission_count,
                replacement_count=self._buffer.replacement_count,
                has_pending=self._buffer.has_pending(),
            )

        return put_res

    # -------------------------------------------------------------------------
    # Result Retrieval API
    # -------------------------------------------------------------------------

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent public AI contract result."""
        with self._lock:
            return self._last_result

    def pop_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Pops a result from the internal result queue."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # -------------------------------------------------------------------------
    # Telemetry & Statistics Computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_rolling_fps(timestamps: Sequence[float], window_sec: float = 2.0) -> float:
        """
        Computes rolling frequency (FPS) from a sequence of monotonic timestamps.
        Filters to items within window_sec of the newest timestamp.
        Returns 0.0 if fewer than 2 valid timestamps are present.
        """
        if not timestamps or len(timestamps) < 2:
            return 0.0
        t_latest = timestamps[-1]
        recent = [t for t in timestamps if (t_latest - t) <= window_sec]
        if len(recent) < 2:
            return 0.0
        duration = recent[-1] - recent[0]
        if duration <= 0.0:
            return 0.0
        return float((len(recent) - 1) / duration)

    @staticmethod
    def _compute_mean(values: Sequence[float]) -> float:
        """Returns mean of sequence or 0.0 if empty."""
        if not values or len(values) == 0:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _compute_p95(values: Sequence[float]) -> float:
        """Returns 95th percentile of sequence or 0.0 if empty."""
        if not values or len(values) == 0:
            return 0.0
        return float(np.percentile(values, 95))

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns full telemetry snapshot containing counters, rolling FPS, and latency metrics.
        All latencies and frame ages are reported in milliseconds (ms) using time.monotonic().
        """
        with self._telemetry_lock:
            sub_times = list(self._submission_times)
            proc_times = list(self._processing_times)
            infer_durs = list(self._inference_durations_ms)
            frame_ages = list(self._frame_ages_ms)
            disp_lats = list(self._dispatch_latencies_ms)
            e2e_lats = list(self._end_to_end_latencies_ms)
            proc_cnt = self._processed_count
            err_cnt = self._error_count
            emit_cnt = self._emitted_count
            last_infer = self._latest_inference_duration_ms
            last_age = self._latest_frame_age_ms
            last_e2e = self._latest_end_to_end_ms

        sub_cnt = self._buffer.submission_count
        rep_cnt = self._buffer.replacement_count

        ingest_fps = self._compute_rolling_fps(sub_times)
        ai_fps = self._compute_rolling_fps(proc_times)
        drop_rate = (rep_cnt / sub_cnt) if sub_cnt > 0 else 0.0

        return {
            "rate_strategy": self.rate_strategy.value,
            "min_ai_interval_sec": round(self.min_ai_interval_sec, 4) if self.min_ai_interval_sec > 0.0 else 0.0,
            "target_ai_fps": round(self.target_ai_fps, 2) if self.target_ai_fps is not None else None,
            "frames_submitted": sub_cnt,
            "frames_replaced": rep_cnt,
            "frames_processed": proc_cnt,
            "error_count": err_cnt,
            "results_emitted": emit_cnt,
            "camera_ingest_fps": round(ingest_fps, 2),
            "ai_processing_fps": round(ai_fps, 2),
            "frame_drop_rate": round(drop_rate, 4),
            "mean_inference_ms": round(self._compute_mean(infer_durs), 3),
            "p95_inference_ms": round(self._compute_p95(infer_durs), 3),
            "mean_frame_age_ms": round(self._compute_mean(frame_ages), 3),
            "p95_frame_age_ms": round(self._compute_p95(frame_ages), 3),
            "mean_dispatch_latency_ms": round(self._compute_mean(disp_lats), 3),
            "p95_dispatch_latency_ms": round(self._compute_p95(disp_lats), 3),
            "mean_end_to_end_ms": round(self._compute_mean(e2e_lats), 3),
            "p95_end_to_end_ms": round(self._compute_p95(e2e_lats), 3),
            "latest_inference_duration_ms": round(last_infer, 3) if last_infer is not None else None,
            "latest_frame_age_ms": round(last_age, 3) if last_age is not None else None,
            "latest_end_to_end_ms": round(last_e2e, 3) if last_e2e is not None else None,
            "target_fps": {
                "camera": 30.0,
                "display": 30.0,
                "ai_min": 10.0,
                "ai_max": 15.0,
                "target_ai_fps": round(self.target_ai_fps, 2) if self.target_ai_fps is not None else None,
            },
        }

    # -------------------------------------------------------------------------
    # Worker Execution Loop
    # -------------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Continuous background thread loop.
        Confines InferencePipeline execution to this single thread.
        Instruments monotonic timestamping across pickup, inference, and result dispatch.
        Enforces rate-matching pacing according to configured RateStrategy.
        """
        while self._running.is_set():
            # If paused, sleep briefly until resumed or stopped
            if self._paused.is_set():
                time.sleep(0.02)
                continue

            # Enforce strategy pacing interval before popping frame from buffer
            min_interval = self.min_ai_interval_sec
            if min_interval > 0.0 and self._last_process_start > 0.0:
                elapsed = time.monotonic() - self._last_process_start
                remaining = min_interval - elapsed
                if remaining > 0.0005:
                    self._pacing_event.wait(timeout=remaining)
                    if not self._running.is_set():
                        break
                    if self._paused.is_set():
                        continue

            frame_data = self._buffer.get(timeout=0.1)
            if frame_data is None:
                continue

            # Stage C: Worker Pickup Timestamp
            t_pickup = time.monotonic()
            capture_mono = self._buffer.last_capture_monotonic

            # If worker was paused while waiting for frame in get()
            if self._paused.is_set():
                with self._buffer._condition:
                    if self._buffer._pending_data is None:
                        self._buffer._pending_data = frame_data
                        self._buffer._pending_capture_mono = capture_mono
                time.sleep(0.02)
                continue

            frame, timestamp, metadata, expected_step = frame_data

            if capture_mono is None and metadata and isinstance(metadata, dict):
                capture_mono = metadata.get("capture_monotonic")

            frame_age_ms = max(0.0, (t_pickup - capture_mono) * 1000.0) if capture_mono is not None else None

            # Stage D: Inference Start Timestamp
            t_infer_start = time.monotonic()
            self._last_process_start = t_infer_start

            try:
                # Synchronously process frame through the full perception graph
                # and obtain frozen 8-field public AI contract dictionary
                public_result = self.pipeline.process_frame_public(
                    frame=frame,
                    annotate=True,
                    timestamp=timestamp,
                    metadata=metadata,
                    expected_step=expected_step,
                )

                # Stage E: Inference Completion Timestamp
                t_infer_end = time.monotonic()
                infer_dur_ms = max(0.0, (t_infer_end - t_infer_start) * 1000.0)

                with self._telemetry_lock:
                    self._processed_count += 1
                    self._processing_times.append(t_infer_end)
                    self._inference_durations_ms.append(infer_dur_ms)
                    self._latest_inference_duration_ms = infer_dur_ms
                    if frame_age_ms is not None:
                        self._frame_ages_ms.append(frame_age_ms)
                        self._latest_frame_age_ms = frame_age_ms

                with self._lock:
                    self._last_result = public_result

                # Put into bounded result queue (drop oldest if full)
                try:
                    self._result_queue.put_nowait(public_result)
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                        self._result_queue.put_nowait(public_result)
                    except (queue.Empty, queue.Full):
                        pass

                # Stage F1: Recovery Callback Dispatch (Out-of-band B12.2 channel)
                rec_event = getattr(self.pipeline, "last_recovery_event", None)
                if rec_event is not None and self.recovery_callback is not None:
                    try:
                        rec_dict = rec_event.to_dict() if hasattr(rec_event, "to_dict") else rec_event
                        self.recovery_callback(rec_dict)
                    except Exception as rec_cb_err:
                        logger.warning(f"InferenceWorker recovery callback exception: {rec_cb_err}")

                # Stage F2: Public-Result Callback Dispatch (Frozen 8-field contract)
                t_dispatch_start = time.monotonic()
                if self.result_callback is not None:
                    try:
                        self.result_callback(public_result)
                        with self._telemetry_lock:
                            self._emitted_count += 1
                    except Exception as cb_err:
                        logger.warning(f"InferenceWorker callback exception: {cb_err}")

                t_dispatch_end = time.monotonic()
                dispatch_ms = max(0.0, (t_dispatch_end - t_dispatch_start) * 1000.0)

                with self._telemetry_lock:
                    self._dispatch_latencies_ms.append(dispatch_ms)
                    if capture_mono is not None:
                        e2e_ms = max(0.0, (t_dispatch_end - capture_mono) * 1000.0)
                        self._end_to_end_latencies_ms.append(e2e_ms)
                        self._latest_end_to_end_ms = e2e_ms
                    else:
                        e2e_ms = None

                if self.telemetry_aggregator is not None:
                    self.telemetry_aggregator.record_worker_timings(
                        process_end_mono=t_infer_end,
                        infer_dur_ms=infer_dur_ms,
                        frame_age_ms=frame_age_ms,
                        dispatch_ms=dispatch_ms,
                        e2e_ms=e2e_ms,
                    )
                    self.telemetry_aggregator.record_buffer_status(
                        submission_count=self._buffer.submission_count,
                        replacement_count=self._buffer.replacement_count,
                        has_pending=self._buffer.has_pending(),
                    )
                    if self.result_callback is not None:
                        self.telemetry_aggregator.record_result_emitted()

            except Exception as err:
                t_infer_end = time.monotonic()
                with self._telemetry_lock:
                    self._error_count += 1

                if self.telemetry_aggregator is not None:
                    self.telemetry_aggregator.record_error()

                logger.error(f"InferenceWorker exception during frame execution: {err}")
                # Authoritative fallback
                fallback_exp_step = expected_step
                if fallback_exp_step is None and self.validator is not None:
                    curr_step = self.validator.get_current_expected_step()
                    fallback_exp_step = curr_step.step_id if curr_step else None

                fallback_result = AIResultAdapter.adapt(
                    action="IDLE",
                    confidence=0.0,
                    timestamp=timestamp,
                    expected_step=fallback_exp_step,
                    fsm_status="OUT_OF_SEQUENCE",
                )
                with self._lock:
                    self._last_result = fallback_result
                if self.result_callback is not None:
                    try:
                        self.result_callback(fallback_result)
                        with self._telemetry_lock:
                            self._emitted_count += 1
                        if self.telemetry_aggregator is not None:
                            self.telemetry_aggregator.record_result_emitted()
                    except Exception:
                        pass

    def get_telemetry_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Returns complete unified diagnostic and telemetry snapshot dictionary
        if telemetry aggregator is attached, otherwise None.
        """
        if self.telemetry_aggregator is not None:
            return self.telemetry_aggregator.get_snapshot_dict()
        return None

