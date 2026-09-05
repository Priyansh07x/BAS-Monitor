"""
VideoRecorder — Thread-safe local video storage for BAS experiments.

Captures frames from OpenCV and writes them to MP4 files on a background
thread so that the AI inference pipeline is never blocked by disk I/O.

File naming convention:  REC_YYYYMMDD_HHMMSS.mp4
Output directory:        data/videos/

Architecture (from GUIDE.md §47):
    Camera → Local storage → REC_20260905_072800.mp4
"""

from __future__ import annotations

import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_VIDEO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "videos"


class VideoRecorder:
    """
    Non-blocking video recorder that writes frames on a background thread.

    Usage:
        recorder = VideoRecorder()
        recorder.start_recording(fps=30.0, frame_size=(1280, 720))
        # ... in your capture loop ...
        recorder.enqueue_frame(frame)
        # ... when done ...
        filepath = recorder.stop_recording()
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        codec: str = "avc1",
        max_queue_size: int = 300,
    ):
        """
        Args:
            output_dir:      Where to save MP4 files.  Defaults to data/videos/.
            codec:           FourCC codec string (default: mp4v).
            max_queue_size:  Maximum buffered frames before dropping oldest.
        """
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_VIDEO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.codec = codec
        self.max_queue_size = max_queue_size

        # ---- internal state ----
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._recording = threading.Event()
        self._lock = threading.Lock()

        self._current_path: Optional[Path] = None
        self._frame_count: int = 0
        self._start_time: Optional[float] = None
        self._fps: float = 30.0
        self._frame_size: tuple[int, int] = (1280, 720)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """True while a recording session is active."""
        return self._recording.is_set()

    @property
    def current_filepath(self) -> Optional[Path]:
        """Path of the file currently being written, or None."""
        return self._current_path

    @property
    def frame_count(self) -> int:
        """Number of frames written in the current (or last) session."""
        return self._frame_count

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since recording started (0.0 if idle)."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def start_recording(
        self,
        fps: float = 30.0,
        frame_size: tuple[int, int] = (1280, 720),
        filename: Optional[str] = None,
    ) -> Path:
        """
        Begin a new recording session.

        Args:
            fps:         Target frames-per-second for the output file.
            frame_size:  (width, height) of the frames that will be enqueued.
            filename:    Override the auto-generated filename (optional).

        Returns:
            Path to the MP4 file that will be created.

        Raises:
            RuntimeError: If a recording is already in progress.
        """
        with self._lock:
            if self._recording.is_set():
                raise RuntimeError(
                    "Recording already in progress. Call stop_recording() first."
                )

            self._fps = fps
            self._frame_size = frame_size
            self._frame_count = 0
            self._start_time = time.time()

            # Generate filename: REC_YYYYMMDD_HHMMSS.mp4
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"REC_{timestamp}.mp4"

            self._current_path = self.output_dir / filename

            # Create the VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self._current_path),
                fourcc,
                self._fps,
                self._frame_size,
            )

            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(
                    f"Failed to open VideoWriter for {self._current_path}"
                )

            # Clear the queue of any stale frames
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            # Start background writer thread
            self._recording.set()
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="VideoRecorderWorker",
                daemon=True,
            )
            self._thread.start()

            return self._current_path

    def enqueue_frame(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the write queue (non-blocking).

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            True if the frame was enqueued, False if dropped (queue full)
            or not currently recording.
        """
        if not self._recording.is_set():
            return False

        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            # Drop the frame rather than blocking the inference pipeline
            return False

    def stop_recording(self) -> dict:
        """
        Stop the current recording and finalize the MP4 file.

        Returns:
            A summary dict with filepath, frame_count, duration, and file_size.
        """
        with self._lock:
            if not self._recording.is_set():
                return {"status": "no_active_recording"}

            # Signal the writer thread to stop
            self._recording.clear()

            # Send a sentinel (None) so the thread wakes up if blocked on get()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

        # Wait for the writer thread to finish flushing
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        # Release the writer
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        elapsed = time.time() - self._start_time if self._start_time else 0.0
        file_size = (
            self._current_path.stat().st_size
            if self._current_path and self._current_path.exists()
            else 0
        )

        summary = {
            "status": "saved",
            "filepath": self._current_path,
            "frame_count": self._frame_count,
            "duration_seconds": round(elapsed, 2),
            "file_size_bytes": file_size,
            "fps": self._fps,
        }

        self._start_time = None
        return summary

    def shutdown(self):
        """Stop any active recording and clean up resources."""
        if self._recording.is_set():
            self.stop_recording()

    # ------------------------------------------------------------------
    # Background writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self):
        """
        Continuously pop frames from the queue and write to disk.
        Runs on a daemon thread until stop_recording() is called.
        """
        while self._recording.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Sentinel value — time to exit
            if frame is None:
                break

            # Resize if the frame doesn't match the expected size
            h, w = frame.shape[:2]
            if (w, h) != self._frame_size:
                frame = cv2.resize(frame, self._frame_size)

            if self._writer is not None:
                self._writer.write(frame)
                self._frame_count += 1


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_recorder_instance: Optional[VideoRecorder] = None
_recorder_lock = threading.Lock()


def get_video_recorder(**kwargs) -> VideoRecorder:
    """Return or create a module-level VideoRecorder singleton."""
    global _recorder_instance
    with _recorder_lock:
        if _recorder_instance is None:
            _recorder_instance = VideoRecorder(**kwargs)
        return _recorder_instance
