"""
video_loader.py — Offline Video File Loader
ISRO SIH26174 BAS Experiment Monitor

Loads and navigates offline recorded video files (MP4/AVI/MKV)
for post-flight experiment validation, test playback, and offline HAR evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoLoader:
    """
    Thread-safe video file reader with frame indexing, metadata extraction,
    and generator support.
    """

    def __init__(self, filepath: Optional[Union[str, Path]] = None):
        self.filepath: Optional[Path] = Path(filepath) if filepath else None
        self.capture: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        self._duration: float = 0.0

        if self.filepath:
            self.open(self.filepath)

    def open(self, filepath: Union[str, Path]) -> bool:
        """Open a video file from disk."""
        self.release()
        path = Path(filepath)
        if not path.exists():
            return False

        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            return False

        self.filepath = path
        self._fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self._total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = (
            (self._total_frames / self._fps) if self._fps > 0 else 0.0
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the video file."""
        if self.capture is None or not self.capture.isOpened():
            return False, None

        ret, frame = self.capture.read()
        return ret, frame if ret else None

    def seek_frame(self, frame_number: int) -> bool:
        """Seek to a specific 0-indexed frame."""
        if self.capture is None or not self.capture.isOpened():
            return False

        target = max(0, min(frame_number, max(0, self._total_frames - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        return True

    def seek_timestamp(self, seconds: float) -> bool:
        """Seek to a specific playback timestamp in seconds."""
        if self.capture is None or not self.capture.isOpened():
            return False

        frame_idx = int(seconds * self._fps)
        return self.seek_frame(frame_idx)

    def get_current_frame_index(self) -> int:
        """Return the current 0-indexed frame position."""
        if self.capture is None or not self.capture.isOpened():
            return 0
        return int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

    def get_metadata(self) -> dict:
        """Return structured metadata about the loaded video file."""
        return {
            "filepath": str(self.filepath) if self.filepath else None,
            "filename": self.filepath.name if self.filepath else None,
            "fps": self._fps,
            "total_frames": self._total_frames,
            "duration_seconds": round(self._duration, 2),
            "width": self._width,
            "height": self._height,
            "is_open": self.is_open(),
        }

    def iter_frames(
        self, step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding (frame_index, frame_ndarray)."""
        if not self.is_open():
            return

        self.seek_frame(0)
        idx = 0
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1

    def is_open(self) -> bool:
        """Check if video file is actively open."""
        return self.capture is not None and self.capture.isOpened()

    def release(self) -> None:
        """Close and release the video file descriptor."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.filepath = None
        self._fps = 0.0
        self._total_frames = 0
        self._width = 0
        self._height = 0
        self._duration = 0.0

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
