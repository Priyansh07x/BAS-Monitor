"""
Unit tests for backend.video.recorder — VideoRecorder.
"""

import re
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.video.recorder import VideoRecorder


class TestVideoRecorder(unittest.TestCase):
    """Tests for the VideoRecorder class."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.recorder = VideoRecorder(output_dir=self.tmp_dir)

    def tearDown(self):
        self.recorder.shutdown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Test 1: Basic record / stop lifecycle produces a valid MP4
    # ------------------------------------------------------------------ #
    def test_record_and_stop_creates_valid_mp4(self):
        """Recording and stopping should create a non-empty MP4 file."""
        frame_size = (640, 480)
        fps = 15.0

        filepath = self.recorder.start_recording(fps=fps, frame_size=frame_size)
        self.assertTrue(self.recorder.is_recording)
        self.assertIsNotNone(filepath)

        # Write 30 dummy frames (2 seconds at 15fps)
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            enqueued = self.recorder.enqueue_frame(frame)
            self.assertTrue(enqueued)

        summary = self.recorder.stop_recording()

        self.assertEqual(summary["status"], "saved")
        self.assertEqual(summary["frame_count"], 30)
        self.assertGreater(summary["file_size_bytes"], 0)
        self.assertTrue(filepath.exists())

        # Verify the file is a valid video with OpenCV
        cap = cv2.VideoCapture(str(filepath))
        self.assertTrue(cap.isOpened())
        read_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            read_count += 1
        cap.release()
        self.assertEqual(read_count, 30)

    # ------------------------------------------------------------------ #
    # Test 2: Filename convention is REC_YYYYMMDD_HHMMSS.mp4
    # ------------------------------------------------------------------ #
    def test_filename_convention(self):
        """Auto-generated filename should match REC_YYYYMMDD_HHMMSS.mp4."""
        filepath = self.recorder.start_recording(fps=10.0, frame_size=(320, 240))
        self.recorder.stop_recording()

        pattern = r"^REC_\d{8}_\d{6}\.mp4$"
        self.assertRegex(filepath.name, pattern)

    # ------------------------------------------------------------------ #
    # Test 3: Cannot start two recordings simultaneously
    # ------------------------------------------------------------------ #
    def test_double_start_raises(self):
        """Starting a second recording without stopping should raise."""
        self.recorder.start_recording(fps=10.0, frame_size=(320, 240))
        with self.assertRaises(RuntimeError):
            self.recorder.start_recording(fps=10.0, frame_size=(320, 240))
        self.recorder.stop_recording()

    # ------------------------------------------------------------------ #
    # Test 4: enqueue_frame returns False when not recording
    # ------------------------------------------------------------------ #
    def test_enqueue_when_not_recording(self):
        """Enqueuing a frame when not recording should return False."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.recorder.enqueue_frame(frame)
        self.assertFalse(result)

    # ------------------------------------------------------------------ #
    # Test 5: Custom filename override works
    # ------------------------------------------------------------------ #
    def test_custom_filename(self):
        """Passing a custom filename should override the default."""
        filepath = self.recorder.start_recording(
            fps=10.0, frame_size=(320, 240), filename="my_custom_video.mp4"
        )
        self.recorder.stop_recording()
        self.assertEqual(filepath.name, "my_custom_video.mp4")

    # ------------------------------------------------------------------ #
    # Test 6: stop_recording when idle returns no_active_recording
    # ------------------------------------------------------------------ #
    def test_stop_when_not_recording(self):
        """Stopping when no recording is active should return a safe dict."""
        result = self.recorder.stop_recording()
        self.assertEqual(result["status"], "no_active_recording")


if __name__ == "__main__":
    unittest.main()
