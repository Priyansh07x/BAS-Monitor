"""Backend video subsystem — camera capture and local recording."""

from backend.video.recorder import VideoRecorder, get_video_recorder

__all__ = ["VideoRecorder", "get_video_recorder"]
