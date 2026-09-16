"""
action_recognizer.py — Action Recognition Module Interface
ISRO SIH26174 BAS Experiment Monitor

Provides the ActionRecognizer interface for temporal action classification,
wrapping the 1D-TCN ActionClassifier engine.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np

from .action_classifier import ActionClassifier


class ActionRecognizer:
    """
    Action recognition facade conforming to GUIDE.md §65 and test suites.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        window_size: int = 30,
        actions_list: Optional[List[str]] = None,
    ):
        self.classifier = ActionClassifier(
            model_path=model_path,
            window_size=window_size,
            actions_list=actions_list,
        )

    def recognize(
        self,
        keypoint_vector: np.ndarray,
        interaction_state: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Feed a vector and return the classified action and confidence score."""
        self.classifier.push_frame_vector(keypoint_vector)
        return self.classifier.classify(interaction_state=interaction_state)

    def reset(self) -> None:
        self.classifier.reset_buffer()
