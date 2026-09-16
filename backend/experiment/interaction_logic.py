"""
interaction_logic.py — Hand-Object Geometric & Spatial Interaction Engine
ISRO SIH26174 BAS Experiment Monitor

Analyzes spatial proximity, bounding box overlap (IoU), and 3D hand/joint vectors
relative to payload experiment objects to classify physical astronaut interactions
(e.g., APPROACHING, GRASPING, MANIPULATING, RELEASING).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class InteractionEngine:
    """
    Evaluates spatial relationships between detected hands and target objects.
    """

    def __init__(
        self,
        proximity_threshold: float = 0.15,  # normalized frame distance (0.0 to 1.0)
        iou_contact_threshold: float = 0.05,
    ):
        self.proximity_threshold = proximity_threshold
        self.iou_contact_threshold = iou_contact_threshold
        self._last_interaction_state: str = "IDLE"
        self._active_target_object: Optional[str] = None

    @staticmethod
    def compute_iou(boxA: Dict[str, float], boxB: Dict[str, float]) -> float:
        """Compute Intersection over Union between two normalized bounding boxes."""
        xA = max(boxA.get("x1", 0.0), boxB.get("x1", 0.0))
        yA = max(boxA.get("y1", 0.0), boxB.get("y1", 0.0))
        xB = min(boxA.get("x2", 0.0), boxB.get("x2", 0.0))
        yB = min(boxA.get("y2", 0.0), boxB.get("y2", 0.0))

        inter_width = max(0.0, xB - xA)
        inter_height = max(0.0, yB - yA)
        inter_area = inter_width * inter_height

        boxA_area = max(0.0, boxA.get("x2", 0.0) - boxA.get("x1", 0.0)) * max(
            0.0, boxA.get("y2", 0.0) - boxA.get("y1", 0.0)
        )
        boxB_area = max(0.0, boxB.get("x2", 0.0) - boxB.get("x1", 0.0)) * max(
            0.0, boxB.get("y2", 0.0) - boxB.get("y1", 0.0)
        )

        union_area = boxA_area + boxB_area - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def compute_centroid_distance(
        ptA: Tuple[float, float], ptB: Tuple[float, float]
    ) -> float:
        """Normalized Euclidean distance between two 2D points."""
        return math.hypot(ptA[0] - ptB[0], ptA[1] - ptB[1])

    def evaluate_interaction(
        self,
        hand_boxes: List[Dict[str, Any]],
        object_boxes: List[Dict[str, Any]],
        hand_landmarks: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates interaction dynamics between hands and objects.
        Returns state: 'IDLE', 'APPROACHING', 'HOLDING', 'OPERATING'
        """
        if not hand_boxes or not object_boxes:
            self._last_interaction_state = "IDLE"
            self._active_target_object = None
            return {
                "state": "IDLE",
                "target_object": None,
                "confidence": 0.0,
                "closest_distance": 1.0,
                "max_iou": 0.0,
            }

        closest_distance = 1.0
        max_iou = 0.0
        active_obj = None

        for hand in hand_boxes:
            hx = (hand.get("x1", 0.0) + hand.get("x2", 0.0)) / 2.0
            hy = (hand.get("y1", 0.0) + hand.get("y2", 0.0)) / 2.0

            for obj in object_boxes:
                ox = (obj.get("x1", 0.0) + obj.get("x2", 0.0)) / 2.0
                oy = (obj.get("y1", 0.0) + obj.get("y2", 0.0)) / 2.0

                dist = self.compute_centroid_distance((hx, hy), (ox, oy))
                iou = self.compute_iou(hand, obj)

                if iou > max_iou:
                    max_iou = iou
                    active_obj = obj.get("label", "OBJECT")

                if dist < closest_distance:
                    closest_distance = dist
                    if max_iou <= 0:
                        active_obj = obj.get("label", "OBJECT")

        # Classify state
        if max_iou >= self.iou_contact_threshold or closest_distance < (self.proximity_threshold * 0.5):
            state = "HOLDING"
            confidence = min(0.99, max(0.70, 0.70 + (max_iou * 0.3)))
        elif closest_distance <= self.proximity_threshold:
            state = "APPROACHING"
            confidence = 0.80
        else:
            state = "IDLE"
            confidence = 0.90
            active_obj = None

        self._last_interaction_state = state
        self._active_target_object = active_obj

        return {
            "state": state,
            "target_object": active_obj,
            "confidence": round(confidence, 2),
            "closest_distance": round(closest_distance, 3),
            "max_iou": round(max_iou, 3),
        }
