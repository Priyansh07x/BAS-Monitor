"""
interaction_logic.py — Orientation-Robust Hand-Object Spatial Interaction Engine
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B4.1b)

Analyzes spatial proximity, scale-normalized contact distances, bounding box overlap (IoU),
and optional 3D/2D hand landmark vectors relative to experiment objects to classify physical
astronaut interactions (e.g., IDLE, APPROACHING, HOLDING, OPERATING).

Orientation-Robust Enhancements (Gate B4.1b):
- Uses 2D Euclidean centroid distance as primary orientation-invariant spatial evidence.
- Normalizes distance by combined hand + object characteristic scale (radii) to ensure
  invariance under diagonal rotations (45, 135, 225 deg) and camera distance scaling.
- Retains standard IoU as secondary evidence for backward compatibility on upright scenes.
- Utilizes landmark keypoints when provided to measure direct fingertip-to-object proximity.
- Conservative fail-safe policy: ambiguous geometry defaults to APPROACHING / IDLE.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class InteractionEngine:
    """
    Evaluates spatial relationships between detected hands and target objects
    with orientation-tolerant and scale-aware geometric criteria.
    """

    def __init__(
        self,
        proximity_threshold: float = 0.15,  # Normalized frame distance threshold for APPROACHING
        iou_contact_threshold: float = 0.05,  # Standard IoU contact threshold for upright boxes
        contact_scale_threshold: float = 0.85,  # PROPOSED DEFAULT — scale-normalized contact ratio for HOLDING
        reach_scale_threshold: float = 2.00,  # PROPOSED DEFAULT — scale-normalized reach ratio for APPROACHING
    ):
        self.proximity_threshold = proximity_threshold
        self.iou_contact_threshold = iou_contact_threshold
        self.contact_scale_threshold = contact_scale_threshold
        self.reach_scale_threshold = reach_scale_threshold

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
        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def compute_centroid_distance(
        ptA: Tuple[float, float], ptB: Tuple[float, float]
    ) -> float:
        """
        Calculates normalized Euclidean distance between two 2D points.
        This Euclidean distance is mathematically invariant under 2D rotations in R^2.
        """
        return math.hypot(ptA[0] - ptB[0], ptA[1] - ptB[1])

    @staticmethod
    def compute_characteristic_radius(box: Dict[str, Any]) -> float:
        """
        Calculates the half-diagonal characteristic radius of a normalized bounding box.
        Represents the physical reach envelope from the box centroid.
        """
        w = max(0.0, float(box.get("x2", 0.0)) - float(box.get("x1", 0.0)))
        h = max(0.0, float(box.get("y2", 0.0)) - float(box.get("y1", 0.0)))
        return 0.5 * math.hypot(w, h)

    def evaluate_interaction(
        self,
        hand_boxes: List[Dict[str, Any]],
        object_boxes: List[Dict[str, Any]],
        hand_landmarks: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates spatial interaction dynamics between detected hands and objects.

        Returns state:
          - 'IDLE': No hand near target objects
          - 'APPROACHING': Hand within reach envelope of an object
          - 'HOLDING': Hand in physical contact / grasping scale of an object
          - 'OPERATING': Reserved for dual-hand tool manipulation sequences

        Backward compatible return dictionary:
          {
            "state": str,
            "target_object": Optional[str],
            "confidence": float,
            "closest_distance": float,
            "max_iou": float,
            "normalized_scale_proximity": float
          }
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
                "normalized_scale_proximity": 1.0,
            }

        closest_distance = 1.0
        min_scale_proximity = 999.0
        min_edge_distance = 1.0
        max_iou = 0.0
        active_obj = None

        for hand in hand_boxes:
            hx = (float(hand.get("x1", 0.0)) + float(hand.get("x2", 0.0))) / 2.0
            hy = (float(hand.get("y1", 0.0)) + float(hand.get("y2", 0.0))) / 2.0
            r_hand = self.compute_characteristic_radius(hand)

            for obj in object_boxes:
                ox = (float(obj.get("x1", 0.0)) + float(obj.get("x2", 0.0))) / 2.0
                oy = (float(obj.get("y1", 0.0)) + float(obj.get("y2", 0.0))) / 2.0
                r_obj = self.compute_characteristic_radius(obj)

                # 1. Primary Evidence: Rotation-invariant Euclidean centroid distance
                dist = self.compute_centroid_distance((hx, hy), (ox, oy))

                # 2. Scale-normalized contact ratio: distance / combined reach radii
                combined_radius = max(0.01, r_hand + r_obj)
                scale_proximity = dist / combined_radius
                edge_dist = max(0.0, dist - combined_radius)

                # 3. Secondary Evidence: Axis-aligned IoU overlap
                iou = self.compute_iou(hand, obj)

                # 4. Landmark refinement (if detailed finger landmarks provided)
                if hand_landmarks:
                    for lm in hand_landmarks:
                        lx = float(lm.get("x", hx))
                        ly = float(lm.get("y", hy))
                        lm_dist = self.compute_centroid_distance((lx, ly), (ox, oy))
                        if lm_dist < dist:
                            dist = lm_dist
                            scale_proximity = min(scale_proximity, lm_dist / max(0.01, r_obj))
                            edge_dist = max(0.0, lm_dist - r_obj)

                if iou > max_iou:
                    max_iou = iou
                    active_obj = obj.get("label", "OBJECT")

                if scale_proximity < min_scale_proximity:
                    min_scale_proximity = scale_proximity
                    if max_iou <= 0.0:
                        active_obj = obj.get("label", "OBJECT")

                if dist < closest_distance:
                    closest_distance = dist
                    if max_iou <= 0.0 and min_scale_proximity >= 1.0:
                        active_obj = obj.get("label", "OBJECT")

                if edge_dist < min_edge_distance:
                    min_edge_distance = edge_dist

        # -------------------------------------------------------------------
        # Hybrid Decision Logic (Conservative & Orientation-Tolerant)
        # -------------------------------------------------------------------
        is_holding = (
            # Criterion A: Scale-normalized contact (rotation and scale invariant)
            (min_scale_proximity <= self.contact_scale_threshold)
            # Criterion B: Traditional absolute distance contact
            or (closest_distance < (self.proximity_threshold * 0.5))
            # Criterion C: Axis-aligned IoU contact (backward compatible)
            or (max_iou >= self.iou_contact_threshold)
        )

        is_approaching = (
            # Criterion A: Scale-normalized reach envelope
            (min_scale_proximity <= self.reach_scale_threshold)
            # Criterion B: Boundary edge proximity
            or (min_edge_distance <= self.proximity_threshold)
            # Criterion C: Absolute centroid distance threshold
            or (closest_distance <= self.proximity_threshold)
        )

        if is_holding:
            state = "HOLDING"
            # Smooth confidence estimation bounded in [0.70, 0.99]
            conf_scale = max(0.0, 1.0 - (min_scale_proximity / max(0.01, self.contact_scale_threshold)))
            conf_iou = max_iou * 0.30
            confidence = min(0.99, max(0.70, 0.70 + max(conf_scale * 0.25, conf_iou)))
        elif is_approaching:
            state = "APPROACHING"
            confidence = 0.80
        else:
            state = "IDLE"
            confidence = 0.90
            active_obj = None

        # -------------------------------------------------------------------
        # Secondary Container Spatial Evaluation (Gate B6.2)
        # -------------------------------------------------------------------
        container_box = next((b for b in object_boxes if b.get("label") == "SAMPLE_CONTAINER"), None)
        container_dist = 1.0
        container_iou = 0.0
        container_scale_prox = 999.0
        if container_box and hand_boxes:
            cx = (float(container_box.get("x1", 0.0)) + float(container_box.get("x2", 0.0))) / 2.0
            cy = (float(container_box.get("y1", 0.0)) + float(container_box.get("y2", 0.0))) / 2.0
            r_c = self.compute_characteristic_radius(container_box)
            for hand in hand_boxes:
                hx = (float(hand.get("x1", 0.0)) + float(hand.get("x2", 0.0))) / 2.0
                hy = (float(hand.get("y1", 0.0)) + float(hand.get("y2", 0.0))) / 2.0
                r_h = self.compute_characteristic_radius(hand)
                d = self.compute_centroid_distance((hx, hy), (cx, cy))
                if d < container_dist:
                    container_dist = d
                comb_r = max(0.01, r_h + r_c)
                sp = d / comb_r
                if sp < container_scale_prox:
                    container_scale_prox = sp
                iou = self.compute_iou(hand, container_box)
                if iou > container_iou:
                    container_iou = iou

        self._last_interaction_state = state
        self._active_target_object = active_obj

        return {
            "state": state,
            "target_object": active_obj,
            "confidence": round(confidence, 2),
            "closest_distance": round(closest_distance, 3),
            "max_iou": round(max_iou, 3),
            "normalized_scale_proximity": round(min_scale_proximity, 3),
            "container_distance": round(container_dist, 3) if container_box else None,
            "container_iou": round(container_iou, 3) if container_box else 0.0,
            "container_scale_proximity": round(container_scale_prox, 3) if container_box else None,
        }

    def reset(self) -> None:
        """Reset internal temporal tracking state."""
        self._last_interaction_state = "IDLE"
        self._active_target_object = None
