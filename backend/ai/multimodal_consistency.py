"""
multimodal_consistency.py — Deterministic Multimodal Consistency Evaluator Core
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B11.2.1)

Evaluates cross-modal consistency across Action Classification, Object Detection,
and Spatial Interaction Dynamics for a single observation frame.

Answers:
  "Are the available perception modalities mutually consistent enough to trust this observation?"
Does NOT answer:
  "Is this the correct procedure step?" (SequenceValidatorFSM sole authority).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .result_adapter import (
    CANONICAL_ACTIONS,
    CANONICAL_OBJECTS,
    normalize_action,
    normalize_confidence,
    normalize_object,
)


# Canonical EXP-001 Action-to-Object semantic mapping
CANONICAL_ACTION_OBJECT_MAP: Dict[str, Set[str]] = {
    "PICK_RED": {"RED_SAMPLE"},
    "PLACE_RED": {"RED_SAMPLE"},
    "PICK_BLUE": {"BLUE_SAMPLE"},
    "PLACE_BLUE": {"BLUE_SAMPLE"},
    "CLOSE_LID": {"CONTAINER_LID"},
}

# Physical manipulation actions requiring spatial proximity/contact
MANIPULATION_ACTIONS: Set[str] = {
    "PICK_RED",
    "PLACE_RED",
    "PICK_BLUE",
    "PLACE_BLUE",
    "CLOSE_LID",
}

# Internal consistency categories
CATEGORY_RELIABLE_ALIGNED = "RELIABLE_ALIGNED"
CATEGORY_CROSS_MODAL_CONFLICT = "CROSS_MODAL_CONFLICT"
CATEGORY_UNCERTAIN_MISSING_OBJECT = "UNCERTAIN_MISSING_OBJECT"
CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE = "UNCERTAIN_LOW_OBJECT_CONFIDENCE"
CATEGORY_UNCERTAIN_STATIC_OBJECT = "UNCERTAIN_STATIC_OBJECT"
CATEGORY_INSTABILITY_OBJECT_FLICKER = "INSTABILITY_OBJECT_FLICKER"
CATEGORY_INSTABILITY_ACTION_FLICKER = "INSTABILITY_ACTION_FLICKER"
CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION = "UNCERTAIN_SPATIAL_CONTRADICTION"
CATEGORY_IDLE = "IDLE"

VALID_CATEGORIES: Set[str] = {
    CATEGORY_RELIABLE_ALIGNED,
    CATEGORY_CROSS_MODAL_CONFLICT,
    CATEGORY_UNCERTAIN_MISSING_OBJECT,
    CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE,
    CATEGORY_UNCERTAIN_STATIC_OBJECT,
    CATEGORY_INSTABILITY_OBJECT_FLICKER,
    CATEGORY_INSTABILITY_ACTION_FLICKER,
    CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION,
    CATEGORY_IDLE,
}


class MultimodalConsistencyEvaluator:
    """
    Deterministic, stateless evaluator for cross-modal perception consistency.

    Guarantees:
    1. Pure Python with zero Qt, threading, queue, or FSM dependencies.
    2. Completely stateless per observation.
    3. Evaluates semantic alignment between action and object.
    4. Evaluates spatial plausibility (geometric veto) using InteractionEngine metrics.
    5. Returns structured internal evaluation dictionaries without mutating any procedural state.
    """

    def __init__(
        self,
        action_confidence_threshold: float = 0.70,
        object_confidence_threshold: float = 0.50,
        max_reach_scale_threshold: float = 1.50,
        max_distance_threshold: float = 0.30,
    ):
        """
        Args:
            action_confidence_threshold: Minimum confidence for an action to be considered active.
            object_confidence_threshold: Minimum confidence for an object detection to be considered valid.
            max_reach_scale_threshold: Maximum normalized scale proximity for a manipulation action
                                       to be considered plausible when interaction state is IDLE (B11.2 tuning param).
            max_distance_threshold: Maximum 2D Euclidean centroid distance if scale proximity is absent.
        """
        if action_confidence_threshold <= 0.0 or action_confidence_threshold > 1.0:
            raise ValueError(f"action_confidence_threshold must be in (0.0, 1.0], got {action_confidence_threshold}")
        if object_confidence_threshold <= 0.0 or object_confidence_threshold > 1.0:
            raise ValueError(f"object_confidence_threshold must be in (0.0, 1.0], got {object_confidence_threshold}")
        if max_reach_scale_threshold <= 0.0:
            raise ValueError(f"max_reach_scale_threshold must be > 0.0, got {max_reach_scale_threshold}")
        if max_distance_threshold <= 0.0:
            raise ValueError(f"max_distance_threshold must be > 0.0, got {max_distance_threshold}")

        self._action_conf_thresh = float(action_confidence_threshold)
        self._obj_conf_thresh = float(object_confidence_threshold)
        self._max_reach_scale = float(max_reach_scale_threshold)
        self._max_distance = float(max_distance_threshold)

    @property
    def action_confidence_threshold(self) -> float:
        return self._action_conf_thresh

    @property
    def object_confidence_threshold(self) -> float:
        return self._obj_conf_thresh

    @property
    def max_reach_scale_threshold(self) -> float:
        return self._max_reach_scale

    @property
    def max_distance_threshold(self) -> float:
        return self._max_distance

    def evaluate(
        self,
        action: Optional[str],
        action_confidence: Optional[float] = None,
        object_name: Optional[str] = None,
        object_confidence: Optional[float] = None,
        interaction: Optional[Dict[str, Any]] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
        prior_action: Optional[str] = None,
        prior_object: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the multimodal coherence of a single frame observation.

        Args:
            action: Detected action label.
            action_confidence: Confidence score of detected action in [0.0, 1.0].
            object_name: Detected object label.
            object_confidence: Confidence score of detected object in [0.0, 1.0].
            interaction: Spatial interaction dict from InteractionEngine.
            objects: Full list of detected objects from ObjectDetector.
            prior_action: Optional prior action label (for flicker diagnostic).
            prior_object: Optional prior object label (for flicker diagnostic).

        Returns:
            Dict containing:
                "category": str,
                "is_reliable": bool,
                "action": str,
                "object": str,
                "action_confidence": float,
                "object_confidence": float,
                "semantic_consistent": bool,
                "spatially_plausible": bool,
                "reason": str,
                "details": Dict[str, Any],
        """
        norm_act = normalize_action(action)
        act_conf = normalize_confidence(action_confidence)

        # Resolve raw object and confidence
        raw_obj_name = object_name
        resolved_obj_conf = object_confidence

        # If object confidence not explicitly provided, try resolving from objects list or interaction
        if resolved_obj_conf is None and objects and raw_obj_name:
            for o in objects:
                if str(o.get("label", "")).strip().upper() == str(raw_obj_name).strip().upper():
                    resolved_obj_conf = o.get("confidence")
                    break

        if resolved_obj_conf is None and interaction is not None:
            if interaction.get("target_object") == raw_obj_name:
                resolved_obj_conf = interaction.get("confidence")

        obj_conf = normalize_confidence(resolved_obj_conf) if resolved_obj_conf is not None else 1.0

        # Normalization of object string
        norm_obj = str(raw_obj_name).strip().upper() if raw_obj_name else "NONE"
        is_known_object = norm_obj in CANONICAL_OBJECTS and norm_obj != "NONE"

        # ---------------------------------------------------------------------
        # 1. Neutral / IDLE Observation
        # ---------------------------------------------------------------------
        if norm_act == "IDLE" and (not is_known_object or norm_obj == "NONE"):
            return {
                "category": CATEGORY_IDLE,
                "is_reliable": False,
                "action": "IDLE",
                "object": "NONE",
                "action_confidence": act_conf,
                "object_confidence": 0.0,
                "semantic_consistent": True,
                "spatially_plausible": True,
                "reason": "Neutral observation (no active action or target object).",
                "details": {"interaction_state": (interaction or {}).get("state", "IDLE")},
            }

        # ---------------------------------------------------------------------
        # 2. Static Object Presence (Action IDLE / Low Conf, but Object Present)
        # ---------------------------------------------------------------------
        if norm_act == "IDLE" and is_known_object:
            return {
                "category": CATEGORY_UNCERTAIN_STATIC_OBJECT,
                "is_reliable": False,
                "action": "IDLE",
                "object": norm_obj,
                "action_confidence": act_conf,
                "object_confidence": obj_conf,
                "semantic_consistent": True,
                "spatially_plausible": True,
                "reason": f"Passive static object '{norm_obj}' detected without active manipulation action.",
                "details": {
                    "object_confidence": obj_conf,
                    "interaction_state": (interaction or {}).get("state", "IDLE"),
                },
            }

        # ---------------------------------------------------------------------
        # 3. Manipulation Action with Missing / Unknown Object
        # ---------------------------------------------------------------------
        if norm_act in MANIPULATION_ACTIONS and (not is_known_object or norm_obj == "NONE"):
            return {
                "category": CATEGORY_UNCERTAIN_MISSING_OBJECT,
                "is_reliable": False,
                "action": norm_act,
                "object": "NONE",
                "action_confidence": act_conf,
                "object_confidence": 0.0,
                "semantic_consistent": False,
                "spatially_plausible": True,
                "reason": f"Action '{norm_act}' requires object evidence, but no valid object was detected.",
                "details": {
                    "raw_object": raw_obj_name,
                    "required_objects": list(CANONICAL_ACTION_OBJECT_MAP.get(norm_act, set())),
                },
            }

        # ---------------------------------------------------------------------
        # 4. Semantic Action / Object Conflict (Cross-Modal Mismatch)
        # ---------------------------------------------------------------------
        expected_objects = CANONICAL_ACTION_OBJECT_MAP.get(norm_act, set())
        if norm_act in MANIPULATION_ACTIONS and norm_obj not in expected_objects:
            return {
                "category": CATEGORY_CROSS_MODAL_CONFLICT,
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "action_confidence": act_conf,
                "object_confidence": obj_conf,
                "semantic_consistent": False,
                "spatially_plausible": True,
                "reason": (
                    f"Cross-modal semantic conflict: Action '{norm_act}' requires object in "
                    f"{sorted(list(expected_objects))}, but detected object was '{norm_obj}'."
                ),
                "details": {
                    "expected_objects": sorted(list(expected_objects)),
                    "detected_object": norm_obj,
                },
            }

        # ---------------------------------------------------------------------
        # 5. Low Object Confidence (< threshold)
        # ---------------------------------------------------------------------
        if resolved_obj_conf is not None and obj_conf < self._obj_conf_thresh:
            return {
                "category": CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE,
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "action_confidence": act_conf,
                "object_confidence": obj_conf,
                "semantic_consistent": True,
                "spatially_plausible": True,
                "reason": (
                    f"Object '{norm_obj}' detection confidence ({obj_conf:.2f}) is below "
                    f"minimum threshold ({self._obj_conf_thresh:.2f})."
                ),
                "details": {
                    "object_confidence": obj_conf,
                    "threshold": self._obj_conf_thresh,
                },
            }

        # ---------------------------------------------------------------------
        # 6. Spatial Interaction Plausibility / Geometric Veto
        # ---------------------------------------------------------------------
        spatially_plausible = True
        spatial_reason = ""
        if interaction is not None and norm_act in MANIPULATION_ACTIONS:
            inter_state = str(interaction.get("state", "IDLE")).upper()
            scale_prox = interaction.get("normalized_scale_proximity")
            dist = interaction.get("closest_distance")

            # Condition for geometric veto:
            # Interaction engine reports IDLE (no hand contact) AND
            # normalized scale proximity indicates hand is outside reach envelope (> max_reach_scale_threshold)
            # OR centroid distance > max_distance_threshold
            is_outside_reach = False
            if scale_prox is not None and scale_prox > self._max_reach_scale:
                is_outside_reach = True
            elif scale_prox is None and dist is not None and dist > self._max_distance:
                is_outside_reach = True

            if inter_state == "IDLE" and is_outside_reach:
                spatially_plausible = False
                spatial_reason = (
                    f"Spatial contradiction: Action '{norm_act}' requires physical reach/contact, "
                    f"but InteractionEngine reports IDLE with normalized scale proximity "
                    f"{scale_prox if scale_prox is not None else dist:.2f} > {self._max_reach_scale:.2f}."
                )

        if not spatially_plausible:
            return {
                "category": CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION,
                "is_reliable": False,
                "action": norm_act,
                "object": norm_obj,
                "action_confidence": act_conf,
                "object_confidence": obj_conf,
                "semantic_consistent": True,
                "spatially_plausible": False,
                "reason": spatial_reason,
                "details": {
                    "interaction": interaction,
                    "max_reach_scale_threshold": self._max_reach_scale,
                },
            }

        # ---------------------------------------------------------------------
        # 7. Optional Flicker Diagnostics (Single-frame context check)
        # ---------------------------------------------------------------------
        if prior_action is not None and prior_object is not None:
            norm_prior_act = normalize_action(prior_action)
            norm_prior_obj = str(prior_object).strip().upper()

            # Object flicker: Action identical, but object switched
            if norm_act == norm_prior_act and norm_act in MANIPULATION_ACTIONS and norm_obj != norm_prior_obj and norm_prior_obj in CANONICAL_OBJECTS:
                return {
                    "category": CATEGORY_INSTABILITY_OBJECT_FLICKER,
                    "is_reliable": False,
                    "action": norm_act,
                    "object": norm_obj,
                    "action_confidence": act_conf,
                    "object_confidence": obj_conf,
                    "semantic_consistent": True,
                    "spatially_plausible": True,
                    "reason": f"Perceptual instability: Object rapidly alternated from '{norm_prior_obj}' to '{norm_obj}' during '{norm_act}'.",
                    "details": {
                        "current_object": norm_obj,
                        "prior_object": norm_prior_obj,
                    },
                }

            # Action flicker: Object identical, but action switched
            if norm_obj == norm_prior_obj and is_known_object and norm_act != norm_prior_act and norm_prior_act in MANIPULATION_ACTIONS and norm_act in MANIPULATION_ACTIONS:
                return {
                    "category": CATEGORY_INSTABILITY_ACTION_FLICKER,
                    "is_reliable": False,
                    "action": norm_act,
                    "object": norm_obj,
                    "action_confidence": act_conf,
                    "object_confidence": obj_conf,
                    "semantic_consistent": True,
                    "spatially_plausible": True,
                    "reason": f"Perceptual instability: Action rapidly alternated from '{norm_prior_act}' to '{norm_act}' on object '{norm_obj}'.",
                    "details": {
                        "current_action": norm_act,
                        "prior_action": norm_prior_act,
                    },
                }

        # ---------------------------------------------------------------------
        # 8. Reliable & Aligned Observation
        # ---------------------------------------------------------------------
        return {
            "category": CATEGORY_RELIABLE_ALIGNED,
            "is_reliable": True,
            "action": norm_act,
            "object": norm_obj,
            "action_confidence": act_conf,
            "object_confidence": obj_conf,
            "semantic_consistent": True,
            "spatially_plausible": True,
            "reason": f"Multimodal modalities aligned and coherent for '{norm_act}' on '{norm_obj}'.",
            "details": {
                "action_confidence": act_conf,
                "object_confidence": obj_conf,
                "interaction_state": (interaction or {}).get("state", "UNKNOWN"),
            },
        }
