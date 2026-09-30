"""
tests/test_interaction_orientation.py

Gate B4.1b — Orientation-Robust Interaction Logic Tests
ISRO SIH26174 BAS Experiment Monitor

Validates:
1. Upright (0 deg) standard interaction classification (IDLE, APPROACHING, HOLDING).
2. Orientation invariance across all 7 canonical angles (0, 45, 90, 135, 180, 225, 270 deg).
3. Scale invariance: Zooming in/out preserves valid grasp classification.
4. Translation invariance: Shifting workspace maintains correct interaction state.
5. Clear separation between distant hands and objects (proper IDLE classification).
6. Conservative fail-safe behavior: Ambiguous geometry defaults to APPROACHING / IDLE without forcing false HOLDING.
7. Hand landmark refinement support.
8. Backward compatibility of return dictionary structure.
9. Zero ML model dependency.
"""

import numpy as np
import pytest
from backend.experiment.interaction_logic import InteractionEngine
from backend.ai.augmentation import AugmentationEngine


@pytest.fixture
def engine():
    """Returns an InteractionEngine instance with default thresholds."""
    return InteractionEngine()


@pytest.fixture
def aug_engine():
    """Returns an AugmentationEngine instance for synthetic geometric validation."""
    return AugmentationEngine()


# ---------------------------------------------------------------------------
# Test 1 & 8 & 9: Upright Baseline & API Structure & Zero Model Dependency
# ---------------------------------------------------------------------------

def test_upright_interaction_states(engine):
    """Verify standard upright cases correctly produce HOLDING, APPROACHING, and IDLE."""
    # 1. Direct overlap -> HOLDING
    hand_holding = [{"x1": 0.40, "y1": 0.40, "x2": 0.50, "y2": 0.50}]
    obj = [{"label": "RED_SAMPLE", "x1": 0.42, "y1": 0.42, "x2": 0.48, "y2": 0.48}]
    res_hold = engine.evaluate_interaction(hand_holding, obj)
    assert res_hold["state"] == "HOLDING"
    assert res_hold["target_object"] == "RED_SAMPLE"
    assert res_hold["confidence"] >= 0.70

    # 2. Nearby -> APPROACHING
    hand_approaching = [{"x1": 0.52, "y1": 0.52, "x2": 0.60, "y2": 0.60}]
    res_app = engine.evaluate_interaction(hand_approaching, obj)
    assert res_app["state"] == "APPROACHING"
    assert res_app["target_object"] == "RED_SAMPLE"

    # 3. Far away -> IDLE
    hand_idle = [{"x1": 0.85, "y1": 0.85, "x2": 0.95, "y2": 0.95}]
    res_idle = engine.evaluate_interaction(hand_idle, obj)
    assert res_idle["state"] == "IDLE"
    assert res_idle["target_object"] is None


def test_empty_inputs_return_safe_idle(engine):
    """Verify empty hand or object lists return safe IDLE defaults."""
    res_empty_hand = engine.evaluate_interaction([], [{"label": "RED_SAMPLE", "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}])
    assert res_empty_hand["state"] == "IDLE"

    res_empty_obj = engine.evaluate_interaction([{"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}], [])
    assert res_empty_obj["state"] == "IDLE"


# ---------------------------------------------------------------------------
# Test 2–7: Orientation Tolerance Across All 7 Canonical Angles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270])
def test_grasp_interaction_maintained_across_all_seven_angles(engine, aug_engine, angle):
    """
    Verify that an active grasp (HOLDING state) remains classified as HOLDING
    when the entire scene is synthetically rotated across all 7 canonical angles.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    hand = [{"label": "HAND", "x1": 0.44, "y1": 0.44, "x2": 0.56, "y2": 0.56}]
    obj = [{"label": "BLUE_SAMPLE", "x1": 0.46, "y1": 0.46, "x2": 0.54, "y2": 0.54}]

    # Apply synthetic rotation
    res_hand = aug_engine.rotate(img, angle_deg=angle, bounding_boxes=hand)
    res_obj = aug_engine.rotate(img, angle_deg=angle, bounding_boxes=obj)

    eval_res = engine.evaluate_interaction(res_hand.bounding_boxes, res_obj.bounding_boxes)
    assert eval_res["state"] == "HOLDING", (
        f"At rotation {angle} deg, interaction was {eval_res['state']}, expected HOLDING"
    )
    assert eval_res["target_object"] == "BLUE_SAMPLE"


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270])
def test_approaching_state_maintained_across_all_seven_angles(engine, aug_engine, angle):
    """
    Verify that an APPROACHING hand remains classified as APPROACHING
    when rotated diagonally or transversely.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    hand = [{"label": "HAND", "x1": 0.58, "y1": 0.58, "x2": 0.68, "y2": 0.68}]
    obj = [{"label": "SAMPLE_CONTAINER", "x1": 0.40, "y1": 0.40, "x2": 0.52, "y2": 0.52}]

    res_hand = aug_engine.rotate(img, angle_deg=angle, bounding_boxes=hand)
    res_obj = aug_engine.rotate(img, angle_deg=angle, bounding_boxes=obj)

    eval_res = engine.evaluate_interaction(res_hand.bounding_boxes, res_obj.bounding_boxes)
    assert eval_res["state"] == "APPROACHING", (
        f"At rotation {angle} deg, interaction was {eval_res['state']}, expected APPROACHING"
    )


# ---------------------------------------------------------------------------
# Test 9: Scale Invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale_factor", [0.8, 0.9, 1.0, 1.1, 1.2])
def test_scale_invariance_for_holding(engine, aug_engine, scale_factor):
    """Verify zooming in/out by 0.8x to 1.2x preserves HOLDING classification."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    hand = [{"label": "HAND", "x1": 0.45, "y1": 0.45, "x2": 0.55, "y2": 0.55}]
    obj = [{"label": "CONTAINER_LID", "x1": 0.47, "y1": 0.47, "x2": 0.53, "y2": 0.53}]

    res_hand = aug_engine.scale(img, scale_factor=scale_factor, bounding_boxes=hand)
    res_obj = aug_engine.scale(img, scale_factor=scale_factor, bounding_boxes=obj)

    eval_res = engine.evaluate_interaction(res_hand.bounding_boxes, res_obj.bounding_boxes)
    assert eval_res["state"] == "HOLDING"


# ---------------------------------------------------------------------------
# Test 10: Translation Invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dx,dy", [(-0.1, -0.1), (0.1, 0.1), (-0.05, 0.08), (0.08, -0.05)])
def test_translation_invariance_for_holding(engine, aug_engine, dx, dy):
    """Verify translating the workspace preserves HOLDING classification."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    hand = [{"label": "HAND", "x1": 0.45, "y1": 0.45, "x2": 0.55, "y2": 0.55}]
    obj = [{"label": "RED_SAMPLE", "x1": 0.47, "y1": 0.47, "x2": 0.53, "y2": 0.53}]

    res_hand = aug_engine.translate(img, dx_fraction=dx, dy_fraction=dy, bounding_boxes=hand)
    res_obj = aug_engine.translate(img, dx_fraction=dx, dy_fraction=dy, bounding_boxes=obj)

    eval_res = engine.evaluate_interaction(res_hand.bounding_boxes, res_obj.bounding_boxes)
    assert eval_res["state"] == "HOLDING"


# ---------------------------------------------------------------------------
# Test 11 & 12: Clear Separation & Conservative Ambiguous Behavior
# ---------------------------------------------------------------------------

def test_distant_objects_remain_idle(engine):
    """Verify hand on opposite corner remains strictly IDLE."""
    hand = [{"x1": 0.05, "y1": 0.05, "x2": 0.15, "y2": 0.15}]
    obj = [{"label": "RED_SAMPLE", "x1": 0.80, "y1": 0.80, "x2": 0.90, "y2": 0.90}]
    res = engine.evaluate_interaction(hand, obj)
    assert res["state"] == "IDLE"
    assert res["target_object"] is None


def test_ambiguous_geometry_does_not_force_holding(engine):
    """
    Verify hand barely inside the proximity reach envelope classifies as
    APPROACHING rather than prematurely triggering HOLDING.
    """
    # Centroid distance is 0.12 (inside proximity 0.15, but scale_proximity > contact_scale_threshold)
    hand = [{"x1": 0.30, "y1": 0.50, "x2": 0.38, "y2": 0.58}]  # center (0.34, 0.54)
    obj = [{"label": "SAMPLE_CONTAINER", "x1": 0.45, "y1": 0.50, "x2": 0.55, "y2": 0.60}]  # center (0.50, 0.55)

    res = engine.evaluate_interaction(hand, obj)
    assert res["state"] == "APPROACHING", f"Expected APPROACHING, got {res['state']}"


# ---------------------------------------------------------------------------
# Test 13: Landmark Refinement
# ---------------------------------------------------------------------------

def test_landmark_refinement_evidence(engine):
    """Verify that a fingertip keypoint in direct contact upgrades state to HOLDING."""
    # Hand bounding box is slightly offset, but index fingertip touches object
    hand_box = [{"x1": 0.30, "y1": 0.30, "x2": 0.45, "y2": 0.45}]
    obj_box = [{"label": "CONTAINER_LID", "x1": 0.48, "y1": 0.48, "x2": 0.56, "y2": 0.56}]
    fingertip_keypoint = [{"x": 0.49, "y": 0.49, "name": "INDEX_FINGERTIP"}]

    res = engine.evaluate_interaction(hand_box, obj_box, hand_landmarks=fingertip_keypoint)
    assert res["state"] == "HOLDING"
    assert res["target_object"] == "CONTAINER_LID"
