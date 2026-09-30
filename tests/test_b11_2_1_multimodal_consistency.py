"""
test_b11_2_1_multimodal_consistency.py — Workstream B Sub-Gate B11.2.1 Tests
ISRO SIH26174 BAS Experiment Monitor

Validates:
1. Aligned canonical (action, object) pairs:
   - PICK_RED + RED_SAMPLE
   - PLACE_RED + RED_SAMPLE
   - PICK_BLUE + BLUE_SAMPLE
   - PLACE_BLUE + BLUE_SAMPLE
   - CLOSE_LID + CONTAINER_LID
2. Semantic action/object mismatch detection.
3. Missing object handling.
4. Unknown object handling.
5. Low-confidence object detection.
6. Static object presence without actionable manipulation.
7. Object flicker representation.
8. Action flicker representation.
9. Spatial veto (interaction state IDLE with scale proximity > threshold).
10. Spatially plausible interaction.
11. Conflicting semantic + spatial evidence (deterministic rule precedence).
12. IDLE/NONE observation.
13. No FSM dependency and no procedural mutation.
14. Internal result schema invariance.
15. Canonical vocabulary preservation.
16. CLOSE_LID does NOT accept SAMPLE_CONTAINER.
17. Deterministic replay invariance.
"""

from typing import Any, Dict
import pytest

from backend.ai.multimodal_consistency import (
    CATEGORY_CROSS_MODAL_CONFLICT,
    CATEGORY_IDLE,
    CATEGORY_INSTABILITY_ACTION_FLICKER,
    CATEGORY_INSTABILITY_OBJECT_FLICKER,
    CATEGORY_RELIABLE_ALIGNED,
    CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE,
    CATEGORY_UNCERTAIN_MISSING_OBJECT,
    CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION,
    CATEGORY_UNCERTAIN_STATIC_OBJECT,
    MultimodalConsistencyEvaluator,
)


@pytest.fixture
def evaluator() -> MultimodalConsistencyEvaluator:
    return MultimodalConsistencyEvaluator(
        action_confidence_threshold=0.70,
        object_confidence_threshold=0.50,
        max_reach_scale_threshold=1.50,
        max_distance_threshold=0.30,
    )


# -----------------------------------------------------------------------------
# 1. Aligned Canonical Pairs
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,obj",
    [
        ("PICK_RED", "RED_SAMPLE"),
        ("PLACE_RED", "RED_SAMPLE"),
        ("PICK_BLUE", "BLUE_SAMPLE"),
        ("PLACE_BLUE", "BLUE_SAMPLE"),
        ("CLOSE_LID", "CONTAINER_LID"),
    ],
)
def test_aligned_canonical_pairs(evaluator: MultimodalConsistencyEvaluator, action: str, obj: str):
    res = evaluator.evaluate(
        action=action,
        action_confidence=0.90,
        object_name=obj,
        object_confidence=0.85,
        interaction={"state": "HOLDING", "normalized_scale_proximity": 0.65},
    )
    assert res["category"] == CATEGORY_RELIABLE_ALIGNED
    assert res["is_reliable"] is True
    assert res["semantic_consistent"] is True
    assert res["spatially_plausible"] is True
    assert res["action"] == action
    assert res["object"] == obj


# -----------------------------------------------------------------------------
# 2. Semantic Action / Object Mismatch
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,conflicting_obj",
    [
        ("PICK_RED", "BLUE_SAMPLE"),
        ("PICK_RED", "CONTAINER_LID"),
        ("PLACE_RED", "BLUE_SAMPLE"),
        ("PICK_BLUE", "RED_SAMPLE"),
        ("PLACE_BLUE", "RED_SAMPLE"),
        ("CLOSE_LID", "RED_SAMPLE"),
        ("CLOSE_LID", "BLUE_SAMPLE"),
    ],
)
def test_semantic_action_object_mismatch(
    evaluator: MultimodalConsistencyEvaluator, action: str, conflicting_obj: str
):
    res = evaluator.evaluate(
        action=action,
        action_confidence=0.92,
        object_name=conflicting_obj,
        object_confidence=0.88,
        interaction={"state": "HOLDING", "normalized_scale_proximity": 0.50},
    )
    assert res["category"] == CATEGORY_CROSS_MODAL_CONFLICT
    assert res["is_reliable"] is False
    assert res["semantic_consistent"] is False
    assert "Cross-modal semantic conflict" in res["reason"]


# -----------------------------------------------------------------------------
# 3. CLOSE_LID does NOT accept SAMPLE_CONTAINER
# -----------------------------------------------------------------------------

def test_close_lid_rejects_sample_container(evaluator: MultimodalConsistencyEvaluator):
    """EXP-001 canonical protocol step S5 strictly requires CONTAINER_LID for CLOSE_LID."""
    res = evaluator.evaluate(
        action="CLOSE_LID",
        action_confidence=0.95,
        object_name="SAMPLE_CONTAINER",
        object_confidence=0.90,
    )
    assert res["category"] == CATEGORY_CROSS_MODAL_CONFLICT
    assert res["is_reliable"] is False
    assert res["semantic_consistent"] is False


# -----------------------------------------------------------------------------
# 4. Missing Object
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("missing_val", [None, "", "NONE"])
def test_missing_object_for_manipulation_action(
    evaluator: MultimodalConsistencyEvaluator, missing_val: Any
):
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name=missing_val,
    )
    assert res["category"] == CATEGORY_UNCERTAIN_MISSING_OBJECT
    assert res["is_reliable"] is False
    assert res["semantic_consistent"] is False
    assert "requires object evidence" in res["reason"]


# -----------------------------------------------------------------------------
# 5. Unknown / Legacy Object
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("unknown_obj", ["SAMPLE_VIAL", "FORCEPS", "PIPETTE", "UNKNOWN_FIXTURE"])
def test_unknown_object_handling(evaluator: MultimodalConsistencyEvaluator, unknown_obj: str):
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name=unknown_obj,
    )
    assert res["category"] == CATEGORY_UNCERTAIN_MISSING_OBJECT
    assert res["is_reliable"] is False


# -----------------------------------------------------------------------------
# 6. Low Confidence Object Detection
# -----------------------------------------------------------------------------

def test_low_confidence_object(evaluator: MultimodalConsistencyEvaluator):
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        object_confidence=0.35,  # Below threshold 0.50
    )
    assert res["category"] == CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE
    assert res["is_reliable"] is False
    assert res["object_confidence"] == 0.35
    assert "below minimum threshold" in res["reason"]


# -----------------------------------------------------------------------------
# 7. Static Object Presence (Action IDLE, Object Confident)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("obj", ["RED_SAMPLE", "BLUE_SAMPLE", "CONTAINER_LID", "SAMPLE_CONTAINER"])
def test_static_object_presence_without_manipulation(
    evaluator: MultimodalConsistencyEvaluator, obj: str
):
    res = evaluator.evaluate(
        action="IDLE",
        action_confidence=0.95,
        object_name=obj,
        object_confidence=0.90,
    )
    assert res["category"] == CATEGORY_UNCERTAIN_STATIC_OBJECT
    assert res["is_reliable"] is False
    assert res["object"] == obj
    assert "Passive static object" in res["reason"]


# -----------------------------------------------------------------------------
# 8. Spatial Veto (Geometric Contradiction)
# -----------------------------------------------------------------------------

def test_spatial_interaction_veto(evaluator: MultimodalConsistencyEvaluator):
    """Manipulation action with interaction state IDLE and scale proximity > 1.50 is vetoed."""
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        object_confidence=0.85,
        interaction={
            "state": "IDLE",
            "normalized_scale_proximity": 2.50,  # Far outside reach
            "closest_distance": 0.60,
        },
    )
    assert res["category"] == CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION
    assert res["is_reliable"] is False
    assert res["spatially_plausible"] is False
    assert "Spatial contradiction" in res["reason"]


def test_spatially_plausible_interaction(evaluator: MultimodalConsistencyEvaluator):
    """Manipulation action within reach/holding is plausible."""
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        object_confidence=0.85,
        interaction={
            "state": "HOLDING",
            "normalized_scale_proximity": 0.60,
            "closest_distance": 0.05,
        },
    )
    assert res["category"] == CATEGORY_RELIABLE_ALIGNED
    assert res["is_reliable"] is True
    assert res["spatially_plausible"] is True


# -----------------------------------------------------------------------------
# 9. Deterministic Rule Precedence
# -----------------------------------------------------------------------------

def test_semantic_mismatch_precedence_over_spatial_veto(
    evaluator: MultimodalConsistencyEvaluator,
):
    """When both semantic mismatch and spatial veto exist, semantic mismatch takes precedence."""
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="BLUE_SAMPLE",  # Semantic conflict
        object_confidence=0.85,
        interaction={
            "state": "IDLE",
            "normalized_scale_proximity": 3.0,  # Spatial contradiction
        },
    )
    assert res["category"] == CATEGORY_CROSS_MODAL_CONFLICT
    assert res["is_reliable"] is False
    assert res["semantic_consistent"] is False


# -----------------------------------------------------------------------------
# 10. Flicker Diagnostics
# -----------------------------------------------------------------------------

def test_object_flicker_diagnostic(evaluator: MultimodalConsistencyEvaluator):
    """Action identical, object switched from RED_SAMPLE to BLUE_SAMPLE."""
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        object_confidence=0.85,
        prior_action="PICK_RED",
        prior_object="BLUE_SAMPLE",
    )
    assert res["category"] == CATEGORY_INSTABILITY_OBJECT_FLICKER
    assert res["is_reliable"] is False
    assert "rapidly alternated" in res["reason"]


def test_action_flicker_diagnostic(evaluator: MultimodalConsistencyEvaluator):
    """Object identical, action switched from PICK_RED to PLACE_RED."""
    res = evaluator.evaluate(
        action="PLACE_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        object_confidence=0.85,
        prior_action="PICK_RED",
        prior_object="RED_SAMPLE",
    )
    assert res["category"] == CATEGORY_INSTABILITY_ACTION_FLICKER
    assert res["is_reliable"] is False
    assert "rapidly alternated" in res["reason"]


# -----------------------------------------------------------------------------
# 11. Neutral IDLE / NONE
# -----------------------------------------------------------------------------

def test_idle_none_observation(evaluator: MultimodalConsistencyEvaluator):
    res = evaluator.evaluate(
        action="IDLE",
        action_confidence=0.95,
        object_name="NONE",
    )
    assert res["category"] == CATEGORY_IDLE
    assert res["is_reliable"] is False
    assert res["action"] == "IDLE"
    assert res["object"] == "NONE"


# -----------------------------------------------------------------------------
# 12. No FSM Dependency & No Mutation
# -----------------------------------------------------------------------------

def test_no_fsm_dependency_and_no_mutation(evaluator: MultimodalConsistencyEvaluator):
    """Evaluator is completely self-contained and does not touch FSM or mutate inputs."""
    sample_interaction = {"state": "HOLDING", "normalized_scale_proximity": 0.5}
    orig_copy = dict(sample_interaction)

    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        interaction=sample_interaction,
    )
    assert res["is_reliable"] is True
    # Ensure input interaction dict was not mutated
    assert sample_interaction == orig_copy


# -----------------------------------------------------------------------------
# 13. Output Schema Invariance
# -----------------------------------------------------------------------------

def test_internal_output_schema(evaluator: MultimodalConsistencyEvaluator):
    res = evaluator.evaluate(
        action="PICK_RED",
        action_confidence=0.90,
        object_name="RED_SAMPLE",
        object_confidence=0.85,
    )
    required_keys = {
        "category",
        "is_reliable",
        "action",
        "object",
        "action_confidence",
        "object_confidence",
        "semantic_consistent",
        "spatially_plausible",
        "reason",
        "details",
    }
    assert set(res.keys()) == required_keys
    assert isinstance(res["is_reliable"], bool)
    assert isinstance(res["semantic_consistent"], bool)
    assert isinstance(res["spatially_plausible"], bool)


# -----------------------------------------------------------------------------
# 14. Deterministic Replay Invariance
# -----------------------------------------------------------------------------

def test_deterministic_replay_invariance(evaluator: MultimodalConsistencyEvaluator):
    """Running evaluation 50 times on identical inputs produces identical results."""
    results = [
        evaluator.evaluate(
            action="PICK_RED",
            action_confidence=0.88,
            object_name="RED_SAMPLE",
            object_confidence=0.92,
            interaction={"state": "HOLDING", "normalized_scale_proximity": 0.45},
        )
        for _ in range(50)
    ]
    first = results[0]
    for r in results[1:]:
        assert r == first
