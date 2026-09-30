"""
tests/test_augmentation.py

Gate B4.1a — Orientation & Robustness Augmentation Core Tests
ISRO SIH26174 BAS Experiment Monitor

Validates:
1. Module imports successfully without external ML frameworks.
2. Identity / no-op transformation works.
3. Rotation transforms image and annotations synchronously.
4. All 7 canonical orientation angles (0, 45, 90, 135, 180, 225, 270 deg) are supported.
5. Scale transforms annotations consistently.
6. Translation transforms annotations consistently.
7. Brightness changes image without altering annotation coordinates.
8. Blur changes image without altering annotation coordinates.
9. Occlusion is deterministic when given a fixed random seed.
10. Perspective transforms annotations consistently.
11. Safe handling and clipping of out-of-bounds or edge bounding boxes.
12. 2D/3D landmark keypoints remain synchronized under geometric transformations.
13. Sequential composition of multiple transformations.
14. Reproducibility of seed-controlled multi-transform pipelines.
15. Zero ML model file dependency.
"""

import numpy as np
import pytest
from backend.ai.augmentation import (
    AugmentationEngine,
    AugmentationConfig,
    AugmentationResult,
    get_rotation_matrix_2d,
    transform_normalized_point,
    transform_bounding_box,
    transform_keypoints,
)


@pytest.fixture
def synthetic_image():
    """Generates a simple 100x100 RGB image with distinctive quadrant colors."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]      # Red top-left
    img[:50, 50:] = [0, 255, 0]      # Green top-right
    img[50:, :50] = [0, 0, 255]      # Blue bottom-left
    img[50:, 50:] = [255, 255, 255]  # White bottom-right
    return img


@pytest.fixture
def sample_boxes():
    """Returns sample normalized bounding boxes for RED_SAMPLE and SAMPLE_CONTAINER."""
    return [
        {"label": "RED_SAMPLE", "confidence": 0.95, "x1": 0.10, "y1": 0.10, "x2": 0.40, "y2": 0.40},
        {"label": "SAMPLE_CONTAINER", "confidence": 0.90, "x1": 0.60, "y1": 0.60, "x2": 0.90, "y2": 0.90},
    ]


@pytest.fixture
def sample_keypoints():
    """Returns sample normalized pose/hand keypoints."""
    return [
        {"x": 0.25, "y": 0.25, "z": 0.0, "visibility": 0.99, "name": "RIGHT_WRIST"},
        {"x": 0.75, "y": 0.75, "z": 0.05, "visibility": 0.95, "name": "LEFT_WRIST"},
    ]


@pytest.fixture
def engine():
    """Returns an AugmentationEngine instance."""
    return AugmentationEngine()


# ---------------------------------------------------------------------------
# Test 1 & 15: Import & Zero ML Model Dependency
# ---------------------------------------------------------------------------

def test_module_imports_and_instantiates():
    """Verify engine instantiates without any external weights or models."""
    eng = AugmentationEngine()
    assert eng is not None
    assert len(eng.CANONICAL_ORIENTATION_ANGLES) == 7


# ---------------------------------------------------------------------------
# Test 2: Identity / No-Op Transformation
# ---------------------------------------------------------------------------

def test_identity_transformation(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify 0-deg rotation and unconfigured pipeline returns identical data."""
    res_rot = engine.rotate(synthetic_image, 0.0, sample_boxes, sample_keypoints)
    assert np.array_equal(res_rot.image, synthetic_image)
    assert res_rot.bounding_boxes == sample_boxes
    assert res_rot.keypoints == sample_keypoints

    cfg = AugmentationConfig()  # all disabled
    res_pipe = engine.apply(synthetic_image, sample_boxes, sample_keypoints, config=cfg)
    assert np.array_equal(res_pipe.image, synthetic_image)
    assert res_pipe.bounding_boxes == sample_boxes


# ---------------------------------------------------------------------------
# Test 3 & 4: Rotation & All Seven Canonical Angles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270])
def test_all_seven_canonical_angles_supported(engine, synthetic_image, sample_boxes, sample_keypoints, angle):
    """Verify all 7 canonical angles execute and return synchronized shapes."""
    res = engine.rotate(synthetic_image, angle_deg=angle, bounding_boxes=sample_boxes, keypoints=sample_keypoints)
    assert isinstance(res, AugmentationResult)
    assert res.image.shape == synthetic_image.shape
    assert len(res.bounding_boxes) == len(sample_boxes)
    assert len(res.keypoints) == len(sample_keypoints)

    # Verify bounding boxes remain bounded in [0.0, 1.0]
    for b in res.bounding_boxes:
        assert 0.0 <= b["x1"] <= b["x2"] <= 1.0
        assert 0.0 <= b["y1"] <= b["y2"] <= 1.0


def test_180_degree_rotation_geometry(engine, synthetic_image):
    """Verify 180-degree rotation inverts center-offset points correctly."""
    # Point at (0.2, 0.2) relative to center (0.5, 0.5) should map to (0.8, 0.8)
    boxes = [{"label": "TEST", "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}]
    kps = [{"x": 0.2, "y": 0.2}]
    res = engine.rotate(synthetic_image, 180.0, boxes, kps)

    # Keypoint (0.2, 0.2) inverted around (0.5, 0.5) is (0.8, 0.8)
    assert abs(res.keypoints[0]["x"] - 0.8) < 0.02
    assert abs(res.keypoints[0]["y"] - 0.8) < 0.02

    # Box (0.1, 0.1, 0.3, 0.3) inverted is (0.7, 0.7, 0.9, 0.9)
    assert abs(res.bounding_boxes[0]["x1"] - 0.7) < 0.02
    assert abs(res.bounding_boxes[0]["x2"] - 0.9) < 0.02


# ---------------------------------------------------------------------------
# Test 5: Scale Transformation
# ---------------------------------------------------------------------------

def test_scale_transformation(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify zooming in (scale > 1) expands boxes and spreads keypoints from center."""
    boxes = [{"label": "CENTER_OBJ", "x1": 0.40, "y1": 0.40, "x2": 0.60, "y2": 0.60}]
    kps = [{"x": 0.40, "y": 0.40}]

    # Scale by 1.2x (1.2 times wider from center 0.5)
    res = engine.scale(synthetic_image, scale_factor=1.2, bounding_boxes=boxes, keypoints=kps)
    assert res.image.shape == synthetic_image.shape

    # Original width was 0.20 -> scaled width should be approx 0.24
    scaled_w = res.bounding_boxes[0]["x2"] - res.bounding_boxes[0]["x1"]
    assert abs(scaled_w - 0.24) < 0.02

    # Point (0.40) scaled by 1.2 from 0.5: 0.5 + 1.2 * (0.4 - 0.5) = 0.38
    assert abs(res.keypoints[0]["x"] - 0.38) < 0.02


# ---------------------------------------------------------------------------
# Test 6: Translation Transformation
# ---------------------------------------------------------------------------

def test_translation_transformation(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify positive dx, dy translates bounding boxes and keypoints right and down."""
    dx, dy = 0.05, 0.05
    res = engine.translate(synthetic_image, dx_fraction=dx, dy_fraction=dy, bounding_boxes=sample_boxes, keypoints=sample_keypoints)

    assert abs(res.bounding_boxes[0]["x1"] - (sample_boxes[0]["x1"] + dx)) < 0.01
    assert abs(res.bounding_boxes[0]["y1"] - (sample_boxes[0]["y1"] + dy)) < 0.01
    assert abs(res.keypoints[0]["x"] - (sample_keypoints[0]["x"] + dx)) < 0.01


# ---------------------------------------------------------------------------
# Test 7: Brightness Transformation
# ---------------------------------------------------------------------------

def test_brightness_transformation(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify brightness modifies pixel intensities but preserves exact coordinates."""
    res = engine.adjust_brightness(synthetic_image, gain=1.2, bias=10.0, bounding_boxes=sample_boxes, keypoints=sample_keypoints)

    # Pixel intensities should be brighter
    assert res.image.mean() > synthetic_image.mean()

    # Annotations MUST remain exactly identical
    assert res.bounding_boxes == sample_boxes
    assert res.keypoints == sample_keypoints


# ---------------------------------------------------------------------------
# Test 8: Blur Transformation
# ---------------------------------------------------------------------------

def test_blur_transformation(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify blur smooths pixel transitions but preserves exact coordinates."""
    res = engine.apply_blur(synthetic_image, kernel_size=5, bounding_boxes=sample_boxes, keypoints=sample_keypoints)

    assert res.image.shape == synthetic_image.shape
    # Image content should be modified by convolution
    assert not np.array_equal(res.image, synthetic_image)

    # Coordinates MUST remain exactly identical
    assert res.bounding_boxes == sample_boxes
    assert res.keypoints == sample_keypoints


# ---------------------------------------------------------------------------
# Test 9 & 14: Occlusion Determinism with Seed
# ---------------------------------------------------------------------------

def test_occlusion_determinism_with_seed(engine, synthetic_image, sample_boxes):
    """Verify occlusion produces identical pixel masks with the same seed."""
    res1 = engine.apply_occlusion(synthetic_image, box_fraction=0.20, fill_value=0, seed=42, bounding_boxes=sample_boxes)
    res2 = engine.apply_occlusion(synthetic_image, box_fraction=0.20, fill_value=0, seed=42, bounding_boxes=sample_boxes)
    res3 = engine.apply_occlusion(synthetic_image, box_fraction=0.20, fill_value=0, seed=99, bounding_boxes=sample_boxes)

    assert np.array_equal(res1.image, res2.image), "Identical seed must yield identical occlusion patch"
    assert not np.array_equal(res1.image, res3.image), "Different seed should place occlusion differently"
    assert res1.bounding_boxes == sample_boxes


# ---------------------------------------------------------------------------
# Test 10: Perspective Transformation
# ---------------------------------------------------------------------------

def test_perspective_transformation(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify perspective transform executes and maintains valid bounding boxes."""
    res = engine.apply_perspective(synthetic_image, distortion=0.08, seed=123, bounding_boxes=sample_boxes, keypoints=sample_keypoints)

    assert res.image.shape == synthetic_image.shape
    for b in res.bounding_boxes:
        assert 0.0 <= b["x1"] <= b["x2"] <= 1.0
        assert 0.0 <= b["y1"] <= b["y2"] <= 1.0
    for kp in res.keypoints:
        assert 0.0 <= kp["x"] <= 1.0
        assert 0.0 <= kp["y"] <= 1.0


# ---------------------------------------------------------------------------
# Test 11 & 12: Boundary Clipping & Keypoint Synchronization
# ---------------------------------------------------------------------------

def test_boundary_clipping_and_synchronization(engine, synthetic_image):
    """Verify objects translated past frame boundary are clipped to [0, 1]."""
    corner_box = [{"label": "CORNER", "x1": 0.85, "y1": 0.85, "x2": 0.95, "y2": 0.95}]
    corner_kp = [{"x": 0.90, "y": 0.90, "name": "WRIST"}]

    # Translate by +0.20 (should push box edges past 1.0)
    res = engine.translate(synthetic_image, dx_fraction=0.20, dy_fraction=0.20, bounding_boxes=corner_box, keypoints=corner_kp)

    assert res.bounding_boxes[0]["x2"] == 1.0
    assert res.bounding_boxes[0]["y2"] == 1.0
    assert res.keypoints[0]["x"] == 1.0
    assert res.keypoints[0]["y"] == 1.0
    assert res.keypoints[0]["name"] == "WRIST"


# ---------------------------------------------------------------------------
# Test 13: Composite Pipeline & Seed Reproducibility
# ---------------------------------------------------------------------------

def test_composite_pipeline_execution(engine, synthetic_image, sample_boxes, sample_keypoints):
    """Verify multi-step sequential augmentation pipeline works deterministically."""
    cfg = AugmentationConfig(
        enable_rotation=True,
        rotation_angle_deg=90.0,
        enable_scale=True,
        scale_factor=1.1,
        enable_brightness=True,
        brightness_gain=1.1,
        enable_blur=True,
        blur_kernel_size=3,
        enable_occlusion=True,
        occlusion_size_fraction=0.10,
        seed=777,
    )

    res1 = engine.apply(synthetic_image, sample_boxes, sample_keypoints, config=cfg)
    res2 = engine.apply(synthetic_image, sample_boxes, sample_keypoints, config=cfg)

    assert np.array_equal(res1.image, res2.image)
    assert res1.bounding_boxes == res2.bounding_boxes
    assert res1.keypoints == res2.keypoints
    assert len(res1.transform_metadata.get("applied_transforms", [])) == 5
