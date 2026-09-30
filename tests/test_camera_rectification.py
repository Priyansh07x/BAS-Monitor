"""
test_camera_rectification.py — Gate B4.1c.1 Camera Rectification Core Unit Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Tests:
1. Module imports and dataclass instantiation.
2. Identity / no-op produces equivalent output when disabled or default.
3. Configured canonical 0° transform.
4. Configured canonical 45° transform.
5. Configured canonical 90° transform.
6. Configured canonical 135° transform.
7. Configured canonical 180° transform.
8. Configured canonical 225° transform.
9. Configured canonical 270° transform.
10. Deterministic repeated transform (pure idempotence/reproducibility).
11. Output dimensions handling (preserving source shape vs custom output size).
12. Bounding-box transformation consistency (4-corner envelope mapping, clipping).
13. Keypoint transformation consistency (x/y mapped, z and auxiliary keys preserved).
14. Disabled rectification preserves existing frame processing (identity bypass).
15. Invalid calibration configuration fails safely (singular matrix, invalid shape, missing config file).
16. Standalone execution: zero ML framework / model weight file dependency.
17. Serialization & deserialization: to_dict, from_dict, from_settings, from_config_file.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from backend.video.camera_rectification import (
    CameraRectifier,
    RectificationConfig,
    RectificationResult,
    build_rectification_matrix,
    transform_bounding_box,
    transform_keypoints,
    transform_normalized_point,
    warp_image_numpy,
)


@pytest.fixture
def synthetic_color_frame() -> np.ndarray:
    """Creates a deterministic 100x100 RGB image with distinctive quadrant colors."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]      # Top-Left: Red
    img[:50, 50:] = [0, 255, 0]      # Top-Right: Green
    img[50:, :50] = [0, 0, 255]      # Bottom-Left: Blue
    img[50:, 50:] = [255, 255, 0]    # Bottom-Right: Yellow
    return img


@pytest.fixture
def synthetic_boxes() -> list:
    """Sample normalized bounding boxes."""
    return [
        {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4, "label": "RED_SAMPLE", "confidence": 0.95},
        {"x1": 0.6, "y1": 0.6, "x2": 0.9, "y2": 0.9, "label": "BLUE_SAMPLE", "confidence": 0.90},
    ]


@pytest.fixture
def synthetic_keypoints() -> list:
    """Sample normalized landmark keypoints."""
    return [
        {"x": 0.25, "y": 0.25, "z": 0.05, "visibility": 0.98, "name": "wrist"},
        {"x": 0.75, "y": 0.75, "z": -0.02, "visibility": 0.92, "name": "index_tip"},
    ]


class TestCameraRectificationImportsAndInstantiation:
    """Test 1: Module imports and basic initialization."""

    def test_imports_and_types(self):
        from backend.video import CameraRectifier as CR, RectificationConfig as RC, RectificationResult as RR
        assert CR is CameraRectifier
        assert RC is RectificationConfig
        assert RR is RectificationResult

    def test_default_config_is_identity(self):
        rectifier = CameraRectifier()
        assert rectifier.is_identity()
        assert rectifier.config.enabled is False
        assert rectifier.config.rotation_deg == 0.0
        assert rectifier.config.calibration_status == "UNSPECIFIED"


class TestIdentityAndNoOpMode:
    """Test 2 & 14: Disabled/identity rectification preserves frames without modification."""

    def test_disabled_rectification_returns_exact_copy(self, synthetic_color_frame, synthetic_boxes, synthetic_keypoints):
        rectifier = CameraRectifier(RectificationConfig(enabled=False, rotation_deg=90.0))
        assert rectifier.is_identity()

        res = rectifier.rectify(synthetic_color_frame, synthetic_boxes, synthetic_keypoints)
        assert res.applied is False
        assert np.array_equal(res.image, synthetic_color_frame)
        assert res.bounding_boxes == synthetic_boxes
        assert res.keypoints == synthetic_keypoints
        assert np.allclose(res.transform_matrix, np.eye(3))

    def test_enabled_zero_rotation_is_identity(self, synthetic_color_frame):
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=0.0))
        assert rectifier.is_identity()

        res = rectifier.rectify(synthetic_color_frame)
        assert res.applied is False
        assert np.array_equal(res.image, synthetic_color_frame)


class TestCanonicalAngleRotations:
    """Tests 3–9: Verify all 7 canonical angles rotate frames and annotations correctly."""

    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0])
    def test_canonical_angles_execution(self, angle, synthetic_color_frame, synthetic_boxes, synthetic_keypoints):
        config = RectificationConfig(enabled=True, rotation_deg=angle)
        rectifier = CameraRectifier(config)

        res = rectifier.rectify(synthetic_color_frame, synthetic_boxes, synthetic_keypoints)
        assert res.image.shape == synthetic_color_frame.shape
        assert len(res.bounding_boxes) == len(synthetic_boxes)
        assert len(res.keypoints) == len(synthetic_keypoints)

        # Coordinate range checks
        for box in res.bounding_boxes:
            assert 0.0 <= box["x1"] <= box["x2"] <= 1.0
            assert 0.0 <= box["y1"] <= box["y2"] <= 1.0
            assert box["label"] in ["RED_SAMPLE", "BLUE_SAMPLE"]

        for kp in res.keypoints:
            assert 0.0 <= kp["x"] <= 1.0
            assert 0.0 <= kp["y"] <= 1.0
            assert "z" in kp

    def test_90_degree_rotation_geometry(self, synthetic_color_frame):
        # 90 degrees clockwise rotation:
        # Top-Left (Red) -> Top-Right
        # Top-Right (Green) -> Bottom-Right
        # Bottom-Right (Yellow) -> Bottom-Left
        # Bottom-Left (Blue) -> Top-Left
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=90.0))
        res = rectifier.rectify(synthetic_color_frame)

        # Sample near center of each quadrant in output
        top_right_pixel = res.image[25, 75]
        assert top_right_pixel[0] == 255  # Red moved to Top-Right

        bottom_right_pixel = res.image[75, 75]
        assert bottom_right_pixel[1] == 255  # Green moved to Bottom-Right

        top_left_pixel = res.image[25, 25]
        assert top_left_pixel[2] == 255  # Blue moved to Top-Left

    def test_180_degree_rotation_geometry(self, synthetic_color_frame):
        # 180 degrees: Top-Left (Red) -> Bottom-Right
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=180.0))
        res = rectifier.rectify(synthetic_color_frame)

        bottom_right_pixel = res.image[75, 75]
        assert bottom_right_pixel[0] == 255  # Red is now Bottom-Right

    def test_270_degree_rotation_geometry(self, synthetic_color_frame):
        # 270 degrees clockwise (= 90 deg CCW): Top-Left (Red) -> Bottom-Left
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=270.0))
        res = rectifier.rectify(synthetic_color_frame)

        bottom_left_pixel = res.image[75, 25]
        assert bottom_left_pixel[0] == 255  # Red is now Bottom-Left


class TestDeterminismAndIdempotence:
    """Test 10: Repeated runs with the same config produce identical results."""

    def test_deterministic_repeated_transform(self, synthetic_color_frame, synthetic_boxes, synthetic_keypoints):
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=135.0, scale=1.1))

        res1 = rectifier.rectify(synthetic_color_frame, synthetic_boxes, synthetic_keypoints)
        res2 = rectifier.rectify(synthetic_color_frame, synthetic_boxes, synthetic_keypoints)

        assert np.array_equal(res1.image, res2.image)
        assert res1.bounding_boxes == res2.bounding_boxes
        assert res1.keypoints == res2.keypoints
        assert np.array_equal(res1.transform_matrix, res2.transform_matrix)


class TestOutputDimensions:
    """Test 11: Output size handling."""

    def test_custom_output_size(self, synthetic_color_frame, synthetic_boxes, synthetic_keypoints):
        target_size = (160, 120)  # (width, height)
        config = RectificationConfig(enabled=True, rotation_deg=45.0, output_size=target_size)
        rectifier = CameraRectifier(config)

        res = rectifier.rectify(synthetic_color_frame, synthetic_boxes, synthetic_keypoints)
        assert res.image.shape == (120, 160, 3)
        assert res.output_shape == (120, 160)
        assert res.input_shape == (100, 100)

    def test_grayscale_image_shape(self):
        gray_img = np.ones((80, 80), dtype=np.uint8) * 128
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=90.0, output_size=(60, 40)))
        res = rectifier.rectify(gray_img)
        assert res.image.shape == (40, 60)


class TestAnnotationSynchronization:
    """Tests 12 & 13: Bounding box and keypoint transformations."""

    def test_keypoint_point_center_rotation_90(self):
        # Center keypoint at (0.5, 0.5) remains at (0.5, 0.5) under any center rotation
        kps = [{"x": 0.5, "y": 0.5, "z": 0.123, "name": "center"}]
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=90.0))
        res = rectifier.rectify(np.zeros((100, 100, 3), dtype=np.uint8), keypoints=kps)

        assert len(res.keypoints) == 1
        assert abs(res.keypoints[0]["x"] - 0.5) < 1e-3
        assert abs(res.keypoints[0]["y"] - 0.5) < 1e-3
        assert res.keypoints[0]["z"] == 0.123  # z must be preserved!
        assert res.keypoints[0]["name"] == "center"

    def test_corner_keypoint_rotation_180(self):
        # Keypoint at (0.2, 0.2) after 180-deg rotation around (0.5, 0.5) moves to (0.8, 0.8)
        kps = [{"x": 0.2, "y": 0.2, "z": 0.45}]
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=180.0))
        res = rectifier.rectify(np.zeros((100, 100, 3), dtype=np.uint8), keypoints=kps)

        assert abs(res.keypoints[0]["x"] - 0.8) < 1e-3
        assert abs(res.keypoints[0]["y"] - 0.8) < 1e-3
        assert res.keypoints[0]["z"] == 0.45

    def test_bounding_box_envelope_clipping(self):
        boxes = [{"x1": 0.0, "y1": 0.0, "x2": 0.3, "y2": 0.3, "label": "TEST"}]
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=45.0))
        res = rectifier.rectify(np.zeros((100, 100, 3), dtype=np.uint8), bounding_boxes=boxes)

        assert len(res.bounding_boxes) == 1
        b = res.bounding_boxes[0]
        assert 0.0 <= b["x1"] <= b["x2"] <= 1.0
        assert 0.0 <= b["y1"] <= b["y2"] <= 1.0


class TestFaultToleranceAndEdgeCases:
    """Test 15: Invalid configuration and singular matrix safety."""

    def test_singular_homography_matrix_fails_gracefully(self, synthetic_color_frame):
        # All zeros singular matrix
        singular_matrix = np.zeros((3, 3), dtype=np.float64)
        config = RectificationConfig(enabled=True, homography_matrix=singular_matrix)
        rectifier = CameraRectifier(config)

        res = rectifier.rectify(synthetic_color_frame)
        assert res.image.shape == synthetic_color_frame.shape

    def test_nonexistent_config_file_defaults_to_disabled(self):
        rectifier = CameraRectifier.from_config_file("nonexistent_path/calibration.json")
        assert rectifier.is_identity()
        assert rectifier.config.enabled is False

    def test_invalid_affine_shape_raises_value_error(self):
        with pytest.raises(ValueError):
            build_rectification_matrix(100, 100, 100, 100, affine_matrix=[[1, 2]])


class TestModelDecouplingAndSerialization:
    """Tests 16 & 17: No ML weights required and config serialization."""

    def test_zero_ml_model_dependency(self, synthetic_color_frame):
        # Must execute purely mathematically without PyTorch/TensorFlow/Hailo
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=45.0, scale=1.05))
        res = rectifier.rectify(synthetic_color_frame)
        assert isinstance(res, RectificationResult)
        assert res.applied is True

    def test_to_dict_and_from_dict_roundtrip(self):
        config = RectificationConfig(
            enabled=True,
            rotation_deg=90.0,
            scale=1.1,
            translation_x=0.05,
            translation_y=-0.05,
            output_size=(640, 480),
            border_mode="reflect",
            interpolation="nearest",
            calibration_status="PROPOSED_DEFAULT",
        )
        rectifier = CameraRectifier(config)
        d = rectifier.to_dict()

        reconstructed = CameraRectifier.from_dict(d)
        assert reconstructed.config.enabled is True
        assert reconstructed.config.rotation_deg == 90.0
        assert reconstructed.config.scale == 1.1
        assert reconstructed.config.translation_x == 0.05
        assert reconstructed.config.translation_y == -0.05
        assert reconstructed.config.output_size == (640, 480)
        assert reconstructed.config.border_mode == "reflect"
        assert reconstructed.config.interpolation == "nearest"
        assert reconstructed.config.calibration_status == "PROPOSED_DEFAULT"

    def test_from_settings_dictionary(self):
        settings = {
            "camera": {"source": 0},
            "rectification": {
                "enabled": True,
                "rotation_deg": 180.0,
                "calibration_status": "PROPOSED_DEFAULT",
            }
        }
        rectifier = CameraRectifier.from_settings(settings)
        assert rectifier.config.enabled is True
        assert rectifier.config.rotation_deg == 180.0
        assert rectifier.config.calibration_status == "PROPOSED_DEFAULT"
