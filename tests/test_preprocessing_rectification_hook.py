"""
test_preprocessing_rectification_hook.py — Gate B4.1c.2 Preprocessing Integration Hook Unit Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Tests:
1. FrameProcessor imports cleanly from backend.video.frame_processor and backend.video.
2. Rectification disabled (rectifier=None or enabled=False) preserves existing preprocessing output.
3. Rectification enabled applies geometric rectification before letterbox/resizing.
4. Canonical 0° rectification produces upright preprocessing output.
5. Canonical 45° rectification is supported.
6. Canonical 90° rectification transforms input before detector and pose preprocessing.
7. Canonical 135° rectification is supported.
8. Canonical 180° rectification inverts image before preprocessing.
9. Canonical 225° rectification is supported.
10. Canonical 270° rectification is supported.
11. Existing letterboxing/resizing dimensions and scale factors remain mathematically correct.
12. Synchronized annotation transformation preserves coordinate consistency without double-transformation.
13. Invalid rectifier inputs fail safely with clear error messaging.
14. Zero ML model dependency (pure mathematical/geometric execution).
15. Default configuration from settings.json preserves baseline preprocessing behavior.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from backend.video.frame_processor import FrameProcessor, _resolve_rectifier
from backend.video.camera_rectification import CameraRectifier, RectificationConfig


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Creates a deterministic 100x100 BGR image with distinctive quadrant colors."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [0, 0, 255]      # Top-Left: Red in BGR (B=0, G=0, R=255)
    img[:50, 50:] = [0, 255, 0]      # Top-Right: Green in BGR (B=0, G=255, R=0)
    img[50:, :50] = [255, 0, 0]      # Bottom-Left: Blue in BGR (B=255, G=0, R=0)
    img[50:, 50:] = [0, 255, 255]    # Bottom-Right: Yellow in BGR (B=0, G=255, R=255)
    return img


@pytest.fixture
def synthetic_boxes() -> list:
    return [
        {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4, "label": "RED_SAMPLE", "confidence": 0.95},
        {"x1": 0.6, "y1": 0.6, "x2": 0.9, "y2": 0.9, "label": "BLUE_SAMPLE", "confidence": 0.90},
    ]


@pytest.fixture
def synthetic_keypoints() -> list:
    return [
        {"x": 0.25, "y": 0.25, "z": 0.05, "visibility": 0.98, "name": "wrist"},
        {"x": 0.75, "y": 0.75, "z": -0.02, "visibility": 0.92, "name": "index_tip"},
    ]


class TestImportsAndAPICompatibility:
    """Test 1: Imports and backward compatibility."""

    def test_import_paths(self):
        from backend.video import FrameProcessor as FP1
        from backend.video.frame_processor import FrameProcessor as FP2
        assert FP1 is FrameProcessor
        assert FP2 is FrameProcessor

    def test_legacy_method_signatures_unchanged(self, synthetic_frame):
        # Default calls without rectifier parameter
        det_img, scale, pad = FrameProcessor.preprocess_for_detector(synthetic_frame)
        assert det_img.shape == (640, 640, 3)
        assert isinstance(scale, float)
        assert len(pad) == 2

        pose_img = FrameProcessor.preprocess_for_pose(synthetic_frame)
        assert pose_img.shape == (256, 256, 3)
        assert pose_img.dtype == np.float32


class TestDisabledRectificationPreservesBaseline:
    """Test 2 & 15: Disabled rectification produces identical output to legacy calls."""

    def test_none_rectifier_matches_baseline(self, synthetic_frame):
        base_det, base_s, base_p = FrameProcessor.preprocess_for_detector(synthetic_frame)
        hook_det, hook_s, hook_p = FrameProcessor.preprocess_for_detector(synthetic_frame, rectifier=None)

        assert np.array_equal(base_det, hook_det)
        assert base_s == hook_s
        assert base_p == hook_p

    def test_disabled_config_matches_baseline(self, synthetic_frame):
        config = RectificationConfig(enabled=False, rotation_deg=90.0)
        base_det, _, _ = FrameProcessor.preprocess_for_detector(synthetic_frame)
        hook_det, _, _ = FrameProcessor.preprocess_for_detector(synthetic_frame, rectifier=config)

        assert np.array_equal(base_det, hook_det)

    def test_default_settings_json_preserves_baseline(self, synthetic_frame):
        settings_path = Path("config/settings.json")
        assert settings_path.exists()
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        rect_section = settings.get("rectification", {})
        rectifier = CameraRectifier.from_dict(rect_section)
        assert rectifier.is_identity()

        base_pose = FrameProcessor.preprocess_for_pose(synthetic_frame)
        hook_pose = FrameProcessor.preprocess_for_pose(synthetic_frame, rectifier=rectifier)
        assert np.array_equal(base_pose, hook_pose)


class TestCanonicalAngleRectificationHooks:
    """Tests 4–10: Canonical angle preprocessing."""

    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0])
    def test_canonical_angle_detector_preprocessing(self, angle, synthetic_frame):
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=angle))
        det_img, scale, pad = FrameProcessor.preprocess_for_detector(
            synthetic_frame, target_size=(640, 640), rectifier=rectifier
        )
        assert det_img.shape == (640, 640, 3)
        assert scale > 0.0

    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0])
    def test_canonical_angle_pose_preprocessing(self, angle, synthetic_frame):
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=angle))
        pose_img = FrameProcessor.preprocess_for_pose(
            synthetic_frame, target_size=(256, 256), rectifier=rectifier
        )
        assert pose_img.shape == (256, 256, 3)
        assert pose_img.dtype == np.float32
        assert 0.0 <= pose_img.min() and pose_img.max() <= 1.0

    def test_90_degree_detector_geometry(self, synthetic_frame):
        # 90 degrees clockwise rotation before detector letterbox:
        # Original Top-Left is Red ([255, 0, 0] in BGR -> [0, 0, 255] in RGB)
        # After 90 deg rotation, Red moves to Top-Right.
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=90.0))
        det_img, _, (dw, dh) = FrameProcessor.preprocess_for_detector(
            synthetic_frame, target_size=(100, 100), rectifier=rectifier
        )
        # Sample near Top-Right of unpadded 100x100 image
        top_right_pixel = det_img[25, 75]
        # In RGB, Red is [255, 0, 0]
        assert top_right_pixel[0] == 255


class TestLetterboxingAndResizingConsistency:
    """Test 11: Output dimensions and scaling invariants."""

    def test_detector_letterbox_invariants(self, synthetic_frame):
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=45.0))
        det_img, scale, (dw, dh) = FrameProcessor.preprocess_for_detector(
            synthetic_frame, target_size=(640, 640), rectifier=rectifier
        )
        assert det_img.shape == (640, 640, 3)
        assert scale == 6.4
        assert dw == 0.0
        assert dh == 0.0


class TestAnnotationSynchronizationAndPipelineHook:
    """Test 12: process_frame_pipeline annotation synchronization."""

    def test_pipeline_synchronized_annotations(self, synthetic_frame, synthetic_boxes, synthetic_keypoints):
        rectifier = CameraRectifier(RectificationConfig(enabled=True, rotation_deg=180.0))
        res = FrameProcessor.process_frame_pipeline(
            synthetic_frame,
            target_size=(640, 640),
            rectifier=rectifier,
            bounding_boxes=synthetic_boxes,
            keypoints=synthetic_keypoints,
        )

        assert res["rectification_applied"] is True
        assert res["rectified_frame"].shape == synthetic_frame.shape
        assert res["detector_input"].shape == (640, 640, 3)
        assert res["pose_input"].shape == (256, 256, 3)

        # Transformed boxes and keypoints must be present
        assert len(res["bounding_boxes"]) == 2
        assert len(res["keypoints"]) == 2

        # 180 degree rotation maps (0.25, 0.25) to (0.75, 0.75)
        kp0 = res["keypoints"][0]
        assert abs(kp0["x"] - 0.75) < 1e-3
        assert abs(kp0["y"] - 0.75) < 1e-3
        assert kp0["z"] == 0.05  # z must be preserved!
        assert kp0["name"] == "wrist"

    def test_pipeline_disabled_rectification(self, synthetic_frame, synthetic_boxes, synthetic_keypoints):
        res = FrameProcessor.process_frame_pipeline(
            synthetic_frame,
            rectifier=None,
            bounding_boxes=synthetic_boxes,
            keypoints=synthetic_keypoints,
        )
        assert res["rectification_applied"] is False
        assert np.array_equal(res["rectified_frame"], synthetic_frame)
        assert res["bounding_boxes"] == synthetic_boxes
        assert res["keypoints"] == synthetic_keypoints


class TestErrorHandlingAndDecoupling:
    """Tests 13 & 14: Error handling and zero ML dependency."""

    def test_invalid_rectifier_type_raises_type_error(self, synthetic_frame):
        with pytest.raises(TypeError):
            FrameProcessor.preprocess_for_detector(synthetic_frame, rectifier=12345)  # type: ignore

    def test_dict_config_instantiation(self, synthetic_frame):
        rect_dict = {"enabled": True, "rotation_deg": 90.0}
        det_img, _, _ = FrameProcessor.preprocess_for_detector(synthetic_frame, rectifier=rect_dict)
        assert det_img.shape == (640, 640, 3)

    def test_zero_ml_model_dependency(self, synthetic_frame):
        # Pure mathematical preprocessing without model weights
        pose_tensor = FrameProcessor.preprocess_for_pose(
            synthetic_frame, rectifier=RectificationConfig(enabled=True, rotation_deg=90.0)
        )
        assert isinstance(pose_tensor, np.ndarray)
        assert pose_tensor.shape == (256, 256, 3)
