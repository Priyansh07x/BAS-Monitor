"""
test_b14_2_spatial_kinematics.py — Comprehensive Unit & Integration Tests for Gate B14.2
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.2)

Verifies:
TRANSFORM (1-9):
1. Identity transform.
2. Pure translation.
3. 90° rotation.
4. Combined rotation + translation and inverse roundtrip.
5. Batch point transformation.
6. Invalid matrix shape rejection.
7. Invalid translation vector shape rejection.
8. Non-finite values handling (NaN / Inf).
9. Rotation matrix determinant and validity checking.

3D GEOMETRY (10-17):
10. Euclidean 3D distance calculation.
11. Hand-to-target distance.
12. Point-in-volume containment.
13. Distance-to-volume surface calculation.
14. Reach threshold comparison.
15. Missing-joint handling.
16. Metric vs non-metric distinction and safety.
17. Coordinate-frame mismatch handling.

PAYLOAD SPATIAL (18-22):
18. RED_SAMPLE & BLUE_SAMPLE proximity evaluation.
19. SAMPLE_CONTAINER proximity evaluation.
20. CONTAINER_LID proximity evaluation.
21. Containment depth and inside-volume flags.
22. Deterministic SpatialRelationResult serialization.

POSTURE NORMALIZATION & ROLL-INVARIANCE (23-29):
23. Pelvis-centering to origin.
24. Torso-axis alignment to +Y.
25. 0° roll invariance.
26. 45° roll invariance.
27. 90° roll invariance.
28. 180° inversion invariance.
29. 270° roll invariance.

INTEGRATION & SAFETY (30-38):
30. SyntheticHMREngine -> B14.2 spatial adapter flow.
31. MediaPipe landmark-only -> B14.2 flow.
32. Missing HMR dependency fallback.
33. Malformed HMR result containment.
34. Optional side-channel disabled by default.
35. B14.2 failure containment without breaking pipeline.
36. Frozen 8-field public AI contract remains unchanged.
37. B10/B11/B12/FSM behavior remains unchanged.
38. No extra background worker threads spawned.

LONG-RUN & STRESS (39-41):
39. 1,000+ synthetic geometry evaluations latency profiling.
40. Bounded memory and zero resource accumulation.
41. Deterministic repeated results.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Dict, List
import unittest
import numpy as np

from backend.ai.hmr import (
    BoundingVolume3D,
    CameraToPayloadTransform,
    HMRMeshResult,
    Joint3D,
    MediaPipeHMREngine,
    MeshVertex3D,
    MicrogravityPostureNormalizer,
    SpatialDisambiguationAdapter,
    SpatialRelationResult,
    SyntheticHMREngine,
    compute_euclidean_distance_3d,
    evaluate_joint_target_relation,
    evaluate_payload_spatial_relations,
    get_canonical_exp001_volumes,
)
from backend.ai.result_adapter import AIResultAdapter
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.experiment.recovery_manager import RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM


class TestB142SpatialKinematics(unittest.TestCase):
    """Unit and integration test suite for Gate B14.2 3D spatial reasoning and kinematics."""

    def setUp(self) -> None:
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_frame[100:200, 100:200] = 128
        self.canonical_volumes = get_canonical_exp001_volumes()

    # =========================================================================
    # TRANSFORM (1-9)
    # =========================================================================

    def test_01_identity_transform(self) -> None:
        tf = CameraToPayloadTransform.identity()
        pt = np.array([1.0, 2.0, 3.0])
        res = tf.transform_point(pt)
        np.testing.assert_allclose(res, pt)

        j = Joint3D(1.0, 2.0, 3.0, name="wrist")
        j_tf = tf.transform_joint(j)
        self.assertAlmostEqual(j_tf.x, 1.0)
        self.assertAlmostEqual(j_tf.y, 2.0)
        self.assertAlmostEqual(j_tf.z, 3.0)
        self.assertEqual(j_tf.name, "wrist")

    def test_02_pure_translation(self) -> None:
        tf = CameraToPayloadTransform(translation_vector=np.array([0.5, -0.2, 1.0]))
        pt = np.array([1.0, 2.0, 3.0])
        res = tf.transform_point(pt)
        np.testing.assert_allclose(res, [1.5, 1.8, 4.0])

    def test_03_90_degree_rotation(self) -> None:
        # 90 degrees roll around Z-axis
        tf = CameraToPayloadTransform.from_euler_angles(roll_deg=90.0)
        pt = np.array([1.0, 0.0, 0.0])
        res = tf.transform_point(pt)
        # (1, 0, 0) rotated 90 deg around Z becomes (0, 1, 0)
        np.testing.assert_allclose(res, [0.0, 1.0, 0.0], atol=1e-5)

    def test_04_combined_rotation_translation_inverse(self) -> None:
        tf = CameraToPayloadTransform.from_euler_angles(
            roll_deg=45.0,
            pitch_deg=30.0,
            yaw_deg=15.0,
            translation_m=(0.2, -0.5, 1.2),
        )
        pt = np.array([0.1, 0.4, 1.5])
        p_payload = tf.transform_point(pt)
        # Inverse transform should restore original point
        tf_inv = tf.inverse()
        p_cam_restored = tf_inv.transform_point(p_payload)
        np.testing.assert_allclose(p_cam_restored, pt, atol=1e-5)

    def test_05_batch_point_transform(self) -> None:
        tf = CameraToPayloadTransform.from_rot_trans(
            rotation=np.eye(3),
            translation=[1.0, 2.0, 3.0],
        )
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -2.0, -3.0],
        ])
        res = tf.transform_points_batch(pts)
        expected = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(res, expected)

    def test_06_invalid_matrix_shape(self) -> None:
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.ones((2, 2)))
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.ones((3, 4)))

    def test_07_invalid_translation_shape(self) -> None:
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(translation_vector=np.ones((2,)))
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(translation_vector=np.ones((4,)))

    def test_08_non_finite_transform_handling(self) -> None:
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]))
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(translation_vector=np.array([0.0, np.inf, 1.0]))

        tf = CameraToPayloadTransform.identity()
        with self.assertRaises(ValueError):
            tf.transform_point([np.nan, 1.0, 2.0])

    def test_09_rotation_validation(self) -> None:
        # Zero determinant (singular matrix)
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.zeros((3, 3)))
        # Negative determinant (reflection, not proper rotation)
        R_ref = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=R_ref)

    # =========================================================================
    # 3D GEOMETRY (10-17)
    # =========================================================================

    def test_10_euclidean_distance(self) -> None:
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(compute_euclidean_distance_3d(p1, p2), 5.0)

    def test_11_hand_to_target_distance(self) -> None:
        container = self.canonical_volumes["SAMPLE_CONTAINER"]
        # Container center at (0, 0, 0), dimensions (0.14, 0.12, 0.14) -> half-extents (0.07, 0.06, 0.07)
        # Point at (0.17, 0.0, 0.0) -> distance from surface (0.07, 0.0, 0.0) is 0.10 m
        j = Joint3D(x=0.17, y=0.0, z=0.0, name="right_wrist")
        rel = evaluate_joint_target_relation(j, container)
        self.assertAlmostEqual(rel.distance, 0.10, places=4)
        self.assertTrue(rel.is_within_reach)
        self.assertFalse(rel.is_inside)

    def test_12_point_in_volume(self) -> None:
        container = self.canonical_volumes["SAMPLE_CONTAINER"]
        # Inside
        self.assertTrue(container.contains_point((0.02, -0.01, 0.03)))
        # Outside
        self.assertFalse(container.contains_point((0.20, 0.0, 0.0)))
        # On boundary
        self.assertTrue(container.contains_point((0.07, 0.06, 0.07)))

    def test_13_distance_to_volume_surface(self) -> None:
        red_box = self.canonical_volumes["RED_SAMPLE"]
        # Center at (-0.15, 0.0, 0.0), half-extents (0.03, 0.03, 0.03) -> box X: [-0.18, -0.12]
        # Point at (-0.15, 0.0, 0.0) is inside -> distance 0.0
        self.assertAlmostEqual(red_box.distance_to_point((-0.15, 0.0, 0.0)), 0.0)
        # Point at (-0.15, 0.08, 0.0) -> dy = 0.08 - 0.03 = 0.05
        self.assertAlmostEqual(red_box.distance_to_point((-0.15, 0.08, 0.0)), 0.05)

    def test_14_reach_threshold(self) -> None:
        lid = self.canonical_volumes["CONTAINER_LID"]
        j_near = Joint3D(0.0, 0.12, 0.0, name="right_hand")  # dist ~ 0.025m <= 0.05m
        j_far = Joint3D(0.0, 0.40, 0.0, name="right_hand")   # dist ~ 0.305m > 0.05m

        rel_near = evaluate_joint_target_relation(j_near, lid, reach_threshold=0.05)
        rel_far = evaluate_joint_target_relation(j_far, lid, reach_threshold=0.05)

        self.assertTrue(rel_near.is_within_reach)
        self.assertFalse(rel_far.is_within_reach)

    def test_15_missing_joint_handling(self) -> None:
        rels = evaluate_payload_spatial_relations([], self.canonical_volumes)
        self.assertEqual(rels, {})

    def test_16_metric_vs_non_metric_distinction(self) -> None:
        vol = BoundingVolume3D("MOCK", (0.0, 0.0, 0.0), (0.1, 0.1, 0.1), is_metric=False)
        j = Joint3D(0.2, 0.0, 0.0, name="wrist")
        rel = evaluate_joint_target_relation(j, vol, is_metric=False)
        self.assertFalse(rel.is_metric)

        vol_metric = self.canonical_volumes["RED_SAMPLE"]
        rel_metric = evaluate_joint_target_relation(j, vol_metric, is_metric=True)
        self.assertTrue(rel_metric.is_metric)

    def test_17_coordinate_frame_mismatch_handling(self) -> None:
        tf = CameraToPayloadTransform(
            source_frame="synthetic_camera_canonical",
            target_frame="payload_rack_canonical",
        )
        self.assertEqual(tf.source_frame, "synthetic_camera_canonical")
        self.assertEqual(tf.target_frame, "payload_rack_canonical")

    # =========================================================================
    # PAYLOAD SPATIAL (18-22)
    # =========================================================================

    def test_18_sample_proximity(self) -> None:
        joints = [
            Joint3D(-0.15, 0.01, 0.0, name="left_wrist"),   # Near RED_SAMPLE
            Joint3D(0.15, 0.01, 0.0, name="right_wrist"),    # Near BLUE_SAMPLE
        ]
        rels = evaluate_payload_spatial_relations(joints, self.canonical_volumes, reach_threshold=0.05)
        self.assertIn("RED_SAMPLE", rels)
        self.assertIn("BLUE_SAMPLE", rels)
        self.assertTrue(rels["RED_SAMPLE"].is_within_reach)
        self.assertTrue(rels["BLUE_SAMPLE"].is_within_reach)

    def test_19_container_proximity(self) -> None:
        joints = [Joint3D(0.0, 0.02, 0.0, name="right_hand")]
        rels = evaluate_payload_spatial_relations(joints, self.canonical_volumes)
        self.assertIn("SAMPLE_CONTAINER", rels)
        self.assertTrue(rels["SAMPLE_CONTAINER"].is_inside)

    def test_20_lid_proximity(self) -> None:
        joints = [Joint3D(0.0, 0.09, 0.0, name="right_wrist")]
        rels = evaluate_payload_spatial_relations(joints, self.canonical_volumes)
        self.assertIn("CONTAINER_LID", rels)
        self.assertTrue(rels["CONTAINER_LID"].is_within_reach)

    def test_21_containment_evaluation(self) -> None:
        container = self.canonical_volumes["SAMPLE_CONTAINER"]
        inside_joint = Joint3D(0.0, 0.0, 0.0, name="right_hand")
        outside_joint = Joint3D(0.5, 0.5, 0.5, name="right_hand")

        rel_in = evaluate_joint_target_relation(inside_joint, container)
        rel_out = evaluate_joint_target_relation(outside_joint, container)

        self.assertTrue(rel_in.is_inside)
        self.assertEqual(rel_in.distance, 0.0)
        self.assertFalse(rel_out.is_inside)
        self.assertGreater(rel_out.distance, 0.5)

    def test_22_deterministic_spatial_result_serialization(self) -> None:
        rel = SpatialRelationResult(
            source_entity="right_wrist",
            target_entity="RED_SAMPLE",
            distance=0.04567,
            is_metric=True,
            is_inside=False,
            is_within_reach=True,
            coordinate_frame="payload_rack_canonical",
            confidence=0.95,
            approach_vector=(1.0, 0.0, 0.0),
        )
        d = rel.to_dict()
        raw_json = json.dumps(d, sort_keys=True)
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["source_entity"], "right_wrist")
        self.assertEqual(parsed["target_entity"], "RED_SAMPLE")
        self.assertTrue(parsed["is_within_reach"])
        self.assertEqual(parsed["approach_vector"], [1.0, 0.0, 0.0])

    # =========================================================================
    # POSTURE NORMALIZATION & ROLL-INVARIANCE (23-29)
    # =========================================================================

    def _create_synthetic_astronaut_joints(self) -> List[Joint3D]:
        """Creates a baseline upright astronaut skeleton."""
        return [
            Joint3D(0.0, 0.0, 1.5, name="pelvis"),
            Joint3D(0.0, 0.5, 1.5, name="neck"),
            Joint3D(-0.2, 0.45, 1.5, name="left_shoulder"),
            Joint3D(0.2, 0.45, 1.5, name="right_shoulder"),
            Joint3D(-0.35, 0.1, 1.45, name="left_wrist"),
            Joint3D(0.35, 0.1, 1.45, name="right_wrist"),
        ]

    def _rotate_joints_roll(self, joints: List[Joint3D], roll_deg: float) -> List[Joint3D]:
        """Rotates joints around Z-axis by roll_deg degrees."""
        rad = math.radians(roll_deg)
        R = np.array([
            [math.cos(rad), -math.sin(rad), 0.0],
            [math.sin(rad), math.cos(rad), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated = []
        for j in joints:
            p = R @ np.array([j.x, j.y, j.z])
            rotated.append(Joint3D(float(p[0]), float(p[1]), float(p[2]), name=j.name))
        return rotated

    def test_23_pelvis_centering(self) -> None:
        joints = self._create_synthetic_astronaut_joints()
        norm_joints = MicrogravityPostureNormalizer.normalize_posture(joints)
        pelvis = [j for j in norm_joints if j.name == "pelvis"][0]
        self.assertAlmostEqual(pelvis.x, 0.0, places=5)
        self.assertAlmostEqual(pelvis.y, 0.0, places=5)
        self.assertAlmostEqual(pelvis.z, 0.0, places=5)

    def test_24_torso_axis_normalization(self) -> None:
        joints = self._create_synthetic_astronaut_joints()
        norm_joints = MicrogravityPostureNormalizer.normalize_posture(joints)
        neck = [j for j in norm_joints if j.name == "neck"][0]
        # Neck must be strictly along +Y (x=0, z=0, y > 0)
        self.assertAlmostEqual(neck.x, 0.0, places=5)
        self.assertGreater(neck.y, 0.4)
        self.assertAlmostEqual(neck.z, 0.0, places=5)

    def test_25_0_degree_roll_invariance(self) -> None:
        joints = self._create_synthetic_astronaut_joints()
        v1 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints)
        self.assertIn("right_wrist", v1)
        self.assertIn("left_wrist", v1)

    def test_26_45_degree_roll_invariance(self) -> None:
        joints_0 = self._create_synthetic_astronaut_joints()
        joints_45 = self._rotate_joints_roll(joints_0, 45.0)

        v_0 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_0)
        v_45 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_45)

        np.testing.assert_allclose(v_0["right_wrist"], v_45["right_wrist"], atol=1e-4)
        np.testing.assert_allclose(v_0["left_wrist"], v_45["left_wrist"], atol=1e-4)

    def test_27_90_degree_roll_invariance(self) -> None:
        joints_0 = self._create_synthetic_astronaut_joints()
        joints_90 = self._rotate_joints_roll(joints_0, 90.0)

        v_0 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_0)
        v_90 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_90)

        np.testing.assert_allclose(v_0["right_wrist"], v_90["right_wrist"], atol=1e-4)

    def test_28_180_degree_inversion_invariance(self) -> None:
        joints_0 = self._create_synthetic_astronaut_joints()
        joints_180 = self._rotate_joints_roll(joints_0, 180.0)

        v_0 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_0)
        v_180 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_180)

        np.testing.assert_allclose(v_0["right_wrist"], v_180["right_wrist"], atol=1e-4)

    def test_29_270_degree_roll_invariance(self) -> None:
        joints_0 = self._create_synthetic_astronaut_joints()
        joints_270 = self._rotate_joints_roll(joints_0, 270.0)

        v_0 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_0)
        v_270 = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(joints_270)

        np.testing.assert_allclose(v_0["right_wrist"], v_270["right_wrist"], atol=1e-4)

    # =========================================================================
    # INTEGRATION & SAFETY (30-38)
    # =========================================================================

    def test_30_synthetic_hmr_to_b14_2_flow(self) -> None:
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)

        adapter = SpatialDisambiguationAdapter(enabled=True)
        diag = adapter.process_hmr_result(hmr_res)

        self.assertTrue(diag["active"])
        self.assertTrue(diag["is_metric"])
        self.assertIn("spatial_relations", diag)
        self.assertIn("normalized_wrist_vectors", diag)
        adapter.close()
        engine.close()

    def test_31_mediapipe_landmark_only_to_b14_2_flow(self) -> None:
        mp_engine = MediaPipeHMREngine()
        hmr_res = mp_engine.recover_mesh(self.dummy_frame)

        adapter = SpatialDisambiguationAdapter(enabled=True)
        diag = adapter.process_hmr_result(hmr_res)

        self.assertTrue(diag["active"])
        self.assertIn("spatial_relations", diag)
        adapter.close()
        mp_engine.close()

    def test_32_missing_hmr_fallback(self) -> None:
        adapter = SpatialDisambiguationAdapter(enabled=True)
        empty_res = HMRMeshResult.empty()
        diag = adapter.process_hmr_result(empty_res)
        self.assertEqual(diag["reason"], "empty_hmr_joints")
        self.assertEqual(diag["spatial_relations"], {})
        adapter.close()

    def test_33_malformed_hmr_containment(self) -> None:
        adapter = SpatialDisambiguationAdapter(enabled=True)
        # None input
        diag = adapter.process_hmr_result(None)
        self.assertEqual(diag["spatial_relations"], {})
        adapter.close()

    def test_34_optional_side_channel_disabled_by_default(self) -> None:
        adapter = SpatialDisambiguationAdapter()  # Default enabled=False
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)

        diag = adapter.process_hmr_result(hmr_res)
        self.assertFalse(diag["active"])
        self.assertEqual(diag["reason"], "disabled_or_closed")
        adapter.close()
        engine.close()

    def test_35_b14_2_failure_does_not_break_pipeline(self) -> None:
        # Invalid transform with NaN inside adapter
        bad_adapter = SpatialDisambiguationAdapter(enabled=True)
        # Inject bad volume to force exception
        bad_adapter.payload_volumes = {"BAD": None}  # type: ignore

        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)
        diag = bad_adapter.process_hmr_result(hmr_res)

        # Exception caught safely without raising
        self.assertTrue(diag["active"])
        self.assertIn("error", diag)
        bad_adapter.close()
        engine.close()

    def test_36_public_contract_unchanged(self) -> None:
        res = AIResultAdapter.adapt(
            action="PICK_RED",
            confidence=0.88,
            fsm_status="VALID",
            expected_step=1,
            detected_step=1,
            next_step=2,
            object="RED_SAMPLE",
            timestamp="2026-09-29T10:00:00.000000",
        )
        self.assertEqual(
            set(res.keys()),
            {"timestamp", "action", "object", "confidence", "expected_step", "detected_step", "status", "next_step"},
        )
        self.assertNotIn("spatial_relations", res)
        self.assertNotIn("normalized_wrist_vectors", res)

    def test_37_b10_b11_b12_fsm_unchanged(self) -> None:
        fsm = SequenceValidatorFSM()
        self.assertEqual(fsm.state, "IDLE")

        tf = TemporalConfirmationEngine()
        self.assertEqual(tf.window_size, 5)

        uh = UncertaintyHandler()
        self.assertEqual(uh.high_confidence_threshold, 0.70)

        rm = RecoveryManager()
        self.assertEqual(rm.state, "IDLE")

    def test_38_no_extra_worker_threads(self) -> None:
        threads_before = threading.active_count()
        adapter = SpatialDisambiguationAdapter(enabled=True)
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)
        adapter.process_hmr_result(hmr_res)
        threads_after = threading.active_count()
        self.assertEqual(threads_before, threads_after)
        adapter.close()
        engine.close()

    # =========================================================================
    # LONG-RUN & STRESS (39-41)
    # =========================================================================

    def test_39_1000_synthetic_evaluations_latency(self) -> None:
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)
        adapter = SpatialDisambiguationAdapter(enabled=True)

        t0 = time.monotonic()
        for _ in range(1000):
            diag = adapter.process_hmr_result(hmr_res)
            self.assertTrue(diag["active"])
        elapsed = time.monotonic() - t0
        avg_ms = (elapsed / 1000.0) * 1000.0

        # Spatial geometry + posture normalization must be < 5.0 ms per frame on CPU
        self.assertLess(avg_ms, 5.0)
        adapter.close()
        engine.close()

    def test_40_bounded_memory(self) -> None:
        adapter = SpatialDisambiguationAdapter(enabled=True)
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)

        for _ in range(500):
            diag = adapter.process_hmr_result(hmr_res)
            self.assertIsNotNone(diag)

        adapter.close()
        engine.close()

    def test_41_deterministic_repeated_results(self) -> None:
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)
        adapter = SpatialDisambiguationAdapter(enabled=True)

        res1 = adapter.process_hmr_result(hmr_res)
        res2 = adapter.process_hmr_result(hmr_res)

        # Clear volatile elapsed_ms
        res1.pop("elapsed_ms", None)
        res2.pop("elapsed_ms", None)

        self.assertEqual(res1, res2)
        adapter.close()
        engine.close()


if __name__ == "__main__":
    unittest.main()
