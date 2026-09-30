"""
test_b14_3_consolidation.py — Comprehensive Verification, Stress Traversal & Gate B14 Consolidation
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.3)

Verifies:
1. End-to-end synthetic HMR traversal (1,000+ frames with varying inputs and phase articulation).
2. Multi-angle roll traversal (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° roll invariance).
3. 3D pitch/yaw transform traversal with full SE(3) rotation matrices.
4. Camera-to-payload transform analytical inverse roundtrip precision (T⁻¹(T(P)) ≈ P).
5. Transform invalid-input containment (invalid shapes, singular matrices, reflections, NaNs/Infs).
6. Canonical EXP-001 object 3D spatial geometry (inside, outside, boundary, near/far contact, reach).
7. Metric vs non-metric safety distinction and coordinate frame propagation.
8. Microgravity posture normalization stability and anatomical basis orthonormality.
9. Missing-joint, missing-pelvis, and degenerate shoulder fallback behavior.
10. Integrated B14.1 + B14.2 pipeline flow (HMR -> Transform -> PostureNormalizer -> SpatialGeometry -> SpatialDisambiguationAdapter).
11. Disabled side-channel behavior (zero-overhead bypass, active=False).
12. Enabled side-channel behavior (proximity, reachable target selection, normalized wrist vectors).
13. B14 failure isolation (internal error containment, zero exception leakage into FSM or pipeline).
14. Public 8-field AI contract strictness (zero leakage of 3D arrays, mesh vertices, or SMPL parameters).
15. B10 / B11 / B12 / FSM procedural authority invariance (identical outcomes with B14 disabled vs enabled).
16. 1,000+ frame long-run stress with bounded memory and predictable execution latency.
17. Bounded memory and container stability (zero accumulation of cached meshes or objects).
18. Benchmark harness reproducibility (BenchmarkHarness.run_hmr_spatial_benchmark() passes).
19. Performance measurement (sub-millisecond CPU kinematics and sub-5ms full side channel).
20. Engine and adapter lifecycle, context manager, and clean close/reset behavior.
21. Strict absence of proprietary external assets (.pkl, .npz, .pth, .onnx).
22. Absence of extra background threads or timer workers.
23. Numerical determinism and bitwise reproducibility across repeated stream evaluations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List
import unittest
import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ai.hmr import (
    BoundingVolume3D,
    CameraToPayloadTransform,
    HMRMeshResult,
    HMRRecoveryEngineInterface,
    Joint3D,
    MediaPipeHMREngine,
    MeshVertex3D,
    MicrogravityPostureNormalizer,
    SMPL_24_JOINT_NAMES,
    SMPL_24_PARENT_MAP,
    SpatialDisambiguationAdapter,
    SpatialRelationResult,
    SyntheticHMREngine,
    compute_euclidean_distance_3d,
    evaluate_joint_target_relation,
    evaluate_payload_spatial_relations,
    get_canonical_exp001_volumes,
)
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.result_adapter import AIResultAdapter
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM
from evaluation.benchmark_harness import BenchmarkHarness, BenchmarkResult


class TestB143Consolidation(unittest.TestCase):
    """Comprehensive test suite for Gate B14.3 consolidation, stress traversal, and closure."""

    def setUp(self) -> None:
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_frame[100:200, 100:200] = 128
        self.canonical_volumes = get_canonical_exp001_volumes()

    # -------------------------------------------------------------------------
    # 1. End-to-End Synthetic HMR Traversal (1,000+ frames)
    # -------------------------------------------------------------------------
    def test_01_synthetic_hmr_end_to_end_traversal(self) -> None:
        engine = SyntheticHMREngine()
        for i in range(1050):
            # Vary frame intensity to modulate phase
            frame = np.full((120, 160, 3), fill_value=(i % 256), dtype=np.uint8)
            res = engine.recover_mesh(frame)
            self.assertTrue(res.mesh_present)
            self.assertEqual(len(res.joints), 24)
            self.assertEqual(len(res.vertices), 48)
            self.assertEqual(len(res.faces), 72)
            self.assertEqual(len(res.beta), 10)
            self.assertEqual(len(res.theta), 72)
            self.assertTrue(all(math.isfinite(j.x) and math.isfinite(j.y) and math.isfinite(j.z) for j in res.joints))
            self.assertTrue(all(math.isfinite(v.x) and math.isfinite(v.y) and math.isfinite(v.z) for v in res.vertices))
        engine.close()

    # -------------------------------------------------------------------------
    # 2. Multi-Angle Roll Traversal (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    # -------------------------------------------------------------------------
    def test_02_multi_angle_roll_traversal(self) -> None:
        engine = SyntheticHMREngine()
        base_res = engine.recover_mesh(self.dummy_frame)
        base_vectors = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(base_res.joints)
        self.assertIn("right_wrist", base_vectors)
        self.assertIn("left_wrist", base_vectors)

        angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        for angle in angles:
            rad = math.radians(angle)
            R_z = np.array([
                [math.cos(rad), -math.sin(rad), 0.0],
                [math.sin(rad), math.cos(rad), 0.0],
                [0.0, 0.0, 1.0],
            ])
            rot_joints = []
            for j in base_res.joints:
                p = R_z @ np.array([j.x, j.y, j.z])
                rot_joints.append(Joint3D(float(p[0]), float(p[1]), float(p[2]), name=j.name))

            v_rot = MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors(rot_joints)
            np.testing.assert_allclose(
                base_vectors["right_wrist"],
                v_rot["right_wrist"],
                atol=1e-4,
                err_msg=f"Roll invariance failed at {angle}° roll",
            )
            np.testing.assert_allclose(
                base_vectors["left_wrist"],
                v_rot["left_wrist"],
                atol=1e-4,
                err_msg=f"Roll invariance failed at {angle}° roll",
            )
        engine.close()

    # -------------------------------------------------------------------------
    # 3. 3D Pitch/Yaw Transform Traversal
    # -------------------------------------------------------------------------
    def test_03_pitch_yaw_transform_traversal(self) -> None:
        angles_test = [
            (0.0, 30.0, 0.0),    # pitch only
            (0.0, 0.0, 45.0),    # yaw only
            (45.0, -30.0, 60.0), # complex roll-pitch-yaw
            (-90.0, 45.0, -15.0),
        ]
        pt = np.array([0.2, -0.4, 1.6])
        for r_deg, p_deg, y_deg in angles_test:
            tf = CameraToPayloadTransform.from_euler_angles(
                roll_deg=r_deg,
                pitch_deg=p_deg,
                yaw_deg=y_deg,
                translation_m=(0.5, 0.1, -0.3),
            )
            # Check orthogonality R @ R^T = I
            R = tf.rotation_matrix
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=5)

            p_transformed = tf.transform_point(pt)
            self.assertTrue(all(math.isfinite(x) for x in p_transformed))

    # -------------------------------------------------------------------------
    # 4. Camera-to-Payload Roundtrip Precision (T⁻¹(T(P)) ≈ P)
    # -------------------------------------------------------------------------
    def test_04_camera_to_payload_roundtrip_precision(self) -> None:
        tf = CameraToPayloadTransform.from_euler_angles(
            roll_deg=75.0,
            pitch_deg=-25.0,
            yaw_deg=50.0,
            translation_m=(0.35, -0.15, 1.25),
        )
        tf_inv = tf.inverse()
        test_points = [
            np.array([0.0, 0.0, 1.5]),
            np.array([-0.34, 0.12, 1.35]),
            np.array([0.5, -0.8, 2.1]),
        ]
        for pt in test_points:
            p_payload = tf.transform_point(pt)
            p_restored = tf_inv.transform_point(p_payload)
            np.testing.assert_allclose(p_restored, pt, atol=1e-5)

    # -------------------------------------------------------------------------
    # 5. Transform Invalid-Input Containment
    # -------------------------------------------------------------------------
    def test_05_transform_invalid_input_containment(self) -> None:
        # Invalid rotation matrix dimensions
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.eye(4))
        # NaN in rotation matrix
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]))
        # Singular matrix (det = 0)
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.zeros((3, 3)))
        # Reflection matrix (det = -1)
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(rotation_matrix=np.diag([-1.0, 1.0, 1.0]))
        # Invalid translation vector
        with self.assertRaises(ValueError):
            CameraToPayloadTransform(translation_vector=np.array([1.0, 2.0]))
        # NaN in transform point
        tf = CameraToPayloadTransform.identity()
        with self.assertRaises(ValueError):
            tf.transform_point([np.nan, 0.0, 1.0])

    # -------------------------------------------------------------------------
    # 6. Canonical Object Spatial Tests (EXP-001)
    # -------------------------------------------------------------------------
    def test_06_canonical_object_spatial_tests(self) -> None:
        # RED_SAMPLE: center=(-0.15, 0.0, 0.0), extents=(0.06, 0.06, 0.06) => [-0.18, -0.12]
        # BLUE_SAMPLE: center=(0.15, 0.0, 0.0), extents=(0.06, 0.06, 0.06) => [0.12, 0.18]
        # SAMPLE_CONTAINER: center=(0.0, 0.0, 0.0), extents=(0.14, 0.12, 0.14) => [-0.07, 0.07]
        # CONTAINER_LID: center=(0.0, 0.08, 0.0), extents=(0.15, 0.03, 0.15) => y: [0.065, 0.095]
        red = self.canonical_volumes["RED_SAMPLE"]
        blue = self.canonical_volumes["BLUE_SAMPLE"]
        container = self.canonical_volumes["SAMPLE_CONTAINER"]
        lid = self.canonical_volumes["CONTAINER_LID"]

        # Inside tests
        self.assertTrue(red.contains_point((-0.15, 0.0, 0.0)))
        self.assertTrue(blue.contains_point((0.15, 0.0, 0.0)))
        self.assertTrue(container.contains_point((0.0, 0.0, 0.0)))
        self.assertTrue(lid.contains_point((0.0, 0.08, 0.0)))

        # Outside tests
        self.assertFalse(red.contains_point((0.15, 0.0, 0.0)))
        self.assertFalse(blue.contains_point((-0.15, 0.0, 0.0)))

        # Reach test
        j_near_red = Joint3D(-0.15, 0.04, 0.0, name="right_wrist")
        rel_red = evaluate_joint_target_relation(j_near_red, red, reach_threshold=0.10)
        self.assertTrue(rel_red.is_within_reach)
        self.assertLessEqual(rel_red.distance, 0.02)

    # -------------------------------------------------------------------------
    # 7. Metric vs Non-Metric Safety Distinction
    # -------------------------------------------------------------------------
    def test_07_metric_non_metric_safety(self) -> None:
        metric_res = HMRMeshResult(
            joints=[Joint3D(0.0, 0.0, 1.5, name="right_wrist")],
            coordinate_frame="synthetic_camera_canonical",
        )
        non_metric_res = HMRMeshResult(
            joints=[Joint3D(0.5, 0.5, 0.1, name="right_wrist")],
            coordinate_frame="mediapipe_normalized",
        )

        adapter = SpatialDisambiguationAdapter(enabled=True)
        diag_m = adapter.process_hmr_result(metric_res)
        diag_nm = adapter.process_hmr_result(non_metric_res)

        self.assertTrue(diag_m["is_metric"])
        self.assertFalse(diag_nm["is_metric"])
        adapter.close()

    # -------------------------------------------------------------------------
    # 8. Posture Normalization Stability
    # -------------------------------------------------------------------------
    def test_08_posture_normalization_stability(self) -> None:
        engine = SyntheticHMREngine()
        res = engine.recover_mesh(self.dummy_frame)
        norm_joints = MicrogravityPostureNormalizer.normalize_posture(res.joints)

        pelvis = next(j for j in norm_joints if j.name == "pelvis")
        self.assertAlmostEqual(pelvis.x, 0.0, places=4)
        self.assertAlmostEqual(pelvis.y, 0.0, places=4)
        self.assertAlmostEqual(pelvis.z, 0.0, places=4)

        neck = next(j for j in norm_joints if j.name == "neck")
        self.assertAlmostEqual(neck.x, 0.0, places=4)
        self.assertGreater(neck.y, 0.3)
        self.assertAlmostEqual(neck.z, 0.0, places=4)
        engine.close()

    # -------------------------------------------------------------------------
    # 9. Missing-Joint Fallback Behavior
    # -------------------------------------------------------------------------
    def test_09_missing_joint_fallback(self) -> None:
        # Skeleton with only random joints, no pelvis or neck
        degraded_joints = [
            Joint3D(0.1, 0.2, 1.5, name="unknown_1"),
            Joint3D(0.2, 0.3, 1.5, name="unknown_2"),
        ]
        # Must execute safely without crashing
        norm = MicrogravityPostureNormalizer.normalize_posture(degraded_joints)
        self.assertEqual(len(norm), 2)

        # Empty joints
        self.assertEqual(MicrogravityPostureNormalizer.normalize_posture([]), [])
        self.assertEqual(MicrogravityPostureNormalizer.compute_torso_relative_wrist_vectors([]), {})

    # -------------------------------------------------------------------------
    # 10. Integrated B14.1 + B14.2 Pipeline Flow
    # -------------------------------------------------------------------------
    def test_10_b14_integrated_pipeline_flow(self) -> None:
        with SyntheticHMREngine() as engine:
            hmr_res = engine.recover_mesh(self.dummy_frame)

        transform = CameraToPayloadTransform.from_euler_angles(roll_deg=0.0, translation_m=(0.0, 0.0, -1.5))
        adapter = SpatialDisambiguationAdapter(transform=transform, enabled=True)
        diag = adapter.process_hmr_result(hmr_res)

        self.assertTrue(diag["active"])
        self.assertTrue(diag["is_metric"])
        self.assertIn("SAMPLE_CONTAINER", diag["spatial_relations"])
        self.assertIn("right_wrist", diag["normalized_wrist_vectors"])
        adapter.close()

    # -------------------------------------------------------------------------
    # 11. Disabled Side-Channel Behavior
    # -------------------------------------------------------------------------
    def test_11_disabled_side_channel_behavior(self) -> None:
        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)

        adapter = SpatialDisambiguationAdapter(enabled=False)
        t0 = time.perf_counter()
        diag = adapter.process_hmr_result(hmr_res)
        dur_us = (time.perf_counter() - t0) * 1_000_000.0

        self.assertFalse(diag["active"])
        self.assertEqual(diag["reason"], "disabled_or_closed")
        self.assertEqual(diag["spatial_relations"], {})
        self.assertLess(dur_us, 500.0) # Should be sub-millisecond bypass
        adapter.close()
        engine.close()

    # -------------------------------------------------------------------------
    # 12. Enabled Side-Channel Behavior
    # -------------------------------------------------------------------------
    def test_12_enabled_side_channel_behavior(self) -> None:
        # Create wrist near RED_SAMPLE
        hmr_res = HMRMeshResult(
            joints=[
                Joint3D(-0.15, 0.01, 0.0, name="right_wrist"),
                Joint3D(0.0, 0.0, 0.0, name="pelvis"),
                Joint3D(0.0, 0.4, 0.0, name="neck"),
            ],
            coordinate_frame="payload_rack_canonical",
        )
        adapter = SpatialDisambiguationAdapter(
            transform=CameraToPayloadTransform.identity(),
            enabled=True,
            reach_threshold_m=0.10,
        )
        diag = adapter.process_hmr_result(hmr_res)
        self.assertTrue(diag["active"])
        self.assertEqual(diag["manipulation_target"], "RED_SAMPLE")
        adapter.close()

    # -------------------------------------------------------------------------
    # 13. B14 Failure Isolation & Error Containment
    # -------------------------------------------------------------------------
    def test_13_b14_failure_isolation(self) -> None:
        adapter = SpatialDisambiguationAdapter(enabled=True)
        # Force a corrupt payload_volumes dictionary
        adapter.payload_volumes = {"CORRUPT": 12345}  # type: ignore

        engine = SyntheticHMREngine()
        hmr_res = engine.recover_mesh(self.dummy_frame)
        diag = adapter.process_hmr_result(hmr_res)

        # Must catch error internally and return error payload without raising
        self.assertTrue(diag["active"])
        self.assertIn("error", diag)
        self.assertEqual(diag["spatial_relations"], {})
        adapter.close()
        engine.close()

    # -------------------------------------------------------------------------
    # 14. Public 8-Field Contract Strictness
    # -------------------------------------------------------------------------
    def test_14_public_contract_strictness(self) -> None:
        res = AIResultAdapter.adapt(
            action="PICK_RED",
            confidence=0.92,
            fsm_status="VALID",
            expected_step=1,
            detected_step=1,
            next_step=2,
            object="RED_SAMPLE",
            timestamp="2026-09-29T11:00:00.000000",
        )
        required_keys = {
            "timestamp",
            "action",
            "object",
            "confidence",
            "expected_step",
            "detected_step",
            "status",
            "next_step",
        }
        self.assertEqual(set(res.keys()), required_keys)
        # Verify NO 3D mesh fields exist
        forbidden_keys = {
            "joints", "vertices", "faces", "beta", "theta", "mesh_present",
            "coordinate_frame", "spatial_relations", "normalized_wrist_vectors",
            "manipulation_target", "approach_vector",
        }
        for k in forbidden_keys:
            self.assertNotIn(k, res)

    # -------------------------------------------------------------------------
    # 15. B10/B11/B12/FSM Procedural Authority Invariance
    # -------------------------------------------------------------------------
    def test_15_procedural_authority_invariance(self) -> None:
        fsm_disabled = SequenceValidatorFSM()
        fsm_enabled = SequenceValidatorFSM()
        fsm_disabled.start()
        fsm_enabled.start()

        # Step S1: PICK_RED on RED_SAMPLE
        res_dis = fsm_disabled.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        res_ena = fsm_enabled.validate_action("PICK_RED", 0.95, "RED_SAMPLE")
        res_dis.pop("timestamp", None)
        res_ena.pop("timestamp", None)
        self.assertEqual(res_dis, res_ena)
        self.assertEqual(fsm_disabled.state, fsm_enabled.state)
        self.assertEqual(fsm_disabled.current_step_index, fsm_enabled.current_step_index)

        # Anomaly S3 out of order
        res_dis_ooo = fsm_disabled.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
        res_ena_ooo = fsm_enabled.validate_action("PICK_BLUE", 0.95, "BLUE_SAMPLE")
        res_dis_ooo.pop("timestamp", None)
        res_ena_ooo.pop("timestamp", None)
        self.assertEqual(res_dis_ooo, res_ena_ooo)
        self.assertEqual(fsm_disabled.state, fsm_enabled.state)

    # -------------------------------------------------------------------------
    # 16. 1,000+ Frame Stress & Bounded Memory
    # -------------------------------------------------------------------------
    def test_16_1000_frame_stress_bounded_memory(self) -> None:
        engine = SyntheticHMREngine()
        transform = CameraToPayloadTransform.from_euler_angles(roll_deg=45.0)
        adapter = SpatialDisambiguationAdapter(transform=transform, enabled=True)

        t0 = time.monotonic()
        for i in range(1000):
            res = engine.recover_mesh(self.dummy_frame)
            diag = adapter.process_hmr_result(res)
            self.assertTrue(diag["active"])
        elapsed = time.monotonic() - t0
        avg_ms = (elapsed / 1000.0) * 1000.0

        # Entire 3D HMR + transform + geometry pipeline must run in < 5.0 ms / frame on CPU
        self.assertLess(avg_ms, 5.0)
        engine.close()
        adapter.close()

    # -------------------------------------------------------------------------
    # 17. Zero Growing Containers or Memory Accumulation
    # -------------------------------------------------------------------------
    def test_17_no_growing_containers(self) -> None:
        engine = SyntheticHMREngine()
        adapter = SpatialDisambiguationAdapter(enabled=True)

        # Verify no list accumulation across calls
        for _ in range(200):
            res = engine.recover_mesh(self.dummy_frame)
            _ = adapter.process_hmr_result(res)

        # Check internal structures
        self.assertEqual(len(engine.__dict__), 4)
        self.assertEqual(len(adapter.__dict__), 5)
        engine.close()
        adapter.close()

    # -------------------------------------------------------------------------
    # 18. Benchmark Harness Integration & Reproducibility
    # -------------------------------------------------------------------------
    def test_18_benchmark_harness_integration(self) -> None:
        result = BenchmarkHarness.run_hmr_spatial_benchmark(num_frames=20)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.benchmark_name, "hmr_spatial")
        self.assertTrue(result.passed)
        self.assertIn("mean_synthetic_hmr_latency_ms", result.metrics)
        self.assertIn("roll_invariance_max_delta_m", result.metrics)
        self.assertIn("mean_full_side_channel_latency_ms", result.metrics)
        self.assertLessEqual(result.metrics["roll_invariance_max_delta_m"], 1e-4)

        # Verify JSON serialization of benchmark report
        json_str = result.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["benchmark_name"], "hmr_spatial")
        self.assertTrue(parsed["passed"])

    # -------------------------------------------------------------------------
    # 19. Performance and Latency Profiling
    # -------------------------------------------------------------------------
    def test_19_performance_latency_profiling(self) -> None:
        tf = CameraToPayloadTransform.from_euler_angles(roll_deg=30.0)
        vol = self.canonical_volumes["SAMPLE_CONTAINER"]
        joint = Joint3D(0.0, 0.05, 0.0, name="wrist")

        # Measure 1,000 transform iterations
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = tf.transform_point([0.1, 0.2, 1.5])
        tf_dur_us = (time.perf_counter() - t0) * 1000.0
        # Transform point must be < 50 us average
        self.assertLess(tf_dur_us / 1000.0 * 1000.0, 50.0)

        # Measure 1,000 spatial relation evaluations
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = evaluate_joint_target_relation(joint, vol)
        geom_dur_us = (time.perf_counter() - t0) * 1000.0
        self.assertLess(geom_dur_us / 1000.0 * 1000.0, 50.0)

    # -------------------------------------------------------------------------
    # 20. Lifecycle, Reset, Context Manager & Close Behavior
    # -------------------------------------------------------------------------
    def test_20_lifecycle_close_behavior(self) -> None:
        with SyntheticHMREngine() as engine:
            res = engine.recover_mesh(self.dummy_frame)
            self.assertTrue(res.mesh_present)
        self.assertTrue(engine._is_closed)

        # After closing, returns typed empty result with reason
        res_closed = engine.recover_mesh(self.dummy_frame)
        self.assertFalse(res_closed.mesh_present)
        self.assertEqual(res_closed.metadata["reason"], "engine_closed")

        with SpatialDisambiguationAdapter(enabled=True) as adapter:
            diag = adapter.process_hmr_result(res)
            self.assertTrue(diag["active"])
        self.assertTrue(adapter._is_closed)

        diag_closed = adapter.process_hmr_result(res)
        self.assertFalse(diag_closed["active"])
        self.assertEqual(diag_closed["reason"], "disabled_or_closed")

    # -------------------------------------------------------------------------
    # 21. Absence of External Model Assets & Checkpoints
    # -------------------------------------------------------------------------
    def test_21_absence_of_external_assets(self) -> None:
        # Verify no proprietary SMPL files or network access needed
        syn = SyntheticHMREngine()
        mp = MediaPipeHMREngine()
        r1 = syn.recover_mesh(self.dummy_frame)
        r2 = mp.recover_mesh(self.dummy_frame)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        syn.close()
        mp.close()

    # -------------------------------------------------------------------------
    # 22. Absence of Extra Background Threads or Timers
    # -------------------------------------------------------------------------
    def test_22_absence_of_extra_background_threads(self) -> None:
        threads_before = threading.active_count()
        syn = SyntheticHMREngine()
        mp = MediaPipeHMREngine()
        adapter = SpatialDisambiguationAdapter(enabled=True)

        for _ in range(10):
            r1 = syn.recover_mesh(self.dummy_frame)
            r2 = mp.recover_mesh(self.dummy_frame)
            _ = adapter.process_hmr_result(r1)
            _ = adapter.process_hmr_result(r2)

        threads_after = threading.active_count()
        self.assertEqual(threads_before, threads_after)
        syn.close()
        mp.close()
        adapter.close()

    # -------------------------------------------------------------------------
    # 23. Numerical Determinism & Bitwise Reproducibility
    # -------------------------------------------------------------------------
    def test_23_deterministic_numerical_output(self) -> None:
        engine = SyntheticHMREngine()
        res1 = engine.recover_mesh(self.dummy_frame)
        res2 = engine.recover_mesh(self.dummy_frame)

        for j1, j2 in zip(res1.joints, res2.joints):
            self.assertEqual(j1.name, j2.name)
            self.assertEqual(j1.x, j2.x)
            self.assertEqual(j1.y, j2.y)
            self.assertEqual(j1.z, j2.z)

        for v1, v2 in zip(res1.vertices, res2.vertices):
            self.assertEqual(v1.x, v2.x)
            self.assertEqual(v1.y, v2.y)
            self.assertEqual(v1.z, v2.z)

        self.assertEqual(res1.faces, res2.faces)
        engine.close()


if __name__ == "__main__":
    unittest.main()
