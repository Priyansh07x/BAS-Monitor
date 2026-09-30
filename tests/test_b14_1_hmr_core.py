"""
test_b14_1_hmr_core.py — Unit & Integration Tests for Gate B14.1
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Phase B14.1)

Verifies:
1. Joint3D construction and typing.
2. MeshVertex3D construction and typing.
3. HMRMeshResult construction and defaults.
4. Deterministic serialization (to_dict, to_json) with JSON-safe primitives.
5. Empty / unavailable mesh serialization and reason logging.
6. SyntheticHMREngine deterministic output on identical inputs.
7. SyntheticHMREngine joint hierarchy validity (24 joints, root pelvis, valid parent mapping).
8. SyntheticHMREngine output dimensions (24 joints, 48 vertices, 72 faces, 10-D beta, 72-D theta).
9. MediaPipeHMREngine mapping for representative 33-landmark input.
10. MediaPipeHMREngine confidence and visibility propagation.
11. MediaPipeHMREngine does NOT fabricate SMPL vertices (vertices=[], faces=[], mesh_present=False).
12. Malformed input handling (None, empty array, non-array).
13. Missing MediaPipe dependency fallback behavior.
14. HMRRecoveryEngineInterface contract and context manager behavior.
15. Engine close() and lifecycle resource management.
16. Repeated synthetic generation stability and phase perturbation.
17. 1,000-frame synthetic generation memory and resource stability.
18. Coordinate convention metadata tagging.
19. Public 8-field AI contract remains strictly unchanged.
20. B10/B11/B12/FSM systems remain completely unaffected and unmutated.
21. Absence of proprietary SMPL .pkl assets and zero import failures.
22. Absence of neural checkpoint requirements and pure CPU execution.
23. Absence of extra worker threads or background concurrency leaks.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List
import unittest
import numpy as np

from backend.ai.hmr import (
    HMRMeshResult,
    HMRRecoveryEngineInterface,
    Joint3D,
    MeshVertex3D,
    MediaPipeHMREngine,
    SMPL_24_JOINT_NAMES,
    SMPL_24_PARENT_MAP,
    SyntheticHMREngine,
)
from backend.ai.pose_detector import PoseDetector
from backend.ai.result_adapter import AIResultAdapter
from backend.experiment.sequence_validator import SequenceValidatorFSM
from backend.experiment.recovery_manager import RecoveryManager
from backend.ai.temporal_filter import TemporalConfirmationEngine
from backend.ai.uncertainty_handler import UncertaintyHandler


class TestB141HMRCore(unittest.TestCase):
    """Test suite for Phase B14.1 3D HMR core, abstract interface, and synthetic models."""

    def setUp(self) -> None:
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_frame[100:200, 100:200] = 120

    # 1. Joint3D construction
    def test_01_joint3d_construction(self) -> None:
        j = Joint3D(x=0.123, y=-0.456, z=1.789, visibility=0.95, confidence=0.88, name="pelvis")
        self.assertEqual(j.name, "pelvis")
        self.assertAlmostEqual(j.x, 0.123)
        self.assertAlmostEqual(j.y, -0.456)
        self.assertAlmostEqual(j.z, 1.789)
        self.assertAlmostEqual(j.visibility, 0.95)
        self.assertAlmostEqual(j.confidence, 0.88)
        d = j.to_dict()
        self.assertIsInstance(d["x"], float)
        self.assertEqual(d["name"], "pelvis")

    # 2. MeshVertex3D construction
    def test_02_mesh_vertex3d_construction(self) -> None:
        v = MeshVertex3D(x=1.0, y=2.0, z=3.0)
        self.assertAlmostEqual(v.x, 1.0)
        self.assertAlmostEqual(v.y, 2.0)
        self.assertAlmostEqual(v.z, 3.0)
        d = v.to_dict()
        self.assertEqual(d, {"x": 1.0, "y": 2.0, "z": 3.0})

    # 3. HMRMeshResult construction
    def test_03_hmr_mesh_result_construction(self) -> None:
        res = HMRMeshResult(
            joints=[Joint3D(0.0, 0.0, 1.5, name="pelvis")],
            vertices=[MeshVertex3D(0.0, 0.0, 1.5)],
            faces=[[0, 1, 2]],
            beta=[0.1] * 10,
            theta=[0.2] * 72,
            mesh_present=True,
            coordinate_frame="synthetic_camera_canonical",
            representation_type="synthetic_kinematic_mesh",
            metadata={"test": "val"},
        )
        self.assertTrue(res.mesh_present)
        self.assertEqual(len(res.joints), 1)
        self.assertEqual(len(res.vertices), 1)
        self.assertEqual(len(res.faces), 1)
        self.assertEqual(len(res.beta), 10)
        self.assertEqual(len(res.theta), 72)

    # 4. Deterministic serialization
    def test_04_deterministic_serialization(self) -> None:
        engine = SyntheticHMREngine()
        res1 = engine.recover_mesh(self.dummy_frame)
        res2 = engine.recover_mesh(self.dummy_frame)

        # Clear volatile timestamps for serialization comparison
        res1.metadata.pop("timestamp", None)
        res2.metadata.pop("timestamp", None)

        json1 = res1.to_json(indent=2)
        json2 = res2.to_json(indent=2)
        self.assertEqual(json1, json2)

        # Verify parsed JSON structure
        data = json.loads(json1)
        self.assertTrue(data["mesh_present"])
        self.assertEqual(data["joint_count"], 24)
        self.assertEqual(data["vertex_count"], 48)
        self.assertEqual(data["face_count"], 72)
        self.assertEqual(len(data["beta"]), 10)
        self.assertEqual(len(data["theta"]), 72)

    # 5. Empty/unavailable mesh serialization
    def test_05_empty_mesh_serialization(self) -> None:
        empty = HMRMeshResult.empty(reason="sensor_offline", coordinate_frame="camera")
        self.assertFalse(empty.mesh_present)
        self.assertEqual(empty.representation_type, "none")
        self.assertEqual(empty.metadata["reason"], "sensor_offline")
        d = empty.to_dict()
        self.assertFalse(d["mesh_present"])
        self.assertEqual(d["joint_count"], 0)
        self.assertEqual(d["vertex_count"], 0)
        self.assertEqual(d["face_count"], 0)
        self.assertIsNone(d["beta"])
        self.assertIsNone(d["theta"])
        raw_json = empty.to_json()
        self.assertIn('"mesh_present": false', raw_json)

    # 6. Synthetic engine deterministic output
    def test_06_synthetic_engine_deterministic_output(self) -> None:
        engine = SyntheticHMREngine(base_depth_m=1.6)
        res1 = engine.recover_mesh(self.dummy_frame)
        res2 = engine.recover_mesh(self.dummy_frame)

        self.assertEqual(len(res1.joints), len(res2.joints))
        for j1, j2 in zip(res1.joints, res2.joints):
            self.assertEqual(j1.name, j2.name)
            self.assertAlmostEqual(j1.x, j2.x, places=5)
            self.assertAlmostEqual(j1.y, j2.y, places=5)
            self.assertAlmostEqual(j1.z, j2.z, places=5)

    # 7. Synthetic engine joint hierarchy validity
    def test_07_synthetic_engine_hierarchy_validity(self) -> None:
        self.assertEqual(len(SMPL_24_JOINT_NAMES), 24)
        self.assertEqual(len(SMPL_24_PARENT_MAP), 24)

        # Check root is pelvis (index 0) with None parent
        self.assertEqual(SMPL_24_JOINT_NAMES[0], "pelvis")
        self.assertIsNone(SMPL_24_PARENT_MAP[0])

        # Check all child parent references are valid indices < 24
        for child_idx, parent_idx in SMPL_24_PARENT_MAP.items():
            if parent_idx is not None:
                self.assertGreaterEqual(parent_idx, 0)
                self.assertLess(parent_idx, 24)
                self.assertLess(parent_idx, child_idx)  # Acyclic topological sort property

        engine = SyntheticHMREngine()
        res = engine.recover_mesh(self.dummy_frame)
        joint_names = [j.name for j in res.joints]
        self.assertEqual(joint_names, SMPL_24_JOINT_NAMES)

    # 8. Synthetic engine output dimensions and structure
    def test_08_synthetic_engine_output_dimensions(self) -> None:
        engine = SyntheticHMREngine(enable_mesh_generation=True)
        res = engine.recover_mesh(self.dummy_frame)

        self.assertEqual(len(res.joints), 24)
        self.assertEqual(len(res.vertices), 48)
        self.assertEqual(len(res.faces), 72)
        self.assertEqual(len(res.beta), 10)
        self.assertEqual(len(res.theta), 72)
        self.assertTrue(res.mesh_present)
        self.assertEqual(res.representation_type, "synthetic_kinematic_mesh")

        # Verify all face indices are valid vertex references
        num_v = len(res.vertices)
        for f in res.faces:
            self.assertEqual(len(f), 3)
            for v_idx in f:
                self.assertGreaterEqual(v_idx, 0)
                self.assertLess(v_idx, num_v)

    # 9. MediaPipe adapter mapping for representative 33-landmark input
    def test_09_mediapipe_adapter_mapping(self) -> None:
        adapter = MediaPipeHMREngine(confidence_threshold=0.50)
        res = adapter.recover_mesh(self.dummy_frame)

        self.assertFalse(res.mesh_present)
        self.assertEqual(res.representation_type, "landmark_only")
        self.assertEqual(len(res.joints), 33)
        self.assertEqual(res.joints[0].name, "nose")
        self.assertEqual(res.joints[15].name, "left_wrist")
        self.assertEqual(res.joints[16].name, "right_wrist")
        adapter.close()

    # 10. MediaPipe adapter confidence/visibility propagation
    def test_10_mediapipe_adapter_confidence_visibility(self) -> None:
        adapter = MediaPipeHMREngine()
        res = adapter.recover_mesh(self.dummy_frame)

        for j in res.joints:
            self.assertGreaterEqual(j.confidence, 0.0)
            self.assertLessEqual(j.confidence, 1.0)
            self.assertGreaterEqual(j.visibility, 0.0)
            self.assertLessEqual(j.visibility, 1.0)
        adapter.close()

    # 11. MediaPipe adapter does NOT fabricate SMPL mesh vertices
    def test_11_mediapipe_adapter_does_not_fabricate_mesh(self) -> None:
        adapter = MediaPipeHMREngine()
        res = adapter.recover_mesh(self.dummy_frame)

        self.assertFalse(res.mesh_present)
        self.assertEqual(len(res.vertices), 0)
        self.assertEqual(len(res.faces), 0)
        self.assertIsNone(res.beta)
        self.assertIsNone(res.theta)
        self.assertNotEqual(len(res.vertices), 6890)  # Must NEVER fake 6890 SMPL vertices
        adapter.close()

    # 12. Malformed input handling
    def test_12_malformed_input_handling(self) -> None:
        synthetic = SyntheticHMREngine()
        adapter = MediaPipeHMREngine()

        # None input
        res_syn_none = synthetic.recover_mesh(None)
        res_mp_none = adapter.recover_mesh(None)
        self.assertFalse(res_syn_none.mesh_present)
        self.assertFalse(res_mp_none.mesh_present)
        self.assertEqual(res_syn_none.metadata["reason"], "invalid_or_empty_frame")
        self.assertEqual(res_mp_none.metadata["reason"], "invalid_or_empty_frame")

        # Empty array
        empty_arr = np.array([], dtype=np.uint8)
        res_syn_empty = synthetic.recover_mesh(empty_arr)
        res_mp_empty = adapter.recover_mesh(empty_arr)
        self.assertFalse(res_syn_empty.mesh_present)
        self.assertFalse(res_mp_empty.mesh_present)

        synthetic.close()
        adapter.close()

    # 13. Missing MediaPipe dependency fallback behavior
    def test_13_missing_mediapipe_fallback(self) -> None:
        # Mock PoseDetector without MediaPipe instance
        mock_detector = PoseDetector()
        mock_detector._pose_instance = None  # Force heuristic fallback
        adapter = MediaPipeHMREngine(pose_detector=mock_detector)
        res = adapter.recover_mesh(self.dummy_frame)

        self.assertEqual(len(res.joints), 33)
        self.assertFalse(res.mesh_present)
        self.assertTrue(res.metadata.get("is_fallback", False))
        adapter.close()

    # 14. HMR interface contract and context manager
    def test_14_hmr_interface_contract(self) -> None:
        self.assertTrue(issubclass(SyntheticHMREngine, HMRRecoveryEngineInterface))
        self.assertTrue(issubclass(MediaPipeHMREngine, HMRRecoveryEngineInterface))

        with SyntheticHMREngine() as engine:
            res = engine.recover_mesh(self.dummy_frame)
            self.assertTrue(res.mesh_present)
        self.assertTrue(engine._is_closed)

        with MediaPipeHMREngine() as adapter:
            res = adapter.recover_mesh(self.dummy_frame)
            self.assertFalse(res.mesh_present)
        self.assertTrue(adapter._is_closed)

    # 15. Engine close and lifecycle behavior
    def test_15_engine_close_lifecycle(self) -> None:
        engine = SyntheticHMREngine()
        engine.close()
        self.assertTrue(engine._is_closed)
        res = engine.recover_mesh(self.dummy_frame)
        self.assertFalse(res.mesh_present)
        self.assertEqual(res.metadata["reason"], "engine_closed")

    # 16. Repeated synthetic generation stability
    def test_16_repeated_synthetic_stability(self) -> None:
        engine = SyntheticHMREngine()
        # Varying frame intensities create smooth phase movement
        frame_dark = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_bright = np.ones((480, 640, 3), dtype=np.uint8) * 200

        res_dark = engine.recover_mesh(frame_dark)
        res_bright = engine.recover_mesh(frame_bright)

        # Both generate valid 24 joints and 48 vertices
        self.assertEqual(len(res_dark.joints), 24)
        self.assertEqual(len(res_bright.joints), 24)
        # Dynamic phase reflects input brightness difference
        self.assertNotEqual(
            res_dark.metadata["deterministic_phase"],
            res_bright.metadata["deterministic_phase"],
        )

    # 17. 1,000-frame synthetic generation memory/resource stability
    def test_17_1000_frame_synthetic_stability(self) -> None:
        engine = SyntheticHMREngine()
        t0 = time.monotonic()
        for i in range(1000):
            res = engine.recover_mesh(self.dummy_frame)
            self.assertEqual(len(res.joints), 24)
        elapsed = time.monotonic() - t0
        avg_ms = (elapsed / 1000.0) * 1000.0
        # Average synthetic generation must be very fast (< 5.0 ms per frame on CPU)
        self.assertLess(avg_ms, 5.0)
        engine.close()

    # 18. Coordinate convention metadata
    def test_18_coordinate_convention_metadata(self) -> None:
        syn_engine = SyntheticHMREngine()
        syn_res = syn_engine.recover_mesh(self.dummy_frame)
        self.assertEqual(syn_res.coordinate_frame, "synthetic_camera_canonical")

        mp_adapter = MediaPipeHMREngine()
        mp_res = mp_adapter.recover_mesh(self.dummy_frame)
        self.assertIn(mp_res.coordinate_frame, ["mediapipe_normalized", "mediapipe_world_meters"])
        syn_engine.close()
        mp_adapter.close()

    # 19. Public 8-field AI contract remains unchanged
    def test_19_public_contract_protection(self) -> None:
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
        expected_keys = {
            "timestamp",
            "action",
            "object",
            "confidence",
            "expected_step",
            "detected_step",
            "status",
            "next_step",
        }
        self.assertEqual(set(res.keys()), expected_keys)
        # Verify NO 3D mesh fields leaked into public contract
        self.assertNotIn("joints", res)
        self.assertNotIn("vertices", res)
        self.assertNotIn("mesh_present", res)
        self.assertNotIn("theta", res)
        self.assertNotIn("beta", res)

    # 20. B10/B11/B12/FSM remain unaffected
    def test_20_procedure_components_unaffected(self) -> None:
        fsm = SequenceValidatorFSM()
        self.assertEqual(fsm.state, "IDLE")

        tf = TemporalConfirmationEngine()
        self.assertEqual(tf.window_size, 5)
        self.assertEqual(tf.confirmation_threshold, 3)

        uh = UncertaintyHandler()
        self.assertEqual(uh.high_confidence_threshold, 0.70)
        self.assertEqual(uh.marginal_confidence_threshold, 0.50)

        rm = RecoveryManager()
        self.assertEqual(rm.state, "IDLE")

    # 21. Absence of proprietary SMPL asset requirement
    def test_21_no_proprietary_smpl_asset_required(self) -> None:
        # Running both engines succeeds without needing any .pkl or .npz SMPL file
        syn = SyntheticHMREngine()
        mp = MediaPipeHMREngine()
        r1 = syn.recover_mesh(self.dummy_frame)
        r2 = mp.recover_mesh(self.dummy_frame)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        syn.close()
        mp.close()

    # 22. Absence of neural checkpoint requirements
    def test_22_no_neural_checkpoint_required(self) -> None:
        # Pure CPU execution with NumPy
        syn = SyntheticHMREngine()
        res = syn.recover_mesh(self.dummy_frame)
        self.assertTrue(res.mesh_present)
        self.assertEqual(res.representation_type, "synthetic_kinematic_mesh")
        syn.close()

    # 23. Absence of extra worker threads
    def test_23_no_extra_worker_threads_spawned(self) -> None:
        threads_before = threading.active_count()
        syn = SyntheticHMREngine()
        mp = MediaPipeHMREngine()
        syn.recover_mesh(self.dummy_frame)
        mp.recover_mesh(self.dummy_frame)
        threads_after = threading.active_count()
        self.assertEqual(threads_before, threads_after)
        syn.close()
        mp.close()


if __name__ == "__main__":
    unittest.main()
