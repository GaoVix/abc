"""Checks for direct inference with fixed median camera-relative arm bases."""

import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import abc_sim
from abc_sim.batched_env import model_signature
from abc_sim.randomization.arm_bases import _pose, top_camera_pose


REQUEST = {
    "plate_count": 3, "plate_variant": "plate_0", "dish_rack_variant": "dish_rack_0",
    "randomize_variants": False, "randomize_scales": False,
    "randomize_arm_bases": True, "arm_base_source": "median_camera",
}


class MedianCameraBasesTest(unittest.TestCase):
    def setUp(self):
        self.env = abc_sim.make_env("dishrack", render_cameras=False)
        path = Path(__file__).resolve().parents[1] / "abc_minimal/dishrack_median_base_poses.json"
        with path.open() as stream:
            poses = json.load(stream)["poses"]
        self.median = {side: np.asarray(poses[side]["T_camera_from_base"])
                       for side in ("left", "right")}

    def tearDown(self):
        self.env.close()

    def reset(self, seed, request=REQUEST):
        self.env.forget_arm_state()
        return self.env.reset(seed=seed, options={"randomization": request})

    def test_fixed_median_alignment_and_mounts(self):
        nominal_left = _pose(self.env.data.body("left_arm"))
        nominal_right = _pose(self.env.data.body("right_arm"))
        expected_relative = np.linalg.inv(self.median["left"]) @ self.median["right"]
        configurations, signatures = [], []
        for seed in range(20260511, 20260516):
            _, info = self.reset(seed)
            record = info["randomization"]
            left = _pose(self.env.data.body("left_arm"))
            right = _pose(self.env.data.body("right_arm"))
            camera = np.asarray(top_camera_pose(self.env.model, self.env.data)["world_from_camera"])
            np.testing.assert_allclose(left, nominal_left, atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.linalg.inv(camera) @ left, self.median["left"], atol=1e-12)
            relative = np.linalg.inv(left) @ right
            np.testing.assert_allclose(relative[:2, 3], expected_relative[:2, 3], atol=1e-12)
            np.testing.assert_allclose(right[2, 3], nominal_right[2, 3], atol=1e-12)
            np.testing.assert_allclose(Rotation.from_matrix(right[:3, :3]).as_euler("xyz")[:2],
                                       Rotation.from_matrix(nominal_right[:3, :3]).as_euler("xyz")[:2], atol=1e-12)
            actual_right = np.linalg.inv(camera) @ right
            position_mm = np.linalg.norm(actual_right[:3, 3] - self.median["right"][:3, 3]) * 1000
            rotation_deg = np.rad2deg(Rotation.from_matrix(
                self.median["right"][:3, :3].T @ actual_right[:3, :3]).magnitude())
            self.assertAlmostEqual(position_mm, 0.1586976372, places=7)
            self.assertAlmostEqual(rotation_deg, 0.6359047049, places=7)
            self.assertEqual(record.metadata["arm_base_source"], "median_camera")
            self.assertEqual(record.metadata["arm_base_poses"]["right_arm"]["source"], "median")
            np.testing.assert_allclose(record.metadata["top_camera_pose"]["world_from_camera"], camera)
            configurations.append((camera, right))
            signatures.append(model_signature(self.env.model))
        self.assertEqual(len(set(signatures)), 1)
        for camera, right in configurations[1:]:
            np.testing.assert_array_equal(camera, configurations[0][0])
            np.testing.assert_array_equal(right, configurations[0][1])

    def test_objects_state_and_replay(self):
        _, original_info = self.reset(20260511, dict(REQUEST, randomize_arm_bases=False))
        original_state = self.env.data.qpos.copy()
        original_camera = top_camera_pose(self.env.model, self.env.data)
        _, info = self.reset(20260511)
        record = info["randomization"]
        self.assertEqual(record.object_states, original_info["randomization"].object_states)
        np.testing.assert_array_equal(self.env.data.qpos, original_state)
        camera = top_camera_pose(self.env.model, self.env.data)
        right = _pose(self.env.data.body("right_arm"))
        _, reset_info = self.reset(20260512, dict(REQUEST, randomize_arm_bases=False))
        self.assertNotIn("top_camera_pose", reset_info["randomization"].metadata)
        self.assertEqual(top_camera_pose(self.env.model, self.env.data), original_camera)
        self.env._task_randomizer.apply(self.env.model, self.env.data, record)
        self.assertEqual(top_camera_pose(self.env.model, self.env.data), camera)
        np.testing.assert_allclose(_pose(self.env.data.body("right_arm")), right, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
