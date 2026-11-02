"""Checks for exact, paired training-base sampling without added noise."""

import gzip
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from abc_sim.randomization.arm_bases import sample_training_arm_bases, training_base_pool
from abc_sim.randomization.requests import DishRackResetRequest
from abc_sim.randomization.tasks.dishrack import DishRackRandomizer


ROOT = Path(__file__).resolve().parents[1]


def pose(entity):
    matrix = np.eye(4)
    matrix[:3, :3] = entity.xmat.reshape(3, 3)
    matrix[:3, 3] = entity.xpos
    return matrix


class TrainingArmBasesTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "abc_sim/models/yam_dishrack_base.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_pool_matches_accepted_training_calibrations(self):
        with gzip.open(ROOT / "scripts/dishrack_joint_retarget_calibrations.json.gz", "rt") as stream:
            manifest = json.load(stream)
        expected = {
            record["episode"]: record["calibration"] for record in manifest["records"]
            if record["split"] == "train" and record["calibration_quality"] == "accepted"
        }
        records = training_base_pool()
        self.assertEqual(len(records), 1240)
        self.assertEqual({record["episode"] for record in records}, set(expected))
        for record in records:
            calibration = expected[record["episode"]]
            for side in ("left", "right"):
                target = np.asarray(calibration["left_base_to_camera"]) @ calibration["base_poses"][side]
                np.testing.assert_array_equal(record["camera_from_base"][side], target)

    def test_each_world_samples_one_exact_training_pair_reproducibly(self):
        records = {record["episode"]: record for record in training_base_pool()}
        selected = []
        for world in range(100):
            seed = 20260511 + world
            result = sample_training_arm_bases(self.model, self.data, seed)
            self.assertEqual(result, sample_training_arm_bases(self.model, self.data, seed))
            episode = result["left_arm"]["source_episode"]
            self.assertEqual(episode, result["right_arm"]["source_episode"])
            selected.append(episode)
            world_from_camera = pose(self.data.camera("top"))
            world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
            for side in ("left", "right"):
                actual = np.linalg.inv(world_from_camera) @ pose(self.data.body(f"{side}_arm"))
                np.testing.assert_allclose(actual, records[episode]["camera_from_base"][side], atol=1e-12)
        self.assertGreater(len(set(selected)), 1)

    def test_request_dispatch_and_disabled_mode(self):
        request = DishRackResetRequest.from_value({
            "randomize_arm_bases": True, "arm_base_source": "training",
        })
        result = DishRackRandomizer._randomize_arm_bases(self.model, self.data, request, 20260511)
        self.assertIn("source_episode", result["left_arm"])
        before = self.model.body_pos.copy(), self.model.body_quat.copy()
        disabled = DishRackResetRequest(arm_base_source="training")
        self.assertEqual(DishRackRandomizer._randomize_arm_bases(self.model, self.data, disabled, 1), {})
        np.testing.assert_array_equal(self.model.body_pos, before[0])
        np.testing.assert_array_equal(self.model.body_quat, before[1])
        self.assertEqual(DishRackResetRequest().arm_base_source, "stats")
        with self.assertRaises(ValueError):
            DishRackResetRequest.from_value({"arm_base_source": "unknown"})


if __name__ == "__main__":
    unittest.main()
