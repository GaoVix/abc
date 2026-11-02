"""Checks for paired calibrated bases, fixed cameras and EEF-only contacts."""

import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import abc_sim
from abc_minimal.base_compensation import BaseCompensation
from abc_sim.batched_env import model_signature
from abc_sim.randomization.arm_bases import (
    disable_arm_collisions_except_eef, sample_training_arm_bases, training_base_pool,
)
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

    def test_each_world_samples_paired_full_poses_reproducibly(self):
        records = {record["episode"]: record for record in training_base_pool()}
        camera = pose(self.data.camera("top"))
        world_from_camera = camera.copy()
        world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
        selected = []
        for world in range(100):
            seed = 20260511 + world
            result = sample_training_arm_bases(self.model, self.data, seed)
            repeated = sample_training_arm_bases(self.model, self.data, seed)
            episode = result["left_arm"]["source_episode"]
            self.assertEqual(episode, result["right_arm"]["source_episode"])
            selected.append(episode)
            np.testing.assert_array_equal(pose(self.data.camera("top")), camera)
            for side in ("left", "right"):
                name = f"{side}_arm"
                expected = world_from_camera @ records[episode]["camera_from_base"][side]
                actual = pose(self.data.body(name))
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                self.assertEqual(result[name]["projection"], "full_pose")
                self.assertEqual(repeated[name]["source_episode"], episode)
                np.testing.assert_allclose(result[name]["pos"], actual[:3, 3], atol=1e-12)
                np.testing.assert_allclose(result[name]["quat"], repeated[name]["quat"], atol=1e-12)
        self.assertGreater(len(set(selected)), 1)

    def test_sampling_overwrites_xml_mount_height_and_tilt(self):
        records = {record["episode"]: record for record in training_base_pool()}
        camera = pose(self.data.camera("top"))
        world_from_camera = camera.copy()
        world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
        for side in ("left", "right"):
            body = self.model.body(f"{side}_arm")
            body.pos[2] = 0.81
            body.quat[:] = Rotation.from_euler("xyz", [0.07, -0.09, 0.4]).as_quat()[[3, 0, 1, 2]]
        for seed in (20260511, 20260512, 20260511):
            result = sample_training_arm_bases(self.model, self.data, seed)
            record = records[result["left_arm"]["source_episode"]]
            np.testing.assert_array_equal(pose(self.data.camera("top")), camera)
            for side in ("left", "right"):
                actual = pose(self.data.body(f"{side}_arm"))
                np.testing.assert_allclose(
                    actual, world_from_camera @ record["camera_from_base"][side],
                    atol=1e-12, rtol=0,
                )

    def test_full_pose_matches_both_calibrated_bases_with_fixed_camera(self):
        records = {record["episode"]: record for record in training_base_pool()}
        camera = pose(self.data.camera("top"))
        world_from_camera = camera.copy()
        world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
        for world in range(100):
            result = sample_training_arm_bases(self.model, self.data, 20260511 + world)
            episode = result["left_arm"]["source_episode"]
            self.assertEqual(result["right_arm"]["source_episode"], episode)
            np.testing.assert_array_equal(pose(self.data.camera("top")), camera)
            for side in ("left", "right"):
                actual = pose(self.data.body(f"{side}_arm"))
                np.testing.assert_allclose(
                    np.linalg.inv(world_from_camera) @ actual,
                    records[episode]["camera_from_base"][side], atol=1e-12, rtol=0,
                )
                self.assertEqual(result[f"{side}_arm"]["projection"], "full_pose")
            relative = np.linalg.inv(pose(self.data.body("left_arm"))) @ pose(self.data.body("right_arm"))
            expected_relative = (np.linalg.inv(records[episode]["camera_from_base"]["left"])
                                 @ records[episode]["camera_from_base"]["right"])
            np.testing.assert_allclose(relative, expected_relative, atol=1e-12, rtol=0)

    def test_default_median_is_exact_with_fixed_camera_and_identity_compensation(self):
        env = abc_sim.make_env("dishrack", render_cameras=False)
        conversion = BaseCompensation(ik_backend="mink", reference="median")
        camera = pose(env.data.camera("top"))
        world_from_camera = camera.copy()
        world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
        request = dict(plate_count=3, plate_variant="plate_0", dish_rack_variant="dish_rack_0",
                       randomize_variants=False, randomize_scales=False)
        try:
            for seed in range(20260511, 20260516):
                env.forget_arm_state()
                obs, info = env.reset(seed=seed, options={"randomization": request})
                record = info["randomization"]
                self.assertEqual(record.metadata["arm_base_source"], "median")
                self.assertEqual(record.metadata["arm_collision_mode"], "eef_only")
                np.testing.assert_array_equal(pose(env.data.camera("top")), camera)
                for side in ("left", "right"):
                    for name in (f"{side}_arm", *(f"{side}_link_{i}" for i in range(1, 6)),
                                 f"{side}_camera_d405"):
                        geoms = env.model.geom_bodyid == env.model.body(name).id
                        np.testing.assert_array_equal(env.model.geom_contype[geoms], 0)
                        np.testing.assert_array_equal(env.model.geom_conaffinity[geoms], 0)
                    for name in (f"{side}_link_6", f"{side}_lf_rot", f"{side}_lf_down",
                                 f"{side}_rf_rot", f"{side}_rf_down"):
                        geoms = env.model.geom_bodyid == env.model.body(name).id
                        self.assertTrue(np.any(env.model.geom_contype[geoms]))
                        self.assertTrue(np.any(env.model.geom_conaffinity[geoms]))
                for arm, side in enumerate(("left", "right")):
                    actual = pose(env.data.body(f"{side}_arm"))
                    np.testing.assert_allclose(np.linalg.inv(world_from_camera) @ actual,
                                               conversion.camera_from_reference[arm], atol=1e-12, rtol=0)
                relative = np.linalg.inv(pose(env.data.body("left_arm"))) @ pose(env.data.body("right_arm"))
                expected_relative = (np.linalg.inv(conversion.camera_from_reference[0])
                                     @ conversion.camera_from_reference[1])
                np.testing.assert_allclose(relative, expected_relative, atol=1e-12, rtol=0)
                conversion.reset([record], obs["state"])
                np.testing.assert_allclose(conversion.input_transform[0], [np.eye(4)] * 2, atol=1e-12)
                with patch.object(conversion, "_solve", side_effect=AssertionError("Unexpected IK")):
                    np.testing.assert_array_equal(conversion.convert(obs["state"], state=True), obs["state"])
                    np.testing.assert_array_equal(conversion.convert(obs["state"]), obs["state"])
            # Explicit XML selection restores both original bases without moving the camera.
            env.forget_arm_state()
            env.reset(seed=20260511, options={"randomization": dict(request, arm_base_source="xml")})
            np.testing.assert_array_equal(pose(env.data.camera("top")), camera)
            for arm, side in enumerate(("left", "right")):
                np.testing.assert_allclose(pose(env.data.body(f"{side}_arm")),
                                           conversion.nominal_bases[arm], atol=1e-12)
            env._task_randomizer.apply(env.model, env.data, record)
            np.testing.assert_array_equal(pose(env.data.camera("top")), camera)
            for arm, side in enumerate(("left", "right")):
                actual = np.linalg.inv(world_from_camera) @ pose(env.data.body(f"{side}_arm"))
                np.testing.assert_allclose(actual, conversion.camera_from_reference[arm], atol=1e-12, rtol=0)
        finally:
            env.close()

    def test_collision_mask_retains_eef_and_scene(self):
        original = self.model.geom_contype.copy(), self.model.geom_conaffinity.copy()
        mass, inertia = self.model.body_mass.copy(), self.model.body_inertia.copy()
        signature = model_signature(self.model)
        disable_arm_collisions_except_eef(self.model)
        self.assertNotEqual(model_signature(self.model), signature)
        # The wrist (link_6) stays active, but its camera subtree must be disabled.
        bodies = {self.model.body(name).id for side in ("left", "right")
                  for name in (f"{side}_arm", *(f"{side}_link_{i}" for i in range(1, 6)),
                               f"{side}_camera_d405")}
        disabled = np.isin(self.model.geom_bodyid, list(bodies))
        for actual, before in zip((self.model.geom_contype, self.model.geom_conaffinity), original):
            np.testing.assert_array_equal(actual[disabled], 0)
            np.testing.assert_array_equal(actual[~disabled], before[~disabled])
        for side in ("left", "right"):
            fingertip = self.model.body(f"{side}_lf_down").id
            self.assertTrue(np.any(self.model.geom_contype[self.model.geom_bodyid == fingertip]))
        np.testing.assert_array_equal(self.model.body_mass, mass)
        np.testing.assert_array_equal(self.model.body_inertia, inertia)

    def test_reset_replay_and_compensation_with_full_poses(self):
        env = abc_sim.make_env("dishrack", render_cameras=False)
        request = dict(plate_count=3, plate_variant="plate_0", dish_rack_variant="dish_rack_0",
                       randomize_variants=False, randomize_scales=False, randomize_arm_bases=True,
                       arm_base_source="training_6d", arm_collision_mode="eef_only")
        try:
            env.forget_arm_state()
            env.reset(seed=20260511, options={"randomization": dict(
                request, randomize_arm_bases=False, arm_collision_mode="full",
            )})
            camera = pose(env.data.camera("top"))
            nominal_masks = env.model.geom_contype.copy(), env.model.geom_conaffinity.copy()
            env.forget_arm_state()
            obs, info = env.reset(seed=20260511, options={"randomization": request})
            record = info["randomization"]
            self.assertEqual(record.metadata["arm_collision_mode"], "eef_only")
            np.testing.assert_array_equal(pose(env.data.camera("top")), camera)
            bases = [pose(env.data.body(f"{side}_arm")) for side in ("left", "right")]
            masks = env.model.geom_contype.copy(), env.model.geom_conaffinity.copy()
            world_from_camera = camera.copy()
            world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
            camera_from_world = np.linalg.inv(world_from_camera)
            for reference in ("xml", "median"):
                conversion = BaseCompensation(ik_backend="mink", reference=reference)
                conversion.reset([record], obs["state"])
                canonical = conversion._fk(obs["state"])
                for arm, side in enumerate(("left", "right")):
                    nominal = conversion.camera_from_world @ conversion.nominal_bases[arm]
                    actual = camera_from_world @ bases[arm]
                    target = conversion.camera_from_reference[arm]
                    actual_tcp = camera_from_world @ pose(env.data.site(f"{side}_tcp_site"))
                    np.testing.assert_allclose(conversion.input_transform[0, arm] @ canonical[arm],
                                               nominal @ np.linalg.inv(target) @ actual_tcp, atol=1e-10)
                    np.testing.assert_allclose(conversion.output_transform[0, arm] @ canonical[arm],
                                               nominal @ np.linalg.inv(actual) @ target
                                               @ np.linalg.inv(nominal) @ canonical[arm], atol=1e-10)
            env.forget_arm_state()
            env.reset(seed=20260512, options={"randomization": dict(
                request, randomize_arm_bases=False, arm_collision_mode="full",
            )})
            for actual, before in zip((env.model.geom_contype, env.model.geom_conaffinity), nominal_masks):
                np.testing.assert_array_equal(actual, before)
            env._task_randomizer.apply(env.model, env.data, record)
            np.testing.assert_array_equal(pose(env.data.camera("top")), camera)
            for side, expected in zip(("left", "right"), bases):
                np.testing.assert_allclose(pose(env.data.body(f"{side}_arm")), expected, atol=1e-12)
            for actual, expected in zip((env.model.geom_contype, env.model.geom_conaffinity), masks):
                np.testing.assert_array_equal(actual, expected)
        finally:
            env.close()

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
        self.assertEqual(DishRackResetRequest().arm_base_source, "median")
        self.assertEqual(DishRackResetRequest.from_value({}).arm_base_source, "median")
        self.assertEqual(DishRackResetRequest.from_value(None).arm_base_source, "median")
        self.assertEqual(DishRackResetRequest().arm_collision_mode, "eef_only")
        self.assertEqual(DishRackResetRequest.from_value({}).arm_collision_mode, "eef_only")
        self.assertEqual(DishRackResetRequest.from_value(None).arm_collision_mode, "eef_only")
        for source in ("unknown", "stats", "training_camera", "median_camera"):
            with self.assertRaises(ValueError):
                DishRackResetRequest.from_value({"arm_base_source": source})
        request = DishRackResetRequest.from_value({
            "randomize_arm_bases": True, "arm_base_source": "training_6d", "arm_collision_mode": "eef_only",
        })
        result = DishRackRandomizer._randomize_arm_bases(self.model, self.data, request, 20260511)
        self.assertEqual(result["left_arm"]["projection"], "full_pose")
        with self.assertRaises(ValueError):
            DishRackResetRequest.from_value({"arm_collision_mode": "unknown"})

    def test_training_aliases_select_the_same_full_poses(self):
        for seed in range(20260511, 20260611):
            results = [DishRackRandomizer._randomize_arm_bases(
                self.model, self.data,
                DishRackResetRequest(randomize_arm_bases=True, arm_base_source=source), seed,
            ) for source in ("training", "training_6d")]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["left_arm"]["projection"], "full_pose")
            self.assertEqual(results[0]["right_arm"]["projection"], "full_pose")


if __name__ == "__main__":
    unittest.main()
