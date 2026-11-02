"""Run with the inference-ik extra: python -m unittest discover -s tests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from abc_minimal.base_compensation import BaseCompensation, BaseCompensatedEnv, _pose


HAS_PINK = all(importlib.util.find_spec(name) for name in ("pinocchio", "pink"))


@unittest.skipUnless(HAS_PINK, "requires the inference-ik extra")
class PinocchioIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conversion = BaseCompensation(ik_backend="pink")

    def test_both_arm_fk_matches_mujoco(self):
        rng = np.random.default_rng(123)
        for arm, controller in enumerate(self.conversion.controllers):
            lower = controller.model.lowerPositionLimit + controller.reference
            upper = controller.model.upperPositionLimit + controller.reference
            for joints in rng.uniform(lower - 0.2, upper + 0.2, (100, 6)):
                self.conversion.data.qpos[self.conversion.qpos[arm]] = joints
                mujoco.mj_kinematics(self.conversion.model, self.conversion.data)
                expected = _pose(self.conversion.data.body(
                    f"{self.conversion.sides[arm]}_link_6"
                ))
                position, quaternion = controller.fk(joints)
                rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
                np.testing.assert_allclose(position, expected[:3, 3], atol=1e-12)
                np.testing.assert_allclose(rotation, expected[:3, :3], atol=1e-12)

    def test_analytic_jacobians_match_mujoco_and_finite_difference(self):
        import pinocchio as pin

        rng = np.random.default_rng(7)
        epsilon = 1e-6
        for arm, controller in enumerate(self.conversion.controllers):
            for _ in range(5):
                joints = controller.home + rng.uniform(-0.1, 0.1, 6)
                pin.computeJointJacobians(
                    controller.model, controller.data, joints - controller.reference
                )
                pin.updateFramePlacements(controller.model, controller.data)
                analytic = pin.getFrameJacobian(
                    controller.model, controller.data, controller.frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                ).copy()
                self.conversion.data.qpos[self.conversion.qpos[arm]] = joints
                mujoco.mj_kinematics(self.conversion.model, self.conversion.data)
                mujoco.mj_comPos(self.conversion.model, self.conversion.data)
                body = self.conversion.model.body(f"{self.conversion.sides[arm]}_link_6")
                linear = np.zeros((3, self.conversion.model.nv))
                angular = np.zeros_like(linear)
                mujoco.mj_jacBody(
                    self.conversion.model, self.conversion.data, linear, angular, body.id
                )
                dofs = [self.conversion.model.joint(
                    f"{self.conversion.sides[arm]}_joint{i}"
                ).dofadr[0] for i in range(1, 7)]
                np.testing.assert_allclose(
                    analytic, np.vstack((linear[:, dofs], angular[:, dofs])), atol=1e-12,
                )
                numeric = np.empty((6, 6))
                for joint in range(6):
                    delta = np.eye(6)[joint] * epsilon
                    plus_pos, plus_quat = controller.fk(joints + delta)
                    minus_pos, minus_quat = controller.fk(joints - delta)
                    plus_rot = Rotation.from_quat(plus_quat[[1, 2, 3, 0]])
                    minus_rot = Rotation.from_quat(minus_quat[[1, 2, 3, 0]])
                    numeric[:3, joint] = (plus_pos - minus_pos) / (2 * epsilon)
                    numeric[3:, joint] = (plus_rot * minus_rot.inv()).as_rotvec() / (2 * epsilon)
                np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_solve_matches_supplied_five_step_algorithm(self):
        import pinocchio as pin
        from pink import Configuration, solve_ik
        from pink.tasks import FrameTask, PostureTask

        rng = np.random.default_rng(20)
        for controller in self.conversion.controllers:
            for _ in range(5):
                seed = controller.home + rng.uniform(-0.05, 0.05, 6)
                position, quaternion = controller.fk(seed)
                position += rng.uniform(-0.01, 0.01, 3)
                initial = seed - controller.reference
                frame = FrameTask(
                    frame=controller.target_link, position_cost=50.0,
                    orientation_cost=1.0, lm_damping=0.0, gain=0.2,
                )
                frame.set_target(pin.SE3(
                    Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix(), position,
                ))
                smooth = PostureTask(cost=0.2, lm_damping=0.0, gain=0.2)
                smooth.set_target(initial)
                posture = PostureTask(cost=0.05, lm_damping=0.0, gain=0.2)
                posture.set_target(controller.home - controller.reference)
                configuration = Configuration(
                    controller.model, controller.model.createData(), initial,
                    copy_data=True, forward_kinematics=True,
                )
                for _ in range(5):
                    velocity = solve_ik(
                        configuration, [frame, smooth, posture], dt=0.05,
                        solver="daqp", damping=0.0,
                    )
                    configuration.integrate_inplace(velocity, 0.05)
                    configuration.update(np.clip(
                        configuration.q, controller.model.lowerPositionLimit,
                        controller.model.upperPositionLimit,
                    ))
                actual = controller.solve(position, quaternion, seed)
                np.testing.assert_allclose(
                    actual, configuration.q + controller.reference, atol=1e-11,
                )

    def test_unreachable_target_returns_finite_bounded_joints(self):
        for controller in self.conversion.controllers:
            joints = controller.solve(np.array([5., 5., 5.]), [1., 0., 0., 0.], controller.home)
            self.assertTrue(np.isfinite(joints).all())
            self.assertTrue(np.all(joints - controller.reference >= controller.model.lowerPositionLimit))
            self.assertTrue(np.all(joints - controller.reference <= controller.model.upperPositionLimit))

    def test_conversion_preserves_grippers_and_uses_one_five_step_solve(self):
        state = self._reset_conversion()
        for is_state in (True, False):
            before = self.conversion.stats["state" if is_state else "action"].copy()
            output = self.conversion.convert(state, state=is_state, physical_state=state)
            stats = self.conversion.stats["state" if is_state else "action"]
            self.assertEqual(output.shape, state.shape)
            np.testing.assert_allclose(output[:, [6, 13]], state[:, [6, 13]], atol=1e-7)
            self.assertEqual(stats["arm_solves"] - before["arm_solves"], 4)
            self.assertEqual(stats["solve_calls"] - before["solve_calls"], 4)

    def _reset_conversion(self):
        state = np.zeros((2, 14))
        for arm, controller in enumerate(self.conversion.controllers):
            state[:, 7 * arm:7 * arm + 6] = controller.home
        state[:, 6] = [0.2, 0.3]
        state[:, 13] = [0.7, 0.8]
        records = []
        for _ in state:
            poses = {
                f"{side}_arm": {
                    "pos": (self.conversion.nominal_bases[arm][:3, 3] + [0.01, 0, 0]).tolist(),
                    "quat": [1, 0, 0, 0],
                }
                for arm, side in enumerate(self.conversion.sides)
            }
            records.append(SimpleNamespace(metadata={"arm_base_poses": poses}))
        self.conversion.reset(records, state)
        return state

    def test_pink_always_initializes_from_passed_state(self):
        measured = self._reset_conversion()
        # Poison the legacy caches to catch any use of a previous IK result.
        self.conversion.state_seed[:] = 100.0
        self.conversion.action_seed[:] = 100.0
        for is_state in (True, False):
            target = measured.copy()
            if not is_state:
                target[:, 0] += 0.08  # A policy target is not the measured state.
            kwargs = {"state": True} if is_state else {"physical_state": measured}
            observed = []
            original_solve = self.conversion._solve

            def capture(arm, target, seed):
                observed.append((arm, seed.copy()))
                return original_solve(arm, target, seed)

            with patch.object(self.conversion, "_solve", side_effect=capture):
                first = self.conversion.convert(target, **kwargs)
                second = self.conversion.convert(target, **kwargs)
            np.testing.assert_array_equal(first, second)
            for call_index, (arm, seed) in enumerate(observed):
                world = (call_index % 4) // 2
                np.testing.assert_array_equal(
                    seed, measured[world, 7 * arm:7 * arm + 6],
                )
        with self.assertRaisesRegex(ValueError, "current physical_state"):
            self.conversion.convert(measured)

    def test_each_action_step_reads_fresh_physical_state_without_rendering(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                physical = np.ones((2, 14) if batched else (14,))
                recorded = []

                def read_state():
                    return physical.copy()

                def convert(actions, *, physical_state):
                    recorded.append(physical_state.copy())
                    return actions

                def step(actions):
                    physical[:] += 0.03

                underlying = SimpleNamespace(get_obs=Mock(side_effect=AssertionError("rendered")))
                if batched:
                    underlying.base_env = object()
                    underlying.state = read_state
                    underlying.step = step
                else:
                    underlying.env = SimpleNamespace(get_state=read_state)
                    underlying.step_one = step
                wrapper = BaseCompensatedEnv.__new__(BaseCompensatedEnv)
                wrapper._env = underlying
                wrapper._conversion = SimpleNamespace(convert=convert)
                step_fn = wrapper.step if batched else wrapper.step_one
                for _ in range(3):
                    expected = physical.copy()
                    step_fn(np.zeros_like(physical))
                    np.testing.assert_array_equal(recorded[-1], expected)
                underlying.get_obs.assert_not_called()

    def test_median_reference_identity_when_physical_bases_match(self):
        conversion = BaseCompensation(ik_backend="pink", reference="median")
        state = np.zeros((1, 14))
        poses = {}
        for arm, side in enumerate(conversion.sides):
            state[0, 7 * arm:7 * arm + 6] = conversion.controllers[arm].home
            actual = conversion.world_from_camera @ conversion.camera_from_reference[arm]
            poses[f"{side}_arm"] = {
                "pos": actual[:3, 3],
                "quat": Rotation.from_matrix(actual[:3, :3]).as_quat()[[3, 0, 1, 2]],
            }
        conversion.reset([SimpleNamespace(metadata={"arm_base_poses": poses})], state)
        np.testing.assert_allclose(conversion.input_transform[0], [np.eye(4)] * 2, atol=1e-12)
        np.testing.assert_array_equal(conversion.convert(state, state=True), state.astype(np.float32))
        np.testing.assert_array_equal(
            conversion.convert(state, physical_state=state), state.astype(np.float32),
        )

    def test_median_state_and_action_targets_preserve_camera_frame_pose(self):
        conversion = BaseCompensation(ik_backend="pink", reference="median")
        state = np.zeros((1, 14))
        actual_bases, poses = [], {}
        for arm, side in enumerate(conversion.sides):
            state[0, 7 * arm:7 * arm + 6] = conversion.controllers[arm].home
            actual = conversion.nominal_bases[arm].copy()
            actual[:3, 3] += [0.02, -0.01, 0.005]
            actual[:3, :3] = Rotation.from_euler("xyz", [0.03, -0.02, 0.01]).as_matrix()
            actual_bases.append(conversion.camera_from_world @ actual)
            poses[f"{side}_arm"] = {
                "pos": actual[:3, 3],
                "quat": Rotation.from_matrix(actual[:3, :3]).as_quat()[[3, 0, 1, 2]],
            }
        conversion.reset([SimpleNamespace(metadata={"arm_base_poses": poses})], state)
        for is_state in (True, False):
            joints = state.copy()
            if not is_state:
                joints[:, [0, 7]] += 0.08
            source_poses = conversion._fk(joints[0])
            targets = []

            def capture(arm, target, seed):
                targets.append(target.copy())
                return seed.copy(), 0.0, 0.0, 1

            with patch.object(conversion, "_solve", side_effect=capture):
                conversion.convert(joints, state=is_state, physical_state=state)
            for arm in range(2):
                nominal = conversion.camera_from_world @ conversion.nominal_bases[arm]
                reference, actual = conversion.camera_from_reference[arm], actual_bases[arm]
                source_base = actual if is_state else reference
                target_base = reference if is_state else actual
                source_camera_pose = source_base @ np.linalg.inv(nominal) @ source_poses[arm]
                target_camera_pose = target_base @ np.linalg.inv(nominal) @ targets[arm]
                np.testing.assert_allclose(source_camera_pose, target_camera_pose, atol=1e-12)

    def test_median_fk_matches_mujoco_with_repositioned_bases(self):
        conversion = BaseCompensation(ik_backend="pink", reference="median")
        model_path = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_dishrack_base.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        for arm, side in enumerate(conversion.sides):
            world_from_base = conversion.world_from_camera @ conversion.camera_from_reference[arm]
            body = model.body(f"{side}_arm")
            body.pos[:] = world_from_base[:3, 3]
            body.quat[:] = Rotation.from_matrix(world_from_base[:3, :3]).as_quat()[[3, 0, 1, 2]]
        data = mujoco.MjData(model)
        rng = np.random.default_rng(53)
        for _ in range(20):
            joints = np.zeros(14)
            for arm, controller in enumerate(conversion.controllers):
                joints[7 * arm:7 * arm + 6] = rng.uniform(
                    controller.model.lowerPositionLimit, controller.model.upperPositionLimit,
                ) + controller.reference
            data.qpos[conversion.qpos] = joints.reshape(2, 7)[:, :6]
            mujoco.mj_kinematics(model, data)
            nominal_poses = conversion._fk(joints)
            for arm, side in enumerate(conversion.sides):
                nominal = conversion.camera_from_world @ conversion.nominal_bases[arm]
                expected = conversion.camera_from_world @ _pose(data.site(f"{side}_tcp_site"))
                actual = conversion.camera_from_reference[arm] @ np.linalg.inv(nominal) @ nominal_poses[arm]
                np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
