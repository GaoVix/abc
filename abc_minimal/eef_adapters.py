"""Connect the trained EEF MLPs to DiT inference and Mink IK."""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from deploy.robot.control.mink_ik import MinkArmController
from scripts.train_eef_adapters import JOINT_STATE_INDICES, load_adapters

KINEMATIC_MODEL = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_dishrack_base.xml"


class EEFInferenceAdapter:
    def __init__(self, checkpoint, device, execute_chunk_dim):
        self.device = device
        self.execute_chunk_dim = execute_chunk_dim
        self.mlps = load_adapters(checkpoint, device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        camera_path = Path(saved["config"]["camera_transform"])
        self.camera = np.asarray(json.loads(camera_path.read_text())["T_top_camera_from_world"])
        self.model = mujoco.MjModel.from_xml_path(str(KINEMATIC_MODEL))
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(f"{side}_joint{j}").id for side in ("left", "right") for j in range(1, 7)]
        self.qpos_indices = self.model.jnt_qposadr[joints]
        self.sites = [self.model.site(f"{side}_tcp_site").id for side in ("left", "right")]
        self.site_rotations = []
        self.controllers = []
        for arm, side in enumerate(("left", "right")):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, self.model.site_quat[self.sites[arm]])
            self.site_rotations.append(rotation.reshape(3, 3))
            spec = mujoco.MjSpec.from_file(str(KINEMATIC_MODEL))
            for field in ("sensors", "equalities", "actuators", "tendons", "keys"):
                for element in list(getattr(spec, field)):
                    spec.delete(element)
            for joint in list(spec.joints):
                if joint.name not in {f"{side}_joint{j}" for j in range(1, 7)}:
                    spec.delete(joint)
            home = self.mlps.state_mean[arm * 6 : (arm + 1) * 6].cpu().numpy()
            self.controllers.append(MinkArmController(home, model=spec.compile(), target_link=f"{side}_link_6"))
        self.reset()

    def reset(self):
        self.previous_target = None
        self.previous_action_poses = None

    def fk(self, joints):
        """Exact MuJoCo FK in the same fixed camera frame used to train the MLPs."""
        poses = []
        for values in np.asarray(joints).reshape(-1, 12):
            self.data.qpos[self.qpos_indices] = values
            mujoco.mj_kinematics(self.model, self.data)
            positions = self.data.site_xpos[self.sites] @ self.camera[:3, :3].T + self.camera[:3, 3]
            rotations = self.camera[:3, :3] @ self.data.site_xmat[self.sites].reshape(2, 3, 3)
            poses.append(np.concatenate((positions, rotations[:, :, 0], rotations[:, :, 1]), axis=-1).reshape(18))
        return np.asarray(poses, dtype=np.float32)

    @torch.no_grad()
    def state(self, state):
        result = state.copy()
        poses = self.fk(state[..., JOINT_STATE_INDICES])
        joints = self.mlps.eef_to_joint(torch.as_tensor(poses, device=self.device))
        result[..., JOINT_STATE_INDICES] = joints.cpu().numpy().reshape(*state.shape[:-1], 12)
        return result

    @torch.no_grad()
    def action_prefix(self, actions, prefix_length):
        """Feed the previous MLP_out camera poses directly through MLP_in.

        The cache contains the poses of the commands selected for execution,
        so its tail matches the pending RTC actions even after prefix removal.
        Preserve the original joint targets for execution and IK seeds.
        """
        result = np.asarray(actions, dtype=np.float32).copy()
        if prefix_length:
            valid = result[..., :prefix_length, :]
            if self.previous_action_poses is None:
                # The first synchronous prefixed chunk repeats the initial
                # state; there is no previous MLP_out prediction yet.
                result[..., :prefix_length, :] = self.state(valid)
            else:
                poses = self.previous_action_poses
                num_worlds = valid.reshape(-1, prefix_length, 14).shape[0]
                if poses.shape[0] != num_worlds or poses.shape[1] < prefix_length:
                    raise ValueError(
                        "Cached MLP_out poses do not match the action prefix; "
                        "reset the policy between worlds"
                    )
                joints = self.mlps.eef_to_joint(torch.as_tensor(poses[:, -prefix_length:], device=self.device))
                result[..., :prefix_length, JOINT_STATE_INDICES] = joints.cpu().numpy().reshape(*valid.shape[:-1], 12)
        return result

    @torch.no_grad()
    def actions(self, actions, state, action_prefix, prefix_length):
        result = actions.copy().reshape(-1, actions.shape[-2], 14)
        poses = self.mlps.joint_to_eef(torch.as_tensor(result[..., JOINT_STATE_INDICES], device=self.device))
        poses = poses.cpu().numpy().reshape(*result.shape[:2], 2, 9)
        if prefix_length:
            prefix = np.asarray(action_prefix).reshape(len(result), -1, 14)
            result[:, :prefix_length] = prefix[:, :prefix_length]
            seed = result[:, prefix_length - 1].copy()
        else:
            seed = state.reshape(-1, 14).copy() if self.previous_target is None else self.previous_target.copy()
        for step in range(prefix_length, result.shape[1]):
            for world in range(len(result)):
                for arm, controller in enumerate(self.controllers):
                    pose = poses[world, step, arm]
                    rotation = np.column_stack((pose[3:6], pose[6:9], np.cross(pose[3:6], pose[6:9])))
                    # Camera TCP -> training-world link_6.
                    rotation = self.camera[:3, :3].T @ rotation @ self.site_rotations[arm].T
                    position = self.camera[:3, :3].T @ (pose[:3] - self.camera[:3, 3])
                    position -= rotation @ self.model.site_pos[self.sites[arm]]
                    quaternion = np.empty(4)
                    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
                    columns = slice(arm * 7, arm * 7 + 6)
                    seed[world, columns] = controller.solve(position, quaternion, seed[world, columns])
                    result[world, step, columns] = seed[world, columns]
        # Cache the last command that eval will execute.
        last = min(prefix_length + self.execute_chunk_dim, result.shape[1]) - 1
        self.previous_target = result[:, last].copy()
        # Keep MLP_out poses before IK, aligned with the executed slice. RTC
        # removes the prefix before the next chunk reaches the rollout loop.
        self.previous_action_poses = poses[:, prefix_length : last + 1].reshape(len(result), -1, 18).copy()
        return result.reshape(actions.shape)
