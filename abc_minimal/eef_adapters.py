"""EEF adapters for DiT finetuning and inference."""

import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from abc_sim.rendering.replay.renderer import _put_model_and_data_safe
from deploy.robot.control.mink_ik import MinkArmController
from scripts.train_eef_adapters import CAMERA_TRANSFORM, JOINT_STATE_INDICES, load_adapters, rotation_matrix

KINEMATIC_MODEL = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_dishrack_base.xml"
EEF_POSE_INDICES = [*range(9), *range(10, 19)]
EEF_GRIPPER_INDICES = [9, 19]


class EEFKinematics:
    def __init__(self, camera_transform):
        camera_transform = Path(camera_transform)
        if not camera_transform.is_file() and camera_transform.parts[-2:] == ("scripts", CAMERA_TRANSFORM.name):
            camera_transform = CAMERA_TRANSFORM
        self.camera_transform = str(camera_transform)
        self.camera = np.asarray(json.loads(camera_transform.read_text())["T_top_camera_from_world"])
        self.model = mujoco.MjModel.from_xml_path(str(KINEMATIC_MODEL))
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(f"{side}_joint{j}").id for side in ("left", "right") for j in range(1, 7)]
        self.qpos_indices = self.model.jnt_qposadr[joints]
        self.sites = [self.model.site(f"{side}_tcp_site").id for side in ("left", "right")]

    def fk(self, joints):
        poses = []
        for values in np.asarray(joints).reshape(-1, 12):
            self.data.qpos[self.qpos_indices] = values
            mujoco.mj_kinematics(self.model, self.data)
            positions = self.data.site_xpos[self.sites, :] @ self.camera[:3, :3].T + self.camera[:3, 3]
            rotations = self.camera[:3, :3] @ self.data.site_xmat[self.sites, :].reshape(2, 3, 3)
            poses.append(np.concatenate((positions, rotations[:, :, 0], rotations[:, :, 1]), axis=-1).reshape(18))
        return np.asarray(poses, dtype=np.float32)

    def fk_batch(self, joints):
        batch_key = (len(joints), joints.device, torch.cuda.current_stream(joints.device).cuda_stream)
        if getattr(self, "_batch_key", None) != batch_key:
            wp.set_device(str(joints.device))
            wp.set_stream(wp.stream_from_torch(joints.device), sync=True)
            self.warp_model, self.warp_data = _put_model_and_data_safe(
                mjw, self.model, self.data, nworld=len(joints), nconmax=0, njmax=0
            )
            self.camera_tensor = torch.as_tensor(self.camera, dtype=torch.float32, device=joints.device)
            self._batch_key = batch_key
        wp.to_torch(self.warp_data.qpos)[:, self.qpos_indices] = joints
        mjw.kinematics(self.warp_model, self.warp_data)
        site_pos = wp.to_torch(self.warp_data.site_xpos)[:, self.sites]
        site_rot = wp.to_torch(self.warp_data.site_xmat)[:, self.sites]
        positions = site_pos @ self.camera_tensor[:3, :3].T + self.camera_tensor[:3, 3]
        rotations = self.camera_tensor[:3, :3] @ site_rot
        return torch.cat((positions, rotations[..., 0], rotations[..., 1]), dim=-1).flatten(1)


class EEFTrainingAdapter(torch.nn.Module):
    def __init__(self, checkpoint, norm_stats):
        super().__init__()
        saved = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.mlps = load_adapters(saved)
        self.kinematics = EEFKinematics(saved["config"]["camera_transform"])
        for name in ("state", "actions"):
            self.register_buffer(f"{name}_mean", torch.as_tensor(norm_stats[name]["mean"], dtype=torch.float32).clone())
            self.register_buffer(f"{name}_scale", torch.as_tensor(norm_stats[name]["std"], dtype=torch.float32).clone() + 1e-6)

    @torch.no_grad()
    def prepare(self, batch):
        state = batch["state"] * self.state_scale + self.state_mean
        actions = batch["actions"] * self.actions_scale + self.actions_mean
        joints = torch.cat((state[:, None], actions), dim=1)[..., JOINT_STATE_INDICES]
        poses = self.kinematics.fk_batch(joints.flatten(0, 1)).reshape(*joints.shape[:2], 18)
        return {**batch, "eef_state": poses[:, 0], "eef_actions": poses[:, 1:]}

    def forward(self, batch):
        state = batch["state"].clone()
        joints = self.mlps.eef_to_joint(batch["eef_state"])
        state[..., JOINT_STATE_INDICES] = (
            joints - self.state_mean[JOINT_STATE_INDICES]
        ) / self.state_scale[JOINT_STATE_INDICES]
        state = torch.where(batch["state_is_masked"][:, None], torch.zeros_like(state), state)
        poses = (batch["eef_actions"] - self.mlps.pose_mean) / (self.mlps.pose_std + 1e-6)
        actions = torch.cat((poses[..., :9], batch["actions"][..., 6:7],
                             poses[..., 9:], batch["actions"][..., 13:14]), dim=-1)
        return {**batch, "state": state, "actions": actions}

    def eef_to_joint(self, actions):
        joints = self.mlps.mlp_in(actions[..., EEF_POSE_INDICES]) * self.mlps.state_scale + self.mlps.state_mean
        joints = (joints - self.actions_mean[JOINT_STATE_INDICES]) / self.actions_scale[JOINT_STATE_INDICES]
        return torch.cat((joints[..., :6], actions[..., 9:10], joints[..., 6:], actions[..., 19:20]), dim=-1)

    def joint_to_eef(self, actions):
        """Decode the joint-shaped DiT output as normalized EEF velocity."""
        physical = actions * self.actions_scale + self.actions_mean
        joints = (physical[..., JOINT_STATE_INDICES] - self.mlps.state_mean) / self.mlps.state_scale
        poses = self.mlps.mlp_out(joints)
        return torch.cat((poses[..., :9], actions[..., 6:7], poses[..., 9:], actions[..., 13:14]), dim=-1)


class EEFInferenceAdapter(EEFKinematics):
    def __init__(self, checkpoint, device, execute_chunk_dim, action_dim=14):
        self.device = device
        self.execute_chunk_dim = execute_chunk_dim
        self.action_dim = action_dim
        saved = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.mlps = load_adapters(saved, device)
        super().__init__(saved["config"]["camera_transform"])
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

    @torch.no_grad()
    def state(self, state):
        result = state.copy()
        poses = self.fk(state[..., JOINT_STATE_INDICES])
        joints = self.mlps.eef_to_joint(torch.as_tensor(poses, device=self.device))
        result[..., JOINT_STATE_INDICES] = joints.cpu().numpy().reshape(*state.shape[:-1], 12)
        return result

    @torch.no_grad()
    def action_prefix(self, actions, prefix_length):
        result = np.asarray(actions, dtype=np.float32).copy()
        if self.action_dim == 20:
            result = np.zeros((*result.shape[:-1], 20), dtype=np.float32)
            result[..., EEF_GRIPPER_INDICES] = actions[..., [6, 13]]
            if prefix_length:
                poses = self.previous_action_poses
                if poses is None:
                    poses = self.fk(actions[..., :prefix_length, JOINT_STATE_INDICES])
                else:
                    poses = poses[:, -prefix_length:]
                result[..., :prefix_length, EEF_POSE_INDICES] = poses.reshape(*result.shape[:-2], prefix_length, 18)
            return result
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
        if self.action_dim == 20:
            result = np.zeros((*actions.shape[:-1], 14), dtype=np.float32).reshape(-1, actions.shape[-2], 14)
            result[..., [6, 13]] = actions[..., EEF_GRIPPER_INDICES].reshape(*result.shape[:2], 2)
            poses = torch.as_tensor(actions[..., EEF_POSE_INDICES], device=self.device).reshape(*result.shape[:2], 2, 9)
            rotations = rotation_matrix(poses[..., 3:])
            poses = torch.cat((poses[..., :3], rotations[..., :, 0], rotations[..., :, 1]), dim=-1)
        else:
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
        self.previous_action_poses = poses[:, prefix_length : last + 1].reshape(len(result), -1, 18).copy()
        return result.reshape(*actions.shape[:-1], 14)
