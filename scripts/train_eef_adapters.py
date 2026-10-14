"""Train IK/FK MLPs from official-state joint samples and FK poses.

Pose9 = [position3, rotation column1, rotation column2], in the fixed top-camera frame.
Inputs/outputs use meters and radians; joint scaling is fixed and pose scaling is learned.
FK generation requires CUDA and uses batched mjwarp kinematics.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from abc_minimal.preprocess import parse_norm_stats

REPO = Path(__file__).resolve().parents[1]
CAMERA_TRANSFORM = REPO / "scripts/eef_camera_transform.json"
JOINT_STATE_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class PairGenerator:
    def __init__(self, norm_stats, camera_transform=CAMERA_TRANSFORM):
        self.model = mujoco.MjModel.from_xml_path(str(REPO / "abc_sim/models/yam_dishrack_base.xml"))
        self.data = mujoco.MjData(self.model)
        self.camera_from_world = np.asarray(json.loads(camera_transform.read_text())["T_top_camera_from_world"])
        joints = [self.model.joint(f"{side}_joint{j}").id for side in ("left", "right") for j in range(1, 7)]
        self.qpos_indices = self.model.jnt_qposadr[joints]
        self.limits = self.model.jnt_range[joints].copy()
        self.joint_mean = norm_stats["state"]["mean"][JOINT_STATE_INDICES]
        self.joint_std = norm_stats["state"]["std"][JOINT_STATE_INDICES]
        if not (np.isfinite(self.joint_mean).all() and np.isfinite(self.joint_std).all()
                and (self.joint_std > 0).all()):
            raise ValueError("State sampling mean/std must be finite, with positive std")
        self.rng = np.random.default_rng(0)
        self.sites = [self.model.site(f"{side}_tcp_site").id for side in ("left", "right")]
        self._warp_fk = {}
        self.metadata = {"frame": "fixed initial top camera", "axes": "x right, y down, z forward",
                         "site": "tcp_site", "rotation_6d": "first two matrix columns",
                         "source": str(camera_transform.resolve()), "joint_ranges_rad": self.limits.tolist(),
                         "T_top_camera_from_world": self.camera_from_world.tolist(),
                         "joint_sampling": {"distribution": "independent_truncated_normal",
                                            "stats": "state", "seed": 0,
                                            "mean": self.joint_mean.tolist(),
                                            "std": self.joint_std.tolist()}}

    def __call__(self, joints):
        import mujoco_warp as mjw
        import warp as wp

        from abc_sim.rendering.replay.renderer import _put_model_and_data_safe

        wp.init()
        with wp.ScopedStream(wp.stream_from_torch(joints.device)):
            key = (str(joints.device), len(joints))
            if key not in self._warp_fk:
                model, data = _put_model_and_data_safe(
                    mjw, self.model, self.data, nworld=len(joints), nconmax=0, njmax=0
                )
                camera = torch.as_tensor(self.camera_from_world, dtype=joints.dtype, device=joints.device)
                self._warp_fk[key] = (model, data, camera, wp.to_torch(data.qpos),
                                     wp.to_torch(data.site_xpos), wp.to_torch(data.site_xmat))
            model, data, camera, qpos, site_pos, site_rot = self._warp_fk[key]
            qpos[:, self.qpos_indices] = joints
            mjw.kinematics(model, data)
            positions = site_pos[:, self.sites] @ camera[:3, :3].T + camera[:3, 3]
            rotations = camera[:3, :3] @ site_rot[:, self.sites]
            return torch.cat((positions, rotations[..., :, 0], rotations[..., :, 1]), dim=-1).flatten(1)

    def sample_joints(self, count):
        joints = self.rng.normal(self.joint_mean, self.joint_std, size=(count, 12)).astype(np.float32)
        for _ in range(1000):
            outside = (joints < self.limits[:, 0]) | (joints > self.limits[:, 1])
            if not outside.any():
                return joints
            rows, columns = np.nonzero(outside)
            # Resample rejected coordinates instead of piling samples at limits.
            joints[rows, columns] = self.rng.normal(self.joint_mean[columns], self.joint_std[columns])
        raise RuntimeError("Could not sample joints within limits; check the state stats and robot model")

    def sample(self, count, device):
        joint_values = self.sample_joints(count)
        joints = torch.from_numpy(joint_values).to(device)
        return joints, self(joints)


def rotation_matrix(rotation6):
    first = F.normalize(rotation6[..., :3], dim=-1)
    second = rotation6[..., 3:]
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


class MLP(nn.Sequential):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__(
            nn.Linear(input_dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, output_dim),
        )


class EEFAdapters(nn.Module):
    """Map raw poses/joints using shared state stats and trainable pose scaling."""

    def __init__(self, state_mean, state_scale):
        super().__init__()
        self.mlp_in = MLP(18, 256, 12)
        self.mlp_out = MLP(12, 256, 18)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32).clone())
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32).clone())
        self.pose_mean = nn.Parameter(torch.zeros(18))
        self.pose_log_std = nn.Parameter(torch.zeros(18))

    @property
    def pose_std(self):
        return self.pose_log_std.exp().clamp_min(1e-3)

    def forward(self, joints, poses):
        joint_input = (joints - self.state_mean) / self.state_scale
        joint_prediction = self.mlp_in((poses - self.pose_mean) / self.pose_std)
        pose_prediction = self.predict_pose_values(joints)
        losses = (F.mse_loss(joint_prediction, joint_input), F.mse_loss(pose_prediction, poses))
        return losses, joint_prediction * self.state_scale + self.state_mean, pose_prediction

    def eef_to_joint(self, poses):
        pose_input = (poses - self.pose_mean) / self.pose_std
        return self.mlp_in(pose_input) * self.state_scale + self.state_mean

    def predict_pose_values(self, joints):
        joint_input = (joints - self.state_mean) / self.state_scale
        return self.mlp_out(joint_input) * self.pose_std + self.pose_mean

    def joint_to_eef(self, joints):
        poses = self.predict_pose_values(joints)
        poses = poses.reshape(*poses.shape[:-1], 2, 9)
        rotation = rotation_matrix(poses[..., 3:])
        return torch.cat((poses[..., :3], rotation[..., :, 0], rotation[..., :, 1]), -1).flatten(-2)


def load_adapters(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model_state = saved["model"]
    model = EEFAdapters(model_state["state_mean"], model_state["state_scale"]).to(device)
    model.load_state_dict(model_state)
    return model.eval(), saved["kinematics"]


@torch.no_grad()
def validate(model, joints, poses):
    (loss_in, loss_out), predicted_joints, predicted_poses = model(joints, poses)
    predicted_poses = predicted_poses.reshape(-1, 2, 9)
    expected_poses = poses.reshape(-1, 2, 9)
    position_error = predicted_poses[..., :3] - expected_poses[..., :3]
    r1 = rotation_matrix(predicted_poses[..., 3:])
    r2 = rotation_matrix(expected_poses[..., 3:])
    cosine = (((r1 * r2).sum((-1, -2)) - 1) / 2).clamp(-1, 1)
    return {"loss": (loss_in + loss_out).item(),
            "in_loss": loss_in.item(), "out_loss": loss_out.item(),
            "in_joint_rmse_rad": (predicted_joints - joints).square().mean().sqrt().item(),
            "out_position_rmse_mm": position_error.square().sum(-1).mean().sqrt().item() * 1000,
            "out_rotation_mean_deg": torch.rad2deg(torch.acos(cosine)).mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-transform", type=Path, default=CAMERA_TRANSFORM,
                        help="JSON with T_top_camera_from_world")
    parser.add_argument("--dit-checkpoint", type=Path, default=REPO / "cache/abc_dit_xl_200k_model.pt",
                        help="Checkpoint supplying state mean/std for joint sampling and both MLPs")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/eef_adapters")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or args.lr <= 0:
        parser.error("Steps, batch size and learning rate must be positive")
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mjwarp FK training")
    device = "cuda"
    norm_stats = parse_norm_stats(torch.load(args.dit_checkpoint, map_location="cpu",
                                            mmap=True, weights_only=False)["norm_stats"])
    generator = PairGenerator(norm_stats, args.camera_transform)
    model = EEFAdapters(generator.joint_mean, generator.joint_std + 1e-6).to(device)
    val_joints, val_poses = generator.sample(4096, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    print(f"Two MLPs, {sum(p.numel() for p in model.parameters()):,} parameters; device={device}")
    best = float("inf")
    for step in range(1, args.steps + 1):
        joints, poses = generator.sample(args.batch_size, device)
        (loss_in, loss_out), _, _ = model(joints, poses)
        optimizer.zero_grad(set_to_none=True)
        (loss_in + loss_out).backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:6d}  in_loss {loss_in.item():.6f}  out_loss {loss_out.item():.6f}", flush=True)
        if step % 1000 == 0 or step == args.steps:
            metrics = validate(model, val_joints, val_poses)
            print("val " + json.dumps(metrics), flush=True)
            if metrics["loss"] < best:
                best = metrics["loss"]
                checkpoint = {"model": model.state_dict(), "config": config, "step": step,
                              "kinematics": generator.metadata, "validation": metrics}
                torch.save(checkpoint, args.output / "best.pt")


if __name__ == "__main__":
    main()
