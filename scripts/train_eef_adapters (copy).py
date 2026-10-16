"""Train two MLPs on freshly sampled joint-angle / EEF-pose pairs.

EEF frame: fixed initial top camera. Pose9 = [position3, rotation column1, rotation column2].
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


class PairGenerator:
    def __init__(self, camera_transform=CAMERA_TRANSFORM):
        self.model = mujoco.MjModel.from_xml_path(str(REPO / "abc_sim/models/yam_dishrack_base.xml"))
        self.data = mujoco.MjData(self.model)
        self.camera_from_world = np.asarray(json.loads(camera_transform.read_text())["T_top_camera_from_world"])
        joints = [self.model.joint(f"{side}_joint{j}").id for side in ("left", "right") for j in range(1, 7)]
        self.qpos_indices = self.model.jnt_qposadr[joints]
        self.limits = self.model.jnt_range[joints].copy()
        self.rng = np.random.default_rng(0)
        self.sites = [self.model.site(f"{side}_tcp_site").id for side in ("left", "right")]
        self.metadata = {"frame": "fixed initial top camera", "axes": "x right, y down, z forward",
                         "site": "tcp_site", "rotation_6d": "first two matrix columns",
                         "source": str(camera_transform.resolve()), "joint_ranges_rad": self.limits.tolist(),
                         "T_top_camera_from_world": self.camera_from_world.tolist()}

    def __call__(self, joints):
        joints = np.asarray(joints).reshape(-1, 12)
        poses = np.empty((len(joints), 18), dtype=np.float32)
        for i, row in enumerate(joints):
            self.data.qpos[self.qpos_indices] = row
            mujoco.mj_kinematics(self.model, self.data)
            for arm, site in enumerate(self.sites):
                position = self.camera_from_world[:3, :3] @ self.data.site_xpos[site] + self.camera_from_world[:3, 3]
                rotation = self.camera_from_world[:3, :3] @ self.data.site_xmat[site].reshape(3, 3)
                poses[i, arm * 9:arm * 9 + 9] = np.concatenate((position, rotation[:, 0], rotation[:, 1]))
        return poses

    def sample(self, count, device):
        joints = self.rng.uniform(self.limits[:, 0], self.limits[:, 1], size=(count, 12)).astype(np.float32)
        poses = self(joints)
        return torch.from_numpy(joints).to(device), torch.from_numpy(poses).to(device)


def rotation_matrix(rotation6):
    first = F.normalize(rotation6[..., :3], dim=-1)
    second = rotation6[..., 3:]
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EEFAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_in = MLP(18, 256, 12)
        self.mlp_out = MLP(12, 256, 18)
        self.register_buffer("state_mean", torch.zeros(12))
        self.register_buffer("state_scale", torch.ones(12))
        self.register_buffer("actions_mean", torch.zeros(12))
        self.register_buffer("actions_scale", torch.ones(12))
        self.register_buffer("pose_mean", torch.zeros(18))
        self.register_buffer("pose_std", torch.ones(18))

    def fit_scalers(self, norm_stats, poses):
        indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        for name in ("state", "actions"):
            getattr(self, f"{name}_mean").copy_(torch.from_numpy(norm_stats[name]["mean"][indices]))
            getattr(self, f"{name}_scale").copy_(torch.from_numpy(norm_stats[name]["std"][indices] + 1e-6))
        self.pose_mean.copy_(poses.mean(0))
        self.pose_std.copy_(poses.std(0, unbiased=False).clamp_min(1e-3))

    def normalize(self, joints, poses):
        return ((joints - self.state_mean) / self.state_scale,
                (joints - self.actions_mean) / self.actions_scale,
                (poses - self.pose_mean) / self.pose_std)

    def eef_to_joint(self, poses):
        return self.mlp_in((poses - self.pose_mean) / self.pose_std) * self.state_scale + self.state_mean

    def joint_to_eef(self, joints):
        poses = self.mlp_out((joints - self.actions_mean) / self.actions_scale) * self.pose_std + self.pose_mean
        poses = poses.reshape(*poses.shape[:-1], 2, 9)
        rotation = rotation_matrix(poses[..., 3:])
        return torch.cat((poses[..., :3], rotation[..., :, 0], rotation[..., :, 1]), -1).flatten(-2)


def load_adapters(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = EEFAdapters().to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["kinematics"]


@torch.no_grad()
def validate(model, joints, poses):
    state_input, action_input, pose_input = model.normalize(joints, poses)
    joint_normalized = model.mlp_in(pose_input)
    pose_normalized = model.mlp_out(action_input)
    in_error = joint_normalized - state_input
    out_error = pose_normalized - pose_input
    predicted_joints = joint_normalized * model.state_scale + model.state_mean
    predicted_poses = (pose_normalized * model.pose_std + model.pose_mean).reshape(-1, 2, 9)
    expected_poses = poses.reshape(-1, 2, 9)
    position_error = predicted_poses[..., :3] - expected_poses[..., :3]
    r1 = rotation_matrix(predicted_poses[..., 3:])
    r2 = rotation_matrix(expected_poses[..., 3:])
    cosine = (((r1 * r2).sum((-1, -2)) - 1) / 2).clamp(-1, 1)
    return {"normalized_loss": (in_error.square().mean() + out_error.square().mean()).item(),
            "in_joint_rmse_rad": (predicted_joints - joints).square().mean().sqrt().item(),
            "out_position_rmse_mm": position_error.square().sum(-1).mean().sqrt().item() * 1000,
            "out_rotation_mean_deg": torch.rad2deg(torch.acos(cosine)).mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-transform", type=Path, default=CAMERA_TRANSFORM,
                        help="JSON with T_top_camera_from_world")
    parser.add_argument("--dit-checkpoint", type=Path, default=REPO / "cache/abc_dit_xl_200k_model.pt")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/eef_adapters")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or args.lr <= 0:
        parser.error("Steps, batch size and learning rate must be positive")
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = PairGenerator(args.camera_transform)
    model = EEFAdapters().to(device)
    norm_stats = parse_norm_stats(torch.load(args.dit_checkpoint, map_location="cpu",
                                            mmap=True, weights_only=False)["norm_stats"])
    model.fit_scalers(norm_stats, generator.sample(4096, device)[1])
    val_joints, val_poses = generator.sample(4096, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    print(f"Two MLPs, {sum(p.numel() for p in model.parameters()):,} parameters; device={device}")
    best = float("inf")
    for step in range(1, args.steps + 1):
        joints, poses = generator.sample(args.batch_size, device)
        states, actions, poses = model.normalize(joints, poses)
        loss_in = F.mse_loss(model.mlp_in(poses), states)
        loss_out = F.mse_loss(model.mlp_out(actions), poses)
        optimizer.zero_grad(set_to_none=True)
        (loss_in + loss_out).backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:6d}  in_loss {loss_in.item():.6f}  out_loss {loss_out.item():.6f}", flush=True)
        if step % 1000 == 0 or step == args.steps:
            metrics = validate(model, val_joints, val_poses)
            print("val " + json.dumps(metrics), flush=True)
            if metrics["normalized_loss"] < best:
                best = metrics["normalized_loss"]
                checkpoint = {"model": model.state_dict(), "config": config, "step": step,
                              "kinematics": generator.metadata, "validation": metrics}
                torch.save(checkpoint, args.output / "best.pt")


if __name__ == "__main__":
    main()
