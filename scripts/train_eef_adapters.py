import argparse
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from torch import nn
from torch.nn import functional as F

from abc_minimal.preprocess import parse_norm_stats
from abc_sim.rendering.replay.renderer import _put_model_and_data_safe

REPO = Path(__file__).resolve().parents[1]
CAMERA_TRANSFORM = REPO / "scripts/eef_camera_transform.json"
JOINT_STATE_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class PairGenerator:
    def __init__(self, norm_stats, batch_size, device, camera_transform=CAMERA_TRANSFORM, rotation_dim=6):
        wp.init()
        stream = wp.stream_from_torch(device)
        wp.set_device(stream.device)
        wp.set_stream(stream, sync=True)
        model = mujoco.MjModel.from_xml_path(str(REPO / "abc_sim/models/yam_dishrack_base.xml"))
        data = mujoco.MjData(model)
        self.batch_size = batch_size
        self.device = device
        self.rotation_dim = rotation_dim
        self.camera_from_world = torch.tensor(
            json.loads(camera_transform.read_text())["T_top_camera_from_world"],
            dtype=torch.float32,
            device=device,
        )
        joint_ids = [model.joint(f"{side}_joint{j}").id for side in ("left", "right") for j in range(1, 7)]
        self.qpos_indices = model.jnt_qposadr[joint_ids]
        self.limits = model.jnt_range[joint_ids, :].copy()
        self.joint_mean = norm_stats["state"]["mean"][JOINT_STATE_INDICES]
        self.joint_std = norm_stats["state"]["std"][JOINT_STATE_INDICES]
        self.rng = np.random.default_rng(0)
        self.sites = [model.site(f"{side}_tcp_site").id for side in ("left", "right")]
        self.model, self.data = _put_model_and_data_safe(mjw, model, data, nworld=batch_size, nconmax=0, njmax=0)
        self.qpos = wp.to_torch(self.data.qpos)
        self.site_pos = wp.to_torch(self.data.site_xpos)
        self.site_rot = wp.to_torch(self.data.site_xmat)

    def fk(self, joints):
        self.qpos[:, self.qpos_indices] = joints
        mjw.kinematics(self.model, self.data)
        positions = self.site_pos[:, self.sites, :] @ self.camera_from_world[:3, :3].T + self.camera_from_world[:3, 3]
        rotations = self.camera_from_world[:3, :3] @ self.site_rot[:, self.sites, :, :]
        orientation = (rotation_vector(rotations) if self.rotation_dim == 3
                       else torch.cat((rotations[..., 0], rotations[..., 1]), dim=-1))
        return torch.cat((positions, orientation), dim=-1).flatten(1)

    def __call__(self):
        joints = self.rng.normal(self.joint_mean, self.joint_std, size=(self.batch_size, 12)).astype(np.float32)
        for _ in range(1000):
            outside = (joints < self.limits[:, 0]) | (joints > self.limits[:, 1])
            if not outside.any():
                joints = torch.from_numpy(joints).to(self.device)
                return joints, self.fk(joints)
            rows, columns = np.nonzero(outside)
            # Resample rejected coordinates instead of piling samples at limits.
            joints[rows, columns] = self.rng.normal(self.joint_mean[columns], self.joint_std[columns])
        raise RuntimeError("Could not sample joints within limits; check the state stats and robot model")


def rotation_matrix(rotation6):
    """Decode a rotation vector (3D) or continuous rotation representation (6D)."""
    if rotation6.shape[-1] == 3:
        x, y, z = rotation6.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).unflatten(-1, (3, 3))
        angle = rotation6.norm(dim=-1)[..., None, None]
        return (torch.eye(3, device=rotation6.device, dtype=rotation6.dtype)
                + torch.sinc(angle / torch.pi) * skew
                + .5 * torch.sinc(angle / (2 * torch.pi)).square() * (skew @ skew))
    first = F.normalize(rotation6[..., :3], dim=-1)
    second = rotation6[..., 3:]
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def rotation_vector(matrix):
    """Principal rotation vector (radians), including rotations near zero and pi."""
    a, b, c, d, e, f, g, h, i = matrix.flatten(-2).unbind(-1)
    terms = torch.stack((1 + a + e + i, 1 + a - e - i, 1 - a + e - i, 1 - a - e + i), -1)
    candidates = torch.stack((
        torch.stack((terms[..., 0], h - f, c - g, d - b), -1),
        torch.stack((h - f, terms[..., 1], b + d, c + g), -1),
        torch.stack((c - g, b + d, terms[..., 2], f + h), -1),
        torch.stack((d - b, c + g, f + h, terms[..., 3]), -1),
    ), -2)
    index = terms.argmax(-1, keepdim=True)
    quaternion = candidates.gather(-2, index[..., None].expand(*index.shape, 4)).squeeze(-2)
    quaternion = quaternion / (2 * terms.gather(-1, index).clamp_min(1e-8).sqrt())
    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    angle = 2 * torch.atan2(quaternion[..., 1:].norm(dim=-1, keepdim=True), quaternion[..., :1])
    return quaternion[..., 1:] * (2 / torch.sinc(angle / (2 * torch.pi)))


class MLP(nn.Sequential):
    def __init__(self, sizes):
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(sizes) - 2:
                layers.append(nn.GELU(approximate="tanh"))
        super().__init__(*layers)


class EEFAdapters(nn.Module):

    def __init__(
        self, state_mean, state_scale, pose_mean, pose_std,
        ik_sizes=None, fk_sizes=None,
    ):
        super().__init__()
        self.pose_dim = len(pose_mean)
        self.arm_pose_dim = self.pose_dim // 2
        ik_sizes = ik_sizes or (self.pose_dim, 64, 128, 256, 512, 256, 128, 64, 12)
        fk_sizes = fk_sizes or (12, 64, 128, 64, self.pose_dim)
        self.mlp_in = MLP(ik_sizes)
        self.mlp_out = MLP(fk_sizes)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32).clone())
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32).clone())
        self.register_buffer("pose_mean", torch.as_tensor(pose_mean, dtype=torch.float32).clone())
        self.register_buffer("pose_std", torch.as_tensor(pose_std, dtype=torch.float32).clone())
        if self.pose_mean.shape not in ((12,), (18,)) or self.pose_mean.shape != self.pose_std.shape:
            raise ValueError("Pose mean and std must each contain 12 (rotvec) or 18 (rotation6d) values")
        if not torch.isfinite(self.pose_mean).all() or not torch.isfinite(self.pose_std).all():
            raise ValueError("Pose mean and std must be finite")
        if (self.pose_std < 0).any():
            raise ValueError("Pose std must be nonnegative")

    def forward(self, joints, poses):
        joint_prediction = self.eef_to_joint(poses)
        pose_prediction = self.predict_pose_values(joints)
        pred = pose_prediction.reshape(-1, 2, self.arm_pose_dim)
        target = poses.reshape(-1, 2, self.arm_pose_dim)
        pos_loss = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        rot_loss = F.mse_loss(pred[:, :, 3:], target[:, :, 3:])
        loss_in = ((joint_prediction - joints) / self.state_scale).square().mean()
        return (loss_in, 10 * pos_loss + rot_loss), joint_prediction, pose_prediction

    def eef_to_joint(self, poses):
        pose_input = (poses - self.pose_mean) / (self.pose_std + 1e-6)
        return self.mlp_in(pose_input) * self.state_scale + self.state_mean

    def transform_eef_noise(self, noise):
        arms = noise.unflatten(-1, (2, self.arm_pose_dim + 1))
        poses = self.mlp_out(self.mlp_in(arms[..., :self.arm_pose_dim].flatten(-2))).unflatten(-1, (2, self.arm_pose_dim))
        return torch.cat((poses, arms[..., self.arm_pose_dim:]), dim=-1).flatten(-2)

    def predict_pose_values(self, joints):
        joint_input = (joints - self.state_mean) / self.state_scale
        return self.mlp_out(joint_input) * (self.pose_std + 1e-6) + self.pose_mean

    def joint_to_eef(self, joints):
        poses = self.predict_pose_values(joints)
        if self.arm_pose_dim == 6:
            return poses
        poses = poses.reshape(*poses.shape[:-1], 2, 9)
        rotation = rotation_matrix(poses[..., 3:])
        return torch.cat((poses[..., :3], rotation[..., :, 0], rotation[..., :, 1]), -1).flatten(-2)


def load_adapters(checkpoint, device="cpu"):
    saved = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location=device, weights_only=True)
    model_state = saved["model"]
    sizes = {}
    for name in ("mlp_in", "mlp_out"):
        keys = sorted(
            (key for key in model_state if key.startswith(f"{name}.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        sizes[name] = [model_state[keys[0]].shape[1]] + [model_state[key].shape[0] for key in keys]
    with torch.random.fork_rng(devices=[]):
        model = EEFAdapters(model_state["state_mean"], model_state["state_scale"],
                            model_state["pose_mean"], model_state["pose_std"],
                            ik_sizes=sizes["mlp_in"], fk_sizes=sizes["mlp_out"]).to(device)
    model.load_state_dict(model_state)
    return model.eval()


@torch.no_grad()
def validate(model, joints, poses, fk):
    (loss_in, loss_out), predicted_joints, predicted_poses = model(joints, poses)
    expected_poses = poses.reshape(-1, 2, model.arm_pose_dim)
    r2 = rotation_matrix(expected_poses[:, :, 3:])
    metrics = {"loss": (loss_in + loss_out).item(),
               "in_loss": loss_in.item(), "out_loss": loss_out.item(),
               "in_joint_rmse_rad": (predicted_joints - joints).square().mean().sqrt().item()}
    for name, prediction in (("in", fk(predicted_joints)), ("out", predicted_poses)):
        prediction = prediction.reshape(-1, 2, model.arm_pose_dim)
        position_error = prediction[:, :, :3] - expected_poses[:, :, :3]
        r1 = rotation_matrix(prediction[:, :, 3:])
        cosine = (((r1 * r2).sum((-1, -2)) - 1) / 2).clamp(-1, 1)
        metrics[f"{name}_position_rmse_mm"] = position_error.square().sum(-1).mean().sqrt().item() * 1000
        metrics[f"{name}_rotation_mean_deg"] = torch.rad2deg(torch.acos(cosine)).mean().item()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-transform", type=Path, default=CAMERA_TRANSFORM, help="JSON with T_top_camera_from_world")
    parser.add_argument("--dit-checkpoint", type=Path, default=REPO / "cache/abc_dit_xl_200k_model.pt", help="Checkpoint supplying state mean/std for joint sampling and both MLPs")
    parser.add_argument("--pose-norm-stats", type=Path, required=True, help="JSON with 12D rotvec or 18D rotation6D EEF pose mean/std (grippers excluded)")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/eef_adapters")
    parser.add_argument("--steps", type=int, default=350000)
    parser.add_argument("--batch-size", type=int, default=40960)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1:
        parser.error("Steps and batch size must be positive")
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mjwarp FK training")
    device = "cuda"
    norm_stats = parse_norm_stats(torch.load(args.dit_checkpoint, map_location="cpu", mmap=True, weights_only=False)["norm_stats"])
    pose_stats = json.loads(args.pose_norm_stats.read_text())
    pose_dim = len(pose_stats["mean"])
    if pose_dim not in (12, 18):
        parser.error("Pose statistics must contain 12 (rotvec) or 18 (rotation6D) values")
    generator = PairGenerator(norm_stats, args.batch_size, device, args.camera_transform, rotation_dim=pose_dim // 2 - 3)
    model = EEFAdapters(generator.joint_mean, generator.joint_std + 1e-6,
                        pose_stats["mean"], pose_stats["std"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.wandb:
        import wandb
        wandb.init(project="abc-eef-adapters", config=config, dir=str(args.output))
    print(f"Two MLPs, {sum(p.numel() for p in model.parameters()):,} parameters; device={device}")
    best = float("inf")
    for step in range(1, args.steps + 1):
        joints, poses = generator()
        (loss_in, loss_out), _, _ = model(joints, poses)
        optimizer.zero_grad(set_to_none=True)
        (loss_in + loss_out).backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:6d}  in_loss {loss_in.item():.6f}  out_loss {loss_out.item():.6f}", flush=True)
            if args.wandb:
                wandb.log({"train/in_loss": loss_in.item(), "train/out_loss": loss_out.item()}, step=step)
        if step % 1000 == 0 or step == args.steps:
            val_joints, val_poses = generator()
            metrics = validate(model, val_joints, val_poses, generator.fk)
            print("val " + json.dumps(metrics), flush=True)
            if args.wandb:
                wandb.log({f"val/{key}": value for key, value in metrics.items()}, step=step)
            checkpoint = {"model": model.state_dict(), "config": config, "step": step,
                          "validation": metrics}
            torch.save(checkpoint, args.output / "last.pt")
            if metrics["loss"] < best:
                best = metrics["loss"]
                torch.save(checkpoint, args.output / "best.pt")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
