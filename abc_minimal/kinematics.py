from pathlib import Path

import mujoco
import numpy as np
import torch


def _rotate(vectors, axis, sine, cosine):
    return (vectors * cosine
            + torch.linalg.cross(axis.expand_as(vectors), vectors, dim=-1) * sine
            + axis * (vectors * axis).sum(dim=-1, keepdim=True) * (1 - cosine))


class EEFPoseFK:
    """Batched FK of normalized actions, using MuJoCo's joint geometry."""

    def __init__(self, action_stats, device):
        path = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_dishrack_base.xml"
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        joints = np.array([[model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
                           for side in ("left", "right")])
        sites = [model.site(f"{side}_tcp_site").id for side in ("left", "right")]
        camera = model.camera("top").id
        rotation = np.diag([1., -1., -1.]) @ data.cam_xmat[camera].reshape(3, 3).T
        origin = data.cam_xpos[camera]
        options = {"device": device, "dtype": torch.float32}
        self.mean = torch.as_tensor(action_stats["mean"], **options)
        self.scale = torch.as_tensor(action_stats["std"], **options) + 1e-6
        self.reference = torch.as_tensor(model.qpos0[model.jnt_qposadr[joints]], **options)
        self.axes = torch.as_tensor(data.xaxis[joints] @ rotation.T, **options)
        self.anchors = torch.as_tensor((data.xanchor[joints] - origin) @ rotation.T, **options)
        self.position = torch.as_tensor((data.site_xpos[sites] - origin) @ rotation.T, **options)
        self.rotation = torch.as_tensor(rotation @ data.site_xmat[sites].reshape(2, 3, 3), **options)

    @torch.no_grad()
    def __call__(self, actions):
        joints = actions * self.scale + self.mean
        angles = joints.reshape(*joints.shape[:-1], 2, 7)[..., :6] - self.reference
        position = self.position.expand(*actions.shape[:-1], 2, 3)
        rotation = self.rotation.transpose(-1, -2).expand(*actions.shape[:-1], 2, 3, 3)
        for index in range(5, -1, -1):
            axis, anchor = self.axes[:, index], self.anchors[:, index]
            angle = angles[..., index, None]
            sine, cosine = angle.sin(), angle.cos()
            position = anchor + _rotate(position - anchor, axis, sine, cosine)
            rotation = _rotate(rotation, axis[:, None], sine[..., None], cosine[..., None])
        return position, rotation.transpose(-1, -2)
