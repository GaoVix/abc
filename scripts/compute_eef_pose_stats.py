"""Compute fixed EEF mean/std from every raw arm-state message in MCAP episodes."""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abc_minimal.export_mcap import STATE_TOPICS
from abc_minimal.stats import RunningStats

REPO = Path(__file__).resolve().parents[1]
CAMERA_TRANSFORM = REPO / "scripts/eef_camera_transform.json"
ARM_TOPICS = [topic for topic, dim in STATE_TOPICS if dim == 6]


class EEFKinematics:
    def __init__(self, camera_transform):
        self.camera = np.asarray(json.loads(Path(camera_transform).read_text())["T_top_camera_from_world"])
        self.model = mujoco.MjModel.from_xml_path(str(REPO / "abc_sim/models/yam_dishrack_base.xml"))
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(f"{side}_joint{j}").id for side in ("left", "right") for j in range(1, 7)]
        self.qpos_indices = self.model.jnt_qposadr[joints]
        self.sites = [self.model.site(f"{side}_tcp_site").id for side in ("left", "right")]

    def fk(self, joints):
        poses = []
        for values in np.asarray(joints).reshape(-1, 12):
            self.data.qpos[self.qpos_indices] = values
            mujoco.mj_kinematics(self.model, self.data)
            positions = self.data.site_xpos[self.sites] @ self.camera[:3, :3].T + self.camera[:3, 3]
            rotations = self.camera[:3, :3] @ self.data.site_xmat[self.sites].reshape(2, 3, 3)
            poses.append(np.concatenate((positions, rotations[:, :, 0], rotations[:, :, 1]), axis=-1).reshape(18))
        return np.asarray(poses, dtype=np.float32)


def read_joint_states(path):
    states = {topic: [] for topic in ARM_TOPICS}
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, decoded in reader.iter_decoded_messages(topics=ARM_TOPICS):
            states[channel.topic].append(list(decoded.position))
    result = [np.asarray(states[topic], dtype=np.float64) for topic in ARM_TOPICS]
    for topic, values in zip(ARM_TOPICS, result):
        if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
            raise ValueError(f"{path}: missing or invalid joint states for {topic}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="ABC-130k root containing data/train and data/val")
    parser.add_argument("--output", type=Path, default=REPO / "cache/eef_pose_norm_stats.json")
    parser.add_argument("--camera-transform", type=Path, default=CAMERA_TRANSFORM)
    args = parser.parse_args()
    episodes = sorted(args.dataset.glob("data/*/*/episode_*/episode.mcap"))
    if not episodes:
        parser.error(f"No episode.mcap files found under {args.dataset}")

    kinematics = EEFKinematics(args.camera_transform)
    stats = [RunningStats(), RunningStats()]
    started = time.monotonic()
    for index, path in enumerate(episodes, start=1):
        for arm, values in enumerate(read_joint_states(path)):
            joints = np.zeros((len(values), 12), dtype=np.float64)
            joints[:, arm * 6 : (arm + 1) * 6] = values
            # Each arm's FK is independent; keep both raw message streams intact.
            poses = kinematics.fk(joints)[:, arm * 9 : (arm + 1) * 9]
            stats[arm].update(poses)
        if index == 1 or index % 25 == 0 or index == len(episodes):
            elapsed = time.monotonic() - started
            print(f"[{index}/{len(episodes)}] left={stats[0].count} right={stats[1].count} elapsed={elapsed:.1f}s",
                  flush=True)

    left, right = [value.statistics() for value in stats]
    result = {key: left[key] + right[key] for key in ("mean", "std")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Saved {args.output.resolve()} (ddof=1)", flush=True)


if __name__ == "__main__":
    main()
