import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import mujoco
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, map_coordinates
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.robot.control.mink_ik import MinkArmController


SIDES = ("left", "right")


def pose(entity):
    result = np.eye(4)
    result[:3, :3] = entity.xmat.reshape(3, 3)
    result[:3, 3] = entity.xpos
    return result


def video_frames(path, topic):
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        decoder = None
        for _, _, message, value in reader.iter_decoded_messages(topics=[topic]):
            if decoder is None:
                decoder = av.CodecContext.create(value.format.lower().replace("h265", "hevc"), "r")
                decoder.thread_count = 2
                origin = message.log_time
            packet = av.Packet(value.data)
            packet.pts = message.log_time - origin
            packet.time_base = Fraction(1, 1_000_000_000)
            for frame in decoder.decode(packet):
                yield origin + round(frame.pts * frame.time_base * 1_000_000_000), frame
        for frame in decoder.decode(None):
            yield origin + round(frame.pts * frame.time_base * 1_000_000_000), frame


class Robot:
    def __init__(self, path, bases=None):
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        if bases is not None:
            for side in SIDES:
                transform = np.asarray(bases[side])
                body = self.model.body(f"{side}_arm")
                body.pos[:] = transform[:3, 3]
                body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.joints = [[self.model.joint(f"{side}_joint{i}") for i in range(1, 7)] for side in SIDES]
        self.qpos = [[int(j.qposadr[0]) for j in joints] for joints in self.joints]
        self.sites = [self.model.site(f"{side}_tcp_site").id for side in SIDES]
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[2] = 1
        self.options.sitegroup[:] = 0
        for geom in range(self.model.ngeom):
            name = self.model.body(int(self.model.geom_bodyid[geom])).name
            if not name.startswith(("left_", "right_")):
                self.model.geom_group[geom] = 0

    def set_state(self, state):
        for arm, side in enumerate(SIDES):
            self.data.qpos[self.qpos[arm]] = state[7 * arm:7 * arm + 6]
            driver, follower = [self.model.joint(f"{side}_{finger}_finger") for finger in ("left", "right")]
            coupling = np.flatnonzero((self.model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
                                      & (self.model.eq_obj1id == driver.id)
                                      & (self.model.eq_obj2id == follower.id)).item()
            constant, ratio = self.model.eq_data[coupling, :2]
            displacement = 0.0475 * state[7 * arm + 6]
            self.data.qpos[int(driver.qposadr[0])] = displacement
            self.data.qpos[int(follower.qposadr[0])] = (displacement - constant) / ratio
        mujoco.mj_kinematics(self.model, self.data)

    def tcp(self, arm):
        return pose(self.data.site(self.sites[arm]))

    def setup_ik(self, path, state):
        self.controllers, self.tcp_from_link = [], []
        for arm, side in enumerate(SIDES):
            link = f"{side}_link_6"
            self.tcp_from_link.append(np.linalg.inv(self.tcp(arm)) @ pose(self.data.body(link)))
            spec = mujoco.MjSpec.from_file(str(path))
            for field in ("sensors", "equalities", "actuators", "tendons", "keys"):
                for element in list(getattr(spec, field)):
                    spec.delete(element)
            for joint in list(spec.joints):
                if joint.name not in {f"{side}_joint{i}" for i in range(1, 7)}:
                    spec.delete(joint)
            self.controllers.append(MinkArmController(
                state[arm * 7:arm * 7 + 6], model=spec.compile(), target_link=link
            ))

    def solve(self, arm, target, seed):
        link_target = target @ self.tcp_from_link[arm]
        quaternion = Rotation.from_matrix(link_target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        joints = self.controllers[arm].solve(link_target[:3, 3], quaternion, seed)
        self.data.qpos[self.qpos[arm]] = joints
        mujoco.mj_kinematics(self.model, self.data)
        current = self.tcp(arm)
        rotation = Rotation.from_matrix(current[:3, :3] @ target[:3, :3].T).magnitude()
        return joints, np.array([np.linalg.norm(current[:3, 3] - target[:3, 3]) * 1000,
                                 np.rad2deg(rotation)])


def projection_maps(camera, fovy):
    width, height = camera["width"], camera["height"]
    intrinsic = np.asarray(camera["K"]).reshape(3, 3)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) / intrinsic[1, 1]
    k1, k2, p1, p2, k3 = camera["D"]
    radius2 = x * x + y * y
    radial = 1 + ((k3 * radius2 + k2) * radius2 + k1) * radius2
    dx, dy = x * radial, y * radial
    source_u = (dx + 2 * p1 * dx * dy + p2 * (radius2 + 2 * dx * dx)) * intrinsic[0, 0] + intrinsic[0, 2]
    source_v = (dy + 2 * p2 * dx * dy + p1 * (radius2 + 2 * dy * dy)) * intrinsic[1, 1] + intrinsic[1, 2]
    focal = height / (2 * np.tan(np.deg2rad(fovy / 2)))
    return np.array([source_v, source_u]), np.array([focal * y + (height - 1) / 2,
                                                               focal * x + (width - 1) / 2])


def remap_rgb(rgb, coordinates):
    return np.stack([map_coordinates(rgb[..., channel], coordinates, order=1, mode="constant")
                     for channel in range(3)], axis=-1)


def robot_mask(robot, renderer, camera_in_world, coordinates):
    renderer.update_scene(robot.data, "top", scene_option=robot.options)
    for camera in renderer.scene.camera:
        camera.pos[:] = camera_in_world[:3, 3]
        camera.forward[:] = camera_in_world[:3, 2]
        camera.up[:] = -camera_in_world[:3, 1]
    segmentation = renderer.render()
    return map_coordinates((segmentation[..., 0] >= 0).astype(np.uint8), coordinates, order=0).astype(bool)


def overlay(rgb, mask, color, label):
    result = rgb.copy()
    result[mask] = 0.78 * result[mask] + 0.22 * np.array(color)
    result[mask & ~binary_erosion(mask)] = color
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(20, 25, 30))
    draw.text((8, 6), label, fill="white")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--real-model", type=Path, required=True)
    parser.add_argument("--sim-model", type=Path, default=REPO / "abc_sim/models/yam_dishrack_base.xml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((args.calibration / "camera_calibration.json").read_text())
    alignment = np.load(args.calibration / "time_offset.npz")
    states, timestamps = alignment["states"], alignment["timestamps_ns"]
    real = Robot(args.real_model, calibration["base_poses"])
    sim = Robot(args.sim_model)
    sim.set_state(states[0])
    mujoco.mj_forward(sim.model, sim.data)
    sim.setup_ik(args.sim_model, states[0])
    camera = sim.data.camera("top")
    sim_camera_in_world = pose(camera)
    sim_camera_in_world[:3, :3] @= np.diag([1., -1., -1.])
    real_camera_from_base = np.asarray(calibration["left_base_to_camera"])
    sim_world_from_camera = sim_camera_in_world
    real_camera_in_world = np.linalg.inv(real_camera_from_base)
    intrinsics = calibration["recorded_intrinsics"]
    width, height = intrinsics["width"], intrinsics["height"]
    fovy = sim.model.cam_fovy[sim.model.camera("top").id]
    rgb_map, render_map = projection_maps(intrinsics, fovy)
    real.model.cam_fovy[real.model.camera("top").id] = fovy
    total = min(len(states), args.max_frames or len(states))
    sample_ids = set(np.linspace(0, total - 1, 6, dtype=int))
    samples, output_states, targets, actuals, errors = [], [], [], [], []
    seed = states[0].copy()
    output = av.open(str(args.output / "overlay.mp4"), "w", options={"movflags": "+faststart"})
    fps = Fraction(1_000_000_000 / float(np.median(np.diff(timestamps)))).limit_denominator(1001)
    video = output.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    video.options = {"crf": "18", "preset": "fast"}
    with mujoco.Renderer(real.model, height, width) as real_renderer, \
            mujoco.Renderer(sim.model, height, width) as sim_renderer:
        real_renderer.enable_segmentation_rendering()
        sim_renderer.enable_segmentation_rendering()
        for index, (timestamp, frame) in enumerate(video_frames(args.mcap, calibration["camera_topic"])):
            if index >= total:
                break
            if timestamp != timestamps[index] or alignment["frame_indices"][index] != index:
                raise ValueError(f"Video/state alignment mismatch at frame {index}")
            state = states[index]
            real.set_state(state)
            target = np.stack([real_camera_from_base @ real.tcp(arm) for arm in range(2)])
            sim.set_state(seed)
            result, error = state.copy(), []
            for arm in range(2):
                segment = slice(7 * arm, 7 * arm + 6)
                q, e = sim.solve(arm, sim_world_from_camera @ target[arm], seed[segment])
                result[segment] = q
                error.append(e)
            sim.set_state(result)
            actual = np.stack([np.linalg.inv(sim_camera_in_world) @ sim.tcp(arm) for arm in range(2)])
            seed = result
            rgb = remap_rgb(frame.to_ndarray(format="rgb24"), rgb_map)
            mask = robot_mask(sim, sim_renderer, sim_camera_in_world, render_map)
            e = np.array(error)
            label = f"Sim Mink IK  f{index}   L {e[0, 0]:.2f}mm/{e[0, 1]:.2f}deg   R {e[1, 0]:.2f}mm/{e[1, 1]:.2f}deg"
            image = overlay(rgb, mask, (255, 150, 35), label)
            for packet in video.encode(av.VideoFrame.from_image(image)):
                output.mux(packet)
            if index in sample_ids:
                baseline_mask = robot_mask(real, real_renderer, real_camera_in_world, render_map)
                baseline = overlay(rgb, baseline_mask, (30, 225, 235), f"Real calibrated FK  f{index}")
                pair = Image.new("RGB", (2 * width, height))
                pair.paste(baseline, (0, 0))
                pair.paste(image, (width, 0))
                pair.save(args.output / f"frame_{index:06d}.jpg", quality=95)
                samples.append(pair)
            output_states.append(result)
            targets.append(target)
            actuals.append(actual)
            errors.append(error)
            if index % 200 == 0:
                print(f"frame {index}/{total}, max position error {e[:, 0].max():.3f} mm", flush=True)
    for packet in video.encode():
        output.mux(packet)
    output.close()
    errors = np.asarray(errors)
    output_states = np.asarray(output_states)
    np.savez_compressed(args.output / "retargeted.npz", timestamps_ns=timestamps[:len(errors)],
                        real_states=states[:len(errors)], sim_states=output_states,
                        target_camera_poses=targets, actual_camera_poses=actuals, errors_mm_deg=errors)
    sheet = Image.new("RGB", (2 * width, height * len(samples)))
    for index, image in enumerate(samples):
        sheet.paste(image, (0, index * height))
    sheet.save(args.output / "comparison.jpg", quality=95)
    report = {"mcap": str(args.mcap), "calibration": str(args.calibration),
              "real_model": str(args.real_model), "sim_model": str(args.sim_model),
              "ik_solver": "MinkArmController", "ik_settings": {
                  "dt": sim.controllers[0].dt, "iterations": sim.controllers[0].iterations,
                  "solver": sim.controllers[0].solver, "max_joint_delta": sim.controllers[0].max_joint_delta,
                  "other_parameters": "repository defaults"},
              "frame_count": len(errors), "state_time_offset_s": calibration["state_time_offset_s"],
              "eef_frame": "left_tcp_site / right_tcp_site", "intrinsics": intrinsics,
              "background": "RGB undistorted with the calibration's inverse-Brown mapping",
              "T_sim_camera_from_world": np.linalg.inv(sim_camera_in_world).tolist(),
              "T_real_camera_from_left_base": real_camera_from_base.tolist(),
              "real_base_poses": calibration["base_poses"], "arms": {}}
    for arm, side in enumerate(SIDES):
        joint_steps = np.abs(np.diff(output_states[:, arm * 7:arm * 7 + 6], axis=0)).max(axis=1)
        report["arms"][side] = {"mean_position_error_mm": float(errors[:, arm, 0].mean()),
                               "max_position_error_mm": float(errors[:, arm, 0].max()),
                               "mean_rotation_error_deg": float(errors[:, arm, 1].mean()),
                               "max_rotation_error_deg": float(errors[:, arm, 1].max()),
                               "frames_above_1mm_or_0_1deg": int(np.sum((errors[:, arm, 0] > 1)
                                                                          | (errors[:, arm, 1] > 0.1))),
                               "max_joint_step_rad": float(joint_steps.max(initial=0)),
                               "max_joint_step_frame": int(joint_steps.argmax() + 1) if len(joint_steps) else 0}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
