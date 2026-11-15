import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.export_retargeted_joints import aligned_rows
from scripts.verify_camera_fk_ik import Robot, overlay, pose, projection_maps, remap_rgb, robot_mask, video_frames


def mark_tcp(image, points, intrinsics, color):
    intrinsic = np.asarray(intrinsics["K"]).reshape(3, 3)
    draw = ImageDraw.Draw(image)
    for target, actual in points:
        projected = np.stack((target, actual)) @ intrinsic.T
        projected = projected[:, :2] / projected[:, 2:3]
        x, y = projected[0]
        draw.line((x - 6, y, x + 6, y), fill="white", width=2)
        draw.line((x, y - 6, x, y + 6), fill="white", width=2)
        x, y = projected[1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.episode / "joint_conversion_report.json").read_text())
    metadata = json.loads((args.episode / "episode_metadata.json").read_text())
    calibration = report["calibration"]
    converted = np.fromfile(args.episode / "states_actions.bin", dtype=np.float64).reshape(-1, 28)
    original, _ = aligned_rows(Path(report["source_mcap"]), metadata, report["time_offset_s"])
    assert converted.shape == original.shape == (metadata["num_steps"], 28)
    assert np.array_equal(converted[:, [6, 13, 20, 27]], original[:, [6, 13, 20, 27]])
    total = min(len(converted), args.max_frames or len(converted))
    real = Robot(report["real_model"], calibration["base_poses"])
    sim = Robot(report["sim_model"])
    sim.set_state(converted[0, :14])
    mujoco.mj_forward(sim.model, sim.data)
    sim_world_from_camera = pose(sim.data.camera("top"))
    sim_world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
    sim_camera_from_world = np.linalg.inv(sim_world_from_camera)
    real_camera_from_world = np.asarray(calibration["left_base_to_camera"])
    real_world_from_camera = np.linalg.inv(real_camera_from_world)
    errors = np.empty((total, 2, 2, 2))
    tcp_points = np.empty((total, 2, 2, 2, 3))
    for index in range(total):
        for stream in range(2):
            segment = slice(stream * 14, (stream + 1) * 14)
            real.set_state(original[index, segment])
            sim.set_state(converted[index, segment])
            for arm in range(2):
                target = real_camera_from_world @ real.tcp(arm)
                actual = sim_camera_from_world @ sim.tcp(arm)
                tcp_points[index, stream, arm] = target[:3, 3], actual[:3, 3]
                errors[index, stream, arm, 0] = np.linalg.norm(actual[:3, 3] - target[:3, 3]) * 1000
                errors[index, stream, arm, 1] = np.rad2deg(Rotation.from_matrix(
                    actual[:3, :3] @ target[:3, :3].T
                ).magnitude())
    worst_position = int(errors[..., 0].reshape(total, -1).max(axis=1).argmax())
    worst_rotation = int(errors[..., 1].reshape(total, -1).max(axis=1).argmax())
    sample_ids = set(np.linspace(0, total - 1, 6, dtype=int)) | {worst_position, worst_rotation}
    intrinsics = calibration["recorded_intrinsics"]
    width, height = intrinsics["width"], intrinsics["height"]
    fovy = sim.model.cam_fovy[sim.model.camera("top").id]
    real.model.cam_fovy[real.model.camera("top").id] = fovy
    rgb_map, render_map = projection_maps(intrinsics, fovy)
    ticks = metadata["t0_ns"] + (np.arange(total, dtype=np.int64) + 1) * metadata["tick_ns"]
    frames = iter(video_frames(Path(report["source_mcap"]), calibration["camera_topic"]))
    current = next(frames)
    following = next(frames, None)
    selected_times = []
    samples = []
    output = av.open(str(args.output / "overlay.mp4"), "w", options={"movflags": "+faststart"})
    video = output.add_stream("libx264", rate=Fraction(30))
    video.width, video.height, video.pix_fmt = 3 * width, height, "yuv420p"
    video.options = {"crf": "20", "preset": "fast", "threads": "2"}
    with mujoco.Renderer(real.model, height, width) as real_renderer, \
            mujoco.Renderer(sim.model, height, width) as sim_renderer:
        real_renderer.enable_segmentation_rendering()
        sim_renderer.enable_segmentation_rendering()
        for index, tick in enumerate(ticks):
            while following is not None and following[0] <= tick:
                current, following = following, next(frames, None)
            selected_times.append(current[0])
            rgb = remap_rgb(current[1].to_ndarray(format="rgb24"), rgb_map)
            real.set_state(original[index, :14])
            mask = robot_mask(real, real_renderer, real_world_from_camera, render_map)
            baseline = overlay(rgb, mask, (30, 225, 235), f"Real calibrated state  f{index}")
            image = Image.new("RGB", (3 * width, height))
            image.paste(baseline, (0, 0))
            for stream, name, color in ((0, "STATE", (255, 150, 35)), (1, "ACTION", (215, 110, 255))):
                sim.set_state(converted[index, stream * 14:(stream + 1) * 14])
                mask = robot_mask(sim, sim_renderer, sim_world_from_camera, render_map)
                error = errors[index, stream]
                label = (f"Saved sim {name}  f{index}   L {error[0, 0]:.1f}mm/{error[0, 1]:.1f}deg"
                         f"   R {error[1, 0]:.1f}mm/{error[1, 1]:.1f}deg")
                panel = overlay(rgb, mask, color, label)
                mark_tcp(panel, tcp_points[index, stream], intrinsics, color)
                image.paste(panel, ((stream + 1) * width, 0))
            for packet in video.encode(av.VideoFrame.from_image(image)):
                output.mux(packet)
            if index in sample_ids:
                image.save(args.output / f"frame_{index:06d}.jpg", quality=95)
                thumbnail = image.copy()
                thumbnail.thumbnail((1440, 360))
                samples.append(thumbnail)
            if index % 300 == 0:
                print(f"{args.episode.name}: rendered {index}/{total}", flush=True)
    for packet in video.encode():
        output.mux(packet)
    output.close()
    frames.close()
    sheet = Image.new("RGB", (samples[0].width, sum(image.height for image in samples)))
    y = 0
    for image in samples:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(args.output / "comparison.jpg", quality=95)
    np.savez_compressed(args.output / "errors.npz", errors_mm_deg=errors,
                        tcp_points_camera=tcp_points, ticks_ns=ticks, selected_video_times_ns=selected_times)
    statistics = {}
    for stream, name in enumerate(("state", "action")):
        statistics[name] = {}
        for arm, side in enumerate(("left", "right")):
            values = errors[:, stream, arm]
            statistics[name][side] = {
                "mean_position_error_mm": float(values[:, 0].mean()),
                "max_position_error_mm": float(values[:, 0].max()),
                "mean_rotation_error_deg": float(values[:, 1].mean()),
                "max_rotation_error_deg": float(values[:, 1].max()),
                "frames_above_1mm_or_0_1deg": int(np.sum((values[:, 0] > 1) | (values[:, 1] > 0.1))),
            }
    result = {"episode": str(args.episode.resolve()), "frame_count": total,
              "source": "saved states_actions.bin; no IK recomputation",
              "real_model": report["real_model"], "time_offset_s": report["time_offset_s"],
              "grippers_exact": True, "worst_position_frame": worst_position,
              "worst_rotation_frame": worst_rotation, "sample_frames": sorted(map(int, sample_ids)),
              "panels": ["real calibrated state", "saved simulation state", "saved simulation action"],
              "intrinsics": intrinsics, "statistics": statistics}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["statistics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
