"""Convert release-format MCAP episodes into the training data layout.

Writes state/action arrays, combined camera video, and episode metadata per
episode. CLI entrypoint: scripts/export_mcap.py.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

TICK_NS = 33333333  # int(1e9 / 30)
FPS, OUT_W, OUT_H = 30, 224, 224
# Stereo episodes carry `/top-left-camera` + `/top-right-camera` (1920x1200
# each) AND a low-res `/top-camera` preview (640x400 with the left/right
# insets baked in). Mono episodes only have `/top-camera` (640x480, clean).
# Rule: when both stereo channels exist, deterministically pick one eye
# per episode (sha1(eid)[0] % 2) — matches production stereo_top_policy
# "random". Otherwise fall back to the mono `/top-camera`.
TOP_TOPIC_CANDIDATES = ("/top-left-camera", "/top-right-camera", "/top-camera")
CAMERAS = [("left", "/left-wrist-camera"), ("right", "/right-wrist-camera")]
STATE_TOPICS = [("/left-arm-state", 6), ("/left-ee-state", 1), ("/right-arm-state", 6), ("/right-ee-state", 1)]
ACTION_TOPICS = [("/left-arm-action", 6), ("/left-ee-action", 1), ("/right-arm-action", 6), ("/right-ee-action", 1)]
# Subtask segments: each `/subtask-annotation` message marks the start of a
# subtask (free text on `.data`); the segment runs until the next annotation.
# They live either inline in episode.mcap or in a sibling annotation.mcap that
# shares the same absolute-ns timebase.
ANNOTATION_TOPIC = "/subtask-annotation"
ANNOTATION_FILENAME = "annotation.mcap"
X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p"]
# Strict params for the final combined.mp4 the trainer reads. Mirrors
# production dataprocessing/image_ops.py:re_encode_mp4 so the trainer's
# synthesized custom_frame_mappings (pts=512*k, key_frame at k%30==0) is
# byte-for-byte true: timebase 1/15360, integer PTS at 512 ticks/frame,
# strict GOP=30 with no scenecut, no B-frames, faststart for fast random access.
TIMESCALE = 15360
TICKS_PER_FRAME = 512  # = TIMESCALE // FPS at 30fps
X264_STRICT_PARAMS = (
    f"keyint={FPS}:min-keyint={FPS}:scenecut=0:"
    f"fps={FPS}/1:timebase=1/{TIMESCALE}:force-cfr=1"
)
X264_STRICT_FFMPEG_ARGS = [
    "-vsync", "0",
    "-enc_time_base", f"1/{TIMESCALE}",
    "-video_track_timescale", str(TIMESCALE),
    "-bf", "0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-use_editlist", "0",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-x264-params", X264_STRICT_PARAMS,
    "-threads", "1",
]


@dataclass
class ExportMcapConfig:
    """Convert release-format MCAP episodes into the training data layout."""

    root: Annotated[Path, tyro.conf.Positional]
    out_dir: Annotated[Path, tyro.conf.Positional]
    workers: Annotated[int, tyro.conf.Positional] = 4


def floor_indices(source_ts, target_ts):
    """Index of the latest source message at or before each target tick."""
    return np.clip(np.searchsorted(source_ts, target_ts, side="right") - 1, 0, len(source_ts) - 1)


def probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", *entries, "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return [int(x) for x in out.split(",")]


def encode_aligned(h264_path, width, height, needed, out_path):
    """Decode raw h264, emit frame needed[i] at tick i (duplicating as required), re-encode."""
    frame_bytes = width * height * 3
    vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
          f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2")
    dec = subprocess.Popen(
        ["ffmpeg", "-i", h264_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-vsync", "0", "-vf", vf, *X264, "-threads", "1", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    src_idx, frame = -1, None
    try:
        for wanted in needed:
            while src_idx < wanted:
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                src_idx, frame = src_idx + 1, raw
            if frame is None:
                raise RuntimeError(f"decoder produced no frames (wanted index {wanted})")
            enc.stdin.write(frame)
    finally:
        dec.stdout.close(); dec.terminate(); dec.wait()
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def export_episode(job):
    mcap_path, task_name, out_root = job
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    cams, scalars = {}, {}
    annotations = []  # (log_time_ns, label) subtask-segment starts
    operator_id = None
    session = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        # Release episodes carry an "episode-metadata" record with "operator_id";
        # the internal delivery format used "session-metadata" with "operator-id".
        # Support both.
        for m in reader.iter_metadata():
            if m.name in ("episode-metadata", "session-metadata"):
                md = dict(m.metadata)
                operator_id = operator_id or md.get("operator_id") or md.get("operator-id")
                if "session-uuid" in md:
                    session[md["session-uuid"]] = None
                elif "session_id" in md:
                    session[md["session_id"]] = None
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        scalar_names = {t for t, _ in STATE_TOPICS + ACTION_TOPICS}
        cam_names = {t for _, t in CAMERAS} | set(TOP_TOPIC_CANDIDATES)
        for _, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic in cam_names:
                cams.setdefault(channel.topic, []).append((message.log_time, decoded.data))
            elif channel.topic in scalar_names:
                scalars.setdefault(channel.topic, []).append(
                    (message.log_time, np.array(decoded.position, dtype=np.float64)))
            elif channel.topic == ANNOTATION_TOPIC:
                annotations.append((message.log_time, decoded.data))
    for msgs in (*cams.values(), *scalars.values()):
        msgs.sort(key=lambda x: x[0])

    ep_id = f"episode_{next(iter(session))}" if session else Path(mcap_path).parent.name

    # Subtasks may live in a sibling annotation.mcap rather than inline.
    if not annotations:
        annotation_path = Path(mcap_path).with_name(ANNOTATION_FILENAME)
        if annotation_path != Path(mcap_path) and annotation_path.exists():
            with open(annotation_path, "rb") as f:
                ann_reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for _, channel, message, decoded in ann_reader.iter_decoded_messages():
                    if channel.topic == ANNOTATION_TOPIC:
                        annotations.append((message.log_time, decoded.data))
    # Resolve the top stream. Stereo episodes: deterministic per-episode pick
    # of one eye (matches production stereo_top_policy="random"). Mono
    # episodes: fall back to the single `/top-camera` stream.
    if "/top-left-camera" in cams and "/top-right-camera" in cams:
        import hashlib
        top_topic = ("/top-left-camera" if hashlib.sha1(ep_id.encode()).digest()[0] % 2 == 0
                     else "/top-right-camera")
    elif "/top-camera" in cams:
        top_topic = "/top-camera"
    else:
        print(f"[SKIP] {ep_id}: no top camera"); return None
    active_cams = [("top", top_topic)] + [(k, t) for k, t in CAMERAS if t in cams]
    streams = [cams[t] for _, t in active_cams] + [scalars[t] for t, _ in STATE_TOPICS + ACTION_TOPICS if t in scalars]
    if not active_cams or len(streams) == len(active_cams):
        print(f"[SKIP] {ep_id}: missing cameras or states"); return None
    t0 = max(s[0][0] for s in streams)
    t_end = min(s[-1][0] for s in streams)
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    num_steps = len(ticks)
    if num_steps < 10:
        print(f"[SKIP] {ep_id}: too short ({num_steps} steps)"); return None

    out_dir = Path(out_root) / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for topic, dim in STATE_TOPICS + ACTION_TOPICS:
        msgs = scalars.get(topic)
        if not msgs:
            parts.append(np.zeros((num_steps, dim))); continue
        ts = np.array([t for t, _ in msgs], dtype=np.int64)
        parts.append(np.stack([v for _, v in msgs])[floor_indices(ts, ticks)])
    sa = np.concatenate(parts, axis=-1)
    sa.tofile(out_dir / "states_actions.bin")

    # Per-camera mp4s are intermediates that get vstacked into combined.mp4.
    # Keep them in a tempdir so the output dir only carries the trainer-needed files.
    with tempfile.TemporaryDirectory() as work:
        mp4s = []
        for cam_key, topic in active_cams:
            msgs = cams[topic]
            cam_ts = np.array([t for t, _ in msgs], dtype=np.int64)
            with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as tmp:
                tmp.write(b"".join(data for _, data in msgs))
            try:
                width, height = probe(tmp.name, "-show_entries", "stream=width,height")
                (n_frames,) = probe(tmp.name, "-count_frames", "-show_entries", "stream=nb_read_frames")
                if n_frames > 0 and n_frames != len(cam_ts):  # h264 chunks != frames; respace timestamps
                    cam_ts = np.linspace(cam_ts[0], cam_ts[-1], n_frames, dtype=np.int64)
                mp4 = str(Path(work) / f"{cam_key}.mp4")
                needed = floor_indices(cam_ts, ticks)
                if cam_key == "top":
                    top_camera_timestamps = cam_ts[needed].tolist()
                encode_aligned(tmp.name, width, height, needed, mp4)
                mp4s.append(mp4)
            finally:
                os.unlink(tmp.name)

        combined = str(out_dir / "combined_camera-images-rgb.mp4")
        # vstack inputs, then force the timebase + integer PTS the trainer expects.
        # settb/setpts make every frame's pts exactly TICKS_PER_FRAME * k in
        # timebase 1/TIMESCALE (matches the synthesized custom_frame_mappings).
        filt = (
            "".join(f"[{i}:v]" for i in range(len(mp4s)))
            + f"vstack=inputs={len(mp4s)}[v0];"
            + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", *sum((["-i", p] for p in mp4s), []),
             "-filter_complex", filt, "-map", "[out]",
             *X264_STRICT_FFMPEG_ARGS, combined],
            capture_output=True, check=True,
        )
        for mp4 in mp4s + [combined]:
            (n,) = probe(mp4, "-count_frames", "-show_entries", "stream=nb_read_frames")
            if n != num_steps:
                raise RuntimeError(
                    f"{ep_id}: {Path(mp4).name} has {n} frames, expected {num_steps}"
                )

    meta = {"task_name": task_name, "cameras": [k for k, _ in active_cams],
            "camera_resolutions": {k: [OUT_W, OUT_H] for k, _ in active_cams},
            "alignment": "fixed_clock_30hz_causal", "t0_ns": int(t0), "tick_ns": TICK_NS,
            "num_steps": num_steps, "top_camera_timestamps_ns": top_camera_timestamps}
    # operator_id is optional conditioning/filtering signal (see the dataloader's
    # operator prompting); persist it in the metadata when the MCAP carried one.
    if operator_id:
        meta["operator_id"] = operator_id
    (out_dir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))

    # Sidecars for policy conditioning (both optional; absent => task-prompt only):
    #   subtasks.json  {"<frame_idx>": "<label>"}  — the per-frame subtask label,
    #     the exact schema the dataloader's build_prompt() reads. Each annotation
    #     starts a segment that runs until the next one (or episode end).
    #   operator.json  {"operator_id": "<uuid>"}   — convenience mirror of the id
    #     also stored in episode_metadata.json.
    n_annotated = 0
    if annotations:
        frame_labels = _frames_to_subtasks(annotations, ticks)
        n_annotated = len(frame_labels)
        (out_dir / "subtasks.json").write_text(json.dumps(frame_labels))
    if operator_id:
        (out_dir / "operator.json").write_text(json.dumps({"operator_id": operator_id}))

    extra = []
    if operator_id:
        extra.append(f"operator={operator_id[:8]}")
    if n_annotated:
        extra.append(f"subtask-frames={n_annotated}")
    print(f"[OK] {ep_id}: {num_steps} steps, cams={[k for k, _ in active_cams]}"
          + (f", {', '.join(extra)}" if extra else ""))
    return ep_id


def _frames_to_subtasks(annotations, ticks):
    """Map (log_time_ns, label) segment-starts to a {frame_idx: label} dict.

    Each annotation marks the start of a subtask segment that runs until the
    next annotation; every frame whose tick falls in a segment gets that
    segment's label. Frames before the first annotation are left unlabeled.
    """
    anns = sorted(annotations, key=lambda x: x[0])
    ann_ts = np.array([t for t, _ in anns], dtype=np.int64)
    # For each frame tick, the active segment is the latest annotation at or
    # before it; searchsorted(side="right") - 1 gives that segment index (-1 =
    # before the first annotation, left unlabeled).
    seg = np.searchsorted(ann_ts, ticks, side="right") - 1
    out = {}
    for frame_idx, s in enumerate(seg):
        if s >= 0:
            label = anns[int(s)][1]
            if label:
                out[str(frame_idx)] = label
    return out


def main(config: ExportMcapConfig):
    jobs = sorted(
        (str(p), p.parent.parent.name, str(config.out_dir))
        for p in config.root.glob("*/episode_*/episode.mcap")
    )
    print(f"{len(jobs)} episodes")
    with Pool(config.workers) as pool:
        done = [r for r in pool.map(export_episode, jobs) if r]
    print(f"exported {len(done)}/{len(jobs)}")


if __name__ == "__main__":
    main(tyro.cli(ExportMcapConfig))
