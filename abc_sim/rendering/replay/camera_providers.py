"""Camera frame providers for replay viewers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from abc_sim.rendering.replay.renderer import RendererWrapper, WarpReplayRuntime


@dataclass
class CameraProvider:
    """Base interface for camera frame sources."""

    camera_names: tuple[str, ...]

    def initial_frames(self) -> dict[str, np.ndarray]:
        return {}

    def frames_for_step(self, step_idx: int, query_ts: float) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def reset(self) -> None:
        return

    def close(self) -> None:
        return


class NullCameraProvider(CameraProvider):
    """No-op provider used when camera output is disabled."""

    def __init__(self):
        super().__init__(camera_names=())

    def frames_for_step(self, step_idx: int, query_ts: float) -> dict[str, np.ndarray]:
        return {}


class RecordedCameraProvider(CameraProvider):
    """Sample-and-hold provider for recorded MP4 frames."""

    def __init__(self, camera_frames: dict[str, np.ndarray], camera_ts: dict[str, np.ndarray]):
        super().__init__(camera_names=tuple(camera_frames.keys()))
        self._camera_frames = camera_frames
        self._camera_ts = camera_ts

    def initial_frames(self) -> dict[str, np.ndarray]:
        return {name: frames[0] for name, frames in self._camera_frames.items() if len(frames) > 0}

    def frames_for_step(self, step_idx: int, query_ts: float) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for cam_name in self.camera_names:
            cam_frames = self._camera_frames.get(cam_name)
            ts_arr = self._camera_ts.get(cam_name)
            if cam_frames is None or ts_arr is None or len(cam_frames) == 0:
                continue
            idx = int(np.searchsorted(ts_arr, query_ts, side="right")) - 1
            idx = max(0, min(idx, len(cam_frames) - 1))
            frames[cam_name] = cam_frames[idx]
        return frames


class SimRenderCameraProvider(CameraProvider):
    """Render live simulated camera frames from the current MuJoCo state."""

    def __init__(
        self,
        model,
        data,
        *,
        backend: Literal["mujoco", "mjwarp", "madrona"],
        width: int,
        height: int,
        gpu_id: int | None = None,
        camera_names: tuple[str, ...] | None = None,
    ):
        if backend == "mujoco":
            raise ValueError("Use MujocoCameraProvider for backend='mujoco'")
        self._runtime = WarpReplayRuntime(model, data, gpu_id=gpu_id)
        self._renderer = RendererWrapper(
            backend=backend,
            runtime=self._runtime,
            cam_res=(width, height),
            gpu_id=gpu_id,
        )
        available_names = tuple(model.cam(i).name for i in range(model.ncam))
        self._camera_index = {name: idx for idx, name in enumerate(available_names)}
        if camera_names is None:
            camera_names = available_names
        super().__init__(camera_names=tuple(name for name in camera_names if name in available_names))
        self._last_frames: dict[str, np.ndarray] = {}
        self._needs_renderer_reset = True

    def initial_frames(self) -> dict[str, np.ndarray]:
        self._last_frames = self.frames_for_step(0, 0.0)
        return self._last_frames

    def frames_for_step(self, step_idx: int, query_ts: float) -> dict[str, np.ndarray]:
        self._runtime.sync_from_mujoco()
        self._runtime.forward()
        if self._needs_renderer_reset:
            img = self._renderer.reset_numpy(actual_batch=1)[0]
            self._needs_renderer_reset = False
        else:
            img = self._renderer.render_numpy(actual_batch=1)[0]
        frames = {cam_name: img[self._camera_index[cam_name]] for cam_name in self.camera_names}
        self._last_frames = {name: frames[name] for name in self.camera_names if name in frames}
        return self._last_frames

    def reset(self) -> None:
        self._runtime.reset_from_mujoco()
        self._runtime.refresh_geom_poses()
        self._needs_renderer_reset = True

    def close(self) -> None:
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "_runtime", None) is not None:
            self._runtime.close()
            self._runtime = None
        self._last_frames = {}


class MujocoCameraProvider(CameraProvider):
    """Render live simulated camera frames with MuJoCo's offscreen renderer."""

    def __init__(
        self,
        model,
        data,
        *,
        width: int,
        height: int,
        camera_names: tuple[str, ...] | None = None,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo sim camera rendering requires the `mujoco` package.") from exc

        self._data = data
        renderer_cls = getattr(mujoco, "Renderer", None)
        if renderer_cls is None:
            try:
                from mujoco.rendering.classic import gl_context as mj_gl_context
                from mujoco.rendering.classic import renderer as mj_renderer
            except ImportError as exc:
                raise RuntimeError(
                    "MuJoCo sim camera rendering requires a Renderer implementation in the `mujoco` package."
                ) from exc
            importlib.reload(mj_gl_context)
            importlib.reload(mj_renderer)
            renderer_cls = mj_renderer.Renderer

        self._renderer = renderer_cls(model, height=height, width=width)
        available_names = tuple(model.cam(i).name for i in range(model.ncam))
        if camera_names is None:
            camera_names = available_names
        super().__init__(camera_names=tuple(name for name in camera_names if name in available_names))

    def initial_frames(self) -> dict[str, np.ndarray]:
        return self.frames_for_step(0, 0.0)

    def frames_for_step(self, step_idx: int, query_ts: float) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for cam_name in self.camera_names:
            self._renderer.update_scene(self._data, camera=cam_name)
            frames[cam_name] = self._renderer.render().copy()
        return frames

    def close(self) -> None:
        self._renderer.close()


def create_camera_provider(
    *,
    source: Literal["recorded", "sim", "none"],
    sim_backend: Literal["mujoco", "mjwarp", "madrona"],
    episode_camera_frames: dict[str, np.ndarray],
    episode_camera_ts: dict[str, np.ndarray],
    env,
    width: int,
    height: int,
    gpu_id: int | None = None,
) -> CameraProvider:
    """Create the configured camera provider for the viewer."""
    if source == "none":
        return NullCameraProvider()
    if source == "recorded":
        return RecordedCameraProvider(episode_camera_frames, episode_camera_ts)
    if sim_backend == "mujoco":
        return MujocoCameraProvider(
            env.model,
            env.data,
            width=width,
            height=height,
            camera_names=tuple(env.camera_names),
        )
    return SimRenderCameraProvider(
        env.model,
        env.data,
        backend=sim_backend,
        width=width,
        height=height,
        gpu_id=gpu_id,
        camera_names=tuple(env.camera_names),
    )
