"""abc_sim task adapter for the sim-eval rollout loop.

``SimTaskEnv`` presents the surface ``run_eval`` drives — ``reset``/``obs``/
``step_one``/``evaluate``/``close`` plus the ``*_vanilla`` variants — on top of
the vendored ``abc_sim`` catalogue, so any task there can be evaluated with the
same rollout loop as the native put-bottles scene. ``abc_sim`` always steps
physics in CPU MuJoCo and renders from that state, so the ``*_vanilla`` entry
points are aliases rather than a second physics path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

import abc_sim
from abc_sim.task_specs import SimTaskSpec


# Sim episodes are prompted with a "sim " prefix in the training mixture.
SIM_PROMPT_PREFIX = "sim "


def task_spec(task: str) -> SimTaskSpec:
    """Resolve a task name, alias, or prompt into its abc_sim spec."""
    return abc_sim.get_task_spec(task)


def task_prompt(task: str) -> str:
    """Default prompt for a task: the sim prefix plus the task's own prompt."""
    return SIM_PROMPT_PREFIX + task_spec(task).prompt


class SimTaskEnv:
    """One abc_sim world, stepped a single action at a time by the eval loop."""

    def __init__(
        self,
        *,
        task: str,
        height: int,
        width: int,
        camera_keys: tuple[str, ...],
        prompt: str,
        camera_backend: str = "mjwarp",
        gpu_id: int | None = None,
        disable_arm_control_limits: bool = False,
        disable_all_arm_limits: bool = False,
    ):
        self.spec = task_spec(task)
        self.height = height
        self.width = width
        self.render_width = 1024
        self.render_height = round(height * self.render_width / width)
        self.camera_keys = tuple(camera_keys)
        self.prompt = prompt
        self.randomization: Any = None
        if self.spec.evaluator_name is None:
            print(
                f"warning: sim task '{self.spec.name}' has no success evaluator; "
                "reward stays 0.0, success/ever_success stay False",
                flush=True,
            )
        # Pass the canonical spec name rather than spec.env_task: scene lookup
        # resolves through it either way, but scenes shared by several tasks
        # (blocks, drawer, count_into_opaque_box) only attach the right
        # evaluator and prompt when the env knows which task it is running.
        self.env = abc_sim.make_env(
            task=self.spec.name,
            prompt=prompt,
            render_cameras=True,
            camera_backend=camera_backend,
            camera_gpu_id=gpu_id,
            camera_height=self.render_height,
            camera_width=self.render_width,
            disable_arm_control_limits=disable_arm_control_limits,
            disable_all_arm_limits=disable_all_arm_limits,
        )
        print(
            f"camera rendering: {self.render_width}x{self.render_height} "
            f"-> Pillow resize (default) -> {width}x{height}", flush=True,
        )

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> dict[str, Any]:
        obs, info = self.env.reset(seed=seed, options=options, randomize=True)
        self.randomization = info.get("randomization")
        return self._policy_obs(obs)

    def forget_arm_state(self) -> None:
        self.env.forget_arm_state()

    def obs(self) -> dict[str, Any]:
        return self._policy_obs(self.env.get_obs())

    def step_one(self, action: np.ndarray) -> None:
        # render_obs=False: the eval loop renders on its own schedule, and the
        # 15 actions of a chunk are executed without touching the cameras.
        self.env.step(np.asarray(action, dtype=np.float32), render_obs=False)

    def evaluate(self) -> dict[str, Any]:
        result = self.env.evaluate_task()
        if result is None:
            return {"reward": 0.0, "success": False, "ever_success": False}
        return result.to_info(squeeze=True)

    def render_cameras(self) -> dict[str, np.ndarray]:
        return self._camera_images(self.env.get_obs()["images"])

    def close(self) -> None:
        self.env.close()

    # --vanilla-physics asks the native env for CPU physics with MJWarp
    # rendering; that is already how abc_sim runs, so these are the same paths.
    step_one_vanilla = step_one
    evaluate_vanilla = evaluate
    obs_vanilla_state = obs
    render_cameras_vanilla_state = render_cameras

    def _policy_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        # Drop the keys the policy does not read (masks, camera timestamps) so
        # the observation matches what the native env hands the rollout loop.
        # The prompt is the env's own: task randomizers rewrite it per episode
        # for the prompt-varying tasks (put_relative, count_into_opaque_box).
        return {
            "state": np.asarray(obs["state"], dtype=np.float32),
            "images": self._camera_images(obs["images"]),
            "prompt": obs["prompt"],
        }

    def _camera_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [name for name in self.camera_keys if name not in images]
        if missing:
            raise ValueError(
                f"Cameras missing from the {self.spec.name} scene: {', '.join(missing)} "
                f"(scene has {', '.join(images)})"
            )
        selected = {name: images[name] for name in self.camera_keys}

        for name, image in selected.items():
            resized = Image.fromarray(image.transpose(1, 2, 0)).resize(
                (self.width, self.height),
            )
            selected[name] = np.asarray(resized).transpose(2, 0, 1).copy()
        return selected
