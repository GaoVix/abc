"""One terminal progress bar shared by all worlds in a sim evaluation."""

from __future__ import annotations

from tqdm import tqdm


class EvalProgress:
    def __init__(self, num_worlds: int, num_chunks: int, *, detailed: bool = False):
        self.num_worlds = num_worlds
        self.num_chunks = num_chunks
        self.detailed = detailed
        self.completed = 0
        self.successes = 0
        self._bar: tqdm | None = None
        self._status: dict[str, str] = {}

    def start(self, first_world: int, batch_size: int = 1) -> None:
        # Start after the first inference warmup, and reuse this bar for every
        # subsequent world/batch. Its total counts completed worlds, including
        # those whose rollout finishes before the chunk budget.
        if self._bar is None:
            self._bar = tqdm(
                total=self.num_worlds, desc="Evaluating", unit="world", dynamic_ncols=True
            )
        last_world = first_world + batch_size
        world = str(first_world + 1) if batch_size == 1 else f"{first_world + 1}-{last_world}"
        self._status = {
            "world": f"{world}/{self.num_worlds}",
            "chunk": f"0/{self.num_chunks}",
            "success": f"{self.successes}/{self.completed}",
            "success_rate": f"{self.successes / self.completed:.1%}" if self.completed else "--",
        }
        self._bar.set_postfix(self._status)

    def chunk(
        self,
        chunk: int,
        *,
        reward: float | None = None,
        active: int | None = None,
        infer_s: float | None = None,
        steps_s: float | None = None,
        render_s: float | None = None,
        rtc_ready: bool | None = None,
    ) -> None:
        self._status["chunk"] = f"{chunk}/{self.num_chunks}"
        if reward is not None:
            self._status["reward"] = f"{reward:.2f}"
        if active is not None:
            self._status["active"] = str(active)
        if self.detailed:
            for name, seconds in (("infer", infer_s), ("steps", steps_s), ("render", render_s)):
                if seconds is not None:
                    self._status[name] = f"{seconds * 1000:.0f}ms"
            if rtc_ready is not None:
                self._status["rtc_ready"] = str(rtc_ready)
        if self._bar is not None:
            self._bar.set_postfix(self._status)

    def finish_world(self, *, success: bool, reward: float) -> None:
        self.completed += 1
        self.successes += int(success)
        self._status["success"] = f"{self.successes}/{self.completed}"
        self._status["success_rate"] = f"{self.successes / self.completed:.1%}"
        self._status["reward"] = f"{reward:.2f}"
        if self._bar is not None:
            self._bar.set_postfix(self._status, refresh=False)
            self._bar.update(1)

    def write(self, message: str) -> None:
        if self._bar is None:
            print(message, flush=True)
        else:
            self._bar.write(message)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
