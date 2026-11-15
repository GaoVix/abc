"""Summarize recorded/commanded gripper aperture in converted ABC episodes.

Values are kept unchanged: normalized aperture, not millimeters.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abc_minimal.episode_io import GRIPPER_INDICES, load_episode


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("cache/dishrack_real"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for split in ("train_real", "val_real"):
        for episode in sorted((args.root / split).glob("episode_*")):
            metadata, states, actions = load_episode(episode)
            row = {"split": split, "episode": episode.name, "num_steps": len(states)}
            for kind, values in (("state", states), ("action", actions)):
                for side, index in zip(("left", "right"), GRIPPER_INDICES):
                    aperture = values[:, index]
                    if not np.isfinite(aperture).all():
                        raise ValueError(f"{episode}: nonfinite {side} {kind} aperture")
                    row[f"{side}_{kind}_min"] = float(aperture.min())
                    row[f"{side}_{kind}_max"] = float(aperture.max())
            rows.append(row)
    if not rows:
        parser.error(f"no converted episodes under {args.root}")

    columns = [key for key in rows[0] if key.endswith(("_min", "_max"))]
    summaries = []
    for split in ("train_real", "val_real", "all"):
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not selected:
            continue
        summary = {
            "split": split,
            "sequences": len(selected),
            "num_steps": sum(row["num_steps"] for row in selected),
        }
        for column in columns:
            reduce = min if column.endswith("_min") else max
            summary[column] = reduce(row[column] for row in selected)
        summaries.append(summary)

    output = args.output or args.root / "gripper_stats.csv"
    summary_output = output.with_name(output.stem + "_summary.csv")
    write_csv(output, rows)
    write_csv(summary_output, summaries)
    print(f"Sequence table: {output.resolve()}")
    print(f"Summary table: {summary_output.resolve()}")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
