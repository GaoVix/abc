import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def link(source, destination):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name(
        "dishrack_joint_retarget_calibrations.json.gz"))
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    with gzip.open(args.manifest, "rt") as stream:
        records = [r for r in json.load(stream)["records"] if r["calibration_quality"] == "accepted"]
    counts = Counter()
    for record in records:
        relative = Path(f"{record['split']}_real") / record["episode"]
        link(source / relative, output / relative)
        counts[record["split"]] += 1
    for name in ("train_sim", "val_sim"):
        link(source / name, output / name)
    stats = source / "norm_stats.json"
    if not stats.exists():
        stats = source.parent / "norm_stats.json"
    if stats.exists():
        link(stats, output / "norm_stats.json")
    print(f"Linked {len(records)} episodes: {dict(counts)} -> {output}")


if __name__ == "__main__":
    main()
