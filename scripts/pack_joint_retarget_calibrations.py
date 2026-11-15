import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    records = []
    for record in source["records"]:
        packed = {key: record[key] for key in (
            "episode", "source_relative", "split", "task", "calibration_quality"
        )}
        if record["calibration_quality"] == "accepted":
            calibration = json.loads((Path(record["calibration_path"]) / "camera_calibration.json").read_text())
            packed["model_name"] = Path(calibration["model_path"]).name
            packed["calibration"] = {key: calibration[key] for key in (
                "left_base_to_camera", "base_poses", "state_time_offset_s", "camera_topic", "recorded_intrinsics"
            )}
        records.append(packed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump({"counts": source["counts"], "records": records}, stream,
                  separators=(",", ":"), allow_nan=False)
    print(f"{args.output}: {source['counts']['accepted']} calibrations, {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
