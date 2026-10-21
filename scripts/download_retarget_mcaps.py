import argparse
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.export_hf_task import Config, download, list_files, token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with gzip.open(args.manifest, "rt", encoding="utf-8") as stream:
        records = [r for r in json.load(stream)["records"] if r["calibration_quality"] == "accepted"]
    jobs = []
    for task, split in sorted({(r["task"], r["split"]) for r in records}):
        cfg = Config(task=task)
        auth = token(cfg)
        selected = [r for r in records if r["task"] == task and r["split"] == split]
        remote = {item["path"]: item for item in list_files(cfg, split, auth, "/episode.mcap")}
        files = [remote[r["source_relative"]] for r in selected]
        directories = {str(Path(r["source_relative"]).parent) for r in selected}
        files += [item for item in list_files(cfg, split, auth, "/annotation.mcap")
                  if str(Path(item["path"]).parent) in directories]
        jobs.extend((cfg, auth, item, args.output / "data" / split) for item in files)
    size = sum(item["size"] for _, _, item, _ in jobs)
    print(f"{len(records)} episodes, {len(jobs)} files, {size / 2**30:.2f} GiB -> {args.output}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"[{index}/{len(futures)}] downloaded", flush=True)


if __name__ == "__main__":
    main()
