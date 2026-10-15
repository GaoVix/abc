# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support", "tyro"]
# ///
"""Export one ABC-130k Hugging Face task to the training format."""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro

from prepare import download_url


HF, REPO = "https://huggingface.co", "XDOF/ABC-130k"
CACHE, UA = Path(os.environ.get("ABC_CACHE", "/tmp/abc_minimal_cache")), "abc-minimal-hf-task/1.0"


@dataclass
class Config:
    """Download all MCAPs for one HF task, then convert via abc_minimal.export_mcap."""

    task: Annotated[str, tyro.conf.arg(help="HF task folder, e.g. organize_the_condiment_bottles.")]
    split: Annotated[Literal["train", "val", "all"], tyro.conf.arg(help="Split to download.")] = "all"
    repo_id: Annotated[str, tyro.conf.arg(help="HF dataset repo id.")] = REPO
    revision: Annotated[str, tyro.conf.arg(help="HF revision.")] = "main"
    hf_token: Annotated[
        str | None,
        tyro.conf.arg(help="HF token; otherwise use HF_TOKEN or HUGGING_FACE_HUB_TOKEN."),
    ] = None
    cache: Annotated[Path, tyro.conf.arg(help="Cache root.")] = CACHE
    workers: Annotated[int, tyro.conf.arg(help="Conversion worker processes.")] = 4
    max_episodes: Annotated[int | None, tyro.conf.arg(help="Optional per-split cap for smoke tests.")] = None
    dry_run: Annotated[bool, tyro.conf.arg(help="List only; do not download or convert.")] = False
    keep_mcaps: Annotated[bool, tyro.conf.arg(help="Keep staged raw MCAPs after conversion.")] = False


def token(cfg: Config) -> str | None:
    vals = [cfg.hf_token, os.getenv("HF_TOKEN"), os.getenv("HUGGING_FACE_HUB_TOKEN")]
    return next((v.strip() for v in vals if v and v.strip()), None)


def headers(tok: str | None) -> dict[str, str]:
    out = {"User-Agent": UA}
    if tok:
        out["Authorization"] = f"Bearer {tok}"
    return out


def quote(path: str) -> str:
    return urllib.parse.quote(path.strip("/"), safe="/")


def next_page(link: str | None) -> str | None:
    if not link:
        return None
    for part in link.split(","):
        if 'rel="next"' in part:
            return part[part.find("<") + 1 : part.find(">")]
    return None


def open_hf(url: str, tok: str | None):
    req = urllib.request.Request(url, headers=headers(tok))
    try:
        return urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                "HF access denied. Accept XDOF/ABC-130k access and set HF_TOKEN "
                "or pass --hf-token."
            ) from e
        if e.code == 404:
            raise RuntimeError(f"HF path not found: {url}") from e
        raise


def read_json(url: str, tok: str | None):
    with open_hf(url, tok) as r:
        return json.loads(r.read()), r.headers.get("Link")


def list_files(cfg: Config, split: str, tok: str | None, suffix: str) -> list[dict]:
    """List every file under the task/split whose path ends with ``suffix``."""
    path = f"data/{split}/{cfg.task}"
    url = f"{HF}/api/datasets/{cfg.repo_id}/tree/{cfg.revision}/{quote(path)}?recursive=1&expand=1"
    out = []
    while url:
        payload, link = read_json(url, tok)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(payload["error"])
        out += [
            {"path": e["path"], "size": int(e.get("size") or 0)}
            for e in payload
            if e.get("type") == "file" and e.get("path", "").endswith(suffix)
        ]
        url = next_page(link)
    return sorted(out, key=lambda e: e["path"])


def list_mcaps(cfg: Config, split: str, tok: str | None) -> list[dict]:
    out = list_files(cfg, split, tok, "/episode.mcap")
    return out[: cfg.max_episodes] if cfg.max_episodes is not None else out


def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def split_root(cfg: Config, split: str) -> Path:
    return cfg.cache.expanduser() / "hf_tasks" / cfg.task / split


def local_path(root: Path, item: dict) -> Path:
    # Preserve the source filename so annotation.mcap lands beside its
    # episode.mcap (the converter reads subtasks from that sibling file).
    p = Path(item["path"]).parts
    return root / p[2] / p[3] / p[-1]


def download(cfg: Config, tok: str | None, item: dict, root: Path) -> None:
    episode = Path(item["path"]).parts[3]
    url = f"{HF}/datasets/{cfg.repo_id}/resolve/{cfg.revision}/{quote(item['path'])}"
    download_url(
        url,
        local_path(root, item),
        expected=item["size"] or None,
        require_size=False,
        headers=headers(tok),
        label=episode,
    )


def write_manifest(cfg: Config, split: str, files: list[dict],
                   annotations: list[dict], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "repo_id": cfg.repo_id,
        "revision": cfg.revision,
        "task": cfg.task,
        "split": split,
        "count": len(files),
        "bytes": sum(f["size"] for f in files),
        "annotation_count": len(annotations),
        "staged_root": str(root),
        "out_dir": str(cfg.cache / f"{split}_real"),
        "files": files,
        "annotations": annotations,
    }
    (root / "manifest.json").write_text(json.dumps(data, indent=2))
    print(f"[manifest] {root / 'manifest.json'}")


def convert(cfg: Config, split: str, root: Path) -> None:
    from abc_minimal.export_mcap import ExportMcapConfig, main as export_mcap_main

    out_dir = (cfg.cache / f"{split}_real").expanduser()
    print(f"[convert] export_mcap {root} -> {out_dir} ({cfg.workers} workers)")
    export_mcap_main(ExportMcapConfig(root=root, out_dir=out_dir, workers=cfg.workers))


def run_split(cfg: Config, split: str, tok: str | None) -> None:
    files, root = list_mcaps(cfg, split, tok), split_root(cfg, split)
    # Optional per-episode subtask annotations live in a sibling annotation.mcap;
    # fetch the ones belonging to the episodes we're keeping so the converter can
    # emit subtasks.json. Episodes without annotations simply have none.
    episode_dirs = {Path(f["path"]).parts[3] for f in files}
    annotations = [
        a for a in list_files(cfg, split, tok, "/annotation.mcap")
        if Path(a["path"]).parts[3] in episode_dirs
    ]
    print(f"[{split}] {len(files)} episodes ({fmt(sum(f['size'] for f in files))}), "
          f"{len(annotations)} with subtask annotations")
    write_manifest(cfg, split, files, annotations, root)
    if cfg.dry_run:
        return
    for item in files + annotations:
        download(cfg, tok, item, root)
    convert(cfg, split, root)
    if not cfg.keep_mcaps:
        shutil.rmtree(root)


def main(cfg: Config) -> None:
    tok = token(cfg)
    for split in (("train", "val") if cfg.split == "all" else (cfg.split,)):
        run_split(cfg, split, tok)


if __name__ == "__main__":
    main(tyro.cli(Config))
