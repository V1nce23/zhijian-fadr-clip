#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "real-model-benchmark"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SEGMENT_CONNECTIONS = int(os.environ.get("DOWNLOAD_CONNECTIONS", "4"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_command() -> list[str]:
    return [
        "curl",
        "-4",
        "-fL",
        "--retry",
        "30",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
    ]


def download_segment(url: str, destination: Path, start: int, end: int) -> None:
    expected_size = end - start + 1
    if destination.exists() and destination.stat().st_size == expected_size:
        return
    destination.unlink(missing_ok=True)
    completed = subprocess.run(
        curl_command()
        + [
            "--range",
            f"{start}-{end}",
            "--output",
            str(destination),
            "--write-out",
            "%{http_code}",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != "206":
        raise ValueError(f"server did not honor byte range {start}-{end}: {completed.stdout}")
    if destination.stat().st_size != expected_size:
        raise ValueError(f"incomplete byte range {start}-{end}")


def segmented_download(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    prefix_size = partial.stat().st_size if partial.exists() else 0
    if prefix_size > expected_size:
        partial.unlink()
        prefix_size = 0
    if prefix_size == expected_size:
        partial.replace(destination)
        return

    remaining = expected_size - prefix_size
    segment_size = max((remaining + SEGMENT_CONNECTIONS - 1) // SEGMENT_CONNECTIONS, 1)
    ranges = []
    start = prefix_size
    while start < expected_size:
        end = min(start + segment_size - 1, expected_size - 1)
        segment = destination.with_suffix(destination.suffix + f".segment-{start}-{end}")
        ranges.append((segment, start, end))
        start = end + 1
    with ThreadPoolExecutor(max_workers=SEGMENT_CONNECTIONS) as executor:
        list(executor.map(lambda item: download_segment(url, *item), ranges))

    assembled = destination.with_suffix(destination.suffix + ".assembled")
    with assembled.open("wb") as output:
        if partial.exists():
            with partial.open("rb") as prefix:
                shutil.copyfileobj(prefix, output, length=1024 * 1024)
        for segment, _, _ in ranges:
            with segment.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if assembled.stat().st_size != expected_size:
        raise ValueError(f"assembled size mismatch for {destination}")
    assembled.replace(destination)
    partial.unlink(missing_ok=True)
    for segment, _, _ in ranges:
        segment.unlink(missing_ok=True)


def curl(url: str, destination: Path | None = None, expected_size: int | None = None) -> bytes:
    command = curl_command()
    if destination is None:
        return subprocess.run(command + [url], check=True, capture_output=True).stdout
    if expected_size and expected_size >= 1024 * 1024 and SEGMENT_CONNECTIONS > 1:
        segmented_download(url, destination, expected_size)
        return b""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(command + ["--continue-at", "-", "--output", str(partial), url], check=True)
    partial.replace(destination)
    return b""


def verify(path: Path, expected_hash: str | None, expected_size: int | None = None) -> dict:
    actual_hash = sha256(path)
    actual_size = path.stat().st_size
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}")
    if expected_size and actual_size != expected_size:
        raise ValueError(f"size mismatch for {path}: {actual_size} != {expected_size}")
    return {"sha256": actual_hash, "size": actual_size}


def matches_expected(
    path: Path,
    expected_hash: str | None,
    expected_size: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        verify(path, expected_hash, expected_size)
    except ValueError:
        return False
    return True


def download_model(model: dict) -> list[dict]:
    inventory = []
    destination_root = REPOSITORY_ROOT / "models" / model["destination"]
    for item in model["files"]:
        destination = destination_root / item["path"]
        if not matches_expected(destination, item.get("sha256"), item.get("size")):
            if model["kind"] == "local_fadr_clip":
                raise FileNotFoundError("FADR-CLIP 最佳权重未准备好，请先运行 tools/prepare_fadr_clip.py；不会下载其他检测模型")
            elif model["kind"] == "github_asset":
                url = (
                    f"https://github.com/{model['repository']}/releases/download/"
                    f"{model['release']}/{quote(item['path'], safe='')}"
                )
            elif model["kind"] == "huggingface":
                encoded_path = quote(item["path"], safe="/")
                url = (
                    f"{HF_ENDPOINT}/{model['repository']}/resolve/"
                    f"{model['revision']}/{encoded_path}?download=true"
                )
            else:
                raise ValueError(f"unsupported model kind: {model['kind']}")
            print(f"downloading model {model['destination']}/{item['path']}", flush=True)
            curl(url, destination, item.get("size"))
        facts = verify(destination, item.get("sha256"), item.get("size"))
        inventory.append(
            {
                "kind": model["kind"],
                "repository": model.get("repository", "local/FADR_CLIP"),
                "release": model.get("release"),
                "revision": model.get("revision"),
                "path": str(destination.relative_to(REPOSITORY_ROOT)),
                **facts,
            }
        )
    return inventory


def download_models(output: Path) -> list[dict]:
    del output
    manifest = json.loads((HERE / "models.json").read_text())
    with ThreadPoolExecutor(max_workers=3) as executor:
        groups = list(executor.map(download_model, manifest["models"]))
    return [item for group in groups for item in group]


def download_samples(output: Path) -> list[dict]:
    manifest = json.loads((HERE / "manifest.json").read_text())
    inventory = []
    for sample in manifest["samples"]:
        source = manifest["sources"][sample["media_type"]]
        suffix = Path(sample["path"]).suffix
        destination = output / "samples" / sample["media_type"] / sample["label"] / f"{sample['id']}{suffix}"
        if not matches_expected(destination, sample.get("sha256"), sample.get("size")):
            encoded_path = quote(sample["path"], safe="/")
            url = (
                f"{HF_ENDPOINT}/datasets/{source['repository']}/resolve/"
                f"{source['revision']}/{encoded_path}?download=true"
            )
            print(f"downloading sample {sample['id']}", flush=True)
            curl(url, destination)
        facts = verify(destination, sample.get("sha256"), sample.get("size"))
        inventory.append(
            {
                "id": sample["id"],
                "media_type": sample["media_type"],
                "label": sample["label"],
                "repository": source["repository"],
                "source_path": sample["path"],
                "pair_id": sample.get("pair_id"),
                "row_index": sample.get("row_index"),
                "path": str(destination.relative_to(REPOSITORY_ROOT)),
                **facts,
            }
        )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--samples-only", action="store_true")
    args = parser.parse_args()
    if args.models_only and args.samples_only:
        parser.error("--models-only and --samples-only are mutually exclusive")

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.samples_only:
        model_inventory = download_models(args.output)
        (args.output / "model-inventory.json").write_text(
            json.dumps(model_inventory, ensure_ascii=False, indent=2) + "\n"
        )
    if not args.models_only:
        sample_inventory = download_samples(args.output)
        (args.output / "sample-inventory.json").write_text(
            json.dumps(sample_inventory, ensure_ascii=False, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
