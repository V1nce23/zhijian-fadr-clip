#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "services" / "api"))

from app.core.config import settings  # noqa: E402
from app.detectors.base import DetectionResult  # noqa: E402
from app.detectors.catalog import VIDEO_FRAME_COUNT, detector_specs  # noqa: E402
from app.detectors.registry import get_detector  # noqa: E402
from app.services.inference import InferenceControl  # noqa: E402
from app.services.media import prepare_detection_media  # noqa: E402
from app.services.results import aggregate_results  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    import torch
    import transformers

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def summarize(results: list[dict]) -> dict:
    successful = [result for result in results if result["status"] == "succeeded"]
    correct = [result for result in successful if result["binary_correct"]]
    uncertain = [result for result in successful if result["verdict"] == "uncertain"]
    labels = ("real", "fake")
    matrix = {
        expected: {
            predicted: sum(
                result["label"] == expected and result["binary_prediction"] == predicted
                for result in successful
            )
            for predicted in labels
        }
        for expected in labels
    }
    durations = [result["elapsed_seconds"] for result in successful]
    return {
        "total": len(results),
        "succeeded": len(successful),
        "failed": len(results) - len(successful),
        "binary_accuracy_at_0_5": len(correct) / len(successful)
        if successful
        else None,
        "uncertain_count": len(uncertain),
        "confusion_matrix": matrix,
        "mean_inference_seconds": statistics.mean(durations) if durations else None,
        "median_inference_seconds": statistics.median(durations) if durations else None,
    }


def platform_results(results: list[dict], inventory: list[dict]) -> list[dict]:
    reports = []
    for sample in inventory:
        runs = [row for row in results if row["id"] == sample["id"]]
        successful = [row for row in runs if row["status"] == "succeeded"]
        if not successful:
            reports.append({**sample, "status": "failed"})
            continue
        summary = aggregate_results(
            [
                DetectionResult(row["score"], row["verdict"], row["evidence"])
                for row in successful
            ]
        )
        reports.append(
            {
                **sample,
                "status": "completed"
                if len(successful) == len(runs)
                else "partially_completed",
                "score": summary.score,
                "verdict": summary.verdict.value,
                "summary": summary.summary,
                "succeeded_checkers": len(successful),
                "failed_checkers": len(runs) - len(successful),
            }
        )
    return reports


def summarize_platform(reports: list[dict]) -> dict:
    successful = [row for row in reports if row["status"] != "failed"]
    decisive = [row for row in successful if row["verdict"] != "uncertain"]
    correct = sum(
        row["verdict"]
        == ("likely_manipulated" if row["label"] == "fake" else "likely_authentic")
        for row in decisive
    )
    return {
        "total": len(reports),
        "succeeded": len(successful),
        "partially_completed": sum(
            row["status"] == "partially_completed" for row in reports
        ),
        "failed": len(reports) - len(successful),
        "uncertain_count": len(successful) - len(decisive),
        "decisive_count": len(decisive),
        "decisive_coverage": len(decisive) / len(reports) if reports else None,
        "decisive_accuracy": correct / len(decisive) if decisive else None,
    }


def evaluate(inventory: list[dict], models: Path, device: str) -> list[dict]:
    results: list[dict] = []
    for sample in inventory:
        path = REPOSITORY_ROOT / sample["path"]
        if sha256(path) != sample["sha256"]:
            raise ValueError(f"sample changed after download: {path}")
        media_type = sample["media_type"]
        with tempfile.TemporaryDirectory(prefix="deepfake30-") as temporary_directory:
            preprocessing_error = None
            detection_source = None
            try:
                detection_source = prepare_detection_media(
                    path, media_type, Path(temporary_directory)
                )
            except Exception as exc:
                preprocessing_error = exc
            # Match the worker's per-sample model order and lifetime. All models
            # read original media without lossy normalization.
            for spec in detector_specs(media_type):
                adapter = None
                started = time.perf_counter()
                row = {
                    **sample,
                    "detector_id": spec.detector_id,
                    "model_version": spec.model_version,
                    "preprocessing_version": spec.preprocessing_version,
                }
                try:
                    if preprocessing_error:
                        raise preprocessing_error
                    control = InferenceControl(settings.detector_timeout_seconds)
                    adapter = get_detector(
                        media_type,
                        spec,
                        str(models),
                        device,
                        True,
                        settings.detector_authentic_threshold,
                        settings.detector_manipulated_threshold,
                    )
                    result = adapter.predict(
                        detection_source, sha256=sample["sha256"], check=control.check
                    )
                    control.check()
                    prediction = "fake" if result.score >= 0.5 else "real"
                    row.update(
                        {
                            "status": "succeeded",
                            "score": result.score,
                            "verdict": result.verdict,
                            "evidence": result.evidence,
                            "binary_prediction": prediction,
                            "binary_correct": prediction == sample["label"],
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                finally:
                    del adapter
                row["elapsed_seconds"] = time.perf_counter() - started
                results.append(row)
                print(
                    f"{spec.detector_id} {sample['id']}: {row['status']} {row.get('score', row.get('error', ''))}",
                    flush=True,
                )
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "real-model-benchmark",
    )
    parser.add_argument("--models", type=Path, default=REPOSITORY_ROOT / "models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inventory = [sample for sample in json.loads((args.artifacts / "sample-inventory.json").read_text())
                 if sample["media_type"] in ("image", "video")]
    results = evaluate(inventory, args.models, args.device)
    reports = platform_results(results, inventory)
    report = {
        "schema_version": 2,
        "environment": environment(),
        "pipeline": {
            "media_normalization": "app.services.media.prepare_detection_media",
            "video_frame_count": VIDEO_FRAME_COUNT,
            "authentic_threshold": settings.detector_authentic_threshold,
            "manipulated_threshold": settings.detector_manipulated_threshold,
            "platform_aggregation": "app.services.results.aggregate_results",
            "timing": "prediction including model load; excludes shared media conversion",
        },
        "single_model_summary": summarize(results),
        "single_model_by_modality": {
            media: summarize([row for row in results if row["media_type"] == media])
            for media in ("image", "video")
        },
        "platform_summary": summarize_platform(reports),
        "platform_by_modality": {
            media: summarize_platform(
                [row for row in reports if row["media_type"] == media]
            )
            for media in ("image", "video")
        },
        "results": results,
        "platform_results": reports,
    }
    output = args.output or args.artifacts / "results-current.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output}")
    return 0 if report["single_model_summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
