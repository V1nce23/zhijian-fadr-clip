import hashlib
import importlib.util
from pathlib import Path

import pytest

from app.detectors.base import DetectionResult, DetectorSpec


spec = importlib.util.spec_from_file_location(
    "deepfake30_runner",
    Path(__file__).resolve().parents[3] / "benchmarks/deepfake30/run.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_prepares_once_and_reuses_production_input(tmp_path, monkeypatch):
    source = tmp_path / "clip.webm"
    source.write_bytes(b"original video")
    inventory = [
        {
            "id": "video-1",
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "media_type": "video",
            "label": "fake",
        }
    ]
    normalized_calls = []
    predictions = []

    def normalize(path, media, directory):
        normalized_calls.append((path, media))
        return path

    class Detector:
        def predict(self, path, *, sha256, check):
            check()
            assert path.is_file()
            assert path == source
            assert path.read_bytes() == b"original video"
            predictions.append(path)
            return DetectionResult(0.8, "likely_manipulated", {})

    monkeypatch.setattr(benchmark, "prepare_detection_media", normalize)
    monkeypatch.setattr(benchmark, "get_detector", lambda *args: Detector())
    monkeypatch.setattr(
        benchmark,
        "detector_specs",
        lambda media: (
            [DetectorSpec(f"model-{n}", "1", "v2-32frames", media) for n in range(3)]
            if media == "video"
            else []
        ),
    )
    results = benchmark.evaluate(inventory, tmp_path, "cpu")
    assert len(normalized_calls) == 1
    assert len(predictions) == 3
    assert len(set(predictions)) == 1
    assert all(row["status"] == "succeeded" for row in results)
    assert all(row["preprocessing_version"] == "v2-32frames" for row in results)
    assert predictions[0].exists()


def test_platform_metrics_follow_votes_and_keep_uncertainty():
    inventory = [
        {"id": "a", "label": "real"},
        {"id": "b", "label": "fake"},
        {"id": "c", "label": "real"},
    ]
    rows = [
        {
            "id": "a",
            "status": "succeeded",
            "score": score,
            "verdict": verdict,
            "evidence": {},
        }
        for score, verdict in [
            (0.3, "likely_authentic"),
            (0.3, "likely_authentic"),
            (1.0, "likely_manipulated"),
        ]
    ]
    rows += [
        {
            "id": "b",
            "status": "succeeded",
            "score": score,
            "verdict": verdict,
            "evidence": {},
        }
        for score, verdict in [(0.1, "likely_authentic"), (0.9, "likely_manipulated")]
    ]
    rows += [{"id": "c", "status": "failed"}]
    reports = benchmark.platform_results(rows, inventory)
    # Mean score > 0.5 must not override the production vote result.
    assert reports[0]["score"] > 0.5
    assert reports[0]["verdict"] == "likely_authentic"
    assert reports[1]["verdict"] == "uncertain"
    assert reports[2]["status"] == "failed"
    metrics = benchmark.summarize_platform(reports)
    assert metrics["decisive_count"] == 1
    assert metrics["uncertain_count"] == 1
    assert metrics["failed"] == 1
    assert metrics["decisive_accuracy"] == 1.0
    assert metrics["decisive_coverage"] == pytest.approx(1 / 3)


def test_benchmark_rejects_modified_samples(tmp_path):
    source = tmp_path / "sample.png"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sample changed"):
        benchmark.evaluate(
            [
                {
                    "id": "a",
                    "path": str(source),
                    "sha256": "invalid",
                    "media_type": "image",
                }
            ],
            tmp_path,
            "cpu",
        )
