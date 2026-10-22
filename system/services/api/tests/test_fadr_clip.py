import hashlib
import json

import numpy as np
import pytest

from app.detectors.base import DetectionResult, DetectorSpec
from app.detectors.catalog import FADR_CHECKPOINT_SHA256, detector_specs
from app.detectors.face_media import FaceSample, uniform_frame_indices
from app.detectors.fadr_clip.adapter import FADRClipDetector, load_checkpoint_state, load_verified_config, resize_rgb
from app.detectors.registry import get_detector
from app.services.results import aggregate_results


@pytest.mark.parametrize("total,expected", [(1, [0]), (3, [0, 1, 2]), (64, np.linspace(0, 63, 32, dtype=int).tolist())])
def test_sampling_matches_release_linspace_and_short_videos(total, expected):
    assert uniform_frame_indices(total) == expected


def test_release_resize_and_clip_normalization_are_preserved(tmp_path):
    import cv2
    import torch
    # A varying image distinguishes interpolation and channel order mistakes.
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    expected = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    assert np.array_equal(resize_rgb(image), expected)
    detector = FADRClipDetector(detector_specs("image")[0], tmp_path)
    detector.config = {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}
    tensor = detector._input_tensor(image)
    assert tensor.shape == (3, 224, 224)
    expected = torch.from_numpy(expected.copy()).permute(2, 0, 1).float() / 255
    expected = (expected - torch.tensor(detector.config["mean"])[:, None, None]) / torch.tensor(detector.config["std"])[:, None, None]
    assert torch.equal(tensor, expected)


def test_nonsquare_release_input_is_padded_without_distortion():
    image = np.full((56, 224, 3), 255, dtype=np.uint8)
    actual = resize_rgb(image)
    assert actual.shape == (224, 224, 3)
    assert np.all(actual[84:140] == 255)
    assert not actual[:84].any() and not actual[140:].any()


def test_video_report_averages_only_frames_with_faces_and_keeps_scores(tmp_path, monkeypatch):
    detector = FADRClipDetector(detector_specs("video")[0], tmp_path)
    monkeypatch.setattr(detector, "load", lambda: None)
    detector.config = {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}
    samples = [FaceSample(np.full((1, 1, 3), value), index, index / 10, {}) for value, index in [(0.2, 0), (0.8, 10), (0.9, 20)]]
    monkeypatch.setattr(detector.faces, "extract", lambda *args, **kwargs: (samples, 32, 4.0))
    monkeypatch.setattr(detector, "_input_tensor", lambda value: float(value[0, 0, 0]))
    monkeypatch.setattr(detector, "_scores", lambda batch: (batch, [[0.1, 0.2, 0.3, 0.4] for _ in batch]))
    checks = []
    result = detector.predict(tmp_path / "clip.mp4", check=lambda: checks.append(True))
    assert result.score == pytest.approx(0.633333)
    assert result.verdict == "uncertain"
    assert result.evidence["sampled_frames"] == 32
    assert result.evidence["frames_with_face"] == 3
    assert [row["frame_index"] for row in result.evidence["frames"]] == [0, 10, 20]
    assert [row["score"] for row in result.evidence["frames"]] == [0.2, 0.8, 0.9]
    assert result.evidence["source_checkpoint_sha256"] == FADR_CHECKPOINT_SHA256
    assert len(checks) >= 6


def test_modified_weight_file_is_rejected_before_loading_model(tmp_path):
    weights = tmp_path / "model.safetensors"
    config = tmp_path / "model_config.json"
    weights.write_bytes(b"tampered")
    config.write_text("{}")
    facts = lambda p: {"size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest = {"source_checkpoint_sha256": FADR_CHECKPOINT_SHA256, "files": {p.name: facts(p) for p in (weights, config)}}
    manifest["files"][weights.name]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="校验失败"):
        load_verified_config(tmp_path)


def test_gpu_checkpoint_owns_host_storage_and_preserves_exact_values(tmp_path, monkeypatch):
    import torch
    import safetensors.torch

    weights = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
               "indices": torch.tensor([1, 4], dtype=torch.int64)}
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(weights, str(path))
    mapped = safetensors.torch.load_file(str(path), device="cpu")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *args, **kwargs: mapped)
    loaded = load_checkpoint_state(path, "cuda")
    for name, expected in weights.items():
        assert torch.equal(loaded[name], expected)
        assert loaded[name].dtype == expected.dtype
        assert loaded[name].device.type == "cpu"
        assert loaded[name].untyped_storage().data_ptr() != mapped[name].untyped_storage().data_ptr()
    mapped["weight"].zero_()
    assert torch.equal(loaded["weight"], weights["weight"])


def test_old_detector_is_not_available_for_new_inference(tmp_path):
    spec = DetectorSpec("dfbench-xception", "v1", "legacy", "image")
    with pytest.raises(ValueError, match="unsupported detector"):
        get_detector("image", spec, str(tmp_path), "cpu", True, 0.35, 0.65)


@pytest.mark.parametrize("score,verdict", [(0.2, "likely_authentic"), (0.5, "uncertain"), (0.8, "likely_manipulated")])
def test_single_model_summary_preserves_its_score_and_verdict(score, verdict):
    summary = aggregate_results([DetectionResult(score, verdict, {})])
    assert summary.score == score
    assert summary.verdict.value == verdict
    assert "FADR-CLIP" in summary.summary
