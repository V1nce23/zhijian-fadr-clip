import json
import uuid

import cv2
import numpy as np
import pytest
import torch

from app.core.database import SessionLocal
from app.detectors.catalog import detector_specs
from app.detectors.fadr_clip.adapter import FADRClipDetector
from app.detectors.fadr_clip.attention import combine_attention, render_attention
from app.models import DetectorRun
from app.services import attention_gallery as attention
from app.services.evidence_gallery import publish_gallery
from app.services.storage import get_storage
from app.tasks.attention_gallery import prepare_attention_gallery
from test_evidence_gallery import files, report, route
from test_live_visualization import gallery, prepare, running


def payload(flat=False):
    patches = np.full((1, 4, 256), 1 / 256)
    if not flat:
        patches[:, :, 0] += 0.2
        patches /= patches.sum(axis=-1, keepdims=True)
    return combine_attention(patches, [[0.1, 0.2, 0.3, 0.4]])[0]


def endpoint(identity, run_id):
    return f"/api/v1/analyses/{identity}/runs/{run_id}/attention"


def test_combination_uses_actual_routing_and_preserves_patch_orientation():
    patches = np.zeros((1, 4, 256))
    for layer in range(4):
        patches[0, layer, layer] = 1
    original = patches.copy()
    combined = combine_attention(patches, [[0.1, 0.2, 0.3, 0.4]])[0]
    grid = np.asarray(combined["patch_weights"])
    assert grid.shape == (16, 16) and np.allclose(grid[0, :4], [0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(patches, original)
    png, metadata = render_attention(payload())
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape == (256, 256, 4) and image[:, :, 3].min() == 255
    # Maximum (upper-left) and minimum (bottom-right) map to exact TURBO endpoints.
    ends = cv2.applyColorMap(np.array([[255, 0]], np.uint8), cv2.COLORMAP_TURBO)
    assert np.array_equal(image[0, 0, :3], ends[0, 0])
    assert np.array_equal(image[-1, -1, :3], ends[0, 1])
    assert metadata["method"] == attention.METHOD and not metadata["uniform"]


def test_uniform_map_is_transparent_instead_of_inventing_hotspot():
    png, metadata = render_attention(payload(flat=True))
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert metadata["uniform"] and not image[:, :, 3].any()


@pytest.mark.parametrize("bad", ["shape", "nan", "negative", "sum"])
def test_invalid_attention_is_rejected(bad):
    patches, routes = np.full((1, 4, 256), 1 / 256), np.full((1, 4), 0.25)
    if bad == "shape":
        patches = patches[:, :, :100]
    elif bad == "nan":
        patches[0, 0, 0] = np.nan
    elif bad == "negative":
        routes[0, 0] = -0.2
    else:
        routes *= 2
    with pytest.raises(ValueError):
        combine_attention(patches, routes)


def test_hook_reads_same_forward_without_changing_scores_and_always_removes_it(tmp_path):
    class Descriptor(torch.nn.Module):
        def forward(self, value):
            return value.mean(dim=-1), value.softmax(dim=-1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_descriptor = Descriptor()
            self.calls = 0

        def forward(self, data, inference=True):
            self.calls += 1
            batch = data["image"].shape[0]
            self.layer_descriptor(torch.arange(256, dtype=torch.float32).repeat(batch, 4, 1) / 50)
            return {"prob": torch.full((batch,), 0.123456), "attn_weights": torch.tensor([[0.1, 0.2, 0.3, 0.4]]).repeat(batch, 1)}

    detector = FADRClipDetector(detector_specs("image")[0], tmp_path)
    detector.model = Model()
    tensors = [torch.zeros(3, 224, 224)]
    before = detector.model({"image": torch.stack(tensors)})["prob"].tolist()
    scores, _ = detector._scores(tensors)
    assert scores == before and detector.model.calls == 2 and len(detector.batch_attention) == 1
    assert not detector.model.layer_descriptor._forward_hooks
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")
    detector.model.forward = fail
    with pytest.raises(RuntimeError):
        detector._scores(tensors)
    assert not detector.batch_attention and not detector.model.layer_descriptor._forward_hooks


def test_live_attention_and_score_appear_together_and_optional_failure_keeps_score(client, monkeypatch):
    identity, run_id, reporter = running(client)
    prepare(reporter)
    reporter("scores", frames=[{"frame_index": 0, "score": 0.89, "attention": payload()}])
    result = gallery(client, identity, run_id)
    assert result["frames"][0]["score"] == 0.89
    assert result["frames"][0]["heatmap_url"].endswith("attention-v1/frame-0.png")
    assert result["frames"][0]["attention_origin"] == "inference"
    storage = client.app.dependency_overrides[get_storage]()
    def fail(*args):
        raise OSError("optional attention storage failed")
    monkeypatch.setattr(storage, "upload", fail)
    reporter("scores", frames=[{"frame_index": 20, "score": 0.12, "attention": payload()}])
    result = gallery(client, identity, run_id)
    assert result["frames"][2]["score"] == 0.12 and result["frames"][2]["heatmap_url"] is None


def test_explicit_queue_is_deduplicated_and_get_never_runs_model(client, tmp_path, monkeypatch):
    identity, run_id, _ = report(client)
    publish_gallery(identity, run_id, files(tmp_path))
    before = client.get(f"/api/v1/analyses/{identity}").json()
    queued = []
    monkeypatch.setattr(prepare_attention_gallery, "apply_async", lambda **kwargs: queued.append(kwargs))
    for _ in range(3):
        assert client.get(route(identity, run_id)).json()["attention"]["status"] == "not_requested"
    assert not queued
    for _ in range(3):
        assert client.post(endpoint(identity, run_id)).json()["status"] == "pending"
    assert len(queued) == 1
    assert client.get(f"/api/v1/analyses/{identity}").json() == before


def test_backfill_uses_cached_faces_without_realigning_and_never_changes_original_report(client, tmp_path, monkeypatch):
    identity, run_id, _ = report(client)
    publish_gallery(identity, run_id, files(tmp_path))
    before = client.get(f"/api/v1/analyses/{identity}").json()
    monkeypatch.setattr("app.detectors.face_media.FaceMedia.extract", lambda *args: pytest.fail("do not align cached faces again"))
    def load(self):
        self.device = "cpu"
        self.config = {"mean": [0.5] * 3, "std": [0.2] * 3}
    def scores(self, tensors):
        self.batch_attention = [payload() for _ in tensors]
        return [0.999] * len(tensors), [[0.25] * 4 for _ in tensors]
    monkeypatch.setattr(FADRClipDetector, "load", load)
    monkeypatch.setattr(FADRClipDetector, "_scores", scores)
    prepare_attention_gallery(identity, run_id)
    result = client.get(route(identity, run_id)).json()
    assert result["attention"]["status"] == "ready" and result["attention"]["completed_frames"] == 2
    assert all(item["heatmap_url"] and item["attention_origin"] == "supplementary_replay" for item in result["frames"])
    assert [item["score"] for item in result["frames"]] == [0.2, 0.8]
    assert client.get(f"/api/v1/analyses/{identity}").json() == before


def test_failed_attention_is_retryable_and_unsupported_models_are_not_replayed(client, tmp_path, monkeypatch):
    identity, run_id, _ = report(client)
    publish_gallery(identity, run_id, files(tmp_path))
    monkeypatch.setattr("app.tasks.evidence_gallery.prepare_evidence_gallery", lambda *args: None)
    monkeypatch.setattr(FADRClipDetector, "load", lambda *args: (_ for _ in ()).throw(OSError("offline failure")))
    client.post(endpoint(identity, run_id))
    prepare_attention_gallery(identity, run_id)
    assert client.get(route(identity, run_id)).json()["attention"]["status"] == "failed"
    assert client.post(endpoint(identity, run_id)).json()["status"] == "pending"
    with SessionLocal() as db:
        run = db.get(DetectorRun, uuid.UUID(run_id))
        run.model_version = "other-weights"
        db.commit()
    assert client.post(endpoint(identity, run_id)).json()["status"] == "unavailable"


@pytest.mark.parametrize("delete_upload", [False, True])
def test_delete_purges_attention_and_blocks_late_publish(client, tmp_path, delete_upload):
    identity, run_id, upload_id = report(client)
    assert attention.publish_attention(identity, run_id, index=0, payload=payload())
    target = f"/api/v1/uploads/{upload_id}" if delete_upload else f"/api/v1/analyses/{identity}"
    assert client.delete(target).status_code == 200
    assert not attention.publish_attention(identity, run_id, index=10, payload=payload())
    storage = client.app.dependency_overrides[get_storage]()
    assert not any(key.startswith(f"evidence/{identity}/") for key in storage.objects)
    assert client.post(endpoint(identity, run_id)).status_code == 404


def test_attention_requires_owner_and_matching_run(client):
    assert client.post(endpoint(uuid.uuid4(), uuid.uuid4())).status_code == 401
    identity, run_id, _ = report(client)
    assert client.post(endpoint(identity, uuid.uuid4())).status_code == 404
    client.post("/api/v1/users", json={"email": "attention@example.com", "name": "Other", "password": "AttentionPass123!", "role": "analyst"})
    client.post("/api/v1/auth/login", json={"email": "attention@example.com", "password": "AttentionPass123!"})
    assert client.post(endpoint(identity, run_id)).status_code == 404
