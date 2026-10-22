import json
import uuid

import cv2
import numpy as np
import pytest

from app.core.database import SessionLocal
from app.models import DetectorRun
from app.services.evidence_gallery import gallery_prefix, manifest_key, publish_gallery, state_key
from app.services import previews
from app.services.storage import get_storage
from app.tasks.analysis import run_analysis
from app.tasks.evidence_gallery import prepare_evidence_gallery
from test_workflow import WEBM_BYTES, PNG_BYTES, login


def report(client, image=False):
    login(client)
    name, data, mime = ("face.png", PNG_BYTES, "image/png") if image else ("clip.webm", WEBM_BYTES, "video/webm")
    upload = client.post("/api/v1/uploads", files={"file": (name, data, mime)}).json()
    analysis = client.post("/api/v1/analyses", json={"upload_id": upload["id"]}).json()
    run_analysis(analysis["id"])
    run_id = analysis["detector_runs"][0]["id"]
    if not image:
        with SessionLocal() as db:
            run = db.get(DetectorRun, uuid.UUID(run_id))
            run.evidence = {"kind": "video_frames", "sampled_frames": 32, "frames_with_face": 2,
                            "frames": [{"frame_index": 0, "timestamp_seconds": 0.0, "score": 0.2},
                                       {"frame_index": 10, "timestamp_seconds": 0.4, "score": 0.8}]}
            db.commit()
    return analysis["id"], run_id, upload["id"]


def route(identity, run_id):
    return f"/api/v1/analyses/{identity}/runs/{run_id}/gallery"


def files(tmp_path, indices=(0, 10)):
    paths = {}
    for index in indices:
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), np.full((256, 256, 3), index, dtype=np.uint8))
        paths[index] = path
    return paths


def test_polls_queue_once_without_changing_report_or_rerunning_model(client, monkeypatch):
    identity, run_id, _ = report(client)
    before = client.get(f"/api/v1/analyses/{identity}").json()
    jobs = []
    monkeypatch.setattr(prepare_evidence_gallery, "apply_async", lambda **kwargs: jobs.append(kwargs))
    for _ in range(3):
        assert client.get(route(identity, run_id)).json()["status"] == "pending"
    assert len(jobs) == 1 and jobs[0]["args"] == [identity, run_id]
    assert client.get(f"/api/v1/analyses/{identity}").json() == before
    assert len(client.get("/api/v1/analyses").json()) == 1


def test_gallery_uses_authoritative_frame_scores_and_is_available_without_redis(client, tmp_path, monkeypatch):
    identity, run_id, _ = report(client)
    before = client.get(f"/api/v1/analyses/{identity}").json()
    assert publish_gallery(identity, run_id, files(tmp_path))
    monkeypatch.setattr(previews, "preview_registry", lambda: pytest.fail("cached gallery must not need Redis"))
    result = client.get(route(identity, run_id)).json()
    assert result["status"] == "ready" and result["image_kind"] == "aligned_face"
    assert [(f["frame_index"], f["score"]) for f in result["frames"]] == [(0, 0.2), (10, 0.8)]
    assert result["frames"][0]["heatmap_url"] is None
    assert all(f["image_url"].endswith(f"frame-{f['frame_index']}.png") for f in result["frames"])
    assert client.get(f"/api/v1/analyses/{identity}").json() == before


def test_worker_saves_exact_rgb_crops_without_model_inference(client, monkeypatch):
    from app.detectors.face_media import FaceMedia, FaceSample
    identity, run_id, _ = report(client)
    rgb = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    monkeypatch.setattr("app.services.media.prepare_detection_media", lambda source, *args: source)
    monkeypatch.setattr(FaceMedia, "extract", lambda *args: ([FaceSample(rgb, 0, 0.0, {}), FaceSample(rgb, 10, 0.4, {})], 32, 1))
    monkeypatch.setattr("app.detectors.fadr_clip.adapter.FADRClipDetector.load", lambda *args: pytest.fail("gallery must not load model"))
    before = client.get(f"/api/v1/analyses/{identity}").json()
    prepare_evidence_gallery(identity, run_id)
    storage = client.app.dependency_overrides[get_storage]()
    for index in (0, 10):
        png = storage.objects[f"{gallery_prefix(identity, run_id)}frame-{index}.png"]
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), rgb)
    assert client.get(route(identity, run_id)).json()["status"] == "ready"
    assert client.get(f"/api/v1/analyses/{identity}").json() == before


def test_gallery_checks_login_owner_and_run_membership(client):
    assert client.get(route(uuid.uuid4(), uuid.uuid4())).status_code == 401
    identity, run_id, _ = report(client)
    assert client.get(route(identity, uuid.uuid4())).status_code == 404
    client.post("/api/v1/users", json={"email": "gallery@example.com", "name": "Another user", "password": "GalleryPass123!", "role": "analyst"})
    client.post("/api/v1/auth/login", json={"email": "gallery@example.com", "password": "GalleryPass123!"})
    assert client.get(route(identity, run_id)).status_code == 404
    assert client.post(route(identity, run_id)).status_code == 404


def test_failed_gallery_can_retry(client, monkeypatch):
    identity, run_id, _ = report(client)
    previews.preview_registry().set(state_key(identity, run_id), json.dumps({"status": "failed"}))
    assert client.get(route(identity, run_id)).json()["status"] == "failed"
    jobs = []
    monkeypatch.setattr(prepare_evidence_gallery, "apply_async", lambda **kwargs: jobs.append(kwargs))
    assert client.post(route(identity, run_id)).json()["status"] == "pending"
    assert len(jobs) == 1


def test_incomplete_frames_cannot_publish_misleading_gallery(client, tmp_path):
    identity, run_id, _ = report(client)
    with pytest.raises(ValueError, match="完整还原"):
        publish_gallery(identity, run_id, files(tmp_path, (0,)))
    storage = client.app.dependency_overrides[get_storage]()
    assert not storage.exists(manifest_key(identity, run_id))


@pytest.mark.parametrize("delete_upload", [False, True])
def test_deleting_report_or_source_cleans_gallery_and_blocks_late_worker(client, tmp_path, delete_upload):
    identity, run_id, upload_id = report(client)
    paths = files(tmp_path)
    assert publish_gallery(identity, run_id, paths)
    target = f"/api/v1/uploads/{upload_id}" if delete_upload else f"/api/v1/analyses/{identity}"
    assert client.delete(target).status_code == 200
    storage = client.app.dependency_overrides[get_storage]()
    assert not any(key.startswith(gallery_prefix(identity)) for key in storage.objects)
    assert client.get(route(identity, run_id)).status_code == 404
    assert not publish_gallery(identity, run_id, paths)
    prepare_evidence_gallery(identity, run_id)
    assert not any(key.startswith(gallery_prefix(identity)) for key in storage.objects)


def test_old_preprocessing_does_not_claim_reconstructed_faces_are_exact(client):
    identity, run_id, _ = report(client)
    with SessionLocal() as db:
        run = db.get(DetectorRun, uuid.UUID(run_id))
        run.preprocessing_version = "aligned256-clip224-v2"
        db.commit()
    assert client.get(route(identity, run_id)).json()["status"] == "unavailable"


def test_image_gallery_has_one_face_and_original_model_score(client, tmp_path):
    identity, run_id, _ = report(client, image=True)
    before = client.get(f"/api/v1/analyses/{identity}").json()
    assert publish_gallery(identity, run_id, files(tmp_path, (0,)))
    result = client.get(route(identity, run_id)).json()
    assert len(result["frames"]) == 1 and result["frames"][0]["score"] == before["score"]


def test_failed_gallery_generation_preserves_completed_report(client, monkeypatch):
    identity, run_id, _ = report(client)
    before = client.get(f"/api/v1/analyses/{identity}").json()
    monkeypatch.setattr("app.services.media.prepare_detection_media", lambda *args: (_ for _ in ()).throw(ValueError("decode failed")))
    prepare_evidence_gallery(identity, run_id)
    assert client.get(route(identity, run_id)).json()["status"] == "failed"
    assert client.get(f"/api/v1/analyses/{identity}").json() == before
