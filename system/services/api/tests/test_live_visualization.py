import hashlib
import uuid

import cv2
import numpy as np
import pytest

from app.core.database import SessionLocal
from app.detectors.catalog import detector_specs
from app.detectors.face_media import FaceSample
from app.detectors.fadr_clip.adapter import FADRClipDetector
from app.models import Analysis, AnalysisEvent, DetectorRun
from app.services.evidence_gallery import gallery_prefix
from app.services.inference import InferenceCancelled
from app.services.live_visualization import LiveVisualization
from app.services.storage import get_storage
from app.tasks.analysis import _claim_analysis, _mark_run_started
from test_workflow import WEBM_BYTES, login


def running(client):
    login(client)
    upload = client.post("/api/v1/uploads", files={"file": ("video.webm", WEBM_BYTES, "video/webm")}).json()
    analysis = client.post("/api/v1/analyses", json={"upload_id": upload["id"]}).json()
    identity, run_id = analysis["id"], analysis["detector_runs"][0]["id"]
    assert _claim_analysis(uuid.UUID(identity))
    _mark_run_started(uuid.UUID(run_id))
    reporter = LiveVisualization(identity, run_id, "video")
    return identity, run_id, reporter


def gallery(client, identity, run_id):
    return client.get(f"/api/v1/analyses/{identity}/runs/{run_id}/gallery").json()


def prepare(reporter):
    reporter("plan", frames=[{"frame_index": i, "timestamp_seconds": i / 25} for i in (0, 10, 20)])
    rgb = np.full((256, 256, 3), [21, 82, 193], dtype=np.uint8)
    for index in (0, 10, 20):
        reporter("sampled", frame_index=index, image=rgb)
        reporter("aligned", frame_index=index, image=None if index == 10 else rgb)
    return rgb


def test_frames_appear_before_model_scores_and_keep_original_rgb(client):
    identity, run_id, reporter = running(client)
    reporter("plan", frames=[{"frame_index": 0, "timestamp_seconds": 0}])
    pending = gallery(client, identity, run_id)
    assert pending["status"] == "streaming" and pending["frames"][0]["score"] is None
    assert pending["frames"][0]["image_url"] is None
    rgb = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    reporter("sampled", frame_index=0, image=rgb)
    sampled = gallery(client, identity, run_id)
    assert sampled["frames"][0]["image_kind"] == "source_frame"
    assert sampled["frames"][0]["score"] is None
    reporter("aligned", frame_index=0, image=rgb)
    aligned = gallery(client, identity, run_id)
    assert aligned["frames"][0]["status"] == "ready"
    assert aligned["frames"][0]["image_kind"] == "aligned_face"
    assert aligned["progress"]["scored_frames"] == 0
    storage = client.app.dependency_overrides[get_storage]()
    data = storage.objects[gallery_prefix(identity, run_id) + "frame-0.png"]
    decoded = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, rgb) and hashlib.sha256(rgb.tobytes()).hexdigest() == digest


def test_real_scores_increment_and_no_face_is_not_zero(client):
    identity, run_id, reporter = running(client)
    prepare(reporter)
    reporter("model_loading")
    loaded = gallery(client, identity, run_id)
    assert loaded["stage"] == "model_loading"
    assert loaded["progress"] == {"sampled_frames": 3, "extracted_frames": 3, "processed_frames": 3, "valid_frames": 2, "scored_frames": 0}
    assert all(frame["score"] is None for frame in loaded["frames"])
    reporter("scores", frames=[{"frame_index": 0, "score": 0.9}])
    partial = gallery(client, identity, run_id)
    assert [frame["score"] for frame in partial["frames"]] == [0.9, None, None]
    assert partial["frames"][1]["status"] == "no_face" and partial["progress"]["scored_frames"] == 1
    assert client.get(f"/api/v1/analyses/{identity}").json()["score"] is None
    reporter("scores", frames=[{"frame_index": 20, "score": 0.1}])
    assert gallery(client, identity, run_id)["progress"]["scored_frames"] == 2
    with SessionLocal() as db:
        events = db.query(AnalysisEvent).filter(AnalysisEvent.analysis_id == uuid.UUID(identity), AnalysisEvent.event_type == "analysis.visualization").all()
        assert events and events[-1].payload["scored_frames"] == 2


def test_cancelled_detection_keeps_partial_frames_and_rejects_late_updates(client):
    identity, run_id, reporter = running(client)
    rgb = prepare(reporter)
    reporter("scores", frames=[{"frame_index": 0, "score": 0.9}])
    assert client.post(f"/api/v1/analyses/{identity}/cancel").status_code == 200
    partial = gallery(client, identity, run_id)
    assert partial["status"] == "stopped" and partial["stage"] == "cancelled"
    assert [frame["score"] for frame in partial["frames"]] == [0.9, None, None]
    with pytest.raises(InferenceCancelled):
        reporter("sampled", frame_index=20, image=rgb)
    assert client.delete(f"/api/v1/analyses/{identity}").status_code == 200
    with pytest.raises(InferenceCancelled):
        reporter("aligned", frame_index=20, image=rgb)
    storage = client.app.dependency_overrides[get_storage]()
    assert not any(key.startswith(gallery_prefix(identity)) for key in storage.objects)


@pytest.mark.parametrize("batch", [False, True])
def test_delete_active_report_blocks_late_scores_assets_and_completion(client, batch):
    from app.detectors.base import DetectionResult
    from app.tasks.analysis import _is_cancelled, _mark_run_succeeded, run_analysis

    identity, run_id, reporter = running(client)
    rgb = prepare(reporter)
    reporter("scores", frames=[{"frame_index": 0, "score": 0.9}])
    with SessionLocal() as db:
        upload_id = str(db.get(Analysis, uuid.UUID(identity)).upload_id)
    response = (client.post("/api/v1/analyses/batch-delete", json={"analysis_ids": [identity]})
                if batch else client.delete(f"/api/v1/analyses/{identity}"))
    assert response.status_code == 200
    if batch:
        assert response.json()["deleted"] == 1 and response.json()["skipped"] == 0
    assert _is_cancelled(uuid.UUID(identity))
    for event, data in [("aligned", {"frame_index": 20, "image": rgb}),
                        ("scores", {"frames": [{"frame_index": 20, "score": 0.8}]}),
                        ("reporting", {})]:
        with pytest.raises(InferenceCancelled):
            reporter(event, **data)
    with pytest.raises(InferenceCancelled):
        _mark_run_succeeded(uuid.UUID(run_id), DetectionResult(0.8, "likely_manipulated", {}))
    run_analysis(identity)
    assert client.get(f"/api/v1/analyses/{identity}").status_code == 404
    assert client.get(f"/api/v1/uploads/{upload_id}/download-url").status_code == 200
    storage = client.app.dependency_overrides[get_storage]()
    assert not any(key.startswith(gallery_prefix(identity)) for key in storage.objects)


def test_visualization_storage_failure_does_not_drop_progress_or_scores(client, monkeypatch):
    identity, run_id, reporter = running(client)
    storage = client.app.dependency_overrides[get_storage]()
    def unavailable(*args):
        raise OSError("storage temporarily unavailable")
    monkeypatch.setattr(storage, "upload", unavailable)
    prepare(reporter)
    reporter("scores", frames=[{"frame_index": 0, "score": 0.7}])
    result = gallery(client, identity, run_id)
    assert result["progress"]["scored_frames"] == 1
    assert result["frames"][0]["score"] == 0.7 and result["frames"][0]["image_url"] is None


@pytest.mark.parametrize("device,batches", [("cpu", [1, 1, 1, 1, 1]), ("cuda", [4, 1])])
def test_detector_prepares_faces_before_loading_and_reports_real_batch_results(tmp_path, monkeypatch, device, batches):
    detector = FADRClipDetector(detector_specs("video")[0], tmp_path)
    samples = [FaceSample(np.full((1, 1, 3), i / 10), i, i / 25, {}) for i in range(5)]
    events, actual_batches = [], []
    loaded = False
    def extract(*args, **kwargs):
        assert not loaded
        return samples, 5, 1
    def load():
        nonlocal loaded
        loaded = True
        detector.device = device
        detector.config = {"mean": [0.4, 0.5, 0.6], "std": [0.2, 0.2, 0.2]}
    def scores(values):
        assert loaded
        actual_batches.append(len(values))
        return values, [[0.25] * 4 for _ in values]
    monkeypatch.setattr(detector.faces, "extract", extract)
    monkeypatch.setattr(detector, "load", load)
    monkeypatch.setattr(detector, "_input_tensor", lambda rgb: float(rgb[0, 0, 0]))
    monkeypatch.setattr(detector, "_scores", scores)
    detector.progress_callback = lambda event, **data: events.append((event, data))
    result = detector.predict(tmp_path / "original.mp4")
    assert actual_batches == batches
    assert events[0][0] == "model_loading" and events[1][0] == "detecting"
    emitted = [item for event, data in events if event == "scores" for item in data["frames"]]
    assert emitted == [{"frame_index": i, "score": i / 10, "routing_weights": [0.25] * 4,
                        "routing_layers": [6, 12, 18, 24]} for i in range(5)]
    assert result.score == 0.2 and [frame["score"] for frame in result.evidence["frames"]] == [i / 10 for i in range(5)]
