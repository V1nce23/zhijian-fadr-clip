"""Decision presets must not change inference scores or historical rules."""

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from app.core.database import SessionLocal
from app.detectors.base import DetectionResult
from app.models import Analysis
from app.services import threshold_policy
from app.services.live_visualization import LiveVisualization
from app.tasks.analysis import _claim_analysis, _mark_run_started, run_analysis
from test_workflow import PNG_BYTES, WEBM_BYTES, login


def upload(client, media):
    login(client)
    file = ("face.png", PNG_BYTES, "image/png") if media == "image" else ("video.webm", WEBM_BYTES, "video/webm")
    return client.post("/api/v1/uploads", files={"file": file}).json()["id"]


@pytest.mark.parametrize("media,score,expected", [
    ("image", 0.95, ["likely_manipulated", "likely_authentic", "likely_manipulated"]),
    ("video", 0.80, ["likely_manipulated", "likely_authentic", "likely_manipulated"]),
])
def test_each_mode_uses_saved_policy_without_changing_score(client, monkeypatch, media, score, expected):
    upload_id = upload(client, media)
    seen = []

    class Detector:
        def predict(self, path, **kwargs):
            seen.append(path.read_bytes())
            # Deliberately different legacy verdict: the saved policy must win.
            return DetectionResult(score, "uncertain", {"kind": "image_regions", "source": "unchanged"})

    monkeypatch.setattr("app.tasks.analysis.get_detector", lambda *args: Detector())
    for mode, verdict in zip(threshold_policy.MODES, expected):
        created = client.post("/api/v1/analyses", json={"upload_id": upload_id, "threshold_mode": mode.mode}).json()
        policy = created["threshold_policy"]
        assert policy == threshold_policy.make_policy(mode.mode, media)
        assert created["detector_runs"][0]["evidence"]["thresholds"] == threshold_policy.frame_thresholds(policy)
        run_analysis(created["id"])
        report = client.get(f"/api/v1/analyses/{created['id']}").json()
        assert report["status"] == "completed" and report["verdict"] == verdict
        assert report["score"] == score and report["threshold_policy"] == policy
        run = report["detector_runs"][0]
        assert run["score"] == score and run["verdict"] == verdict
        assert run["evidence"]["decision_policy"] == policy and run["evidence"]["source"] == "unchanged"
        assert f"{policy['threshold']:.2f}" in report["summary"]
    assert seen == [PNG_BYTES if media == "image" else WEBM_BYTES] * 3
    assert {row["threshold_policy"]["mode"] for row in client.get("/api/v1/analyses").json()} == {p.mode for p in threshold_policy.MODES}


@pytest.mark.parametrize("mode", ["balanced", "low_false_positive", "low_false_negative"])
@pytest.mark.parametrize("media", ["image", "video"])
def test_strict_threshold_ties_and_frame_unit(mode, media):
    policy = threshold_policy.make_policy(mode, media)
    threshold = policy["threshold"]
    assert threshold_policy.policy_verdict(threshold, policy) == "likely_authentic"
    assert threshold_policy.policy_verdict(threshold - 0.000001, policy) == "likely_authentic"
    assert threshold_policy.policy_verdict(threshold + 0.000001, policy) == "likely_manipulated"
    assert threshold_policy.frame_thresholds(policy)["manipulated_min"] == policy["frame_threshold"]


def test_endpoint_default_and_invalid_mode(client):
    assert client.get("/api/v1/analyses/threshold-modes").status_code == 401
    upload_id = upload(client, "image")
    options = client.get("/api/v1/analyses/threshold-modes")
    assert options.status_code == 200 and len(options.json()) == 4
    assert options.json()[-1]["mode"] == "custom"
    assert options.json()[0]["mode"] == "balanced"
    assert client.post("/api/v1/analyses", json={"upload_id": upload_id, "threshold_mode": "unknown"}).status_code == 422
    created = client.post("/api/v1/analyses", json={"upload_id": upload_id}).json()
    assert created["threshold_policy"]["mode"] == "balanced"
    assert created["threshold_policy"]["threshold"] == 0.85


def test_live_frames_and_queued_report_keep_the_original_policy(client, monkeypatch):
    upload_id = upload(client, "video")
    created = client.post("/api/v1/analyses", json={"upload_id": upload_id, "threshold_mode": "low_false_positive"}).json()
    saved = created["threshold_policy"]
    identity = uuid.UUID(created["id"])
    run_id = uuid.UUID(created["detector_runs"][0]["id"])
    assert _claim_analysis(identity)
    _mark_run_started(run_id)
    # Simulate a future preset edit; an existing job must keep its snapshot.
    monkeypatch.setattr(threshold_policy, "MODES", (threshold_policy.ThresholdMode("low_false_positive", "changed", "", 0.1, 0.2),))
    reporter = LiveVisualization(identity, run_id, "video")
    reporter("plan", frames=[{"frame_index": 0, "timestamp_seconds": 0}])
    run = client.get(f"/api/v1/analyses/{identity}").json()["detector_runs"][0]
    assert run["evidence"]["decision_policy"] == saved
    assert run["evidence"]["thresholds"] == {"authentic_max": 0.99, "manipulated_min": 0.99, "operator": ">"}
    assert run["evidence"]["frames"] == []


def test_worker_uses_snapshot_and_leaves_legacy_rule_unchanged(client, monkeypatch):
    upload_id = upload(client, "video")
    created = [client.post("/api/v1/analyses", json={"upload_id": upload_id}).json() for _ in range(2)]
    with SessionLocal() as db:
        db.get(Analysis, uuid.UUID(created[1]["id"])).threshold_policy = None
        db.commit()

    class Detector:
        def predict(self, *args, **kwargs):
            return DetectionResult(0.60, "uncertain", {"thresholds": {"authentic_max": 0.35, "manipulated_min": 0.65}})

    monkeypatch.setattr("app.tasks.analysis.get_detector", lambda *args: Detector())
    monkeypatch.setattr(threshold_policy, "MODES", (threshold_policy.ThresholdMode("balanced", "changed", "", 0.1, 0.2),))
    for item in created:
        run_analysis(item["id"])
    new, old = [client.get(f"/api/v1/analyses/{item['id']}").json() for item in created]
    assert new["verdict"] == "likely_authentic" and new["threshold_policy"]["threshold"] == 0.67
    assert old["verdict"] == "uncertain" and old["threshold_policy"] is None
    assert old["detector_runs"][0]["evidence"]["thresholds"] == {"authentic_max": 0.35, "manipulated_min": 0.65}


def test_migration_adds_nullable_snapshot_without_relabeling(tmp_path):
    database = tmp_path / "thresholds.db"
    environment = {**os.environ, "DATABASE_URL": f"sqlite:///{database}"}
    service_root = Path(__file__).resolve().parents[1]

    def migrate(target):
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", target], cwd=service_root, env=environment, check=True, capture_output=True)

    migrate("20261020_0004")
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO analyses (id,upload_id,user_id,status,progress,cancel_requested,created_at,score,verdict) VALUES (?,?,?,'completed',100,0,CURRENT_TIMESTAMP,0.6,'uncertain')", [uuid.uuid4().hex for _ in range(3)])
    migrate("head")
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT score,verdict,threshold_policy FROM analyses").fetchone() == (0.6, "uncertain", None)


def test_expired_delivery_does_not_poison_dispatch_retry(monkeypatch):
    from app import dispatcher

    revoked, accepted = set(), []

    def deliver(**kwargs):
        # Celery allocates a UUID when task_id is omitted. A fixed ID that
        # expires would instead cause all retries to be discarded as revoked.
        delivery_id = kwargs.get("task_id") or uuid.uuid4().hex
        if delivery_id not in revoked:
            accepted.append(kwargs["args"])
        revoked.add(delivery_id)

    monkeypatch.setattr(dispatcher.run_analysis, "apply_async", deliver)
    identity = str(uuid.uuid4())
    assert dispatcher.publish_analysis(identity)
    assert dispatcher.publish_analysis(identity)
    assert accepted == [[identity], [identity]]


@pytest.mark.parametrize("media", ["image", "video"])
def test_custom_values_control_verdict_and_keep_full_precision(client, monkeypatch, media):
    upload_id = upload(client, media)
    custom = {"video_threshold": 0.750123, "frame_threshold": 0.9985}

    class Detector:
        def predict(self, *args, **kwargs):
            return DetectionResult(0.8, "uncertain", {})

    monkeypatch.setattr("app.tasks.analysis.get_detector", lambda *args: Detector())
    created = client.post("/api/v1/analyses", json={"upload_id": upload_id, "threshold_mode": "custom", "custom_thresholds": custom})
    assert created.status_code == 201
    identity = created.json()["id"]
    policy = created.json()["threshold_policy"]
    assert policy["video_threshold"] == 0.750123 and policy["frame_threshold"] == 0.9985
    assert policy["selection_status"] == "user_defined"
    run_analysis(identity)
    report = client.get(f"/api/v1/analyses/{identity}").json()
    assert report["score"] == 0.8 and report["threshold_policy"] == policy
    assert report["verdict"] == ("likely_manipulated" if media == "video" else "likely_authentic")
    assert ("0.750123" if media == "video" else "0.9985") in report["summary"]
    assert report["detector_runs"][0]["evidence"]["thresholds"]["manipulated_min"] == 0.9985
    assert client.get("/api/v1/analyses").json()[0]["threshold_policy"] == policy


@pytest.mark.parametrize("value", [-0.1, 1.1, None, True, "0.7", "NaN", "Infinity", 0.1234567])
def test_invalid_custom_values_rejected_without_creating_job(client, value):
    upload_id = upload(client, "image")
    response = client.post("/api/v1/analyses", json={"upload_id": upload_id, "threshold_mode": "custom", "custom_thresholds": {"video_threshold": value, "frame_threshold": 0.85}})
    assert response.status_code == 422
    assert client.get("/api/v1/analyses").json() == []


def test_custom_requires_both_values_and_cannot_override_preset(client):
    upload_id = upload(client, "image")
    base = {"upload_id": upload_id, "threshold_mode": "custom"}
    assert client.post("/api/v1/analyses", json=base).status_code == 422
    assert client.post("/api/v1/analyses", json={**base, "custom_thresholds": {"video_threshold": 0.6}}).status_code == 422
    assert client.post("/api/v1/analyses", json={**base, "threshold_mode": "balanced", "custom_thresholds": {"video_threshold": 0.6, "frame_threshold": 0.8}}).status_code == 422
    assert client.get("/api/v1/analyses").json() == []


def test_custom_zero_one_and_exact_tie_are_valid():
    from app.schemas import AnalysisCreate

    request = AnalysisCreate(upload_id=uuid.uuid4(), threshold_mode="custom", custom_thresholds={"video_threshold": 0, "frame_threshold": 1})
    policy = threshold_policy.make_policy("custom", "image", request.custom_thresholds.model_dump())
    assert threshold_policy.policy_verdict(1, policy) == "likely_authentic"
    assert threshold_policy.policy_verdict(0.000001, threshold_policy.make_policy("custom", "video", request.custom_thresholds.model_dump())) == "likely_manipulated"


def test_image_decoding_never_calls_video_sampler_or_capture(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from app.detectors import face_media

    rgb = np.full((256, 256, 3), [40, 90, 180], dtype=np.uint8)
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    media = face_media.FaceMedia(tmp_path)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: pytest.fail("图片不能走视频解码"))
    monkeypatch.setattr(face_media, "uniform_frame_indices", lambda *args: pytest.fail("图片不能走均匀抽帧"))
    monkeypatch.setattr(media, "_face", lambda image, index, timestamp: face_media.FaceSample(image, index, timestamp, {}))
    faces, count, duration = media.extract(path, "image")
    assert count == 1 and duration == 0 and np.array_equal(faces[0].image, rgb)
