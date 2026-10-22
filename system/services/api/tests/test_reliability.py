import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest
from billiard.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select

from app.core.database import SessionLocal
from app.dispatcher import dispatch_pending_analyses
from app.models import Analysis, AnalysisEvent, DetectorStatus
from app.services.inference import (
    InferenceCancelled,
    InferenceControl,
    InferenceTimedOut,
    run_command,
)
from app.tasks.analysis import _claim_analysis, run_analysis
from test_workflow import PNG_BYTES, login


def create_analysis(client) -> str:
    login(client)
    upload = client.post(
        "/api/v1/uploads", files={"file": ("face.png", PNG_BYTES, "image/png")}
    ).json()
    response = client.post("/api/v1/analyses", json={"upload_id": upload["id"]})
    assert response.status_code == 201
    return response.json()["id"]


def test_dispatcher_recovers_publish_failure_and_unpublished_commit(
    client, monkeypatch
):
    def unavailable(**kwargs):
        raise ConnectionError("broker unavailable")

    monkeypatch.setattr("app.dispatcher.run_analysis.apply_async", unavailable)
    analysis_id = create_analysis(client)
    assert client.get(f"/api/v1/analyses/{analysis_id}").json()["status"] == "queued"
    published = []
    monkeypatch.setattr(
        "app.dispatcher.run_analysis.apply_async",
        lambda **kwargs: published.append(kwargs),
    )
    dispatch_pending_analyses()
    assert [call["args"] for call in published] == [[analysis_id]]
    assert "task_id" not in published[0]  # Celery supplies a fresh delivery ID.
    run_analysis(analysis_id)
    dispatch_pending_analyses()
    assert len(published) == 1
    assert client.get(f"/api/v1/analyses/{analysis_id}").json()["status"] == "completed"


def test_duplicate_delivery_does_not_rerun_completed_report(client, monkeypatch):
    analysis_id = create_analysis(client)
    run_analysis(analysis_id)
    before = client.get(f"/api/v1/analyses/{analysis_id}").json()
    with SessionLocal() as db:
        events_before = len(list(db.scalars(select(AnalysisEvent))))
    monkeypatch.setattr(
        "app.tasks.analysis.get_detector", lambda *args: pytest.fail("must not rerun")
    )
    run_analysis(analysis_id)
    assert client.get(f"/api/v1/analyses/{analysis_id}").json() == before
    with SessionLocal() as db:
        assert len(list(db.scalars(select(AnalysisEvent)))) == events_before


def test_only_one_delivery_can_claim_an_analysis(client):
    analysis_id = uuid.UUID(create_analysis(client))
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(_claim_analysis, [analysis_id] * 4))
    assert claims.count(True) == 1
    run_analysis(str(analysis_id))
    assert (
        client.get(f"/api/v1/analyses/{analysis_id}").json()["status"]
        == "preprocessing"
    )


def test_queued_cancellation_is_terminal_without_worker(client, monkeypatch):
    analysis_id = create_analysis(client)
    assert client.post(f"/api/v1/analyses/{analysis_id}/cancel").status_code == 200
    monkeypatch.setattr(
        "app.tasks.analysis.get_storage",
        lambda: pytest.fail("cancelled task must not read media"),
    )
    run_analysis(analysis_id)
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "cancelled"
    assert report["progress"] == 100
    assert all(run["status"] == "cancelled" for run in report["detector_runs"])


def test_running_cancellation_prevents_late_single_model_result(
    client, monkeypatch
):
    from conftest import fake_get_detector

    analysis_id = create_analysis(client)
    calls = []

    def factory(media_type, spec, *args):
        detector = fake_get_detector(media_type, spec)
        original = detector.predict

        def predict(path, **kwargs):
            calls.append(spec.detector_id)
            result = original(path, **kwargs)
            if len(calls) == 1:
                assert (
                    client.post(f"/api/v1/analyses/{analysis_id}/cancel").status_code
                    == 200
                )
            return result

        detector.predict = predict
        return detector

    monkeypatch.setattr("app.tasks.analysis.get_detector", factory)
    run_analysis(analysis_id)
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "cancelled"
    assert len(calls) == 1
    assert report["detector_runs"][0]["status"] == "cancelled"
    assert report["verdict"] is None


def test_single_model_timeout_marks_task_failed(client, monkeypatch):
    def factory(*args, **kwargs):
        raise InferenceTimedOut("model timeout")
    analysis_id = create_analysis(client)
    monkeypatch.setattr("app.tasks.analysis.get_detector", factory)
    run_analysis(analysis_id)
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "failed"
    assert report["verdict"] is None
    assert len(report["detector_runs"]) == 1
    assert report["detector_runs"][0]["status"] == "timed_out"


def test_soft_time_limit_finishes_all_pending_runs(client, monkeypatch):
    analysis_id = create_analysis(client)

    def timed_out(*args, **kwargs):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr("app.tasks.analysis.prepare_detection_media", timed_out)
    run_analysis(analysis_id)
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "failed"
    assert report["progress"] == 100
    assert report["summary"] == "分析任务执行超时"
    assert all(run["status"] == "timed_out" for run in report["detector_runs"])


def test_dispatcher_finalizes_worker_lost_after_deadline(client):
    analysis_id = uuid.UUID(create_analysis(client))
    assert _claim_analysis(analysis_id)
    with SessionLocal() as db:
        analysis = db.get(Analysis, analysis_id)
        analysis.execution_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
        first = analysis.detector_runs[0]
        first.status = DetectorStatus.SUCCEEDED
        first.score = 0.2
        db.commit()
    dispatch_pending_analyses()
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "failed"
    assert report["detector_runs"][0]["status"] == "succeeded"
    assert all(run["status"] == "timed_out" for run in report["detector_runs"][1:])
    run_analysis(str(analysis_id))
    assert client.get(f"/api/v1/analyses/{analysis_id}").json() == report


def test_deleted_analysis_delivery_is_a_noop(client):
    run_analysis(str(uuid.uuid4()))


def test_cancellation_kills_and_reaps_media_subprocess(tmp_path):
    pid_file = tmp_path / "pid"

    def check():
        if pid_file.exists():
            raise InferenceCancelled("cancelled")

    with pytest.raises(InferenceCancelled):
        run_command(
            [
                sys.executable,
                "-c",
                "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
                str(pid_file),
            ],
            check=check,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_media_subprocess_has_a_real_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.05)


def test_control_checks_cancellation_and_deadline():
    with pytest.raises(InferenceCancelled):
        InferenceControl(60, lambda: True).check()
    with pytest.raises(InferenceTimedOut):
        InferenceControl(0).check()


def test_execution_deadline_migration_preserves_inflight_records(tmp_path):
    import sqlite3
    from pathlib import Path

    database = tmp_path / "migration.db"
    environment = {**os.environ, "DATABASE_URL": f"sqlite:///{database}"}
    service_root = Path(__file__).resolve().parents[1]

    def migrate(revision):
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", revision],
            cwd=service_root,
            env=environment,
            check=True,
            capture_output=True,
        )

    migrate("20260722_0001")
    with sqlite3.connect(database) as db:
        db.execute(
            "INSERT INTO analyses (id, upload_id, user_id, status, progress, cancel_requested, created_at) VALUES (?, ?, ?, 'running', 20, 0, CURRENT_TIMESTAMP)",
            [uuid.uuid4().hex for _ in range(3)],
        )
    migrate("head")
    with sqlite3.connect(database) as db:
        status, deadline = db.execute(
            "SELECT status, execution_deadline FROM analyses"
        ).fetchone()
        assert status == "running"
        assert deadline is not None
    subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "20260722_0001"],
        cwd=service_root,
        env=environment,
        check=True,
        capture_output=True,
    )
    migrate("head")
