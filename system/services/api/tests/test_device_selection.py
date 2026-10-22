"""Per-analysis device choices must survive dispatch and control real inference."""

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from app.services import worker_devices
from app.tasks.analysis import run_analysis
from test_workflow import PNG_BYTES, login


def upload_image(client):
    login(client)
    response = client.post(
        "/api/v1/uploads", files={"file": ("face.png", PNG_BYTES, "image/png")}
    )
    assert response.status_code == 201
    return response.json()["id"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_is_persisted_and_controls_each_detector(client, monkeypatch, device):
    from conftest import fake_get_detector

    upload_id = upload_image(client)
    monkeypatch.setattr("app.tasks.analysis.resolve_execution_device", lambda value: value)
    used = []

    def factory(media_type, spec, models_root, selected, *args):
        used.append(selected)
        return fake_get_detector(media_type, spec)

    monkeypatch.setattr("app.tasks.analysis.get_detector", factory)
    response = client.post("/api/v1/analyses", json={"upload_id": upload_id, "device": device})
    assert response.status_code == 201
    created = response.json()
    assert created["requested_device"] == device
    assert created["executed_device"] is None
    run_analysis(created["id"])
    report = client.get(f"/api/v1/analyses/{created['id']}").json()
    assert report["status"] == "completed"
    assert report["executed_device"] == device
    assert used == [device]
    assert all(run["evidence"]["device"] == device for run in report["detector_runs"])
    assert client.get("/api/v1/analyses").json()[0]["executed_device"] == device


def test_gpu_unavailable_rejects_gpu_but_accepts_cpu(client, monkeypatch):
    upload_id = upload_image(client)
    monkeypatch.setattr(
        "app.api.routes.analyses.get_detection_devices",
        lambda: {"worker_ready": True, "cpu": True, "gpu": False, "gpu_name": None},
    )
    assert client.post("/api/v1/analyses", json={"upload_id": upload_id, "device": "cuda"}).status_code == 409
    assert client.get("/api/v1/analyses").json() == []
    assert client.post("/api/v1/analyses", json={"upload_id": upload_id, "device": "cpu"}).status_code == 201


def test_device_endpoint_requires_login_and_rejects_invalid_device(client):
    assert client.get("/api/v1/analyses/devices").status_code == 401
    upload_id = upload_image(client)
    assert client.get("/api/v1/analyses/devices").json()["gpu"] is True
    assert client.post("/api/v1/analyses", json={"upload_id": upload_id, "device": "bogus"}).status_code == 422


def test_offline_worker_rejects_new_tasks(client, monkeypatch):
    upload_id = upload_image(client)
    monkeypatch.setattr(
        "app.api.routes.analyses.get_detection_devices",
        lambda: {"worker_ready": False, "cpu": False, "gpu": False, "gpu_name": None},
    )
    assert client.post("/api/v1/analyses", json={"upload_id": upload_id, "device": "cpu"}).status_code == 503


def test_lost_gpu_fails_without_silent_cpu_fallback(client, monkeypatch):
    upload_id = upload_image(client)
    created = client.post("/api/v1/analyses", json={"upload_id": upload_id, "device": "cuda"}).json()

    def unavailable(device):
        raise RuntimeError("所选 GPU 当前不可用")

    monkeypatch.setattr("app.tasks.analysis.resolve_execution_device", unavailable)
    monkeypatch.setattr("app.tasks.analysis.get_detector", lambda *args: pytest.fail("must not fall back"))
    run_analysis(created["id"])
    report = client.get(f"/api/v1/analyses/{created['id']}").json()
    assert report["status"] == "failed"
    assert report["requested_device"] == "cuda"
    assert report["executed_device"] is None


def test_mixed_worker_pool_does_not_advertise_gpu(monkeypatch):
    class Registry:
        def scan_iter(self, **kwargs):
            return [b"gpu-worker", b"cpu-worker", b"expired-worker"]

        def mget(self, keys):
            return [json.dumps({"cpu": True, "gpu": True, "gpu_name": "RTX"}), json.dumps({"cpu": True, "gpu": False}), None]

    monkeypatch.setattr(worker_devices, "device_registry", lambda: Registry())
    assert worker_devices.get_detection_devices() == {"worker_ready": True, "cpu": True, "gpu": False, "gpu_name": None}


def test_device_migration_preserves_legacy_records(tmp_path):
    database = tmp_path / "devices.db"
    environment = {**os.environ, "DATABASE_URL": f"sqlite:///{database}"}
    service_root = Path(__file__).resolve().parents[1]

    def migrate(target):
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", target], cwd=service_root, env=environment, check=True, capture_output=True)

    migrate("20261013_0002")
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO analyses (id, upload_id, user_id, status, progress, cancel_requested, created_at) VALUES (?, ?, ?, 'completed', 100, 0, CURRENT_TIMESTAMP)", [uuid.uuid4().hex for _ in range(3)])
    migrate("head")
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT status, requested_device, executed_device FROM analyses").fetchone() == ("completed", None, None)
