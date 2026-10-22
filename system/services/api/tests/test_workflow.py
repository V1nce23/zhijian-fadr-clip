import base64
import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.tasks.analysis import run_analysis


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
)
WEBM_BYTES = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAHs"
    "EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHYTbuMU6uEElTDZ1OsggEl"
    "TbuMU6uEHFO7a1OsggHW7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAVSalmsirXsYMPQkBNgI1MYXZmNjIuMTIuMTAxV0GN"
    "TGF2ZjYyLjEyLjEwMUSJiEBpAAAAAAAAFlSua8iuAQAAAAAAAD/XgQFzxYgenBbv"
    "0WygSpyBACK1nIN1bmSIgQCGhVZfVlA5g4EBI+ODhAvrwgDgkLCBILqBIJqBAlWw"
    "hFW5gQESVMNnQIBzc6BjwIBnyJpFo4dFTkNPREVSRIeNTGF2ZjYyLjEyLjEwMXNz"
    "2mPAi2PFiB6cFu/RbKBKZ8ilRaOHRU5DT0RFUkSHmExhdmM2Mi4yOC4xMDEgbGli"
    "dnB4LXZwOWfIoUWjiERVUkFUSU9ORIeTMDA6MDA6MDAuMjAwMDAwMDAwAB9DtnWm"
    "54EAo6GBAACAgkmDQgAB8AH2ADgkHBhKAAAwYAAAEN///VkVgAAcU7trkbuPs4EA"
    "t4r3gQHxggGr8IED"
)


def silent_wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\x00\x00" * 800)
    return output.getvalue()


def login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "StrongPass123!"},
    )
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_requires_authentication(client: TestClient) -> None:
    response = client.get("/api/v1/analyses")
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("filename", "content", "declared_mime", "expected_type", "evidence_kind"),
    [
        ("sample.png", PNG_BYTES, "image/png", "image", "image_regions"),
        ("sample.webm", WEBM_BYTES, "video/webm", "video", "video_segments"),
    ],
)
def test_complete_upload_and_analysis_workflow(
    client: TestClient,
    filename: str,
    content: bytes,
    declared_mime: str,
    expected_type: str,
    evidence_kind: str,
) -> None:
    login(client)

    upload_response = client.post(
        "/api/v1/uploads",
        files={"file": (filename, content, declared_mime)},
    )
    assert upload_response.status_code == 201
    upload = upload_response.json()
    assert upload["media_type"] == expected_type
    assert upload["size_bytes"] == len(content)

    analysis_response = client.post(
        "/api/v1/analyses",
        json={"upload_id": upload["id"]},
    )
    assert analysis_response.status_code == 201
    analysis_id = analysis_response.json()["id"]
    assert analysis_response.json()["status"] == "queued"

    run_analysis(analysis_id)

    report_response = client.get(f"/api/v1/analyses/{analysis_id}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["status"] == "completed"
    assert report["progress"] == 100
    assert report["verdict"] in {
        "likely_manipulated",
        "uncertain",
        "likely_authentic",
    }
    expected_checkers = 1
    assert len(report["detector_runs"]) == expected_checkers
    assert [run["detector_id"] for run in report["detector_runs"]] == ["fadr-clip"]
    for run in report["detector_runs"]:
        assert run["status"] == "succeeded"
        assert run["progress"] == 100
        assert run["verdict"] in {
            "likely_manipulated",
            "uncertain",
            "likely_authentic",
        }
    assert report["detector_runs"][0]["evidence"]["kind"] == evidence_kind
    assert report["score"] == round(
        sum(run["score"] for run in report["detector_runs"]) / expected_checkers, 6
    )

    history_response = client.get("/api/v1/analyses")
    assert history_response.status_code == 200
    assert [item["id"] for item in history_response.json()] == [analysis_id]

    delete_response = client.delete(f"/api/v1/uploads/{upload['id']}")
    assert delete_response.status_code == 200


def _create_image_analysis(client: TestClient) -> tuple[str, str]:
    """Upload an image and create a completed analysis; returns (upload_id, analysis_id)."""
    login(client)
    upload_response = client.post(
        "/api/v1/uploads",
        files={"file": ("sample.png", PNG_BYTES, "image/png")},
    )
    assert upload_response.status_code == 201
    upload_id = upload_response.json()["id"]
    analysis_response = client.post(
        "/api/v1/analyses",
        json={"upload_id": upload_id},
    )
    assert analysis_response.status_code == 201
    analysis_id = analysis_response.json()["id"]
    run_analysis(analysis_id)
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "completed"
    return upload_id, analysis_id


def test_delete_analysis_record_keeps_upload(client: TestClient) -> None:
    upload_id, analysis_id = _create_image_analysis(client)

    delete_response = client.delete(f"/api/v1/analyses/{analysis_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "分析记录已删除"

    # The analysis record (and its runs/events) is gone…
    assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 404
    history_response = client.get("/api/v1/analyses")
    assert history_response.status_code == 200
    assert [item["id"] for item in history_response.json()] == []

    # …but the uploaded file is kept and still downloadable.
    download_response = client.get(f"/api/v1/uploads/{upload_id}/download-url")
    assert download_response.status_code == 200
    assert download_response.json()["url"]


@pytest.mark.parametrize("state", ["created", "queued", "preprocessing", "running", "aggregating"])
def test_delete_unfinished_analysis_stops_deliveries_and_keeps_upload(client: TestClient, state) -> None:
    import uuid
    from app.core.database import SessionLocal
    from app.models import Analysis, AnalysisEvent, DetectorRun
    from app.tasks.analysis import _is_cancelled

    login(client)
    upload_response = client.post(
        "/api/v1/uploads",
        files={"file": ("sample.png", PNG_BYTES, "image/png")},
    )
    upload_id = upload_response.json()["id"]
    analysis_response = client.post(
        "/api/v1/analyses",
        json={"upload_id": upload_id},
    )
    analysis_id = analysis_response.json()["id"]
    assert analysis_response.json()["status"] == "queued"
    with SessionLocal() as db:
        analysis = db.get(Analysis, uuid.UUID(analysis_id))
        analysis.status = state
        db.commit()

    delete_response = client.delete(f"/api/v1/analyses/{analysis_id}")
    assert delete_response.status_code == 200

    assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 404
    assert _is_cancelled(uuid.UUID(analysis_id))
    # An already queued delivery must not resurrect the deleted record.
    run_analysis(analysis_id)
    with SessionLocal() as db:
        assert db.get(Analysis, uuid.UUID(analysis_id)) is None
        assert db.query(DetectorRun).filter_by(analysis_id=uuid.UUID(analysis_id)).count() == 0
        assert db.query(AnalysisEvent).filter_by(analysis_id=uuid.UUID(analysis_id)).count() == 0
    assert client.get(f"/api/v1/uploads/{upload_id}/download-url").status_code == 200


def test_analyst_cannot_delete_other_users_analysis(client: TestClient) -> None:
    upload_id, analysis_id = _create_image_analysis(client)

    login(client)
    client.post(
        "/api/v1/users",
        json={
            "email": "analyst@example.com",
            "name": "分析员",
            "password": "AnalystPass123!",
            "role": "analyst",
        },
    )
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@example.com", "password": "AnalystPass123!"},
    )
    assert login_response.status_code == 200

    # The analyst cannot see the admin's record, so delete is a 404.
    delete_response = client.delete(f"/api/v1/analyses/{analysis_id}")
    assert delete_response.status_code == 404
    # Admin record is untouched.
    login(client)
    assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 200
    # Cleanup: admin deletes the analysis record, upload stays for later cleanup.
    assert client.delete(f"/api/v1/analyses/{analysis_id}").status_code == 200
    assert client.delete(f"/api/v1/uploads/{upload_id}").status_code == 200


def test_batch_delete_analysis_records(client: TestClient) -> None:
    pairs = [_create_image_analysis(client) for _ in range(3)]
    upload_ids = [pair[0] for pair in pairs]
    ids = [pair[1] for pair in pairs]

    delete_response = client.post(
        "/api/v1/analyses/batch-delete",
        json={"analysis_ids": [ids[0], ids[1]]},
    )
    assert delete_response.status_code == 200
    body = delete_response.json()
    assert body["deleted"] == 2
    assert body["skipped"] == 0

    assert client.get(f"/api/v1/analyses/{ids[0]}").status_code == 404
    assert client.get(f"/api/v1/analyses/{ids[1]}").status_code == 404
    assert client.get(f"/api/v1/analyses/{ids[2]}").status_code == 200
    # Source files of deleted records are kept.
    assert (
        client.get(f"/api/v1/uploads/{upload_ids[0]}/download-url").status_code == 200
    )
    assert (
        client.get(f"/api/v1/uploads/{upload_ids[1]}/download-url").status_code == 200
    )


@pytest.mark.parametrize("state", ["queued", "preprocessing", "running", "aggregating"])
def test_batch_delete_includes_unfinished_records(client: TestClient, state) -> None:
    import uuid
    from app.core.database import SessionLocal
    from app.models import Analysis

    login(client)
    upload_response = client.post(
        "/api/v1/uploads",
        files={"file": ("sample.png", PNG_BYTES, "image/png")},
    )
    running_upload = upload_response.json()["id"]
    running_response = client.post(
        "/api/v1/analyses",
        json={"upload_id": running_upload},
    )
    running_id = running_response.json()["id"]
    with SessionLocal() as db:
        analysis = db.get(Analysis, uuid.UUID(running_id))
        analysis.status = state
        db.commit()
    completed_id = _create_image_analysis(client)[1]

    delete_response = client.post(
        "/api/v1/analyses/batch-delete",
        json={"analysis_ids": [running_id, completed_id, running_id]},
    )
    assert delete_response.status_code == 200
    body = delete_response.json()
    assert body["deleted"] == 2
    assert body["skipped"] == 0

    assert client.get(f"/api/v1/analyses/{running_id}").status_code == 404
    assert client.get(f"/api/v1/analyses/{completed_id}").status_code == 404
    assert client.get(f"/api/v1/uploads/{running_upload}/download-url").status_code == 200
    run_analysis(running_id)
    assert client.get(f"/api/v1/analyses/{running_id}").status_code == 404


def test_batch_delete_ignores_foreign_records_for_analyst(client: TestClient) -> None:
    admin_upload, admin_analysis = _create_image_analysis(client)

    login(client)
    client.post(
        "/api/v1/users",
        json={
            "email": "analyst@example.com",
            "name": "分析员",
            "password": "AnalystPass123!",
            "role": "analyst",
        },
    )
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@example.com", "password": "AnalystPass123!"},
    )
    assert login_response.status_code == 200
    analyst_upload = client.post(
        "/api/v1/uploads",
        files={"file": ("sample.png", PNG_BYTES, "image/png")},
    ).json()["id"]
    analyst_analysis = client.post(
        "/api/v1/analyses",
        json={"upload_id": analyst_upload},
    ).json()["id"]
    run_analysis(analyst_analysis)

    delete_response = client.post(
        "/api/v1/analyses/batch-delete",
        json={"analysis_ids": [admin_analysis, analyst_analysis]},
    )
    assert delete_response.status_code == 200
    body = delete_response.json()
    # The admin's record is not visible to the analyst, so it is skipped.
    assert body["deleted"] == 1
    assert body["skipped"] == 1

    login(client)
    assert client.get(f"/api/v1/analyses/{admin_analysis}").status_code == 200
    assert client.get(f"/api/v1/analyses/{analyst_analysis}").status_code == 404
    assert client.delete(f"/api/v1/analyses/{admin_analysis}").status_code == 200
    assert client.delete(f"/api/v1/uploads/{admin_upload}").status_code == 200


def test_aggregate_verdict_majority_and_tie() -> None:
    from app.services.results import aggregate_verdict
    from app.models import Verdict

    assert aggregate_verdict([]) == Verdict.UNCERTAIN
    assert (
        aggregate_verdict(
            [
                Verdict.LIKELY_MANIPULATED,
                Verdict.LIKELY_MANIPULATED,
                Verdict.LIKELY_AUTHENTIC,
            ]
        )
        == Verdict.LIKELY_MANIPULATED
    )
    assert (
        aggregate_verdict(
            [Verdict.LIKELY_MANIPULATED, Verdict.LIKELY_AUTHENTIC, Verdict.UNCERTAIN]
        )
        == Verdict.UNCERTAIN
    )
    assert (
        aggregate_verdict(
            [Verdict.LIKELY_AUTHENTIC, Verdict.LIKELY_AUTHENTIC, Verdict.UNCERTAIN]
        )
        == Verdict.LIKELY_AUTHENTIC
    )


def test_single_model_failure_is_reported_without_a_fake_result(client, monkeypatch):
    def broken_detector(*args, **kwargs):
        raise RuntimeError("FADR weights unavailable")
    monkeypatch.setattr("app.tasks.analysis.get_detector", broken_detector)
    login(client)
    upload = client.post("/api/v1/uploads", files={"file": ("sample.png", PNG_BYTES, "image/png")}).json()
    analysis_id = client.post("/api/v1/analyses", json={"upload_id": upload["id"]}).json()["id"]
    run_analysis(analysis_id)
    report = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert report["status"] == "failed"
    assert report["verdict"] is None
    assert len(report["detector_runs"]) == 1
    assert report["detector_runs"][0]["error"] == "FADR weights unavailable"


def test_admin_can_create_analyst(client: TestClient) -> None:
    login(client)
    response = client.post(
        "/api/v1/users",
        json={
            "email": "analyst@example.com",
            "name": "分析员",
            "password": "AnalystPass123!",
            "role": "analyst",
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == "analyst"


@pytest.mark.parametrize("filename,declared_mime", [
    ("sample.wav", "audio/wav"),
    ("renamed.png", "image/png"),
    ("renamed.mp4", "video/mp4"),
    ("sample.mp3", "application/octet-stream"),
])
def test_audio_upload_is_rejected_even_when_disguised(client, filename, declared_mime):
    login(client)
    response = client.post("/api/v1/uploads", files={"file": (filename, silent_wav_bytes(), declared_mime)})
    assert response.status_code == 415
    assert "仅支持图片或视频" in response.json()["detail"]
    assert client.get("/api/v1/analyses").json() == []


def test_legacy_audio_upload_cannot_start_new_analysis(client):
    from app.core.database import SessionLocal
    from app.models import Upload, MediaType
    import uuid

    login(client)
    upload = client.post("/api/v1/uploads", files={"file": ("sample.png", PNG_BYTES, "image/png")}).json()
    with SessionLocal() as db:
        record = db.get(Upload, uuid.UUID(upload["id"]))
        record.media_type = MediaType.AUDIO
        db.commit()
    response = client.post("/api/v1/analyses", json={"upload_id": upload["id"], "device": "cpu"})
    assert response.status_code == 415
    assert "已停用" in response.json()["detail"]
