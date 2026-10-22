import json
import subprocess
import uuid

from app.services import previews
from app.services.previews import preview_key, state_key
from app.services.storage import get_storage
from app.tasks.preview import prepare_video_preview
from app.tasks.analysis import run_analysis
from test_workflow import WEBM_BYTES, PNG_BYTES, login


def upload_video(client, content=WEBM_BYTES, name="clip.webm"):
    login(client)
    response = client.post("/api/v1/uploads", files={"file": (name, content, "video/webm")})
    assert response.status_code == 201
    return response.json()["id"]


def test_legacy_preview_is_generated_once_without_new_analysis(client, monkeypatch):
    identity = upload_video(client)
    jobs = []
    monkeypatch.setattr(prepare_video_preview, "apply_async", lambda **kwargs: jobs.append(kwargs))
    for _ in range(3):
        assert client.get(f"/api/v1/uploads/{identity}/preview-url").json() == {"status": "pending"}
    assert len(jobs) == 1
    assert jobs[0]["args"] == [identity]
    assert client.get("/api/v1/analyses").json() == []


def test_mpeg4_mp4_becomes_browser_h264_and_original_is_unchanged(client, tmp_path):
    source = tmp_path / "mpeg4.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=5:d=0.4", "-c:v", "mpeg4", str(source)], check=True)
    original = source.read_bytes()
    identity = upload_video(client, original, "mpeg4.mp4")
    storage = client.app.dependency_overrides[get_storage]()
    original_key = next(iter(storage.objects))
    prepare_video_preview(identity)
    response = client.get(f"/api/v1/uploads/{identity}/preview-url")
    assert response.json()["status"] == "ready"
    assert response.json()["mime_type"] == "video/mp4"
    assert storage.objects[original_key] == original
    converted = tmp_path / "preview.mp4"
    converted.write_bytes(storage.objects[preview_key(identity)])
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(converted)]))
    assert info["streams"][0]["codec_name"] == "h264"
    assert info["streams"][0]["pix_fmt"] == "yuv420p"
    assert len(info["streams"]) == 1  # Video previews do not reintroduce audio processing.
    data = converted.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # Fast start, including seek/range playback.
    assert client.get("/api/v1/analyses").json() == []
    assert client.delete(f"/api/v1/uploads/{identity}").status_code == 200
    assert preview_key(identity) not in storage.objects
    prepare_video_preview(identity)
    assert not storage.objects
    assert client.get(f"/api/v1/uploads/{identity}/preview-url").status_code == 404


def test_preview_requires_login_and_owner_permission(client):
    assert client.get(f"/api/v1/uploads/{uuid.uuid4()}/preview-url").status_code == 401
    identity = upload_video(client)
    client.post("/api/v1/users", json={"email": "preview@example.com", "name": "Another user", "password": "PreviewPass123!", "role": "analyst"})
    client.post("/api/v1/auth/login", json={"email": "preview@example.com", "password": "PreviewPass123!"})
    assert client.get(f"/api/v1/uploads/{identity}/preview-url").status_code == 403
    assert client.post(f"/api/v1/uploads/{identity}/preview-url").status_code == 403


def test_image_preview_request_is_rejected(client):
    login(client)
    upload = client.post("/api/v1/uploads", files={"file": ("face.png", PNG_BYTES, "image/png")}).json()
    assert client.get(f"/api/v1/uploads/{upload['id']}/preview-url").status_code == 415


def test_failed_preview_can_be_retried(client, monkeypatch):
    identity = upload_video(client)
    previews.preview_registry().set(state_key(identity), json.dumps({"status": "failed", "detail": "temporary failure"}))
    assert client.get(f"/api/v1/uploads/{identity}/preview-url").json()["status"] == "failed"
    jobs = []
    monkeypatch.setattr(prepare_video_preview, "apply_async", lambda **kwargs: jobs.append(kwargs))
    assert client.post(f"/api/v1/uploads/{identity}/preview-url").json()["status"] == "pending"
    assert len(jobs) == 1


def test_detection_does_not_request_preview_conversion(client, monkeypatch):
    identity = upload_video(client)
    def unexpected(*args, **kwargs):
        raise AssertionError("detection must not encode or publish previews")
    monkeypatch.setattr("app.services.media.normalize_media", unexpected)
    monkeypatch.setattr("app.services.previews.publish_video_preview", unexpected)
    analysis = client.post("/api/v1/analyses", json={"upload_id": identity}).json()
    run_analysis(analysis["id"])
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["status"] == "completed"


def test_ready_preview_does_not_require_redis(client, monkeypatch):
    identity = upload_video(client)
    storage = client.app.dependency_overrides[get_storage]()
    storage.objects[preview_key(identity)] = b"cached preview"
    def offline():
        raise ConnectionError("Redis offline")
    monkeypatch.setattr("app.services.previews.preview_registry", offline)
    assert client.get(f"/api/v1/uploads/{identity}/preview-url").json()["status"] == "ready"


def test_bad_video_reports_preview_failure(client):
    identity = upload_video(client, WEBM_BYTES[:35])
    prepare_video_preview(identity)
    response = client.get(f"/api/v1/uploads/{identity}/preview-url").json()
    assert response["status"] == "failed"
    assert "重试" in response["detail"]
