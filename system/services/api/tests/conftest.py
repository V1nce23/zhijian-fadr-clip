import os
import hashlib
from collections.abc import Generator

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "sqlite:////tmp/deeptrace-test.db"
)
os.environ["S3_ENDPOINT_URL"] = "http://storage.invalid"
os.environ["DETECTOR_QUEUE"] = "celery"

import pytest
from fastapi.testclient import TestClient

from app.detectors.base import DetectionResult, DetectorSpec, verdict_from_score
from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.main import app
from app.models import User, UserRole
from app.services.storage import get_storage


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def ensure_bucket(self) -> None:
        return None

    def upload(self, key: str, file_obj, content_type: str) -> None:
        self.objects[key] = file_obj.read()

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def delete_prefix(self, prefix):
        for key in list(self.objects):
            if key.startswith(prefix):
                del self.objects[key]

    def read_json(self, key):
        import json
        return json.loads(self.objects[key])

    def exists(self, key: str) -> bool:
        return key in self.objects

    def download(self, key: str, destination) -> None:
        destination.write_bytes(self.objects[key])

    def download_url(self, key: str, expires_seconds: int = 300) -> str:
        return f"http://storage.invalid/{key}"


class FakeDetector:
    """Deterministic stand-in used by workflow tests; real inference is offline."""

    def __init__(self, media_type: str, spec: DetectorSpec, **kwargs) -> None:
        self.media_type = media_type
        self.spec = spec
        digest = hashlib.sha256(spec.detector_id.encode()).digest()
        self._score = round(int.from_bytes(digest[:4], "big") / 0xFFFFFFFF, 6)

    def predict(self, path, *, sha256: str = "", check=None) -> DetectionResult:
        if check:
            check()
        score = self._score
        if self.media_type == "image":
            evidence = {
                "kind": "image_regions",
                "regions": [
                    {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "score": score}
                ],
            }
        elif self.media_type == "video":
            evidence = {
                "kind": "video_segments",
                "segments": [
                    {"start_seconds": 0.0, "end_seconds": 1.0, "score": score}
                ],
            }
        else:
            raise ValueError("unsupported test media type")
        return DetectionResult(
            score=score,
            verdict=verdict_from_score(score),
            evidence=evidence,
        )


def fake_get_detector(
    media_type: str, spec: DetectorSpec, *args, **kwargs
) -> FakeDetector:
    return FakeDetector(media_type, spec, **kwargs)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        db.add(
            User(
                email="admin@example.com",
                name="测试管理员",
                password_hash=hash_password("StrongPass123!"),
                role=UserRole.ADMIN,
            )
        )
        db.commit()

    storage = MemoryStorage()
    app.dependency_overrides[get_storage] = lambda: storage
    monkeypatch.setattr("app.main.get_storage", lambda: storage)
    monkeypatch.setattr("app.tasks.analysis.get_storage", lambda: storage)
    monkeypatch.setattr("app.tasks.preview.get_storage", lambda: storage)
    monkeypatch.setattr("app.tasks.evidence_gallery.get_storage", lambda: storage)
    monkeypatch.setattr("app.services.evidence_gallery.get_storage", lambda: storage)
    monkeypatch.setattr("app.services.attention_gallery.get_storage", lambda: storage)
    monkeypatch.setattr("app.tasks.attention_gallery.get_storage", lambda: storage)
    monkeypatch.setattr("app.services.live_visualization.get_storage", lambda: storage)
    monkeypatch.setattr("app.api.routes.analyses.get_storage", lambda: storage)
    monkeypatch.setattr("app.services.previews.get_storage", lambda: storage)
    preview_state = {}

    class PreviewRegistry:
        def get(self, key):
            return preview_state.get(key)

        def set(self, key, value, *, nx=False, ex=None):
            if nx and key in preview_state:
                return False
            preview_state[key] = value
            return True

        def delete(self, key):
            preview_state.pop(key, None)

    monkeypatch.setattr("app.services.previews.preview_registry", lambda: PreviewRegistry())
    monkeypatch.setattr("app.api.routes.uploads.preview_registry", lambda: PreviewRegistry())
    monkeypatch.setattr("app.tasks.preview.prepare_video_preview.apply_async", lambda **kwargs: None)
    monkeypatch.setattr("app.tasks.evidence_gallery.prepare_evidence_gallery.apply_async", lambda **kwargs: None)
    monkeypatch.setattr("app.tasks.attention_gallery.prepare_attention_gallery.apply_async", lambda **kwargs: None)
    monkeypatch.setattr(
        "app.dispatcher.run_analysis.apply_async", lambda **kwargs: None
    )
    monkeypatch.setattr("app.tasks.analysis.get_detector", fake_get_detector)
    monkeypatch.setattr(
        "app.api.routes.analyses.get_detection_devices",
        lambda: {"worker_ready": True, "cpu": True, "gpu": True, "gpu_name": "Test GPU"},
    )
    # Workflow fixtures contain only enough header bytes for upload classification;
    # actual decoding is covered by test_media.py.
    monkeypatch.setattr(
        "app.tasks.analysis.prepare_detection_media",
        lambda source, media_type, directory, **kwargs: source,
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
