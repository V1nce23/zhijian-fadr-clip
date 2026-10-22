"""Persistent browser-compatible video previews; original uploads stay unchanged."""

import json
import uuid
from functools import lru_cache
from pathlib import Path

from redis import Redis
from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Upload, UploadStatus
from app.services.storage import get_storage

PREVIEW_VERSION = "h264-v1"
PENDING_TTL = 900


def preview_key(upload_id) -> str:
    return f"previews/{upload_id}/{PREVIEW_VERSION}.mp4"


def state_key(upload_id) -> str:
    return f"deeptrace:preview:{PREVIEW_VERSION}:{upload_id}"


@lru_cache
def preview_registry():
    return Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)


def set_preview_state(upload_id, status: str, detail: str | None = None):
    value = {"status": status}
    if detail:
        value["detail"] = detail
    preview_registry().set(state_key(upload_id), json.dumps(value), ex=PENDING_TTL)


def request_preview(upload: Upload, storage, *, retry: bool = False) -> dict:
    key = preview_key(upload.id)
    # Persistent objects take precedence over transient queue/Redis state.
    if storage.exists(key):
        return {"status": "ready", "url": storage.download_url(key), "mime_type": "video/mp4"}
    registry = preview_registry()
    raw = registry.get(state_key(upload.id))
    state = json.loads(raw) if raw else None
    if state and state.get("status") == "failed" and retry:
        registry.delete(state_key(upload.id))
        state = None
    if state:
        return state
    if registry.set(state_key(upload.id), json.dumps({"status": "pending"}), nx=True, ex=PENDING_TTL):
        from app.tasks.preview import prepare_video_preview
        try:
            prepare_video_preview.apply_async(args=[str(upload.id)], queue=settings.detector_queue)
        except Exception:
            registry.delete(state_key(upload.id))
            raise
    return {"status": "pending"}


def publish_video_preview(upload_id, path: Path) -> bool:
    """Publish an already normalized H.264 MP4, coordinating with upload deletion."""
    identity = uuid.UUID(str(upload_id))
    with SessionLocal() as db:
        upload = db.scalar(select(Upload).where(Upload.id == identity).with_for_update())
        if not upload or upload.status == UploadStatus.DELETED:
            return False
        with path.open("rb") as stream:
            get_storage().upload(preview_key(identity), stream, "video/mp4")
        db.commit()
    return True
