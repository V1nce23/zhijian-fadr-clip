import logging
import tempfile
import uuid
from pathlib import Path

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models import Upload, UploadStatus
from app.services.media import normalize_media
from app.services.previews import preview_key, publish_video_preview, set_preview_state
from app.services.storage import get_storage

logger = logging.getLogger(__name__)


@celery_app.task(name="deeptrace.prepare_video_preview", soft_time_limit=600, time_limit=630)
def prepare_video_preview(upload_id: str):
    try:
        with SessionLocal() as db:
            upload = db.get(Upload, uuid.UUID(upload_id))
            if not upload or upload.status == UploadStatus.DELETED or upload.media_type != "video":
                return
            object_key = upload.object_key
        storage = get_storage()
        if storage.exists(preview_key(upload_id)):
            return
        set_preview_state(upload_id, "pending")
        with tempfile.TemporaryDirectory(prefix="deeptrace-preview-") as directory:
            workdir = Path(directory)
            source = workdir / "original"
            storage.download(object_key, source)
            normalized = normalize_media(source, "video", workdir)
            publish_video_preview(upload_id, normalized)
    except Exception:
        logger.exception("视频预览生成失败")
        try:
            set_preview_state(upload_id, "failed", "视频预览生成失败，可重试或下载原文件。")
        except Exception:
            logger.warning("视频预览状态暂时无法保存", exc_info=True)
