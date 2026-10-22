import asyncio
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import magic
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models import Analysis, AnalysisStatus, MediaType, Upload, UploadStatus, User, UserRole
from app.schemas import MessageResponse, UploadPublic
from app.services.evidence_gallery import gallery_prefix
from app.services.storage import ObjectStorage, get_storage
from app.services.previews import preview_key, request_preview, state_key, preview_registry


router = APIRouter(prefix="/uploads", tags=["上传"])
CHUNK_SIZE = 1024 * 1024


_EXTENSION_TYPES = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic", ".heif"},
    "video": {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".wmv", ".mpg", ".mpeg", ".ogv"},
}


def _media_type_for_mime(mime_type: str, filename: str) -> MediaType | None:
    if mime_type.startswith("image/"):
        return MediaType.IMAGE
    if mime_type.startswith("video/"):
        return MediaType.VIDEO
    suffix = _safe_suffix(filename)
    if mime_type in {"application/ogg", "application/octet-stream", "application/x-matroska"}:
        for media_type, extensions in _EXTENSION_TYPES.items():
            if suffix in extensions:
                return MediaType(media_type)
    return None


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) else ""


def _can_access(upload: Upload, user: User) -> bool:
    return user.role == UserRole.ADMIN or upload.user_id == user.id


@router.post("", response_model=UploadPublic, status_code=status.HTTP_201_CREATED)
async def create_upload(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_storage),
) -> Upload:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少文件名")

    digest = hashlib.sha256()
    size = 0
    header = b""
    while chunk := await file.read(CHUNK_SIZE):
        if not header:
            header = chunk[:8192]
        size += len(chunk)
        if size > settings.max_upload_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="文件超过大小限制")
        digest.update(chunk)

    if size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能上传空文件")

    detected_mime = magic.from_buffer(header, mime=True)
    media_type = _media_type_for_mime(detected_mime, file.filename)
    if not media_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"目前仅支持图片或视频文件，不支持的文件类型：{detected_mime}",
        )

    upload_id = uuid.uuid4()
    object_key = f"uploads/{upload_id}{_safe_suffix(file.filename)}"
    await file.seek(0)
    await asyncio.to_thread(storage.upload, object_key, file.file, detected_mime)

    upload = Upload(
        id=upload_id,
        user_id=user.id,
        original_name=file.filename[:255],
        mime_type=detected_mime,
        media_type=media_type,
        size_bytes=size,
        sha256=digest.hexdigest(),
        object_key=object_key,
        status=UploadStatus.VALIDATED,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


@router.get("/{upload_id}/download-url")
def get_download_url(
    upload_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_storage),
) -> dict[str, str]:
    upload = db.get(Upload, upload_id)
    if not upload or upload.status == UploadStatus.DELETED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文件不存在")
    if not _can_access(upload, user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该文件")
    return {"url": storage.download_url(upload.object_key)}


def _video_preview(upload_id, user, db, storage, retry=False):
    upload = db.get(Upload, upload_id)
    if not upload or upload.status == UploadStatus.DELETED:
        raise HTTPException(status_code=404, detail="文件不存在")
    if not _can_access(upload, user):
        raise HTTPException(status_code=403, detail="无权访问该文件")
    if upload.media_type != "video":
        raise HTTPException(status_code=415, detail="仅视频文件需要兼容预览")
    try:
        return request_preview(upload, storage, retry=retry)
    except Exception:
        raise HTTPException(status_code=503, detail="视频预览服务暂时不可用，请稍后重试")


@router.get("/{upload_id}/preview-url")
def get_preview_url(upload_id: uuid.UUID, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db), storage: ObjectStorage = Depends(get_storage)):
    return _video_preview(upload_id, user, db, storage)


@router.post("/{upload_id}/preview-url")
def retry_preview(upload_id: uuid.UUID, user: User = Depends(get_current_user),
                  db: Session = Depends(get_db), storage: ObjectStorage = Depends(get_storage)):
    return _video_preview(upload_id, user, db, storage, retry=True)


@router.delete("/{upload_id}", response_model=MessageResponse)
async def delete_upload(
    upload_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_storage),
) -> MessageResponse:
    upload = db.get(Upload, upload_id)
    if not upload or upload.status == UploadStatus.DELETED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文件不存在")
    if not _can_access(upload, user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权删除该文件")

    running = db.scalar(
        select(Analysis).where(
            Analysis.upload_id == upload.id,
            Analysis.status.in_(
                [
                    AnalysisStatus.CREATED,
                    AnalysisStatus.QUEUED,
                    AnalysisStatus.PREPROCESSING,
                    AnalysisStatus.RUNNING,
                    AnalysisStatus.AGGREGATING,
                ]
            ),
        )
    )
    if running:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="文件仍有分析任务正在运行")

    await asyncio.to_thread(storage.delete, upload.object_key)
    upload.status = UploadStatus.DELETED
    upload.deleted_at = datetime.now(timezone.utc)
    db.commit()
    await asyncio.to_thread(storage.delete, preview_key(upload_id))
    for identity in db.scalars(select(Analysis.id).where(Analysis.upload_id == upload_id)):
        await asyncio.to_thread(storage.delete_prefix, gallery_prefix(identity))
    try:
        preview_registry().delete(state_key(upload_id))
    except Exception:
        pass
    return MessageResponse(message="文件已删除")
