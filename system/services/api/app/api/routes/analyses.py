import asyncio
import json
import uuid
from dataclasses import asdict
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import Response

from app.core.database import SessionLocal, get_db
from app.core.dependencies import get_current_user
from app.detectors.catalog import detector_specs
from app.models import (
    Analysis,
    AnalysisEvent,
    AnalysisStatus,
    DetectorRun,
    DetectorStatus,
    Upload,
    UploadStatus,
    User,
    UserRole,
)
from app.schemas import (
    AnalysisBatchDeleteRequest,
    AnalysisBatchDeleteResult,
    AnalysisCreate,
    AnalysisListItem,
    AnalysisPublic,
    DetectionDevices,
    MessageResponse,
    ThresholdModePublic,
)
from app.services.events import add_analysis_event
from app.dispatcher import publish_analysis
from app.services.analysis_state import TERMINAL_STATUSES, finish_analysis
from app.services.worker_devices import get_detection_devices
from app.services.storage import get_storage, ObjectStorage
from app.services.evidence_gallery import request_gallery, gallery_prefix
from app.services.threshold_policy import MODES, CUSTOM_MODE, make_policy, frame_thresholds


router = APIRouter(prefix="/analyses", tags=["分析"])
TERMINAL = TERMINAL_STATUSES


def _authorized_query(user: User):
    query = select(Analysis)
    if user.role != UserRole.ADMIN:
        query = query.where(Analysis.user_id == user.id)
    return query


def _get_analysis(db: Session, analysis_id: uuid.UUID, user: User, *, for_update=False) -> Analysis:
    query = (
        _authorized_query(user)
        .where(Analysis.id == analysis_id)
        .options(selectinload(Analysis.upload), selectinload(Analysis.detector_runs))
    )
    if for_update:
        query = query.with_for_update(of=Analysis).execution_options(populate_existing=True)
    analysis = db.scalar(query)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="分析任务不存在"
        )
    return analysis


@router.get("/devices", response_model=DetectionDevices)
def detection_devices(user: User = Depends(get_current_user)) -> dict:
    return get_detection_devices()


@router.post("", response_model=AnalysisPublic, status_code=status.HTTP_201_CREATED)
def create_analysis(
    payload: AnalysisCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Analysis:
    upload = db.get(Upload, payload.upload_id)
    if not upload or upload.status != UploadStatus.VALIDATED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="上传文件不存在"
        )
    if user.role != UserRole.ADMIN and upload.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="无权分析该文件"
        )

    if upload.media_type not in ("image", "video"):
        raise HTTPException(status_code=415, detail="目前仅支持图片或视频检测，该历史文件类型已停用")

    devices = get_detection_devices()
    if not devices["worker_ready"]:
        raise HTTPException(status_code=503, detail="检测服务尚未就绪，请稍后重试")
    if payload.device == "cuda" and not devices["gpu"]:
        raise HTTPException(status_code=409, detail="GPU 当前不可用，请选择 CPU 或等待 GPU 恢复")

    analysis = Analysis(
        upload_id=upload.id,
        user_id=user.id,
        status=AnalysisStatus.QUEUED,
        progress=5,
        requested_device=payload.device,
        input_face_mode="raw",
        threshold_policy=make_policy(payload.threshold_mode, upload.media_type,
            payload.custom_thresholds.model_dump() if payload.custom_thresholds else None),
    )
    db.add(analysis)
    db.flush()
    media_type = getattr(upload.media_type, "value", upload.media_type)
    for spec in detector_specs(media_type):
        db.add(
            DetectorRun(
                analysis_id=analysis.id,
                detector_id=spec.detector_id,
                model_version=spec.model_version,
                preprocessing_version=spec.preprocessing_version,
                status=DetectorStatus.QUEUED,
                evidence={"decision_policy": analysis.threshold_policy,
                          "thresholds": frame_thresholds(analysis.threshold_policy)},
            )
        )
    add_analysis_event(
        db, analysis, "analysis.queued", {"message": "分析任务已进入队列", "threshold_policy": analysis.threshold_policy}
    )
    db.commit()
    analysis = _get_analysis(db, analysis.id, user)
    publish_analysis(str(analysis.id))
    return analysis


@router.get("/threshold-modes", response_model=list[ThresholdModePublic])
def threshold_modes(user: User = Depends(get_current_user)) -> list[dict]:
    return [asdict(item) for item in (*MODES, CUSTOM_MODE)]


@router.get("", response_model=list[AnalysisListItem])
def list_analyses(
    limit: int = Query(default=50, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Analysis]:
    query = (
        _authorized_query(user)
        .options(selectinload(Analysis.upload))
        .order_by(Analysis.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(query))


@router.get("/{analysis_id}", response_model=AnalysisPublic)
def get_analysis(
    analysis_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Analysis:
    return _get_analysis(db, analysis_id, user)


@router.get("/{analysis_id}/export")
def export_analysis(analysis_id: uuid.UUID, format: Literal["html", "json", "csv", "pdf", "docx"] = "html",
                    include_technical: bool = False,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db),
                    storage: ObjectStorage = Depends(get_storage)):
    analysis = _get_analysis(db, analysis_id, user)
    from app.services.report_export import export_report
    content, media_type = export_report(analysis, storage, format, include_technical=include_technical)
    return Response(content, media_type=media_type, headers={
        "Content-Disposition": f'attachment; filename="zhijian-report-{analysis.id}.{format}"',
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
    })


def _gallery(analysis_id, run_id, user, db, storage, retry=False):
    analysis = _get_analysis(db, analysis_id, user)
    if analysis.upload.status == UploadStatus.DELETED:
        raise HTTPException(status_code=404, detail="原文件已删除")
    run = next((item for item in analysis.detector_runs if item.id == run_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="模型结果不存在")
    try:
        gallery = request_gallery(analysis, run, storage, retry=retry)
        try:
            from app.services.attention_gallery import attach_attention
            attach_attention(analysis, run, storage, gallery)
        except Exception:
            gallery["attention"] = {"status": "failed", "can_generate": run.status == DetectorStatus.SUCCEEDED,
                "detail": "注意力图暂时不可用，原检测结果已保留。"}
        from app.services.routing_evidence import attach_routing
        attach_routing(analysis, run, storage, gallery)
        return gallery
    except Exception:
        raise HTTPException(status_code=503, detail="检测帧可视化暂时不可用，请稍后重试")


@router.get("/{analysis_id}/runs/{run_id}/gallery")
def get_gallery(analysis_id: uuid.UUID, run_id: uuid.UUID,
                user: User = Depends(get_current_user), db: Session = Depends(get_db),
                storage: ObjectStorage = Depends(get_storage)):
    return _gallery(analysis_id, run_id, user, db, storage)


@router.post("/{analysis_id}/runs/{run_id}/gallery")
def retry_gallery(analysis_id: uuid.UUID, run_id: uuid.UUID,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db),
                  storage: ObjectStorage = Depends(get_storage)):
    return _gallery(analysis_id, run_id, user, db, storage, retry=True)


@router.post("/{analysis_id}/runs/{run_id}/attention")
def generate_attention(analysis_id: uuid.UUID, run_id: uuid.UUID,
                       user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analysis = _get_analysis(db, analysis_id, user)
    if analysis.upload.status == UploadStatus.DELETED:
        raise HTTPException(status_code=404, detail="原文件已删除")
    if not any(run.id == run_id for run in analysis.detector_runs):
        raise HTTPException(status_code=404, detail="模型结果不存在")
    from app.services.attention_gallery import request_attention
    try:
        return request_attention(analysis_id, run_id)
    except Exception:
        raise HTTPException(status_code=503, detail="注意力图任务暂时无法提交，请稍后重试")


@router.post("/batch-delete", response_model=AnalysisBatchDeleteResult)
def batch_delete_analyses(
    payload: AnalysisBatchDeleteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisBatchDeleteResult:
    ids = list(dict.fromkeys(payload.analysis_ids))
    query = (
        _authorized_query(user)
        .where(Analysis.id.in_(ids))
        .options(selectinload(Analysis.detector_runs), selectinload(Analysis.events))
        .order_by(Analysis.id)
        .with_for_update(of=Analysis)
    )
    analyses = list(db.scalars(query))
    # The same row lock guards worker progress/result/asset publication. After
    # commit, missing records cancel active workers and queued deliveries no-op.
    for analysis in analyses:
        db.delete(analysis)
    db.commit()
    for analysis in analyses:
        get_storage().delete_prefix(gallery_prefix(analysis.id))

    deleted = len(analyses)
    skipped = len(ids) - deleted
    message = f"已删除 {deleted} 条分析记录"
    if skipped:
        message += f"，跳过 {skipped} 条不存在或无权限的记录"
    return AnalysisBatchDeleteResult(message=message, deleted=deleted, skipped=skipped)


@router.delete("/{analysis_id}", response_model=MessageResponse)
def delete_analysis(
    analysis_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    # Serialize deletion with the worker's check-and-write transactions.
    # Missing analysis is a cancellation signal at its next control check.
    analysis = _get_analysis(db, analysis_id, user, for_update=True)
    db.delete(analysis)
    db.commit()
    get_storage().delete_prefix(gallery_prefix(analysis_id))
    return MessageResponse(message="分析记录已删除")


@router.post("/{analysis_id}/cancel", response_model=MessageResponse)
def cancel_analysis(
    analysis_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    analysis = _get_analysis(db, analysis_id, user)
    db.refresh(analysis, with_for_update=True)
    if analysis.status in TERMINAL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="分析任务已经结束"
        )
    analysis.cancel_requested = True
    finish_analysis(
        db, analysis, AnalysisStatus.CANCELLED, "任务已取消", DetectorStatus.CANCELLED
    )
    db.commit()
    return MessageResponse(message="任务已取消")


@router.get("/{analysis_id}/events")
def stream_events(
    analysis_id: uuid.UUID,
    last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EventSourceResponse:
    _get_analysis(db, analysis_id, user)
    user_id = user.id
    is_admin = user.role == UserRole.ADMIN

    async def event_generator() -> AsyncGenerator[dict, None]:
        cursor = last_event_id or 0
        terminal_idle_cycles = 0
        while True:
            with SessionLocal() as event_db:
                access_query = select(Analysis).where(Analysis.id == analysis_id)
                if not is_admin:
                    access_query = access_query.where(Analysis.user_id == user_id)
                analysis = event_db.scalar(access_query)
                if not analysis:
                    return
                events = list(
                    event_db.scalars(
                        select(AnalysisEvent)
                        .where(
                            AnalysisEvent.analysis_id == analysis_id,
                            AnalysisEvent.id > cursor,
                        )
                        .order_by(AnalysisEvent.id)
                    )
                )
                terminal = analysis.status in TERMINAL

            if events:
                terminal_idle_cycles = 0
                for event in events:
                    cursor = event.id
                    yield {
                        "id": str(event.id),
                        "event": event.event_type,
                        "data": json.dumps(
                            event.payload, ensure_ascii=False, default=str
                        ),
                    }
            elif terminal:
                terminal_idle_cycles += 1
                if terminal_idle_cycles >= 2:
                    return
            await asyncio.sleep(0.75)

    return EventSourceResponse(event_generator(), ping=15)
