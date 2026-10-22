import tempfile
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from billiard.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.detectors.catalog import detector_specs
from app.detectors.registry import get_detector
from app.models import Analysis, AnalysisStatus, DetectorRun, DetectorStatus, Verdict
from app.services.analysis_state import TERMINAL_STATUSES, finish_analysis
from app.services.events import add_analysis_event
from app.services.inference import (
    InferenceCancelled,
    InferenceControl,
    InferenceTimedOut,
)
from app.services.media import prepare_detection_media
from app.services.results import aggregate_results
from app.services.storage import get_storage
from app.services.live_visualization import LiveVisualization
from app.services.threshold_policy import frame_thresholds, policy_verdict, policy_summary


def _active_analysis(db: Session, analysis_id: uuid.UUID) -> Analysis:
    analysis = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id).with_for_update()
    )
    if (
        not analysis
        or analysis.status in TERMINAL_STATUSES
        or analysis.cancel_requested
    ):
        raise InferenceCancelled("任务已结束")
    return analysis


def _claim_analysis(analysis_id: uuid.UUID) -> bool:
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        claimed = db.execute(
            update(Analysis)
            .where(
                Analysis.id == analysis_id,
                Analysis.status == AnalysisStatus.QUEUED,
                Analysis.cancel_requested.is_(False),
            )
            .values(
                status=AnalysisStatus.PREPROCESSING,
                progress=15,
                started_at=now,
                # Celery kills the process after soft limit + 30s. Recovery waits
                # another 30s so a stale process cannot overlap a terminal write.
                execution_deadline=now
                + timedelta(seconds=settings.analysis_timeout_seconds + 60),
            )
        )
        if claimed.rowcount != 1:
            return False
        analysis = db.get(Analysis, analysis_id)
        add_analysis_event(
            db,
            analysis,
            "analysis.preprocessing",
            {"message": "正在读取原始素材，准备人脸预处理"},
        )
        db.commit()
        return True


def _is_cancelled(analysis_id: uuid.UUID) -> bool:
    with SessionLocal() as db:
        analysis = db.get(Analysis, analysis_id)
        return (
            not analysis
            or analysis.cancel_requested
            or analysis.status in TERMINAL_STATUSES
        )


def _set_stage(
    analysis_id: uuid.UUID,
    status: AnalysisStatus,
    progress: int,
    event_type: str,
    message: str,
) -> None:
    with SessionLocal() as db:
        analysis = _active_analysis(db, analysis_id)
        analysis.status = status
        analysis.progress = max(analysis.progress, progress)
        add_analysis_event(db, analysis, event_type, {"message": message})
        db.commit()


def _mark_run_started(run_id: uuid.UUID) -> None:
    with SessionLocal() as db:
        run = db.get(DetectorRun, run_id)
        if not run:
            raise InferenceCancelled("任务已删除")
        _active_analysis(db, run.analysis_id)
        run.status = DetectorStatus.RUNNING
        run.progress = 20
        run.started_at = datetime.now(timezone.utc)
        db.commit()


def _mark_run_succeeded(run_id: uuid.UUID, result) -> None:
    with SessionLocal() as db:
        run = db.get(DetectorRun, run_id)
        if not run:
            raise InferenceCancelled("任务已删除")
        _active_analysis(db, run.analysis_id)
        run.status = DetectorStatus.SUCCEEDED
        run.progress = 100
        run.score = result.score
        run.verdict = Verdict(result.verdict)
        run.evidence = result.evidence
        run.completed_at = datetime.now(timezone.utc)
        db.commit()


def _mark_run_failed(run_id: uuid.UUID, error: Exception) -> None:
    with SessionLocal() as db:
        run = db.get(DetectorRun, run_id)
        if not run:
            raise InferenceCancelled("任务已删除")
        _active_analysis(db, run.analysis_id)
        run.status = (
            DetectorStatus.TIMED_OUT
            if isinstance(error, InferenceTimedOut)
            else DetectorStatus.FAILED
        )
        run.progress = 100
        run.error = str(error)
        run.completed_at = datetime.now(timezone.utc)
        db.commit()


def resolve_execution_device(requested_device: str | None) -> str:
    device = requested_device or settings.detector_device
    if device == "cpu":
        return "cpu"
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("所选 GPU 当前不可用，本次检测未改用 CPU，请重新选择设备提交")


@celery_app.task(
    name="deeptrace.run_analysis",
    soft_time_limit=settings.analysis_timeout_seconds,
    time_limit=settings.analysis_timeout_seconds + 30,
)
def run_analysis(analysis_id: str) -> None:
    analysis_uuid = uuid.UUID(analysis_id)
    if not _claim_analysis(analysis_uuid):
        # A duplicated message, a terminal report or a deleted record is a no-op.
        return
    control = InferenceControl(
        settings.analysis_timeout_seconds, lambda: _is_cancelled(analysis_uuid)
    )
    try:
        control.check()
        with SessionLocal() as db:
            analysis = _active_analysis(db, analysis_uuid)
            upload = analysis.upload
            media_type = getattr(upload.media_type, "value", upload.media_type)
            object_key = upload.object_key
            original_name = upload.original_name
            upload_sha256 = upload.sha256
            runs = list(analysis.detector_runs)
            requested_device = analysis.requested_device
            threshold_policy = analysis.threshold_policy
        device = resolve_execution_device(requested_device)
        with SessionLocal() as db:
            analysis = _active_analysis(db, analysis_uuid)
            analysis.executed_device = device
            add_analysis_event(db, analysis, "analysis.device_selected", {"device": device})
            db.commit()
        if not runs:
            raise ValueError("分析任务没有检测器")
        spec_by_id = {spec.detector_id: spec for spec in detector_specs(media_type)}
        results = []
        with tempfile.TemporaryDirectory(
            prefix="deeptrace-inference-"
        ) as temporary_directory:
            workdir = Path(temporary_directory)
            source = workdir / f"input{Path(original_name).suffix[:16]}"
            get_storage().download(object_key, source)
            control.check()
            detection_source = prepare_detection_media(
                source, media_type, workdir, check=control.check
            )
            _set_stage(
                analysis_uuid,
                AnalysisStatus.RUNNING,
                20,
                "detector.started",
                "原始素材校验完成，正在抽帧、准备人脸并检测" if media_type == "video" else "图片读取完成，正在准备人脸并检测",
            )
            for index, run in enumerate(runs):
                control.check()
                _mark_run_started(run.id)
                detector = None
                try:
                    spec = spec_by_id.get(run.detector_id)
                    if spec is None:
                        raise LookupError(f"检测器未配置: {run.detector_id}")
                    detector_control = InferenceControl(
                        settings.detector_timeout_seconds
                    )

                    def check() -> None:
                        control.check()
                        detector_control.check()

                    check()
                    detector = get_detector(
                        media_type,
                        spec,
                        settings.detector_models_root,
                        device,
                        settings.detector_local_files_only,
                        settings.detector_authentic_threshold,
                        settings.detector_manipulated_threshold,
                    )
                    visualization = LiveVisualization(analysis_uuid, run.id, media_type)
                    detector.progress_callback = visualization
                    result = detector.predict(
                        detection_source, sha256=upload_sha256, check=check
                    )
                    visualization("reporting")
                    result = replace(result, evidence={**result.evidence, "device": device,
                        "visualization": visualization.snapshot("completed")})
                    if threshold_policy:
                        result = replace(result, verdict=policy_verdict(result.score, threshold_policy),
                            evidence={**result.evidence, "decision_policy": threshold_policy,
                                      "thresholds": frame_thresholds(threshold_policy)})
                    check()
                    _mark_run_succeeded(run.id, result)
                    results.append(result)
                except (InferenceCancelled, SoftTimeLimitExceeded):
                    raise
                except Exception as exc:
                    _mark_run_failed(run.id, exc)
                finally:
                    # Release this model before constructing the next one.
                    del detector
                control.check()
                _set_stage(
                    analysis_uuid,
                    AnalysisStatus.RUNNING,
                    20 + round((index + 1) * 62 / len(runs)),
                    "detector.progress",
                    "FADR-CLIP 检测已完成",
                )
        control.check()
        _set_stage(
            analysis_uuid,
            AnalysisStatus.AGGREGATING,
            94,
            "analysis.aggregating",
            "正在生成检测报告",
        )
        with SessionLocal() as db:
            analysis = _active_analysis(db, analysis_uuid)
            if not results:
                finish_analysis(
                    db,
                    analysis,
                    AnalysisStatus.FAILED,
                    "FADR-CLIP 未能完成检测，请查看模型错误详情。",
                    DetectorStatus.FAILED,
                )
            else:
                summary = aggregate_results(results)
                analysis.status = (
                    AnalysisStatus.COMPLETED
                    if len(results) == len(runs)
                    else AnalysisStatus.PARTIALLY_COMPLETED
                )
                analysis.progress = 100
                analysis.completed_at = datetime.now(timezone.utc)
                analysis.verdict = summary.verdict
                analysis.score = summary.score
                analysis.summary = policy_summary(threshold_policy, summary.verdict.value) if threshold_policy else summary.summary
                add_analysis_event(
                    db,
                    analysis,
                    "analysis.completed",
                    {
                        "message": "分析完成",
                        "score": summary.score,
                        "verdict": summary.verdict.value,
                        "succeeded_checkers": len(results),
                        "failed_checkers": len(runs) - len(results),
                    },
                )
            db.commit()
    except InferenceCancelled:
        return
    except Exception as exc:
        timed_out = isinstance(exc, (InferenceTimedOut, SoftTimeLimitExceeded))
        message = "分析任务执行超时" if timed_out else str(exc)
        with SessionLocal() as db:
            analysis = db.scalar(
                select(Analysis).where(Analysis.id == analysis_uuid).with_for_update()
            )
            if analysis:
                finish_analysis(
                    db,
                    analysis,
                    AnalysisStatus.FAILED,
                    message,
                    DetectorStatus.TIMED_OUT if timed_out else DetectorStatus.FAILED,
                )
                db.commit()
