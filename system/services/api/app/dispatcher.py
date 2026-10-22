"""Publish durable queued analyses and finalize abandoned executions."""

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Analysis, AnalysisStatus, DetectorStatus
from app.services.analysis_state import TERMINAL_STATUSES, finish_analysis
from app.tasks.analysis import run_analysis


logger = logging.getLogger(__name__)
DISPATCH_INTERVAL_SECONDS = 15


def publish_analysis(analysis_id: str) -> bool:
    try:
        run_analysis.apply_async(
            args=[analysis_id],
            queue=settings.detector_queue,
            # A delivery needs its own Celery ID: expired IDs stay revoked in
            # the worker. Database _claim_analysis still deduplicates the job.
            expires=DISPATCH_INTERVAL_SECONDS * 2,
            retry=False,
        )
        return True
    except Exception:
        # The database row is the durable work item. The dispatcher will retry
        # even if the API dies between its database commit and queue publish.
        logger.warning(
            "分析任务 %s 暂未投递，等待调度器重试", analysis_id, exc_info=True
        )
        return False


def dispatch_pending_analyses() -> None:
    with SessionLocal() as db:
        abandoned = list(
            db.scalars(
                select(Analysis)
                .where(
                    Analysis.status.not_in(TERMINAL_STATUSES),
                    Analysis.execution_deadline <= datetime.now(timezone.utc),
                )
                .with_for_update(skip_locked=True)
            )
        )
        for analysis in abandoned:
            finish_analysis(
                db,
                analysis,
                AnalysisStatus.FAILED,
                "分析任务执行超时或 Worker 异常退出",
                DetectorStatus.TIMED_OUT,
            )
        db.commit()
        queued = list(
            db.scalars(
                select(Analysis.id)
                .where(Analysis.status == AnalysisStatus.QUEUED)
                .order_by(Analysis.created_at)
            )
        )
    for analysis_id in queued:
        if not publish_analysis(str(analysis_id)):
            break


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            dispatch_pending_analyses()
        except Exception:
            logger.exception("任务调度失败，将重试")
        time.sleep(DISPATCH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
