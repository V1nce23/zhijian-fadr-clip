"""Terminal transitions shared by the API, worker and dispatcher."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Analysis, AnalysisStatus, DetectorStatus
from app.services.events import add_analysis_event


TERMINAL_STATUSES = {
    AnalysisStatus.COMPLETED,
    AnalysisStatus.PARTIALLY_COMPLETED,
    AnalysisStatus.FAILED,
    AnalysisStatus.CANCELLED,
}
FINISHED_RUN_STATUSES = {
    DetectorStatus.SUCCEEDED,
    DetectorStatus.FAILED,
    DetectorStatus.TIMED_OUT,
    DetectorStatus.CANCELLED,
    DetectorStatus.SKIPPED,
}


def finish_analysis(
    db: Session,
    analysis: Analysis,
    status: AnalysisStatus,
    message: str,
    run_status: DetectorStatus,
) -> None:
    """Caller holds the analysis row lock; completed model results are preserved."""
    if analysis.status in TERMINAL_STATUSES:
        return
    analysis.status = status
    analysis.progress = 100
    analysis.summary = message
    analysis.completed_at = datetime.now(timezone.utc)
    for run in analysis.detector_runs:
        if run.status not in FINISHED_RUN_STATUSES:
            run.status = run_status
            run.progress = 100
            run.completed_at = analysis.completed_at
            run.error = message if run_status != DetectorStatus.CANCELLED else None
    add_analysis_event(db, analysis, f"analysis.{status.value}", {"message": message})
