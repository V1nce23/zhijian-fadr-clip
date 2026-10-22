from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, AnalysisEvent


def add_analysis_event(
    db: Session,
    analysis: Analysis,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> AnalysisEvent:
    event_payload = {
        "analysis_id": str(analysis.id),
        "status": analysis.status.value if hasattr(analysis.status, "value") else analysis.status,
        "progress": analysis.progress,
        **(payload or {}),
    }
    event = AnalysisEvent(
        analysis_id=analysis.id,
        event_type=event_type,
        payload=event_payload,
    )
    db.add(event)
    return event
