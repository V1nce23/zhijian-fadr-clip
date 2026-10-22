from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "deeptrace",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.analysis", "app.tasks.preview", "app.tasks.evidence_gallery", "app.tasks.attention_gallery"],
)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Register worker lifecycle signals without importing PyTorch into API processes.
from app.services import worker_devices  # noqa: E402,F401
