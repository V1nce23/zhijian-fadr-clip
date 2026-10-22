import logging
import tempfile
import uuid
from pathlib import Path

from app.celery_app import celery_app
from app.core.database import SessionLocal
from app.models import Analysis, DetectorRun, DetectorStatus, UploadStatus
from app.services.evidence_gallery import expected_frames, manifest_key, publish_gallery, set_state
from app.services.storage import get_storage

logger = logging.getLogger(__name__)


@celery_app.task(name="deeptrace.prepare_evidence_gallery", soft_time_limit=600, time_limit=630)
def prepare_evidence_gallery(analysis_id, run_id):
    try:
        with SessionLocal() as db:
            analysis = db.get(Analysis, uuid.UUID(analysis_id))
            run = db.get(DetectorRun, uuid.UUID(run_id))
            if not analysis or not run or run.analysis_id != analysis.id or run.status != DetectorStatus.SUCCEEDED:
                return
            upload = analysis.upload
            if upload.status == UploadStatus.DELETED or not run.preprocessing_version.startswith("dfb-face256-clip224-v3"):
                return
            required = {item["frame_index"] for item in expected_frames(run)}
            object_key, media_type = upload.object_key, upload.media_type
        storage = get_storage()
        if storage.exists(manifest_key(analysis_id, run_id)):
            return
        set_state(analysis_id, run_id, "pending")
        # Heavy image dependencies belong to the inference worker, not the API.
        import cv2
        from app.core.config import settings
        from app.detectors.face_media import FaceMedia
        from app.services.media import prepare_detection_media
        with tempfile.TemporaryDirectory(prefix="deeptrace-gallery-") as directory:
            workdir = Path(directory)
            source = workdir / "original"
            storage.download(object_key, source)
            prepared = prepare_detection_media(source, media_type, workdir)
            samples, _, _ = FaceMedia(Path(settings.detector_models_root)).extract(prepared, media_type)
            files = {}
            for sample in samples:
                if sample.index not in required:
                    continue
                path = workdir / f"frame-{sample.index}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)):
                    raise ValueError("无法保存检测帧")
                files[sample.index] = path
            publish_gallery(analysis_id, run_id, files)
    except Exception:
        logger.exception("检测帧可视化生成失败")
        try:
            set_state(analysis_id, run_id, "failed", "检测帧暂时无法显示，请重试。检测分数已保留。")
        except Exception:
            logger.warning("检测帧状态暂时无法保存", exc_info=True)
