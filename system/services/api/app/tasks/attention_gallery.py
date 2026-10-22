"""Explicit supplementary attention replay for historical completed reports."""

import logging
import tempfile
import uuid
from pathlib import Path

from app.celery_app import celery_app
from app.core.config import settings
from app.core.database import SessionLocal
from app.detectors.catalog import detector_specs
from app.models import Analysis, DetectorRun, DetectorStatus, UploadStatus
from app.services import previews
from app.services.attention_gallery import publish_attention, read_manifest, state_key, supported
from app.services.evidence_gallery import expected_frames, gallery_prefix
from app.services.storage import get_storage

logger = logging.getLogger(__name__)


@celery_app.task(name="deeptrace.prepare_attention_gallery", soft_time_limit=3000, time_limit=3030)
def prepare_attention_gallery(analysis_id, run_id):
    try:
        identity, run_identity = uuid.UUID(analysis_id), uuid.UUID(run_id)

        def check():
            with SessionLocal() as db:
                analysis = db.get(Analysis, identity)
                run = db.get(DetectorRun, run_identity)
                if (not analysis or not run or run.analysis_id != identity or run.status != DetectorStatus.SUCCEEDED
                        or analysis.upload.status == UploadStatus.DELETED or not supported(run)):
                    raise ValueError("原报告或文件已删除，停止生成注意力图")

        check()
        with SessionLocal() as db:
            analysis, run = db.get(Analysis, identity), db.get(DetectorRun, run_identity)
            media_type = analysis.upload.media_type
            required = [item["frame_index"] for item in expected_frames(run)]
        storage = get_storage()
        face_prefix = gallery_prefix(analysis_id, run_id)
        if any(not storage.exists(f"{face_prefix}frame-{index}.png") for index in required):
            from app.tasks.evidence_gallery import prepare_evidence_gallery
            prepare_evidence_gallery(analysis_id, run_id)
        check()
        if any(not storage.exists(f"{face_prefix}frame-{index}.png") for index in required):
            raise ValueError("原检测人脸缓存暂时无法还原")
        cached = {item["frame_index"] for item in read_manifest(storage, analysis_id, run_id)["frames"]}
        remaining = [index for index in required if index not in cached]
        if remaining:
            import cv2
            from app.detectors.fadr_clip.adapter import FADRClipDetector

            detector = FADRClipDetector(detector_specs(media_type)[0], Path(settings.detector_models_root), device="auto")
            detector.load()
            check()
            batch_size = 4 if detector.device == "cuda" else 1
            with tempfile.TemporaryDirectory(prefix="deeptrace-attention-") as directory:
                for offset in range(0, len(remaining), batch_size):
                    check()
                    indices = remaining[offset:offset + batch_size]
                    tensors = []
                    for index in indices:
                        path = Path(directory) / f"{index}.png"
                        storage.download(f"{face_prefix}frame-{index}.png", path)
                        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                        if bgr is None or bgr.shape != (256, 256, 3):
                            raise ValueError("无法还原原检测的人脸输入")
                        # Already aligned exact face PNG; never align a second time.
                        tensors.append(detector._input_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                    detector._scores(tensors)
                    check()
                    if len(detector.batch_attention) != len(indices):
                        raise ValueError("模型未提供空间注意力")
                    for index, payload in zip(indices, detector.batch_attention):
                        if not publish_attention(analysis_id, run_id, index=index, payload=payload):
                            return
            del detector
        publish_attention(analysis_id, run_id, status="ready")
    except Exception:
        logger.exception("补充注意力图生成失败；原报告未修改")
        publish_attention(analysis_id, run_id, status="failed", detail="注意力图暂时无法生成，可重试；原检测分数已保留。")
    finally:
        try:
            previews.preview_registry().delete(state_key(analysis_id, run_id))
        except Exception:
            logger.warning("注意力任务状态清理暂时失败", exc_info=True)
