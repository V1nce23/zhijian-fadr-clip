"""Publish genuine preprocessing / inference milestones without altering model input."""

import copy
import io
import logging
import uuid

from sqlalchemy import select
from billiard.exceptions import SoftTimeLimitExceeded

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Analysis, AnalysisStatus, DetectorRun, Upload, UploadStatus
from app.services.analysis_state import TERMINAL_STATUSES
from app.services.evidence_gallery import gallery_prefix
from app.services.events import add_analysis_event
from app.services.inference import InferenceCancelled
from app.services.storage import get_storage
from app.services.threshold_policy import frame_thresholds

logger = logging.getLogger(__name__)
LIVE_VERSION = "live-v1"


class LiveVisualization:
    def __init__(self, analysis_id, run_id, media_type):
        self.analysis_id = uuid.UUID(str(analysis_id))
        self.run_id = uuid.UUID(str(run_id))
        self.media_type = media_type
        self.state = {"version": LIVE_VERSION, "stage": "reading", "revision": 0, "frames": []}
        self("reading")

    def snapshot(self, stage=None):
        state = copy.deepcopy(self.state)
        if stage:
            state["stage"] = stage
        return state

    def __call__(self, event, **data):
        with SessionLocal() as db:
            analysis = db.scalar(select(Analysis).where(Analysis.id == self.analysis_id).with_for_update())
            if not analysis or analysis.cancel_requested or analysis.status in TERMINAL_STATUSES:
                raise InferenceCancelled("任务已结束")
            upload = db.scalar(select(Upload).where(Upload.id == analysis.upload_id).with_for_update())
            run = db.get(DetectorRun, self.run_id)
            if not upload or upload.status == UploadStatus.DELETED or not run or run.analysis_id != analysis.id:
                raise InferenceCancelled("任务已删除")
            if event == "plan":
                self.state["frames"] = [{**item, "status": "pending", "score": None,
                    "has_source": False, "has_face": False} for item in data["frames"]]
                self.state["stage"] = "preprocessing"
            elif event in ("sampled", "aligned"):
                frame = next(item for item in self.state["frames"] if item["frame_index"] == data["frame_index"])
                image = data["image"]
                frame["status"] = "sampled" if event == "sampled" else "ready" if image is not None else "no_face"
                if image is not None:
                    # Presentation copies only; the detector receives the untouched RGB.
                    try:
                        import cv2
                        if event == "sampled":
                            height, width = image.shape[:2]
                            scale = min(1.0, 320 / max(height, width))
                            shown = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
                            suffix, options, flag = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85], "has_source"
                            name = f"source-{frame['frame_index']}.jpg"
                        else:
                            shown, suffix, options, flag = image, ".png", [], "has_face"
                            name = f"frame-{frame['frame_index']}.png"
                        ok, encoded = cv2.imencode(suffix, cv2.cvtColor(shown, cv2.COLOR_RGB2BGR), options)
                        if not ok:
                            raise ValueError("无法保存可视化帧")
                        get_storage().upload(gallery_prefix(self.analysis_id, self.run_id) + name,
                            io.BytesIO(encoded.tobytes()), "image/png" if suffix == ".png" else "image/jpeg")
                        frame[flag] = True
                    except (InferenceCancelled, SoftTimeLimitExceeded):
                        raise
                    except Exception:
                        # Optional visualization storage must not invalidate inference.
                        logger.warning("帧图暂时无法保存，继续检测", exc_info=True)
            elif event == "scores":
                for result in data["frames"]:
                    frame = next(item for item in self.state["frames"] if item["frame_index"] == result["frame_index"])
                    if frame["status"] not in ("ready", "scored"):
                        raise ValueError("未对齐的帧不能发布检测分数")
                    frame.update(status="scored", score=result["score"])
                    if "routing_weights" in result:
                        frame.update(routing_weights=result["routing_weights"], routing_layers=result.get("routing_layers"))
                    if result.get("attention") is not None:
                        try:
                            from app.services.attention_gallery import save_frame
                            save_frame(get_storage(), self.analysis_id, self.run_id, frame["frame_index"], result["attention"])
                        except (InferenceCancelled, SoftTimeLimitExceeded):
                            raise
                        except Exception:
                            logger.warning("注意力图暂时无法保存，继续保留检测分数", exc_info=True)
                self.state["stage"] = "detecting"
            else:
                self.state["stage"] = event
            frames = self.state["frames"]
            total = len(frames)
            extracted = sum(item["status"] != "pending" for item in frames)
            processed = sum(item["status"] in ("ready", "scored", "no_face") for item in frames)
            valid = sum(item["status"] in ("ready", "scored") for item in frames)
            scored = sum(item["score"] is not None for item in frames)
            self.state.update(sampled_frames=total, extracted_frames=extracted, processed_frames=processed,
                valid_frames=valid, scored_frames=scored, revision=self.state["revision"] + 1)
            stage = self.state["stage"]
            progress = 5 + round(45 * processed / max(total, 1)) if stage == "preprocessing" else (
                55 + round(35 * scored / max(valid, 1)) if stage == "detecting" else 55 if stage == "model_loading" else 95 if stage == "reporting" else 5)
            run.progress = max(run.progress, progress)
            run.evidence = {"kind": "video_frames" if self.media_type == "video" else "image_regions",
                "sampled_frames": total, "frames_with_face": valid, "visualization": self.snapshot(),
                "decision_policy": analysis.threshold_policy,
                "thresholds": frame_thresholds(analysis.threshold_policy) if analysis.threshold_policy else
                    {"authentic_max": settings.detector_authentic_threshold, "manipulated_min": settings.detector_manipulated_threshold},
                "frames": [{key: item[key] for key in ("frame_index", "timestamp_seconds", "score")}
                           for item in frames if item["score"] is not None]}
            analysis.progress = max(analysis.progress, min(90, 20 + round(run.progress * 0.7)))
            analysis.status = AnalysisStatus.PREPROCESSING if stage in ("reading", "preprocessing") else AnalysisStatus.RUNNING
            add_analysis_event(db, analysis, "analysis.visualization", {"stage": stage, "processed_frames": processed,
                "sampled_frames": total, "scored_frames": scored, "valid_frames": valid})
            db.commit()
