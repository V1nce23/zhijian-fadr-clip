"""Cached visual face evidence, addressed by report/run and authorized by the API."""

import io
import json
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Analysis, AnalysisStatus, DetectorRun, DetectorStatus, Upload, UploadStatus
from app.services import previews
from app.services.storage import get_storage

VERSION = "faces-v1"


def gallery_prefix(analysis_id, run_id=None):
    root = f"evidence/{analysis_id}/"
    return root if run_id is None else f"{root}{run_id}/{VERSION}/"


def manifest_key(analysis_id, run_id):
    return gallery_prefix(analysis_id, run_id) + "manifest.json"


def state_key(analysis_id, run_id):
    return f"deeptrace:gallery:{VERSION}:{analysis_id}:{run_id}"


def set_state(analysis_id, run_id, status, detail=None):
    state = {"status": status}
    if detail:
        state["detail"] = detail
    previews.preview_registry().set(state_key(analysis_id, run_id), json.dumps(state), ex=900)


def expected_frames(run):
    evidence = run.evidence or {}
    if evidence.get("kind") == "video_frames":
        return evidence.get("frames", [])
    if evidence.get("kind") == "image_regions" and run.score is not None:
        return [{"frame_index": 0, "timestamp_seconds": 0.0, "score": run.score}]
    return []


def request_gallery(analysis, run, storage, *, retry=False):
    live = (run.evidence or {}).get("visualization")
    if live and live.get("version") == "live-v1" and (live.get("frames") or run.status != DetectorStatus.SUCCEEDED):
        prefix = gallery_prefix(analysis.id, run.id)
        frames = []
        # Optional backfilled face assets can repair failed live-image uploads.
        repaired = set()
        missing = any(item["status"] == "scored" and not item.get("has_face") for item in live["frames"])
        if missing and run.status == DetectorStatus.SUCCEEDED and storage.exists(manifest_key(analysis.id, run.id)):
            repaired = {item["frame_index"] for item in storage.read_json(manifest_key(analysis.id, run.id))["frames"]}
        for item in live["frames"]:
            face = item.get("has_face") or item["frame_index"] in repaired
            source = item.get("has_source")
            asset = (f"frame-{item['frame_index']}.png" if face else f"source-{item['frame_index']}.jpg") if face or source else None
            frames.append({"frame_index": item["frame_index"], "timestamp_seconds": item["timestamp_seconds"],
                "score": item.get("score"), "status": item["status"], "image_kind": "aligned_face" if face else "source_frame",
                "image_url": storage.download_url(prefix + asset) if asset else None, "heatmap_url": None})
        stopped = run.status in ("failed", "timed_out", "cancelled") or analysis.status in ("failed", "cancelled")
        stage = "cancelled" if analysis.status == "cancelled" else "failed" if stopped else (
            "reporting" if analysis.status == "aggregating" else live["stage"])
        return {"status": "stopped" if stopped else "ready" if run.status == DetectorStatus.SUCCEEDED else "streaming",
            "stage": stage, "frames": frames, "progress": {key: live.get(key, 0) for key in
                ("sampled_frames", "extracted_frames", "processed_frames", "valid_frames", "scored_frames")},
            "image_kind": "aligned_face", "size": 256}
    if run.status != DetectorStatus.SUCCEEDED:
        return {"status": "waiting" if run.status in ("queued", "running") else "unavailable", "frames": []}
    # Only this deterministic original-pixel pipeline can reproduce the exact face
    # inputs. Never silently substitute a crop from a different legacy pipeline.
    if not run.preprocessing_version.startswith("dfb-face256-clip224-v3"):
        return {"status": "unavailable", "frames": [], "detail": "此报告使用较早的预处理方式，重新检测后可查看人脸可视化。"}
    key = manifest_key(analysis.id, run.id)
    if storage.exists(key):
        manifest = storage.read_json(key)
        prefix = gallery_prefix(analysis.id, run.id)
        assets = {item["frame_index"]: item["key"] for item in manifest["frames"]}
        frames = []
        for item in expected_frames(run):
            asset = assets.get(item["frame_index"])
            if asset != f"{prefix}frame-{item['frame_index']}.png":
                raise ValueError("Evidence manifest does not match this report")
            frames.append({**item, "status": "scored", "image_kind": "aligned_face", "image_url": storage.download_url(asset), "heatmap_url": None})
        return {"status": "ready", "frames": frames, "image_kind": "aligned_face", "size": 256}
    registry = previews.preview_registry()
    state_identity = state_key(analysis.id, run.id)
    raw = registry.get(state_identity)
    state = json.loads(raw) if raw else None
    if state and state.get("status") == "failed" and retry:
        registry.delete(state_identity)
        state = None
    if state:
        return state
    if registry.set(state_identity, json.dumps({"status": "pending"}), nx=True, ex=900):
        from app.tasks.evidence_gallery import prepare_evidence_gallery
        try:
            prepare_evidence_gallery.apply_async(args=[str(analysis.id), str(run.id)], queue=settings.detector_queue)
        except Exception:
            registry.delete(state_identity)
            raise
    return {"status": "pending"}


def publish_gallery(analysis_id, run_id, files):
    """Publish manifest last; row locks coordinate with report/upload deletion."""
    identity, run_identity = uuid.UUID(str(analysis_id)), uuid.UUID(str(run_id))
    with SessionLocal() as db:
        analysis = db.scalar(select(Analysis).where(Analysis.id == identity).with_for_update())
        if not analysis:
            return False
        upload = db.scalar(select(Upload).where(Upload.id == analysis.upload_id).with_for_update())
        run = db.get(DetectorRun, run_identity)
        if not upload or upload.status == UploadStatus.DELETED or not run or run.analysis_id != identity or run.status != DetectorStatus.SUCCEEDED:
            return False
        required = {item["frame_index"] for item in expected_frames(run)}
        if set(files) != required or not required:
            raise ValueError("无法完整还原报告中的检测帧")
        storage = get_storage()
        manifest = {"version": VERSION, "frames": []}
        for index, path in files.items():
            key = f"{gallery_prefix(identity, run_identity)}frame-{index}.png"
            with path.open("rb") as stream:
                storage.upload(key, stream, "image/png")
            manifest["frames"].append({"frame_index": index, "key": key})
        storage.upload(manifest_key(identity, run_identity), io.BytesIO(json.dumps(manifest).encode()), "application/json")
        db.commit()
    return True
