"""Optional attention assets, separate from authoritative report scores."""

import io
import json
import time
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.database import SessionLocal
from app.detectors.catalog import FADR_MODEL_VERSION
from app.models import Analysis, DetectorRun, DetectorStatus, Upload, UploadStatus
from app.services import previews
from app.services.storage import get_storage

METHOD = "fadr-route-patch-v1"
LABEL = "路由加权空间注意力"


def prefix(analysis_id, run_id):
    return f"evidence/{analysis_id}/{run_id}/attention-v1/"


def manifest_key(analysis_id, run_id):
    return prefix(analysis_id, run_id) + "manifest.json"


def state_key(analysis_id, run_id):
    return f"deeptrace:attention:v1:{analysis_id}:{run_id}"


def supported(run):
    return (run.detector_id == "fadr-clip" and run.model_version == FADR_MODEL_VERSION
            and run.preprocessing_version.startswith("dfb-face256-clip224-v3"))


def read_manifest(storage, analysis_id, run_id):
    key = manifest_key(analysis_id, run_id)
    manifest = storage.read_json(key) if storage.exists(key) else {"method": METHOD, "frames": []}
    if manifest.get("method") != METHOD:
        raise ValueError("Attention manifest method mismatch")
    for item in manifest.get("frames", []):
        if item["key"] != f"{prefix(analysis_id, run_id)}frame-{int(item['frame_index'])}.png":
            raise ValueError("Attention asset does not belong to this report")
    return manifest


def save_manifest(storage, analysis_id, run_id, manifest):
    storage.upload(manifest_key(analysis_id, run_id),
                   io.BytesIO(json.dumps(manifest, allow_nan=False).encode()), "application/json")


def save_frame(storage, analysis_id, run_id, index, payload, *, origin="inference"):
    """Caller holds Analysis then Upload row locks, coordinating with deletion."""
    from app.detectors.fadr_clip.attention import render_attention

    png, metadata = render_attention(payload)
    manifest = read_manifest(storage, analysis_id, run_id)
    key = f"{prefix(analysis_id, run_id)}frame-{index}.png"
    storage.upload(key, io.BytesIO(png), "image/png")
    # Store raw weights for traceability; no replay score ever enters the report.
    raw_key = f"{prefix(analysis_id, run_id)}frame-{index}.json"
    storage.upload(raw_key, io.BytesIO(json.dumps({**metadata, "origin": origin}, allow_nan=False).encode()), "application/json")
    item = {"frame_index": index, "key": key, "uniform": metadata["uniform"], "origin": origin,
            "routing_weights": metadata["routing_weights"], "routing_layers": metadata["layers"]}
    manifest["frames"] = [entry for entry in manifest["frames"] if entry["frame_index"] != index] + [item]
    save_manifest(storage, analysis_id, run_id, manifest)


def attach_attention(analysis, run, storage, gallery):
    """GET only reads assets. It never schedules a second model inference."""
    if not supported(run):
        gallery["attention"] = {"status": "unavailable", "detail": "此历史模型或预处理版本不支持，重新检测后可查看。"}
        return gallery
    manifest = read_manifest(storage, analysis.id, run.id)
    assets = {item["frame_index"]: item for item in manifest["frames"]}
    frames = gallery.get("frames", [])
    for frame in frames:
        item = assets.get(frame["frame_index"])
        if item and frame.get("image_kind") == "aligned_face" and frame.get("score") is not None:
            frame.update(heatmap_url=storage.download_url(item["key"]), attention_uniform=item["uniform"], attention_origin=item["origin"])
    from app.services.evidence_gallery import expected_frames
    required = {item["frame_index"] for item in expected_frames(run)}
    count = len(required & assets.keys())
    ready = run.status == DetectorStatus.SUCCEEDED and bool(required) and count == len(required)
    pending = manifest.get("status") == "pending"
    expired = pending and time.time() - manifest.get("started_at", 0) > 3600
    status = "ready" if ready else "failed" if expired else "pending" if pending else manifest.get("status", "not_requested")
    gallery["attention"] = {"status": status, "method": METHOD, "label": LABEL,
        "completed_frames": count, "total_frames": len(required),
        "detail": "生成超时，可重试。" if expired else manifest.get("detail"),
        "can_generate": run.status == DetectorStatus.SUCCEEDED and bool(required) and not ready}
    return gallery


def request_attention(analysis_id, run_id):
    """Explicit, deduplicated supplementary replay for completed compatible reports."""
    with SessionLocal() as db:
        analysis = db.scalar(select(Analysis).where(Analysis.id == analysis_id).with_for_update())
        if not analysis:
            return {"status": "unavailable"}
        upload = db.scalar(select(Upload).where(Upload.id == analysis.upload_id).with_for_update())
        run = db.get(DetectorRun, run_id)
        if not upload or upload.status == UploadStatus.DELETED or not run or run.analysis_id != analysis.id or not supported(run) or run.status != DetectorStatus.SUCCEEDED:
            return {"status": "unavailable", "detail": "仅支持已完成的当前 FADR-CLIP 报告。"}
        storage = get_storage()
        current = attach_attention(analysis, run, storage, {})["attention"]
        if current["status"] == "ready":
            return current
        registry = previews.preview_registry()
        identity = state_key(analysis.id, run.id)
        if not registry.set(identity, "pending", nx=True, ex=3600):
            return {**current, "status": "pending"}
        manifest = {"method": METHOD, "frames": []}
        try:
            manifest = read_manifest(storage, analysis.id, run.id)
            manifest.update(status="pending", started_at=time.time(), detail=None)
            save_manifest(storage, analysis.id, run.id, manifest)
            from app.tasks.attention_gallery import prepare_attention_gallery
            prepare_attention_gallery.apply_async(args=[str(analysis.id), str(run.id)], queue=settings.detector_queue, expires=3500)
            db.commit()
        except Exception:
            registry.delete(identity)
            manifest.update(status="failed", detail="注意力图任务暂时无法提交，请重试。")
            save_manifest(storage, analysis.id, run.id, manifest)
            raise
        return {**current, "status": "pending"}


def publish_attention(analysis_id, run_id, *, index=None, payload=None, status=None, detail=None):
    """Historical replay writes only supplementary objects, never report fields."""
    analysis_id, run_id = uuid.UUID(str(analysis_id)), uuid.UUID(str(run_id))
    with SessionLocal() as db:
        analysis = db.scalar(select(Analysis).where(Analysis.id == analysis_id).with_for_update())
        if not analysis:
            return False
        upload = db.scalar(select(Upload).where(Upload.id == analysis.upload_id).with_for_update())
        run = db.get(DetectorRun, run_id)
        if not upload or upload.status == UploadStatus.DELETED or not run or run.analysis_id != analysis.id or run.status != DetectorStatus.SUCCEEDED or not supported(run):
            return False
        storage = get_storage()
        if index is not None:
            from app.services.evidence_gallery import expected_frames
            if index not in {item["frame_index"] for item in expected_frames(run)}:
                raise ValueError("Frame not in original report")
            save_frame(storage, analysis_id, run_id, index, payload, origin="supplementary_replay")
        if status:
            manifest = read_manifest(storage, analysis_id, run_id)
            manifest.update(status=status, detail=detail)
            save_manifest(storage, analysis_id, run_id, manifest)
        db.commit()
        return True
