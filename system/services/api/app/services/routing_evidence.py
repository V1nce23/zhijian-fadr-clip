"""Read routing evidence without inference, queueing or report mutation."""

import logging
import math

from app.services.evidence_gallery import expected_frames

logger = logging.getLogger(__name__)
LAYERS = [6, 12, 18, 24]


def validated_routing(weights, layers, origin="original_detection"):
    if layers != LAYERS or not isinstance(weights, (list, tuple)) or len(weights) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in weights):
        return None
    if not math.isclose(sum(weights), 1, abs_tol=1e-4):
        return None
    return {"layers": LAYERS.copy(), "weights": list(weights), "origin": origin}


def report_frames(run):
    """Keep skipped/pending frames where recorded; never turn missing scores into zero."""
    evidence = run.evidence or {}
    live = evidence.get("visualization", {})
    authoritative = {item["frame_index"]: item for item in expected_frames(run)}
    frames = live.get("frames") if live.get("version") == "live-v1" else None
    if frames:
        return [{**item, **authoritative.get(item["frame_index"], {})} for item in frames]
    return [{**item, "status": "scored" if item.get("score") is not None else "pending"} for item in authoritative.values()]


def routing_for_frames(analysis_id, run, storage, frames):
    evidence = run.evidence or {}
    layers = evidence.get("routing_layers")
    saved = {item["frame_index"]: item for item in report_frames(run)}
    assets = {}
    # Old attention JSON has true per-frame routes even when old DB rows do not.
    if storage is not None and any(item.get("score") is not None and not validated_routing(saved.get(item["frame_index"], {}).get("routing_weights"), layers or saved.get(item["frame_index"], {}).get("routing_layers")) for item in frames):
        try:
            from app.services.attention_gallery import read_manifest, supported
            if supported(run):
                assets = {item["frame_index"]: item for item in read_manifest(storage, analysis_id, run.id)["frames"]}
        except Exception:
            logger.warning("已保存的逐帧路由暂时无法读取", exc_info=True)
    result = {}
    for frame in frames:
        index = frame["frame_index"]
        if frame.get("score") is None:
            continue
        item = saved.get(index, {})
        routing = validated_routing(item.get("routing_weights"), layers or item.get("routing_layers"))
        if routing is None and (run.evidence or {}).get("kind") == "image_regions":
            routing = validated_routing(evidence.get("routing_weights"), layers)
        if routing is None and index in assets:
            try:
                asset = assets[index]
                origin = "supplementary_replay" if asset.get("origin") == "supplementary_replay" else "original_detection"
                routing = validated_routing(asset.get("routing_weights"), asset.get("routing_layers"), origin)
                if routing is None:
                    from app.services.attention_gallery import prefix, METHOD
                    key = f"{prefix(analysis_id, run.id)}frame-{index}.json"
                    if storage.exists(key):
                        raw = storage.read_json(key)
                        if raw.get("method") == METHOD:
                            routing = validated_routing(raw.get("routing_weights"), raw.get("layers"), origin)
            except Exception:
                logger.warning("某帧路由暂时无法读取，保留原分数", exc_info=True)
        if routing is not None:
            result[index] = routing
    return result


def attach_routing(analysis, run, storage, gallery):
    frames = gallery.get("frames", [])
    available = routing_for_frames(analysis.id, run, storage, frames)
    for frame in frames:
        frame["routing"] = available.get(frame["frame_index"])
    return gallery
