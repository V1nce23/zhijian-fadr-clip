from pathlib import Path

from app.detectors.base import DetectorAdapter, DetectorSpec


def get_detector(
    media_type: str, spec: DetectorSpec, models_root: str, device: str,
    local_files_only: bool, authentic_threshold: float, manipulated_threshold: float,
) -> DetectorAdapter:
    if media_type not in ("image", "video") or spec.detector_id != "fadr-clip":
        raise ValueError(f"unsupported detector: {media_type}/{spec.detector_id}")
    from app.detectors.fadr_clip.adapter import FADRClipDetector
    return FADRClipDetector(
        spec, Path(models_root), device=device,
        authentic_threshold=authentic_threshold, manipulated_threshold=manipulated_threshold,
    )
