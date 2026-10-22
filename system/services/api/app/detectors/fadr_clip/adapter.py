"""Single-model FADR-CLIP image/video inference using the verified best weights."""

import hashlib
import json
import math
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.detectors.base import DetectionResult, verdict_from_score
from app.detectors.catalog import FADR_CHECKPOINT_SHA256
from app.detectors.face_media import FaceMedia

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _checked_hash(path: str, size: int, modified_ns: int, expected: str) -> None:
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("FADR-CLIP 模型文件校验失败，请重新准备最佳权重")


def load_verified_config(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_checkpoint_sha256"] != FADR_CHECKPOINT_SHA256:
        raise ValueError("当前权重不是已确认的 FADR-CLIP 最佳主模型")
    for name in ("model.safetensors", "model_config.json"):
        path = directory / name
        info = manifest["files"][name]
        stat = path.stat()
        if stat.st_size != info["size"]:
            raise ValueError("FADR-CLIP 模型文件不完整")
        _checked_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, info["sha256"])
    return json.loads((directory / "model_config.json").read_text(encoding="utf-8"))


def resize_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the release's IsotropicResize + black pad + center crop test pipeline."""
    import cv2

    height, width = image.shape[:2]
    if max(height, width) != size:
        scale = size / max(height, width)
        new_width, new_height = (size, int(height * scale)) if width > height else (int(width * scale), size)
        image = cv2.resize(image, (max(new_width, 1), max(new_height, 1)),
                           interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    height, width = image.shape[:2]
    top, left = (size - height) // 2, (size - width) // 2
    return np.pad(image, ((top, size - height - top), (left, size - width - left), (0, 0)), mode="constant")


def load_checkpoint_state(path: Path, device: str):
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    if device == "cuda":
        # Detach from the Windows bind-mount's memory mapping before CUDA copies.
        # Owned host tensors avoid stalled driver copies in WSL/Celery workers;
        # clone preserves every checkpoint value and dtype exactly.
        state = {name: tensor.clone() for name, tensor in state.items()}
    return state


class FADRClipDetector:
    def __init__(self, spec, models_root: Path, *, device="cpu", authentic_threshold=0.35, manipulated_threshold=0.65):
        self.spec = spec
        self.directory = models_root / "fadr_clip"
        self.requested_device = device
        self.device = "cpu"
        self.model = None
        self.config = None
        self.authentic_threshold = authentic_threshold
        self.manipulated_threshold = manipulated_threshold
        self.faces = FaceMedia(models_root)
        self.progress_callback = None
        self.batch_attention = []

    def load(self):
        if self.model is not None:
            return
        import torch
        from .model import CLIPLoRAClassifier

        self.config = load_verified_config(self.directory)
        if self.requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.requested_device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU 当前不可用，未改用 CPU")
        model = CLIPLoRAClassifier(self.config)
        state = load_checkpoint_state(self.directory / "model.safetensors", self.device)
        # No missing or unexpected keys are tolerated, including frozen backbone tensors.
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.eval().to(self.device)

    def _input_tensor(self, rgb):
        import torch
        image = resize_rgb(rgb, int(self.config.get("clip_input_resolution", 224)))
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)
        mean = tensor.new_tensor(self.config["mean"])[:, None, None]
        std = tensor.new_tensor(self.config["std"])[:, None, None]
        return (tensor - mean) / std

    def _scores(self, tensors):
        import torch
        batch = torch.stack(tensors).to(self.device)
        self.batch_attention = []
        captured = []
        descriptor = getattr(self.model, "layer_descriptor", None)
        hook = descriptor.register_forward_hook(lambda module, inputs, output: captured.append(output[1].detach())) if descriptor is not None else None
        try:
            with torch.inference_mode():
                prediction = self.model({"image": batch}, inference=True)
        finally:
            if hook is not None:
                hook.remove()
        scores = prediction["prob"].float().cpu().tolist()
        routing = prediction["attn_weights"].float().cpu().tolist()
        if len(scores) != len(tensors) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            raise ValueError("FADR-CLIP 返回了无效分数")
        if captured:
            try:
                from .attention import combine_attention
                self.batch_attention = combine_attention(captured[-1].float().cpu().numpy(), routing)
            except (ValueError, TypeError, KeyError):
                logger.warning("空间注意力暂时不可用，检测分数已保留", exc_info=True)
        return scores, routing

    def predict(self, path, *, sha256="", check=None):
        if path is None:
            raise ValueError("FADR-CLIP 需要本地图片或视频文件")
        if check:
            check()
        progress = self.progress_callback
        samples, sampled_count, duration = self.faces.extract(
            Path(path), self.spec.media_type, check, **({"on_progress": progress} if progress else {}),
        )
        if progress:
            progress("model_loading")
        self.load()
        if check:
            check()
        if progress:
            progress("detecting")
        scores, routes = [], []
        batch_size = 4 if self.device == "cuda" else 1
        for offset in range(0, len(samples), batch_size):
            if check:
                check()
            batch = [self._input_tensor(sample.image) for sample in samples[offset:offset + batch_size]]
            self.batch_attention = []
            batch_scores, batch_routes = self._scores(batch)
            scores.extend(batch_scores)
            routes.extend(batch_routes)
            if progress:
                progress("scores", frames=[{"frame_index": sample.index, "score": round(value, 6),
                    "routing_weights": batch_routes[i], "routing_layers": [6, 12, 18, 24],
                    **({"attention": self.batch_attention[i]} if i < len(self.batch_attention) else {})}
                    for i, (sample, value) in enumerate(zip(samples[offset:offset + batch_size], batch_scores))])
            if check:
                check()
        score = sum(scores) / len(scores)
        evidence = {
            "source_checkpoint_sha256": FADR_CHECKPOINT_SHA256,
            "input_resolution": 224, "face_crop_resolution": 256,
            "input_face_mode": "raw",
            "preprocessing": {
                "source": "original upload; no detection-time downscale or lossy transcode",
                "face_alignment": "Always extract and align largest face; DeepfakeBench dlib 81 / five points / scale 1.3 / OpenCV warpAffine",
                "resize": "IsotropicResize INTER_AREA down / INTER_CUBIC up; black pad; center crop 224",
                "channel_order": "RGB",
                "mean": self.config["mean"], "std": self.config["std"],
            },
            "routing_layers": [6, 12, 18, 24],
            "routing_weights": np.mean(routes, axis=0).tolist(),
            "precision": "float32",
            "thresholds": {"authentic_max": self.authentic_threshold, "manipulated_min": self.manipulated_threshold},
        }
        if self.spec.media_type == "image":
            evidence.update(kind="image_regions", regions=[{**samples[0].region, "score": round(score, 6)}],
                            method="release RGB 224 preprocessing; whole-face FADR-CLIP score (not pixel localization)")
        else:
            evidence.update(
                kind="video_frames", sampled_frames=sampled_count, frames_with_face=len(samples),
                duration_seconds=duration,
                frames=[{"frame_index": sample.index, "timestamp_seconds": round(sample.timestamp, 3),
                         "score": round(value, 6), "routing_weights": route}
                        for sample, value, route in zip(samples, scores, routes)],
                method="original-video uniform frame-index sampling (up to 32); arithmetic mean of valid face-frame fake scores",
                timing="frame timestamps estimated from original frame index and nominal FPS; approximate for variable frame rate",
            )
        return DetectionResult(
            score=round(score, 6),
            verdict=verdict_from_score(score, authentic_threshold=self.authentic_threshold, manipulated_threshold=self.manipulated_threshold),
            evidence=evidence,
        )
