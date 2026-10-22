"""Original-pixel decoding and DeepfakeBench frame sampling / face alignment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from app.detectors.catalog import VIDEO_FRAME_COUNT
from app.detectors.faces import DlibFaceAligner


@dataclass
class FaceSample:
    image: np.ndarray
    index: int
    timestamp: float
    region: dict


def uniform_frame_indices(total: int, limit: int = VIDEO_FRAME_COUNT) -> list[int]:
    if total < 1:
        raise ValueError("视频中没有可检测的画面")
    return np.linspace(0, total - 1, min(total, limit), endpoint=True, dtype=int).tolist()


class FaceMedia:
    def __init__(self, models_root: Path):
        self.predictor_path = models_root / "face" / "shape_predictor_81_face_landmarks.dat"
        self._aligner = None

    def _face(self, rgb, index, timestamp):
        if self._aligner is None:
            self._aligner = DlibFaceAligner(self.predictor_path)
        result = self._aligner.align_with_region(rgb, size=256)
        return FaceSample(result[0], index, timestamp, result[1]) if result else None

    def extract(self, path: Path, media_type: str, check=None, *, on_progress=None):
        if check:
            check()
        if media_type == "image":
            # Match the original release's disk decoder and BGR -> RGB conversion.
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("图片无法解码，请检查文件是否完整")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if on_progress:
                on_progress("plan", frames=[{"frame_index": 0, "timestamp_seconds": 0.0}])
                on_progress("sampled", frame_index=0, image=rgb)
            sample = self._face(rgb, 0, 0.0)
            if on_progress:
                on_progress("aligned", frame_index=0, image=sample.image if sample else None)
            if sample is None:
                raise ValueError("未在图片中检测到可对齐的人脸")
            return [sample], 1, 0.0
        if media_type != "video":
            raise ValueError("仅支持图片或视频")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("视频无法解码，请检查文件是否完整或编码是否受支持")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if total < 1:
                total = 0
                # Containers without frame-count metadata are counted without encoding.
                while capture.grab():
                    total += 1
                    if check:
                        check()
                capture.release()
                capture = cv2.VideoCapture(str(path))
            indices = uniform_frame_indices(total)
            if on_progress:
                on_progress("plan", frames=[{"frame_index": index, "timestamp_seconds": round(index / fps, 3)
                    if np.isfinite(fps) and fps > 0 else 0.0} for index in indices])
            selected = set(indices)
            samples, decoded_selected = [], 0
            # Sequential reads match DeepfakeBench and avoid inexact codec seeking.
            # Keep only one full-resolution frame in memory at a time.
            for index in range(total):
                if check:
                    check()
                success, bgr = capture.read()
                if not success:
                    raise ValueError("视频抽帧不完整，请检查文件是否完整")
                if index not in selected:
                    continue
                decoded_selected += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = index / fps if np.isfinite(fps) and fps > 0 else 0.0
                if on_progress:
                    on_progress("sampled", frame_index=index, image=rgb)
                sample = self._face(rgb, index, timestamp)
                if on_progress:
                    on_progress("aligned", frame_index=index, image=sample.image if sample else None)
                if sample is not None:
                    samples.append(sample)
            if decoded_selected != len(indices):
                raise ValueError("视频抽帧不完整，请检查文件是否完整")
            if not samples:
                raise ValueError("未在视频帧中检测到可对齐的人脸")
            duration = total / fps if np.isfinite(fps) and fps > 0 else 0.0
            return samples, len(indices), duration
        finally:
            capture.release()
