"""Keep original pixels for detection; encode compatible previews separately."""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from app.services.inference import InferenceTimedOut, run_command


class MediaConversionError(ValueError):
    pass


OUTPUT_SUFFIX = {"image": ".png", "video": ".mp4"}


def _run(
    command: list[str], *, timeout: int = 600, check: Callable[[], None] | None = None
) -> bytes:
    try:
        return run_command(command, timeout=timeout, check=check)
    except InferenceTimedOut:
        raise
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise MediaConversionError(
            "媒体文件无法解码或转换，请检查文件是否完整"
        ) from exc


def normalize_media(
    source: Path,
    media_type: str,
    directory: Path,
    *,
    check: Callable[[], None] | None = None,
) -> Path:
    """Compatibility conversion for previews only; never feed its video to inference."""
    if media_type not in OUTPUT_SUFFIX:
        raise MediaConversionError("不支持的媒体类型")

    try:
        probe = json.loads(
            _run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source)],
                timeout=30,
                check=check,
            )
        )
    except (ValueError, KeyError) as exc:
        raise MediaConversionError("无法读取媒体文件信息") from exc

    if not any(
        stream.get("codec_type") == "video" for stream in probe.get("streams", [])
    ):
        raise MediaConversionError(
            "文件中没有可检测的图像内容"
        )

    destination = directory / f"normalized{OUTPUT_SUFFIX[media_type]}"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source)]
    if media_type == "image":
        command += [
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-vf",
            "scale=w='min(2048,iw)':h='min(2048,ih)':force_original_aspect_ratio=decrease,format=rgb24",
            "-c:v",
            "png",
            "-update",
            "1",
            str(destination),
        ]
    elif media_type == "video":
        command += [
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            "scale=w='min(1024,iw)':h='min(1024,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(destination),
        ]

    _run(command, check=check)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise MediaConversionError("媒体转换后没有可检测的内容")
    return destination


def prepare_detection_media(source: Path, media_type: str, directory: Path, *, check=None) -> Path:
    """Validate original media; decode unusual still formats without downscaling."""
    if media_type not in OUTPUT_SUFFIX:
        raise MediaConversionError("不支持的媒体类型")
    try:
        probe = json.loads(_run([
            "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source),
        ], timeout=30, check=check))
    except (ValueError, KeyError) as exc:
        raise MediaConversionError("无法读取媒体文件信息") from exc
    if not any(s.get("codec_type") == "video" for s in probe.get("streams", [])):
        raise MediaConversionError("文件中没有可检测的图像内容")
    if media_type == "video":
        return source  # Original codec, resolution and frame order.
    import cv2
    if cv2.imread(str(source), cv2.IMREAD_COLOR) is not None:
        return source
    # Decode first still frame losslessly, with no resize or lossy re-encoding.
    destination = directory / "decoded.png"
    _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
          "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb24",
          "-c:v", "png", "-update", "1", str(destination)], check=check)
    if not destination.is_file() or cv2.imread(str(destination), cv2.IMREAD_COLOR) is None:
        raise MediaConversionError("图片无法解码，请检查文件是否完整")
    return destination
