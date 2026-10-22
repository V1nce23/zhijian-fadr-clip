import json
import subprocess
from pathlib import Path

import pytest

from app.services.media import MediaConversionError, normalize_media, prepare_detection_media
from app.api.routes.uploads import _media_type_for_mime
from app.models import MediaType
from app.tasks.analysis import run_analysis


def ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True
    )


def stream(path: Path) -> dict:
    output = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
    )
    return json.loads(output)["streams"][0]


@pytest.mark.parametrize("suffix", ["bmp", "gif", "jpg", "tiff", "webp"])
def test_image_formats_are_normalized_to_rgb_png(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"source.{suffix}"
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=64x48:d=1", "-frames:v", "1", str(source)
    )

    result = normalize_media(source, "image", tmp_path)

    assert result.suffix == ".png"
    assert stream(result)["pix_fmt"] == "rgb24"
    assert result.stat().st_size > 0


@pytest.mark.parametrize("suffix", ["avi", "mov", "webm"])
def test_video_formats_are_normalized_to_h264_mp4(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"source.{suffix}"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x48:r=5:d=0.4", str(source))

    result = normalize_media(source, "video", tmp_path)

    assert result.suffix == ".mp4"
    assert stream(result)["codec_name"] == "h264"
    assert stream(result)["pix_fmt"] == "yuv420p"


def test_audio_normalization_is_rejected_before_any_decode(tmp_path, monkeypatch):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("retired modality must not run ffprobe or ffmpeg")
    monkeypatch.setattr("app.services.media._run", unexpected_decode)
    with pytest.raises(MediaConversionError, match="不支持的媒体类型"):
        normalize_media(tmp_path / "source.wav", "audio", tmp_path)


def test_audio_only_mp4_is_rejected_before_inference(tmp_path):
    source = tmp_path / "sound.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", "-c:a", "aac", str(source))
    with pytest.raises(MediaConversionError, match="没有可检测的图像"):
        normalize_media(source, "video", tmp_path)


def test_invalid_media_fails_before_inference(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp4"
    source.write_text("not a video")

    with pytest.raises(MediaConversionError):
        normalize_media(source, "video", tmp_path)


def test_generic_mime_uses_known_media_extension() -> None:
    assert (
        _media_type_for_mime("application/octet-stream", "clip.mkv") == MediaType.VIDEO
    )
    assert _media_type_for_mime("application/ogg", "voice.ogg") is None
    assert _media_type_for_mime("application/ogg", "clip.ogv") == MediaType.VIDEO
    assert _media_type_for_mime("application/octet-stream", "report.pdf") is None


def test_video_analysis_reads_original_bytes_without_transcoding(
    tmp_path: Path, client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import fake_get_detector

    source = tmp_path / "source.webm"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x48:r=5:d=0.4", str(source))
    monkeypatch.setattr("app.tasks.analysis.prepare_detection_media", prepare_detection_media)
    seen_paths: set[Path] = set()

    def detector_factory(media_type, spec, *args, **kwargs):
        detector = fake_get_detector(media_type, spec, *args, **kwargs)
        original_predict = detector.predict

        def predict(path, *, sha256="", check=None):
            assert path.suffix == ".webm"
            assert path.read_bytes() == source.read_bytes()
            assert path.is_file()
            seen_paths.add(path)
            return original_predict(path, sha256=sha256, check=check)

        detector.predict = predict
        return detector

    monkeypatch.setattr("app.tasks.analysis.get_detector", detector_factory)
    client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "StrongPass123!"},
    )
    upload = client.post(
        "/api/v1/uploads",
        files={"file": ("source.webm", source.read_bytes(), "video/webm")},
    )
    assert upload.status_code == 201
    analysis = client.post("/api/v1/analyses", json={"upload_id": upload.json()["id"]})
    assert analysis.status_code == 201

    run_analysis(analysis.json()["id"])

    report = client.get(f"/api/v1/analyses/{analysis.json()['id']}").json()
    assert report["status"] == "completed"
    assert len(seen_paths) == 1


def test_corrupt_upload_reports_preprocessing_failure(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_workflow import PNG_BYTES

    monkeypatch.setattr("app.tasks.analysis.prepare_detection_media", prepare_detection_media)
    client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "StrongPass123!"},
    )
    upload = client.post(
        "/api/v1/uploads",
        files={"file": ("broken.png", PNG_BYTES, "image/png")},
    )
    assert upload.status_code == 201
    analysis = client.post("/api/v1/analyses", json={"upload_id": upload.json()["id"]})
    run_analysis(analysis.json()["id"])

    report = client.get(f"/api/v1/analyses/{analysis.json()['id']}").json()
    assert report["status"] == "failed"
    assert "无法解码或转换" in report["summary"]
    assert all(run["status"] == "failed" for run in report["detector_runs"])


@pytest.mark.parametrize("mime,suffix", [("audio/wav","wav"), ("audio/mpeg","mp3"), ("audio/flac","flac"), ("audio/ogg","ogg"), ("audio/mp4","m4a")])
def test_audio_mime_and_extension_fallback_are_unsupported(mime, suffix):
    assert _media_type_for_mime(mime, f"sound.{suffix}") is None
    assert _media_type_for_mime("application/octet-stream", f"sound.{suffix}") is None
