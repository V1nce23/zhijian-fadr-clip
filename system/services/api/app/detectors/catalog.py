from app.detectors.base import DetectorSpec

VIDEO_FRAME_COUNT = 32
FADR_CHECKPOINT_SHA256 = "b0ba99d06f0399e544bd45dbff4cdaa48f771d2324e3bacc78007fbff0597a09"
FADR_MODEL_VERSION = "best-b0ba99d0"


def detector_specs(media_type: str) -> list[DetectorSpec]:
    """New analyses run only the supplied FADR-CLIP best main model."""
    if media_type not in ("image", "video"):
        raise ValueError(f"unsupported media type: {media_type}")
    preprocessing = "dfb-face256-clip224-v3"
    if media_type == "video":
        preprocessing += f"-uniform{VIDEO_FRAME_COUNT}-mean"
    return [DetectorSpec(
        detector_id="fadr-clip", model_version=FADR_MODEL_VERSION,
        preprocessing_version=preprocessing, media_type=media_type,
        repository="local/FADR_CLIP", revision=FADR_CHECKPOINT_SHA256,
    )]
