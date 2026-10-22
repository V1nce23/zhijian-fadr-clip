import pytest

from app.detectors.base import verdict_from_score
from app.detectors.catalog import FADR_CHECKPOINT_SHA256, detector_specs


@pytest.mark.parametrize("media", ["image", "video"])
def test_only_the_best_fadr_main_model_is_configured(media):
    specs = detector_specs(media)
    assert len(specs) == 1
    assert specs[0].detector_id == "fadr-clip"
    assert specs[0].model_version == "best-b0ba99d0"
    assert specs[0].revision == FADR_CHECKPOINT_SHA256
    assert specs[0].media_type == media


@pytest.mark.parametrize("media_type", ["text", "audio"])
def test_unsupported_media_type_raises(media_type) -> None:
    with pytest.raises(ValueError):
        detector_specs(media_type)


def test_uncertain_band_is_preserved() -> None:
    assert verdict_from_score(0.2) == "likely_authentic"
    assert verdict_from_score(0.5) == "uncertain"
    assert verdict_from_score(0.8) == "likely_manipulated"
