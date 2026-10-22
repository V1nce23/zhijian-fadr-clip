from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    model_version: str
    preprocessing_version: str
    media_type: str
    repository: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class DetectionResult:
    score: float
    verdict: str
    evidence: dict[str, Any]


class DetectorAdapter(Protocol):
    spec: DetectorSpec

    def predict(
        self,
        path: Path | None,
        *,
        sha256: str = "",
        check: Callable[[], None] | None = None,
    ) -> DetectionResult: ...


def verdict_from_score(
    score: float,
    *,
    authentic_threshold: float = 0.35,
    manipulated_threshold: float = 0.65,
) -> str:
    if score >= manipulated_threshold:
        return "likely_manipulated"
    if score <= authentic_threshold:
        return "likely_authentic"
    return "uncertain"
