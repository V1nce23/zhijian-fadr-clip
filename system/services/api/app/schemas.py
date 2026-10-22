import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

from app.models import AnalysisStatus, DetectorStatus, MediaType, UploadStatus, UserRole, Verdict


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    name: str
    role: UserRole
    is_active: bool
    created_at: datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=10, max_length=128)
    role: UserRole = UserRole.ANALYST


class UploadPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_name: str
    mime_type: str
    media_type: MediaType
    size_bytes: int
    sha256: str
    status: UploadStatus
    created_at: datetime


class DetectorRunPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    detector_id: str
    model_version: str
    preprocessing_version: str
    status: DetectorStatus
    progress: int
    score: float | None
    verdict: Verdict | None
    evidence: dict[str, Any] | None
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None


class CustomThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_threshold: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    frame_threshold: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def check_precision(self):
        if any(value != round(value, 6) for value in (self.video_threshold, self.frame_threshold)):
            raise ValueError("阈值最多支持 6 位小数")
        return self


class AnalysisCreate(BaseModel):
    upload_id: uuid.UUID
    device: Literal["cpu", "cuda"] = "cpu"
    threshold_mode: Literal["balanced", "low_false_positive", "low_false_negative", "custom"] = "balanced"
    custom_thresholds: CustomThresholds | None = None

    @model_validator(mode="after")
    def check_custom_mode(self):
        if (self.threshold_mode == "custom") != (self.custom_thresholds is not None):
            raise ValueError("自定义模式必须填写视频与图片／单帧阈值；预设模式不能携带自定义阈值")
        return self


class ThresholdModePublic(BaseModel):
    mode: Literal["balanced", "low_false_positive", "low_false_negative", "custom"]
    label: str
    description: str
    video_threshold: float
    frame_threshold: float


class DetectionDevices(BaseModel):
    worker_ready: bool
    cpu: bool
    gpu: bool
    gpu_name: str | None = None


class AnalysisBatchDeleteRequest(BaseModel):
    analysis_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)


class AnalysisBatchDeleteResult(BaseModel):
    message: str
    deleted: int
    skipped: int


class AnalysisPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: AnalysisStatus
    progress: int
    requested_device: Literal["cpu", "cuda"] | None
    executed_device: Literal["cpu", "cuda"] | None
    input_face_mode: Literal["raw", "aligned"] | None
    threshold_policy: dict[str, Any] | None
    verdict: Verdict | None
    score: float | None
    summary: str | None
    cancel_requested: bool
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    upload: UploadPublic
    detector_runs: list[DetectorRunPublic]


class AnalysisListItem(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus
    progress: int
    requested_device: Literal["cpu", "cuda"] | None
    executed_device: Literal["cpu", "cuda"] | None
    input_face_mode: Literal["raw", "aligned"] | None
    threshold_policy: dict[str, Any] | None
    verdict: Verdict | None
    score: float | None
    created_at: datetime
    completed_at: datetime | None
    upload: UploadPublic


class MessageResponse(BaseModel):
    message: str
