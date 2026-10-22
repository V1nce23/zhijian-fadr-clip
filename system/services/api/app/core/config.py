from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "智鉴：伪造人脸检测系统 API"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./deeptrace.db"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    s3_endpoint_url: str = "http://localhost:9000"
    s3_public_endpoint_url: str | None = None
    s3_access_key: str = "deeptrace"
    s3_secret_key: str = "deeptrace-local-secret"
    s3_bucket: str = "deeptrace-uploads"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    session_cookie_name: str = "deeptrace_session"
    session_days: int = 7
    cookie_secure: bool = False
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    max_upload_bytes: int = 256 * 1024 * 1024
    detector_queue: str = "detect.gpu"
    detector_models_root: str = "/models"
    detector_device: str = "auto"
    detector_local_files_only: bool = True
    detector_authentic_threshold: float = 0.35
    detector_manipulated_threshold: float = 0.65
    detector_timeout_seconds: int = Field(default=120, ge=1)
    analysis_timeout_seconds: int = Field(default=900, ge=1)

    admin_email: str | None = None
    admin_password: str | None = None
    admin_name: str = "系统管理员"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
