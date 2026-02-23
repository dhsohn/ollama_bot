"""Pydantic 기반 중앙 설정 로더.

.env 파일과 config.yaml을 병합하여 검증된 AppSettings를 반환한다.
모든 모듈은 이 설정 객체를 생성자 주입으로 받는다.
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta, timezone

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    ZoneInfo = None

    class ZoneInfoNotFoundError(Exception):
        """zoneinfo 미지원 환경에서의 대체 예외."""

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class BotConfig(BaseModel):
    name: str = "ollama_bot"
    language: str = "ko"
    max_conversation_length: int = 50
    response_timeout: int = 60


class OllamaConfig(BaseModel):
    host: str = "http://host.docker.internal:11434"
    model: str = "gpt-oss:20b"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = (
        "당신은 유용한 AI 어시스턴트입니다.\n"
        "한국어로 답변하며, 간결하고 정확한 정보를 제공합니다.\n"
    )


class TelegramConfig(BaseModel):
    polling_interval: int = 1
    max_message_length: int = 4096


class SecurityConfig(BaseModel):
    allowed_users: list[int] = Field(default_factory=list)
    rate_limit: int = 30
    max_file_size: int = 10_485_760
    blocked_paths: list[str] = Field(
        default_factory=lambda: ["/etc/*", "/proc/*", "/sys/*"]
    )


class MemoryConfig(BaseModel):
    backend: str = "sqlite"
    max_long_term_entries: int = 1000
    conversation_retention_days: int = 30


class SchedulerConfig(BaseModel):
    timezone: str = "Asia/Seoul"

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        if ZoneInfo is not None:
            try:
                ZoneInfo(value)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Invalid timezone: {value}") from exc
            return value

        # zoneinfo 미지원(Python 3.8 등) 환경 최소 지원
        fallback_timezones = {
            "UTC": timezone.utc,
            "Asia/Seoul": timezone(timedelta(hours=9), name="Asia/Seoul"),
        }
        if value not in fallback_timezones:
            raise ValueError(f"Invalid timezone: {value}")
        return value


class AppSettings(BaseSettings):
    """루트 설정. .env에서 시크릿을, config.yaml에서 나머지를 로드한다."""

    telegram_bot_token: str = ""
    allowed_telegram_users: str = ""
    ollama_host: str = "http://host.docker.internal:11434"
    ollama_model: str = "gpt-oss:20b"
    scheduler_timezone: str = "Asia/Seoul"
    log_level: str = "INFO"
    data_dir: str = "/app/data"

    bot: BotConfig = Field(default_factory=BotConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def load_config(
    config_path: str = "config/config.yaml",
    env_file: str | None = ".env",
) -> AppSettings:
    """config.yaml과 .env를 병합하여 AppSettings를 반환한다."""
    yaml_data: dict = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    # .env로부터 BaseSettings가 자동 로드 (env_file이 존재하면)
    env_kwargs: dict = {}
    if env_file and Path(env_file).exists():
        env_kwargs["_env_file"] = env_file

    settings = AppSettings(**env_kwargs)
    explicit_env_fields = set(settings.__pydantic_fields_set__)

    # YAML 데이터를 서브 모델에 오버레이
    if "bot" in yaml_data:
        settings.bot = BotConfig(**yaml_data["bot"])
    if "ollama" in yaml_data:
        settings.ollama = OllamaConfig(**yaml_data["ollama"])
    if "telegram" in yaml_data:
        settings.telegram = TelegramConfig(**yaml_data["telegram"])
    if "security" in yaml_data:
        settings.security = SecurityConfig(**yaml_data["security"])
    if "memory" in yaml_data:
        settings.memory = MemoryConfig(**yaml_data["memory"])
    if "scheduler" in yaml_data:
        settings.scheduler = SchedulerConfig(**yaml_data["scheduler"])

    # 명시적으로 지정된 env 값만 YAML보다 우선
    if "ollama_host" in explicit_env_fields and settings.ollama_host:
        settings.ollama.host = settings.ollama_host
    if "ollama_model" in explicit_env_fields and settings.ollama_model:
        settings.ollama.model = settings.ollama_model
    if "scheduler_timezone" in explicit_env_fields and settings.scheduler_timezone:
        settings.scheduler = SchedulerConfig(timezone=settings.scheduler_timezone)

    # ALLOWED_TELEGRAM_USERS CSV → security.allowed_users 리스트
    if settings.allowed_telegram_users:
        raw_ids = [uid.strip() for uid in settings.allowed_telegram_users.split(",") if uid.strip()]
        invalid_ids = [uid for uid in raw_ids if not uid.isdigit()]
        if invalid_ids:
            raise ValueError(
                "ALLOWED_TELEGRAM_USERS에는 숫자 Chat ID만 사용할 수 있습니다: "
                f"{', '.join(invalid_ids)}"
            )
        user_ids = [int(uid) for uid in raw_ids]
        settings.security.allowed_users = user_ids

    return settings
