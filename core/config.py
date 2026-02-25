"""Pydantic 기반 중앙 설정 로더.

.env 파일과 config.yaml을 병합하여 검증된 AppSettings를 반환한다.
모든 모듈은 이 설정 객체를 생성자 주입으로 받는다.
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta, timezone
from typing import Any

_ZoneInfo: Any
_ZoneInfoNotFoundError: type[Exception]
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    from zoneinfo import ZoneInfoNotFoundError as _ZoneInfoNotFoundError
except ImportError:
    _ZoneInfo = None

    class _ZoneInfoNotFoundErrorFallback(Exception):
        """zoneinfo 미지원 환경에서의 대체 예외."""

    _ZoneInfoNotFoundError = _ZoneInfoNotFoundErrorFallback

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
    num_ctx: int = 8192
    prompt_version: str = "v1"
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
    max_concurrent_requests: int = 4
    max_input_length: int = 10_000
    max_file_size: int = 10_485_760
    blocked_paths: list[str] = Field(
        default_factory=lambda: ["/etc/*", "/proc/*", "/sys/*"]
    )

    @field_validator(
        "rate_limit", "max_concurrent_requests", "max_input_length", "max_file_size"
    )
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("security numeric settings must be >= 1")
        return value


class MemoryConfig(BaseModel):
    backend: str = "sqlite"
    max_long_term_entries: int = 1000
    conversation_retention_days: int = 30


class FeedbackConfig(BaseModel):
    enabled: bool = True
    show_buttons: bool = True
    min_feedback_for_analysis: int = 5
    max_guidelines: int = 5
    preview_max_chars: int = 300
    preview_cache_max_size: int = 500
    preview_cache_ttl_hours: int = 24
    retention_days: int = 90
    collect_reason: bool = True
    reason_min_chars: int = 3
    reason_max_chars: int = 500
    reason_timeout_seconds: int = 120
    dicl_enabled: bool = True
    dicl_max_examples: int = 2
    dicl_max_keywords: int = 5
    dicl_max_total_chars: int = 2000
    dicl_recent_days: int = 180

    @field_validator(
        "min_feedback_for_analysis",
        "max_guidelines",
        "preview_max_chars",
        "preview_cache_max_size",
        "preview_cache_ttl_hours",
        "retention_days",
        "reason_min_chars",
        "reason_max_chars",
        "reason_timeout_seconds",
        "dicl_max_examples",
        "dicl_max_keywords",
        "dicl_max_total_chars",
        "dicl_recent_days",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("feedback numeric settings must be >= 1")
        return value


class AutoEvaluationConfig(BaseModel):
    enabled: bool = False
    daily_limit: int = 50
    min_response_length: int = 50
    max_concurrency: int = 2
    cooldown_seconds: int = 300

    @field_validator(
        "daily_limit",
        "min_response_length",
        "max_concurrency",
        "cooldown_seconds",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("auto_evaluation numeric settings must be >= 1")
        return value


class SchedulerConfig(BaseModel):
    timezone: str = "Asia/Seoul"

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        if _ZoneInfo is not None:
            try:
                _ZoneInfo(value)
            except _ZoneInfoNotFoundError as exc:
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


class InstantResponderConfig(BaseModel):
    enabled: bool = True
    rules_path: str = "config/instant_rules.yaml"


class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    model_name: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    similarity_threshold: float = 0.92
    min_query_chars: int = 4
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        r"(지금|현재)\s*몇\s*시",
        r"오늘\s*(날짜|며칠|요일)",
    ])
    max_entries: int = 5000
    ttl_hours: int = 168
    invalidate_on_negative_feedback: bool = True


class IntentRouterConfig(BaseModel):
    enabled: bool = True
    routes_path: str = "config/intent_routes.yaml"
    min_confidence: float = 0.75
    encoder_model: str = "intfloat/multilingual-e5-small"


class ContextCompressorConfig(BaseModel):
    enabled: bool = True
    recent_keep: int = 10
    summary_refresh_interval: int = 10
    summary_max_tokens: int = 200
    background_summarize: bool = True
    archive_enabled: bool = True
    summarize_concurrency: int = 1
    run_only_when_idle: bool = True


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
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    auto_evaluation: AutoEvaluationConfig = Field(default_factory=AutoEvaluationConfig)
    instant_responder: InstantResponderConfig = Field(default_factory=InstantResponderConfig)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    intent_router: IntentRouterConfig = Field(default_factory=IntentRouterConfig)
    context_compressor: ContextCompressorConfig = Field(default_factory=ContextCompressorConfig)

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
    if "feedback" in yaml_data:
        settings.feedback = FeedbackConfig(**yaml_data["feedback"])
    if "auto_evaluation" in yaml_data:
        settings.auto_evaluation = AutoEvaluationConfig(**yaml_data["auto_evaluation"])
    if "instant_responder" in yaml_data:
        settings.instant_responder = InstantResponderConfig(**yaml_data["instant_responder"])
    if "semantic_cache" in yaml_data:
        settings.semantic_cache = SemanticCacheConfig(**yaml_data["semantic_cache"])
    if "intent_router" in yaml_data:
        settings.intent_router = IntentRouterConfig(**yaml_data["intent_router"])
    if "context_compressor" in yaml_data:
        settings.context_compressor = ContextCompressorConfig(**yaml_data["context_compressor"])

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
