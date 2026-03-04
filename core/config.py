"""Pydantic 기반 중앙 설정 로더.

.env 파일과 config.yaml을 병합하여 검증된 AppSettings를 반환한다.
모든 모듈은 이 설정 객체를 생성자 주입으로 받는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    num_ctx: int = 8192
    prompt_version: str = "v1"
    system_prompt: str = (
        "당신은 유용한 AI 어시스턴트입니다.\n"
        "한국어로 답변하며, 간결하고 정확한 정보를 제공합니다.\n"
    )


class LemonadeConfig(BaseModel):
    host: str = "http://host.docker.internal:8000"
    api_key: str = ""
    default_model: str = "gpt-oss-20b-NPU"
    base_path: str = "/api/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    prompt_version: str = "v1"
    system_prompt: str = (
        "당신은 유용한 AI 어시스턴트입니다.\n"
        "한국어로 답변하며, 간결하고 정확한 정보를 제공합니다.\n"
    )
    timeout_seconds: int = 60
    model_load_timeout_seconds: int = 120
    heavy_model_load_timeout_seconds: int = 420
    reconnect_cooldown_seconds: float = 15.0

    @field_validator(
        "timeout_seconds",
        "model_load_timeout_seconds",
        "heavy_model_load_timeout_seconds",
    )
    @classmethod
    def validate_timeout_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lemonade timeout settings must be >= 1")
        return value


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


class RuntimeMaintenanceConfig(BaseModel):
    """런타임 유지보수 루프 주기 설정."""

    memory_maintenance_interval_seconds: int = 6 * 60 * 60
    llm_recovery_interval_seconds: int = 60
    memory_maintenance_jitter_ratio: float = 0.1

    @field_validator(
        "memory_maintenance_interval_seconds",
        "llm_recovery_interval_seconds",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("runtime maintenance intervals must be >= 1")
        return value

    @field_validator("memory_maintenance_jitter_ratio")
    @classmethod
    def validate_jitter_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("memory_maintenance_jitter_ratio must be between 0.0 and 1.0")
        return value


class SchedulerConfig(BaseModel):
    timezone: str = "Asia/Seoul"

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid timezone: {value}") from exc
        return value


class InstantResponderConfig(BaseModel):
    enabled: bool = True
    rules_path: str = "config/instant_rules.yaml"


class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
    encoder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class ContextCompressorConfig(BaseModel):
    enabled: bool = True
    recent_keep: int = 10
    summary_refresh_interval: int = 10
    summary_max_tokens: int = 200
    background_summarize: bool = True
    archive_enabled: bool = True
    summarize_concurrency: int = 1
    run_only_when_idle: bool = True


class RetrievalProviderConfig(BaseModel):
    """Ollama 기반 retrieval(임베딩/리랭킹) 전용 프로바이더 설정."""

    host: str = "http://host.docker.internal:11434"
    embedding_model: str = "Qwen3-Embedding-0.6B-GGUF"
    reranker_model: str = "bge-reranker-v2-m3-GGUF"


class ModelRegistryConfig(BaseModel):
    """모델 레지스트리 설정 (단일 기본 모델 + retrieval 모델)."""

    default_model: str = "gpt-oss-20b-NPU"
    embedding_model: str = "Qwen3-Embedding-0.6B-GGUF"
    reranker_model: str = "bge-reranker-v2-m3-GGUF"


class RAGConfig(BaseModel):
    """RAG 파이프라인 설정."""

    enabled: bool = True
    kb_dirs: list[str] = Field(default_factory=lambda: ["./kb"])
    startup_index_enabled: bool = True
    index_dir: str = ""
    max_file_size_mb: int = 16
    chunk_min_tokens: int = 500
    chunk_max_tokens: int = 1200
    chunk_overlap_ratio: float = 0.15
    retrieve_k0: int = 40
    rerank_enabled: bool = True
    rerank_topk: int = 8
    rerank_budget_ms: int = 1200
    retrieval_score_floor: float = 0.3
    supported_extensions: list[str] = Field(default_factory=lambda: [
        ".md", ".txt", ".docx", ".html", ".htm",
        ".json", ".csv", ".py", ".js", ".ts",
        ".out", ".log",
    ])
    trigger_keywords: list[str] = Field(default_factory=lambda: [
        "문서", "프로젝트", "레포", "노트", "폴더", "논문",
        "결과", "출처", "인용", "어디에 적혀", "내 파일",
        "지식베이스", "kb", "검색해",
    ])

    @field_validator("chunk_overlap_ratio")
    @classmethod
    def validate_overlap(cls, value: float) -> float:
        if not 0.0 <= value <= 0.5:
            raise ValueError("chunk_overlap_ratio must be between 0.0 and 0.5")
        return value

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_max_file_size_mb(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_file_size_mb must be >= 1")
        return value


class DFTConfig(BaseModel):
    """DFT 계산 결과 인덱스 설정."""

    enabled: bool = True
    auto_index_on_startup: bool = True
    max_file_size_mb: int = 64


class SimToolConfig(BaseModel):
    """시뮬레이션 도구별 설정."""

    enabled: bool = True
    executable: str
    cli_template: str
    command_prefix: str = ""
    default_cores: int = 4
    default_memory_mb: int = 8192
    max_cores: int = 16
    max_memory_mb: int = 65536
    output_extension: str = ".out"
    env_vars: dict[str, str] = Field(default_factory=dict)

    @field_validator("default_cores", "max_cores")
    @classmethod
    def validate_positive_cores(cls, value: int) -> int:
        if value < 1:
            raise ValueError("core count must be >= 1")
        return value

    @field_validator("default_memory_mb", "max_memory_mb")
    @classmethod
    def validate_positive_memory(cls, value: int) -> int:
        if value < 1:
            raise ValueError("memory must be >= 1 MB")
        return value


class SimQueueConfig(BaseModel):
    """시뮬레이션 작업 큐 설정."""

    enabled: bool = False
    total_cores: int = 16
    total_memory_mb: int = 131072
    max_concurrent_jobs: int = 4
    default_retry_count: int = 2
    max_retry_count: int = 5
    retry_delay_seconds: int = 30
    queue_check_interval_seconds: int = 5
    external_agent_enabled: bool = False
    external_agent_base_url: str = "http://sim_host_agent:18081"
    external_agent_timeout_seconds: float = 3.0
    external_agent_token_env: str = "SIM_EXTERNAL_AGENT_TOKEN"
    job_work_dir: str = "data/sim_jobs"
    tools: dict[str, SimToolConfig] = Field(default_factory=dict)

    @field_validator("total_cores", "max_concurrent_jobs")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("simulation queue numeric settings must be >= 1")
        return value

    @field_validator("total_memory_mb")
    @classmethod
    def validate_memory(cls, value: int) -> int:
        if value < 1:
            raise ValueError("total_memory_mb must be >= 1")
        return value

    @field_validator("external_agent_timeout_seconds")
    @classmethod
    def validate_external_agent_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("external_agent_timeout_seconds must be > 0")
        return value


class AppSettings(BaseModel):
    """루트 설정. 런타임 설정은 YAML에서, 텔레그램 시크릿은 .env에서 로드한다."""

    telegram_bot_token: str = ""
    allowed_telegram_users: str = ""
    log_level: str = "INFO"
    data_dir: str = "/app/data"

    bot: BotConfig = Field(default_factory=BotConfig)
    lemonade: LemonadeConfig = Field(default_factory=LemonadeConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    runtime_maintenance: RuntimeMaintenanceConfig = Field(default_factory=RuntimeMaintenanceConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    auto_evaluation: AutoEvaluationConfig = Field(default_factory=AutoEvaluationConfig)
    instant_responder: InstantResponderConfig = Field(default_factory=InstantResponderConfig)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    intent_router: IntentRouterConfig = Field(default_factory=IntentRouterConfig)
    context_compressor: ContextCompressorConfig = Field(default_factory=ContextCompressorConfig)
    ollama: RetrievalProviderConfig = Field(default_factory=RetrievalProviderConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    dft: DFTConfig = Field(default_factory=DFTConfig)
    sim_queue: SimQueueConfig = Field(default_factory=SimQueueConfig)


class _TelegramEnvSecrets(BaseSettings):
    """텔레그램 관련 시크릿만 .env에서 로드한다."""

    telegram_bot_token: str = ""
    allowed_telegram_users: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def load_config(
    config_path: str = "config/config.yaml",
    env_file: str | Sequence[str] | None = ".env",
) -> AppSettings:
    """config.yaml과 .env(텔레그램 시크릿 전용)를 병합하여 AppSettings를 반환한다."""
    yaml_data: dict = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    # .env 계열 파일로부터 BaseSettings를 로드한다.
    env_kwargs: dict = {}
    env_files: list[str] = []
    if env_file is not None:
        if isinstance(env_file, str):
            candidates = [env_file]
        else:
            candidates = [str(item) for item in env_file]
        env_files = [candidate for candidate in candidates if candidate and Path(candidate).exists()]

    if len(env_files) == 1:
        env_kwargs["_env_file"] = env_files[0]
    elif len(env_files) > 1:
        env_kwargs["_env_file"] = tuple(env_files)
    else:
        env_kwargs["_env_file"] = None

    # 일반 런타임 설정은 YAML 기준으로만 로드한다.
    if not isinstance(yaml_data, dict):
        raise ValueError("config.yaml 최상위 구조는 mapping(dict)이어야 합니다.")
    settings = AppSettings.model_validate(yaml_data)

    # .env는 텔레그램 관련 시크릿만 반영한다.
    env_secrets = _TelegramEnvSecrets(**env_kwargs)
    if env_secrets.telegram_bot_token:
        settings.telegram_bot_token = env_secrets.telegram_bot_token
    if env_secrets.allowed_telegram_users:
        settings.allowed_telegram_users = env_secrets.allowed_telegram_users

    # ALLOWED_TELEGRAM_USERS CSV → security.allowed_users 리스트
    if settings.allowed_telegram_users:
        raw_ids = [uid.strip() for uid in settings.allowed_telegram_users.split(",") if uid.strip()]
        user_ids: list[int] = []
        invalid_ids: list[str] = []
        for raw in raw_ids:
            try:
                user_ids.append(int(raw))
            except ValueError:
                invalid_ids.append(raw)
        if invalid_ids:
            raise ValueError(
                "ALLOWED_TELEGRAM_USERS에는 정수 Chat ID만 사용할 수 있습니다: "
                f"{', '.join(invalid_ids)}"
            )
        settings.security.allowed_users = user_ids

    return settings
