"""Pydantic 기반 중앙 설정 로더.

config.yaml 단일 파일로부터 검증된 AppSettings를 반환한다.
모든 모듈은 이 설정 객체를 생성자 주입으로 받는다.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, field_validator


class BotConfig(BaseModel):
    name: str = "ollama_bot"
    language: str = "ko"
    llm_provider: str = "lemonade"  # "lemonade" | "openai" | "ollama"
    max_conversation_length: int = 50
    response_timeout: int = 60

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"lemonade", "openai", "ollama"}:
            raise ValueError(
                f"llm_provider must be one of: lemonade, openai, ollama (got '{value}')"
            )
        return normalized


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
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
    host: str = "http://localhost:8000"
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


class OpenAIConfig(BaseModel):
    """OpenAI-compatible API provider settings."""

    host: str = "https://api.openai.com"
    api_key: str = ""
    default_model: str = "gpt-4o-mini"
    base_path: str = "/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = (
        "You are a helpful AI assistant.\n"
        "Provide concise and accurate information.\n"
    )
    timeout_seconds: int = 120
    model_load_timeout_seconds: int = 120
    heavy_model_load_timeout_seconds: int = 420
    reconnect_cooldown_seconds: float = 15.0


class TelegramConfig(BaseModel):
    bot_token: str = ""
    allowed_users: str = ""
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
    """Ollama 기반 retrieval(임베딩/리랭킹) 전용 프로바이더 설정.

    When ``bot.llm_provider`` is ``"ollama"``, the ``chat_model`` field
    is also used for chat completions via OllamaClient.
    """

    host: str = "http://localhost:11434"
    embedding_model: str = "Qwen3-Embedding-0.6B-GGUF"
    reranker_model: str = "bge-reranker-v2-m3-GGUF"
    chat_model: str = ""
    chat_temperature: float = 0.7
    chat_max_tokens: int = 4096
    chat_num_ctx: int = 8192
    chat_system_prompt: str = (
        "You are a helpful AI assistant.\n"
        "Provide concise and accurate information.\n"
    )


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


class AppSettings(BaseModel):
    """루트 설정. config.yaml 단일 파일에서 로드한다."""

    strict_startup: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"

    bot: BotConfig = Field(default_factory=BotConfig)
    lemonade: LemonadeConfig = Field(default_factory=LemonadeConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    runtime_maintenance: RuntimeMaintenanceConfig = Field(default_factory=RuntimeMaintenanceConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    instant_responder: InstantResponderConfig = Field(default_factory=InstantResponderConfig)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    intent_router: IntentRouterConfig = Field(default_factory=IntentRouterConfig)
    context_compressor: ContextCompressorConfig = Field(default_factory=ContextCompressorConfig)
    ollama: RetrievalProviderConfig = Field(default_factory=RetrievalProviderConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


def load_config(
    config_path: str = "config/config.yaml",
) -> AppSettings:
    """config.yaml 단일 파일에서 AppSettings를 로드한다."""
    yaml_data: dict = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    if not isinstance(yaml_data, dict):
        raise ValueError("config.yaml 최상위 구조는 mapping(dict)이어야 합니다.")
    settings = AppSettings.model_validate(yaml_data)

    # telegram.allowed_users CSV → security.allowed_users 리스트
    if settings.telegram.allowed_users:
        raw_ids = [
            uid.strip()
            for uid in settings.telegram.allowed_users.split(",")
            if uid.strip()
        ]
        user_ids: list[int] = []
        invalid_ids: list[str] = []
        for raw in raw_ids:
            try:
                user_ids.append(int(raw))
            except ValueError:
                invalid_ids.append(raw)
        if invalid_ids:
            raise ValueError(
                "telegram.allowed_users에는 정수 Chat ID만 사용할 수 있습니다: "
                f"{', '.join(invalid_ids)}"
            )
        settings.security.allowed_users = user_ids

    return settings
