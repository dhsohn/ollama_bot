"""공통 앱 런타임 부트스트랩.

`apps/ollama_bot` 엔트리포인트가
코어 초기화/실행 로직을 공유하도록 제공한다.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import inspect
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, cast

import aiosqlite

from core.async_utils import run_in_thread
from core.auto_evaluator import AutoEvaluator
from core.automation_callables import register_builtin_callables
from core.auto_scheduler import AutoScheduler
from core.config import AppSettings, OllamaConfig, load_config
from core.context_compressor import ContextCompressor
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.instant_responder import InstantResponder
from core.lemonade_multi_client import build_lemonade_client
from core.llm_protocol import LLMClientProtocol, RetrievalClientProtocol
from core.logging_setup import get_logger, setup_logging
from core.memory import MemoryManager
from core.ollama_client import OllamaClient
from core.security import SecurityManager
from core.skill_manager import SkillManager
from core.telegram_handler import TelegramHandler
from packages.hw_amd_npu import apply_npu_profile

_ALLOW_LOCAL_RUN_ENV = "ALLOW_LOCAL_RUN"
_SUPPORTED_PROVIDERS = {"ollama", "lemonade"}


class StartupError(RuntimeError):
    """초기화 실패 시 사용자 노출용 메시지를 전달한다."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class RuntimeState:
    """초기화된 런타임 의존성을 묶어 관리한다."""

    config: AppSettings
    logger: Any
    memory: MemoryManager
    llm: LLMClientProtocol
    app: Any
    scheduler: AutoScheduler
    skill_count: int
    auto_count: int
    llm_provider: str
    cleanup_stack: AsyncExitStack
    feedback: FeedbackManager | None = None
    auto_evaluator: AutoEvaluator | None = None
    semantic_cache: Any = None  # SemanticCache | None
    rag_startup_index_task: asyncio.Task[Any] | None = None


def _is_running_in_container() -> bool:
    """현재 프로세스가 컨테이너 내부에서 실행 중인지 판단한다."""
    if Path("/.dockerenv").exists():
        return True

    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False

    markers = ("docker", "containerd", "kubepods", "podman")
    return any(marker in cgroup for marker in markers)


def _is_truthy(value: str | None) -> bool:
    """환경변수 문자열을 bool로 변환한다."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(value: str) -> str:
    return value.strip().lower()


def _runtime_env_files() -> str | tuple[str, ...] | None:
    """런타임에서 사용할 env 파일 목록을 결정한다."""
    env_files_csv = os.environ.get("APP_ENV_FILES", "").strip()
    if env_files_csv:
        env_files = tuple(
            item.strip()
            for item in env_files_csv.split(",")
            if item.strip()
        )
        if env_files:
            return env_files

    env_file = os.environ.get("APP_ENV_FILE", "").strip()
    if env_file:
        return env_file

    return ".env"


def _resolve_provider(
    config: AppSettings,
    logger: Any,
    forced_provider: str | None,
) -> str:
    """실행 대상 provider를 결정한다."""
    configured = _normalize_provider(config.llm_provider)
    if configured not in _SUPPORTED_PROVIDERS:
        raise StartupError(
            f"오류: 지원하지 않는 llm_provider='{configured}' "
            f"(지원: {', '.join(sorted(_SUPPORTED_PROVIDERS))})"
        )

    if forced_provider is None:
        return configured

    chosen = _normalize_provider(forced_provider)
    if chosen not in _SUPPORTED_PROVIDERS:
        raise StartupError(
            f"오류: 지원하지 않는 provider='{chosen}' "
            f"(지원: {', '.join(sorted(_SUPPORTED_PROVIDERS))})"
        )
    if chosen != configured:
        logger.warning(
            "llm_provider_overridden_by_app",
            configured=configured,
            forced=chosen,
        )
    return chosen


def _model_for_provider(config: AppSettings, provider: str) -> str:
    if provider == "lemonade":
        return config.model_registry.default_model
    return config.ollama.model


def _create_llm_client(config: AppSettings, provider: str) -> LLMClientProtocol:
    if provider == "ollama":
        return OllamaClient(config.ollama)
    if provider == "lemonade":
        return build_lemonade_client(
            config.lemonade,
            fallback_ollama=config.ollama,
        )
    raise StartupError(
        f"오류: 지원하지 않는 provider='{provider}' "
        f"(지원: {', '.join(sorted(_SUPPORTED_PROVIDERS))})"
    )


def _create_retrieval_client(config: AppSettings) -> OllamaClient:
    """Ollama 기반 retrieval 전용 클라이언트를 생성한다."""
    retrieval_config = OllamaConfig(
        host=config.retrieval_provider.host,
        model=config.retrieval_provider.embedding_model,
    )
    return OllamaClient(retrieval_config)


async def _open_sqlite_db(path: Path) -> aiosqlite.Connection:
    """WAL 모드 SQLite 연결을 생성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.commit()
    return db


async def _memory_maintenance_loop(
    memory: MemoryManager,
    logger,
    interval_seconds: int,
    jitter_ratio: float,
    feedback: FeedbackManager | None = None,
    feedback_retention_days: int | None = None,
    semantic_cache: Any = None,
) -> None:
    """주기적으로 오래된 대화/피드백/캐시 데이터를 정리한다."""
    while True:
        try:
            deleted = await memory.prune_old_conversations()
            logger.debug("memory_retention_pruned", deleted=deleted)
        except Exception as exc:
            logger.error("memory_retention_prune_failed", error=str(exc))

        if feedback is not None and feedback_retention_days is not None:
            try:
                fb_deleted = await feedback.prune_old_feedback(feedback_retention_days)
                logger.debug("feedback_retention_pruned", deleted=fb_deleted)
            except Exception as exc:
                logger.error("feedback_retention_prune_failed", error=str(exc))

            try:
                eval_deleted = await feedback.prune_old_auto_evaluations(feedback_retention_days)
                if eval_deleted:
                    logger.debug("auto_eval_retention_pruned", deleted=eval_deleted)
            except Exception as exc:
                logger.error("auto_eval_retention_prune_failed", error=str(exc))

        if semantic_cache is not None:
            try:
                cache_pruned = await semantic_cache.prune_expired()
                if cache_pruned:
                    logger.debug("semantic_cache_pruned", deleted=cache_pruned)
            except Exception as exc:
                logger.error("semantic_cache_prune_failed", error=str(exc))

        base_interval = max(1, interval_seconds)
        jitter = random.uniform(0.0, base_interval * max(0.0, jitter_ratio))
        await asyncio.sleep(base_interval + jitter)


async def _llm_recovery_loop(
    llm: LLMClientProtocol,
    logger,
    interval_seconds: int,
) -> None:
    """LLM 백엔드 상태를 주기 점검하고 장애 시 재연결을 시도한다."""
    while True:
        try:
            health = await llm.health_check(attempt_recovery=False)
            if health.get("status") != "ok":
                recovered = await llm.recover_connection(force=True)
                if recovered:
                    logger.info("llm_recovered_by_loop")
                else:
                    logger.warning(
                        "llm_still_unhealthy",
                        error=health.get("error"),
                    )
        except Exception as exc:
            logger.error("llm_recovery_loop_failed", error=str(exc))

        await asyncio.sleep(interval_seconds)


def _validate_required_settings(config: AppSettings, logger: Any) -> None:
    """필수 런타임 설정을 검사한다."""
    if (
        not config.telegram_bot_token
        or config.telegram_bot_token == "your_telegram_bot_token_here"
    ):
        logger.error("telegram_bot_token_not_set")
        raise StartupError(
            "오류: TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.\n"
            ".env 파일에 유효한 텔레그램 봇 토큰을 입력하세요."
        )

    if not config.security.allowed_users:
        logger.error("allowed_users_not_set")
        raise StartupError(
            "오류: ALLOWED_TELEGRAM_USERS가 비어 있습니다.\n"
            "private chat에서 허용할 사용자 ID를 .env에 설정하세요."
        )


async def _build_runtime(
    config: AppSettings,
    logger: Any,
    *,
    llm_provider: str,
) -> RuntimeState:
    """의존성 순서대로 모듈을 초기화하고 런타임 상태를 반환한다."""
    cleanup_stack = AsyncExitStack()
    try:
        security = SecurityManager(config.security)
        logger.info(
            "security_initialized",
            allowed_users=len(config.security.allowed_users),
        )

        memory = MemoryManager(
            config=config.memory,
            data_dir=config.data_dir,
            max_conversation_length=config.bot.max_conversation_length,
            archive_enabled=config.context_compressor.enabled and config.context_compressor.archive_enabled,
        )
        await memory.initialize()
        cleanup_stack.push_async_callback(memory.close)
        try:
            pruned = await memory.prune_old_conversations()
            logger.info("memory_retention_pruned_on_start", deleted=pruned)
        except Exception as exc:
            logger.error("memory_retention_prune_failed_on_start", error=str(exc))

        feedback: FeedbackManager | None = None
        if config.feedback.enabled:
            feedback_db_path = Path(config.data_dir) / "memory" / "feedback.db"
            feedback_db = await _open_sqlite_db(feedback_db_path)
            cleanup_stack.push_async_callback(feedback_db.close)
            feedback = FeedbackManager(feedback_db)
            await feedback.initialize_schema()
            try:
                fb_pruned = await feedback.prune_old_feedback(config.feedback.retention_days)
                if fb_pruned:
                    logger.info("feedback_pruned", count=fb_pruned)
            except Exception as exc:
                logger.error("feedback_prune_failed", error=str(exc))

        llm = _create_llm_client(config, llm_provider)
        # 기본 모델명 설정 (lemonade 클라이언트는 빈 문자열로 시작)
        if llm_provider == "lemonade":
            llm.default_model = config.model_registry.default_model
        llm_host = config.ollama.host if llm_provider == "ollama" else getattr(
            llm,
            "host",
            config.lemonade.host,
        )
        try:
            await llm.initialize()
        except Exception as exc:
            logger.error("llm_init_failed", provider=llm_provider, error=str(exc))
            hint = ""
            if llm_provider == "lemonade":
                hint = (
                    "\n힌트: WSL 환경에서는 lemonade-server가 0.0.0.0에 바인딩되어야 합니다."
                    "\n      Windows 방화벽에서 해당 포트의 인바운드를 허용했는지 확인하세요."
                )
            raise StartupError(
                f"오류: {llm_provider} 초기화 실패 ({llm_host})\n"
                f"{exc}{hint}\n"
                "백엔드 실행 상태와 기본 모델 준비 상태를 확인하세요. 봇 시작을 중단합니다."
            ) from exc
        cleanup_stack.push_async_callback(llm.close)

        skills = SkillManager(security=security, skills_dir="skills")
        try:
            skill_count = await skills.load_skills(strict=True)
        except Exception as exc:
            logger.error("skills_init_failed", error=str(exc))
            raise StartupError(
                f"오류: 스킬 로드 실패\n{exc}\n"
                "중복 이름/트리거 또는 YAML 형식을 확인하세요."
            ) from exc
        logger.info("skills_loaded", count=skill_count)
        skill_load_errors = skills.get_last_load_errors()
        if skill_load_errors:
            logger.warning(
                "skills_loaded_with_partial_failures",
                error_count=len(skill_load_errors),
                sample=skill_load_errors[:3],
            )

        instant_responder = None
        semantic_cache = None
        intent_router = None
        context_compressor = None

        if config.instant_responder.enabled:
            instant_responder = InstantResponder(
                rules_path=config.instant_responder.rules_path,
            )
            logger.info("instant_responder_initialized", rules=instant_responder.rules_count)

        if config.semantic_cache.enabled:
            cache_db: aiosqlite.Connection | None = None
            try:
                from core.semantic_cache import SemanticCache

                cache_db_path = Path(config.data_dir) / "memory" / "cache.db"
                cache_db = await _open_sqlite_db(cache_db_path)
                semantic_cache = SemanticCache(
                    db=cache_db,
                    model_name=config.semantic_cache.model_name,
                    similarity_threshold=config.semantic_cache.similarity_threshold,
                    max_entries=config.semantic_cache.max_entries,
                    ttl_hours=config.semantic_cache.ttl_hours,
                    min_query_chars=config.semantic_cache.min_query_chars,
                    exclude_patterns=config.semantic_cache.exclude_patterns,
                )
                await semantic_cache.initialize()
                cleanup_stack.push_async_callback(cache_db.close)
                logger.info("semantic_cache_initialized", enabled=semantic_cache.enabled)
            except Exception as exc:
                if cache_db is not None:
                    await cache_db.close()
                logger.warning("semantic_cache_init_failed", error=str(exc))
                semantic_cache = None

        if config.intent_router.enabled:
            try:
                from core.intent_router import IntentRouter

                shared_encoder = (
                    semantic_cache.encoder
                    if semantic_cache is not None and semantic_cache.enabled
                    else None
                )
                intent_router = await run_in_thread(
                    IntentRouter,
                    routes_path=config.intent_router.routes_path,
                    encoder_model=config.intent_router.encoder_model,
                    min_confidence=config.intent_router.min_confidence,
                    encoder=shared_encoder,
                )
                logger.info(
                    "intent_router_initialized",
                    enabled=intent_router.enabled,
                    routes=intent_router.routes_count,
                )
            except Exception as exc:
                logger.warning("intent_router_init_failed", error=str(exc))
                intent_router = None

        if config.context_compressor.enabled:
            context_compressor = ContextCompressor(
                llm_client=llm,
                memory=memory,
                recent_keep=config.context_compressor.recent_keep,
                summary_refresh_interval=config.context_compressor.summary_refresh_interval,
                summary_max_tokens=config.context_compressor.summary_max_tokens,
                summarize_concurrency=config.context_compressor.summarize_concurrency,
            )
            logger.info("context_compressor_initialized")

        # ── Dual-Provider: Ollama(retrieval) + Lemonade(chat) ──
        rag_pipeline = None
        model_registry = None
        retrieval_client: OllamaClient | None = None
        rag_startup_index_task: asyncio.Task[Any] | None = None

        # Retrieval 클라이언트 초기화 (Ollama — 임베딩/리랭킹 전용)
        if config.rag.enabled:
            try:
                retrieval_client = _create_retrieval_client(config)
                await retrieval_client.initialize()
                cleanup_stack.push_async_callback(retrieval_client.close)
                logger.info(
                    "retrieval_client_initialized",
                    host=config.retrieval_provider.host,
                    embedding_model=config.retrieval_provider.embedding_model,
                    reranker_model=config.retrieval_provider.reranker_model,
                )
            except Exception as exc:
                logger.warning(
                    "retrieval_client_init_failed",
                    host=config.retrieval_provider.host,
                    error=str(exc),
                )
                retrieval_client = None

        # Model Registry 초기화 (retrieval 모델 가용성 관리)
        if retrieval_client is not None:
            try:
                from core.model_registry import ModelRegistry

                retrieval_proto = cast(RetrievalClientProtocol, retrieval_client)
                model_registry = ModelRegistry(
                    config.model_registry, retrieval_proto,
                )
                await model_registry.initialize()
            except Exception as exc:
                logger.warning("model_registry_init_failed", error=str(exc))
                model_registry = None

        # 기본 모델(gpt-oss-20b-NPU) 선로드 — 상주 보장
        if llm_provider == "lemonade":
            default_model = config.model_registry.default_model
            prepare_model = getattr(llm, "prepare_model", None)
            if callable(prepare_model):
                try:
                    maybe_result = prepare_model(
                        model=default_model,
                        role="default",
                    )
                    if inspect.isawaitable(maybe_result):
                        await maybe_result
                    logger.info("default_model_preloaded", model=default_model)
                except Exception as exc:
                    logger.warning(
                        "default_model_preload_failed",
                        model=default_model,
                        error=str(exc),
                    )

        # RAG 파이프라인 초기화 (Ollama retrieval 클라이언트 사용)
        if config.rag.enabled and retrieval_client is not None:
            try:
                from core.rag.context_builder import RAGContextBuilder
                from core.rag.indexer import RAGIndexer
                from core.rag.pipeline import RAGPipeline
                from core.rag.reranker import RAGReranker
                from core.rag.retriever import RAGRetriever

                retrieval_proto = cast(RetrievalClientProtocol, retrieval_client)
                index_dir = config.rag.index_dir or str(
                    Path(config.data_dir) / "rag_index"
                )
                rag_db_path = Path(index_dir) / "rag.db"

                indexer = RAGIndexer(
                    config.rag,
                    retrieval_proto,
                    config.retrieval_provider.embedding_model,
                )
                await indexer.initialize(str(rag_db_path))
                cleanup_stack.push_async_callback(indexer.close)

                # 시작 시 인덱싱
                configured_kb_dirs = list(config.rag.kb_dirs)
                seen_dirs: set[str] = set()
                corpus_roots: list[str] = []
                for root_dir in configured_kb_dirs:
                    path_text = str(root_dir).strip()
                    if not path_text or path_text in seen_dirs:
                        continue
                    seen_dirs.add(path_text)
                    corpus_roots.append(path_text)

                kb_dirs_to_index: list[str] = []
                for root_dir in corpus_roots:
                    kb_path = Path(root_dir)
                    if not kb_path.exists():
                        logger.warning("rag_kb_path_not_found", path=root_dir)
                        continue
                    kb_dirs_to_index.append(root_dir)
                if kb_dirs_to_index and config.rag.startup_index_enabled:
                    _RAG_STARTUP_INDEX_TIMEOUT = 600.0  # 10분

                    async def _run_rag_startup_index() -> None:
                        try:
                            result = await asyncio.wait_for(
                                indexer.index_corpus(kb_dirs_to_index),
                                timeout=_RAG_STARTUP_INDEX_TIMEOUT,
                            )
                            logger.info(
                                "rag_startup_index_completed",
                                indexed=result.get("indexed", 0),
                                skipped=result.get("skipped", 0),
                                removed=result.get("removed", 0),
                                total_chunks=result.get("total_chunks", 0),
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                "rag_startup_index_timeout",
                                timeout_seconds=_RAG_STARTUP_INDEX_TIMEOUT,
                            )
                        except Exception as exc:
                            logger.error("rag_startup_index_failed", error=str(exc))

                    rag_startup_index_task = asyncio.create_task(
                        _run_rag_startup_index(),
                        name="rag_startup_index",
                    )
                    logger.info(
                        "rag_startup_index_started",
                        roots=len(kb_dirs_to_index),
                    )
                elif kb_dirs_to_index:
                    logger.info(
                        "rag_startup_index_skipped",
                        reason="disabled",
                        roots=len(kb_dirs_to_index),
                    )
                else:
                    logger.warning("rag_kb_dirs_empty_or_missing")

                retriever = RAGRetriever(
                    indexer,
                    retrieval_proto,
                    config.retrieval_provider.embedding_model,
                )

                reranker = None
                if config.rag.rerank_enabled:
                    reranker_available = True
                    if model_registry is not None:
                        try:
                            reranker_available = model_registry.is_available("reranker")
                        except Exception:
                            reranker_available = False
                    if reranker_available:
                        reranker = RAGReranker(
                            retrieval_proto,
                            config.retrieval_provider.reranker_model,
                            config.rag,
                        )

                rag_pipeline = RAGPipeline(
                    retriever, reranker, RAGContextBuilder(), config.rag,
                )
                logger.info(
                    "rag_pipeline_initialized",
                    chunks=indexer.chunk_count,
                    reranker=reranker is not None,
                )
            except Exception as exc:
                logger.warning("rag_pipeline_init_failed", error=str(exc))
                rag_pipeline = None

        engine = Engine(
            config=config,
            llm_client=llm,
            memory=memory,
            skills=skills,
            feedback_manager=feedback,
            instant_responder=instant_responder,
            semantic_cache=semantic_cache,
            intent_router=intent_router,
            context_compressor=context_compressor,
            model_router=None,
            rag_pipeline=rag_pipeline,
        )

        auto_evaluator: AutoEvaluator | None = None
        if config.auto_evaluation.enabled and feedback is not None:
            auto_evaluator = AutoEvaluator(
                config=config.auto_evaluation,
                llm_client=llm,
                feedback_manager=feedback,
                timezone_name=config.scheduler.timezone,
            )
            logger.info("auto_evaluator_initialized")

        telegram = TelegramHandler(
            config=config,
            engine=engine,
            security=security,
            feedback=feedback,
            auto_evaluator=auto_evaluator,
            semantic_cache=semantic_cache,
        )

        try:
            scheduler = AutoScheduler(
                config=config,
                security=security,
                auto_dir="auto",
            )
        except Exception as exc:
            logger.error("scheduler_init_failed", error=str(exc))
            raise StartupError(
                f"오류: 자동화 스케줄러 초기화 실패\n{exc}\n"
                "config/config.yaml의 scheduler.timezone 설정을 확인하세요."
            ) from exc

        scheduler.set_dependencies(engine=engine, telegram=telegram)
        assert scheduler.dependencies_ready(), (
            "Scheduler dependencies must be wired before automation loading."
        )
        register_builtin_callables(
            scheduler=scheduler,
            engine=engine,
            memory=memory,
            allowed_users=config.security.allowed_users,
            data_dir=config.data_dir,
            feedback=feedback,
        )
        telegram.set_scheduler(scheduler)
        assert telegram.has_scheduler(), (
            "Telegram handler must receive scheduler before initialization."
        )

        try:
            auto_count = await scheduler.load_automations(strict=True)
        except Exception as exc:
            logger.error("automations_init_failed", error=str(exc))
            raise StartupError(
                f"오류: 자동화 로드 실패\n{exc}\n"
                "중복 이름 또는 YAML 형식을 확인하세요."
            ) from exc
        logger.info("automations_loaded", count=auto_count)
        auto_load_errors = scheduler.get_last_load_errors()
        if auto_load_errors:
            logger.warning(
                "automations_loaded_with_partial_failures",
                error_count=len(auto_load_errors),
                sample=auto_load_errors[:3],
            )

        app = await telegram.initialize()

        return RuntimeState(
            config=config,
            logger=logger,
            memory=memory,
            llm=llm,
            app=app,
            scheduler=scheduler,
            skill_count=skill_count,
            auto_count=auto_count,
            llm_provider=llm_provider,
            cleanup_stack=cleanup_stack,
            feedback=feedback,
            auto_evaluator=auto_evaluator,
            semantic_cache=semantic_cache,
            rag_startup_index_task=rag_startup_index_task,
        )
    except Exception:
        await cleanup_stack.aclose()
        raise


async def _shutdown_runtime(
    runtime: RuntimeState,
    memory_maintenance_task: asyncio.Task | None,
    llm_recovery_task: asyncio.Task | None,
    scheduler_started: bool,
    app_started: bool,
    updater_started: bool,
) -> None:
    """실행 중 자원을 역순으로 정리한다."""
    logger = runtime.logger
    app = runtime.app
    logger.info("shutting_down")

    if scheduler_started:
        try:
            runtime.scheduler.stop()
        except Exception as exc:
            logger.error("scheduler_stop_failed", error=str(exc))

    if runtime.auto_evaluator is not None:
        try:
            await runtime.auto_evaluator.shutdown()
        except Exception as exc:
            logger.error("auto_evaluator_shutdown_failed", error=str(exc))

    if memory_maintenance_task is not None:
        memory_maintenance_task.cancel()
        await asyncio.gather(memory_maintenance_task, return_exceptions=True)

    if llm_recovery_task is not None:
        llm_recovery_task.cancel()
        await asyncio.gather(llm_recovery_task, return_exceptions=True)

    if runtime.rag_startup_index_task is not None:
        runtime.rag_startup_index_task.cancel()
        await asyncio.gather(runtime.rag_startup_index_task, return_exceptions=True)

    if updater_started:
        try:
            await app.updater.stop()
        except Exception as exc:
            logger.error("updater_stop_failed", error=str(exc))

    if app_started:
        try:
            await app.stop()
        except Exception as exc:
            logger.error("app_stop_failed", error=str(exc))

    try:
        await runtime.cleanup_stack.aclose()
    except Exception as exc:
        logger.error("resource_cleanup_failed", error=str(exc))

    logger.info("shutdown_complete")


async def async_main(
    *,
    app_name: str,
    forced_provider: str | None = None,
) -> None:
    """비동기 메인 루프."""
    if not _is_running_in_container() and not _is_truthy(os.environ.get(_ALLOW_LOCAL_RUN_ENV)):
        print(
            "오류: 이 애플리케이션은 Docker 컨테이너에서만 실행됩니다.\n"
            "실행 방법: docker compose up --build -d\n"
            f"로컬 실행 우회: {_ALLOW_LOCAL_RUN_ENV}=1",
            file=sys.stderr,
        )
        sys.exit(1)

    npu_profile = os.environ.get("AMD_NPU_PROFILE", "").strip()
    npu_applied: dict[str, str] = {}
    if npu_profile:
        try:
            npu_applied = apply_npu_profile(npu_profile)
        except ValueError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        config = load_config(env_file=_runtime_env_files())
    except ValueError as exc:
        print(
            f"오류: 설정값이 잘못되었습니다. {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    log_dir = str(Path(config.data_dir) / "logs")
    setup_logging(config.log_level, log_dir=log_dir)
    logger = get_logger(app_name)
    logger.info("starting", app=app_name, version="0.1.0")
    if npu_profile:
        logger.info(
            "amd_npu_profile_configured",
            profile=npu_profile.lower(),
            applied=list(sorted(npu_applied.keys())),
        )

    try:
        _validate_required_settings(config, logger)
        llm_provider = _resolve_provider(config, logger, forced_provider)
        runtime = await _build_runtime(config, logger, llm_provider=llm_provider)
    except StartupError as exc:
        print(exc.message, file=sys.stderr)
        sys.exit(1)

    stop_event = asyncio.Event()
    app = runtime.app

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    memory_maintenance_task: asyncio.Task | None = None
    llm_recovery_task: asyncio.Task | None = None
    scheduler_started = False
    app_started = False
    updater_started = False

    async with app:
        try:
            await app.start()
            app_started = True

            runtime.scheduler.start()
            scheduler_started = True

            memory_maintenance_task = asyncio.create_task(
                _memory_maintenance_loop(
                    runtime.memory,
                    logger,
                    interval_seconds=runtime.config.runtime_maintenance.memory_maintenance_interval_seconds,
                    jitter_ratio=runtime.config.runtime_maintenance.memory_maintenance_jitter_ratio,
                    feedback=runtime.feedback,
                    feedback_retention_days=runtime.config.feedback.retention_days,
                    semantic_cache=runtime.semantic_cache,
                ),
                name="memory_maintenance",
            )
            llm_recovery_task = asyncio.create_task(
                _llm_recovery_loop(
                    runtime.llm,
                    logger,
                    interval_seconds=runtime.config.runtime_maintenance.llm_recovery_interval_seconds,
                ),
                name="llm_recovery",
            )

            logger.info(
                "bot_running",
                provider=runtime.llm_provider,
                model=_model_for_provider(runtime.config, runtime.llm_provider),
                skills=runtime.skill_count,
                automations=runtime.auto_count,
            )

            await app.updater.start_polling(
                poll_interval=runtime.config.telegram.polling_interval,
                drop_pending_updates=True,
            )
            updater_started = True
            await stop_event.wait()
        finally:
            await _shutdown_runtime(
                runtime=runtime,
                memory_maintenance_task=memory_maintenance_task,
                llm_recovery_task=llm_recovery_task,
                scheduler_started=scheduler_started,
                app_started=app_started,
                updater_started=updater_started,
            )


def run_app(
    *,
    app_name: str,
    forced_provider: str | None = None,
) -> None:
    """동기 진입점 래퍼."""
    try:
        asyncio.run(async_main(app_name=app_name, forced_provider=forced_provider))
    except KeyboardInterrupt:
        pass
