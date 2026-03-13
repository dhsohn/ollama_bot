"""Helper steps for runtime construction."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from core.async_utils import run_in_thread
from core.auto_scheduler import AutoScheduler
from core.automation_callables import register_builtin_callables
from core.config import AppSettings, OllamaConfig, get_default_chat_model
from core.context_compressor import ContextCompressor
from core.feedback_manager import FeedbackManager
from core.instant_responder import InstantResponder
from core.llm_protocol import LLMClientProtocol, RetrievalClientProtocol
from core.memory import MemoryManager
from core.ollama_client import OllamaClient
from core.runtime_env import resolve_wsl_loopback_host
from core.runtime_factory_support import (
    StartupError,
    _create_retrieval_client,
    _open_sqlite_db,
    handle_optional_component_failure,
)


@dataclass
class OptionalRuntimeComponents:
    instant_responder: Any = None
    semantic_cache: Any = None
    intent_router: Any = None
    context_compressor: ContextCompressor | None = None


@dataclass
class RetrievalRuntimeComponents:
    retrieval_client: OllamaClient | None = None
    model_registry: Any = None
    rag_pipeline: Any = None
    rag_startup_index_task: asyncio.Task[Any] | None = None


async def initialize_memory_stack(
    config: AppSettings,
    cleanup_stack: AsyncExitStack,
    logger: Any,
) -> tuple[MemoryManager, FeedbackManager | None]:
    memory = MemoryManager(
        config=config.memory,
        data_dir=config.data_dir,
        max_conversation_length=config.bot.max_conversation_length,
        archive_enabled=(
            config.context_compressor.enabled
            and config.context_compressor.archive_enabled
        ),
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
            fb_pruned = await feedback.prune_old_feedback(
                config.feedback.retention_days
            )
            if fb_pruned:
                logger.info("feedback_pruned", count=fb_pruned)
        except Exception as exc:
            logger.error("feedback_prune_failed", error=str(exc))

    return memory, feedback


def rewrite_ollama_host(config: AppSettings, logger: Any) -> None:
    config.ollama.host = resolve_wsl_loopback_host(
        url=config.ollama.host,
        service_name="ollama",
        logger=logger,
    )


async def initialize_chat_client(
    config: AppSettings,
    cleanup_stack: AsyncExitStack,
    logger: Any,
) -> tuple[LLMClientProtocol, str]:
    default_model = get_default_chat_model(config)
    llm = OllamaClient(
        OllamaConfig(
            host=config.ollama.host,
            model=default_model,
            temperature=config.ollama.chat_temperature,
            max_tokens=config.ollama.chat_max_tokens,
            num_ctx=config.ollama.chat_num_ctx,
            system_prompt=config.ollama.chat_system_prompt,
        )
    )
    llm.default_model = default_model
    llm_host = getattr(llm, "host", "unknown")
    try:
        await llm.initialize()
    except Exception as exc:
        logger.error("llm_init_failed", error=str(exc))
        hint = (
            f"\n힌트: Ollama 서버가 {config.ollama.host} 에서 실행 중인지 확인하세요."
            f"\n      채팅 모델({default_model})이 pull 되었는지 확인하세요."
        )
        raise StartupError(
            f"오류: ollama 초기화 실패 ({llm_host})\n"
            f"{exc}{hint}\n"
            "백엔드 실행 상태와 기본 모델 준비 상태를 확인하세요. 봇 시작을 중단합니다."
        ) from exc
    cleanup_stack.push_async_callback(llm.close)
    return llm, default_model


async def initialize_skills(security, logger: Any):
    from core.skill_manager import SkillManager

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
    return skills, skill_count


async def initialize_optional_components(
    config: AppSettings,
    logger: Any,
    degraded_components: list[dict[str, str]],
    *,
    llm: LLMClientProtocol,
    memory: MemoryManager,
    cleanup_stack: AsyncExitStack,
) -> OptionalRuntimeComponents:
    components = OptionalRuntimeComponents()

    if config.instant_responder.enabled:
        components.instant_responder = InstantResponder(
            rules_path=config.instant_responder.rules_path,
        )
        logger.info(
            "instant_responder_initialized",
            rules=components.instant_responder.rules_count,
        )

    if config.semantic_cache.enabled:
        cache_db: Any = None
        try:
            from core.semantic_cache import SemanticCache

            cache_db_path = Path(config.data_dir) / "memory" / "cache.db"
            cache_db = await _open_sqlite_db(cache_db_path)
            components.semantic_cache = SemanticCache(
                db=cache_db,
                model_name=config.semantic_cache.model_name,
                similarity_threshold=config.semantic_cache.similarity_threshold,
                max_entries=config.semantic_cache.max_entries,
                ttl_hours=config.semantic_cache.ttl_hours,
                min_query_chars=config.semantic_cache.min_query_chars,
                exclude_patterns=config.semantic_cache.exclude_patterns,
            )
            await components.semantic_cache.initialize()
            cleanup_stack.push_async_callback(cache_db.close)
            logger.info(
                "semantic_cache_initialized",
                enabled=components.semantic_cache.enabled,
            )
        except Exception as exc:
            if cache_db is not None:
                await cache_db.close()
            handle_optional_component_failure(
                config,
                logger,
                degraded_components,
                component="semantic_cache",
                error=exc,
            )
            components.semantic_cache = None

    if config.intent_router.enabled:
        try:
            from core.intent_router import IntentRouter

            shared_encoder = (
                components.semantic_cache.encoder
                if components.semantic_cache is not None
                and components.semantic_cache.enabled
                else None
            )
            components.intent_router = await run_in_thread(
                IntentRouter,
                routes_path=config.intent_router.routes_path,
                encoder_model=config.intent_router.encoder_model,
                min_confidence=config.intent_router.min_confidence,
                encoder=shared_encoder,
            )
            logger.info(
                "intent_router_initialized",
                enabled=components.intent_router.enabled,
                routes=components.intent_router.routes_count,
            )
        except Exception as exc:
            handle_optional_component_failure(
                config,
                logger,
                degraded_components,
                component="intent_router",
                error=exc,
            )
            components.intent_router = None

    if config.context_compressor.enabled:
        components.context_compressor = ContextCompressor(
            llm_client=llm,
            memory=memory,
            recent_keep=config.context_compressor.recent_keep,
            summary_refresh_interval=config.context_compressor.summary_refresh_interval,
            summary_max_tokens=config.context_compressor.summary_max_tokens,
            summarize_concurrency=config.context_compressor.summarize_concurrency,
        )
        logger.info("context_compressor_initialized")

    return components


async def initialize_retrieval_components(
    config: AppSettings,
    logger: Any,
    degraded_components: list[dict[str, str]],
    cleanup_stack: AsyncExitStack,
) -> RetrievalRuntimeComponents:
    components = RetrievalRuntimeComponents()

    if config.rag.enabled:
        try:
            components.retrieval_client = _create_retrieval_client(config)
            await components.retrieval_client.initialize()
            cleanup_stack.push_async_callback(components.retrieval_client.close)
            logger.info(
                "retrieval_client_initialized",
                host=config.ollama.host,
                embedding_model=config.ollama.embedding_model,
                reranker_model=config.ollama.reranker_model,
            )
        except Exception as exc:
            handle_optional_component_failure(
                config,
                logger,
                degraded_components,
                component="retrieval_client",
                error=exc,
            )
            components.retrieval_client = None

    if components.retrieval_client is not None:
        try:
            from core.config import ModelRegistryConfig
            from core.model_registry import ModelRegistry

            retrieval_proto = cast(RetrievalClientProtocol, components.retrieval_client)
            components.model_registry = ModelRegistry(
                ModelRegistryConfig(
                    default_model=get_default_chat_model(config),
                    embedding_model=config.ollama.embedding_model,
                    reranker_model=config.ollama.reranker_model,
                ),
                retrieval_proto,
            )
            await components.model_registry.initialize()
        except Exception as exc:
            handle_optional_component_failure(
                config,
                logger,
                degraded_components,
                component="model_registry",
                error=exc,
            )
            components.model_registry = None

    return components


async def preload_default_model(
    llm: LLMClientProtocol,
    *,
    default_model: str,
    config: AppSettings,
    logger: Any,
    degraded_components: list[dict[str, str]],
) -> None:
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
            handle_optional_component_failure(
                config,
                logger,
                degraded_components,
                component="default_model_preload",
                error=exc,
            )


async def initialize_rag_pipeline(
    config: AppSettings,
    logger: Any,
    degraded_components: list[dict[str, str]],
    cleanup_stack: AsyncExitStack,
    *,
    retrieval_client: OllamaClient | None,
    model_registry: Any,
) -> tuple[Any, asyncio.Task[Any] | None]:
    rag_pipeline = None
    rag_startup_index_task: asyncio.Task[Any] | None = None

    if config.rag.enabled and retrieval_client is None:
        logger.warning(
            "rag_pipeline_disabled",
            reason="retrieval_client_unavailable",
        )
        return None, None

    if not config.rag.enabled or retrieval_client is None:
        return None, None

    try:
        from core.rag.context_builder import RAGContextBuilder
        from core.rag.indexer import RAGIndexer
        from core.rag.pipeline import RAGPipeline
        from core.rag.reranker import RAGReranker
        from core.rag.retriever import RAGRetriever

        retrieval_proto = cast(RetrievalClientProtocol, retrieval_client)
        index_dir = config.rag.index_dir or str(Path(config.data_dir) / "rag_index")
        rag_db_path = Path(index_dir) / "rag.db"

        indexer = RAGIndexer(
            config.rag,
            retrieval_proto,
            config.ollama.embedding_model,
        )
        await indexer.initialize(str(rag_db_path))
        cleanup_stack.push_async_callback(indexer.close)

        corpus_roots = normalize_corpus_roots(config.rag.kb_dirs)
        kb_dirs_to_index = collect_existing_kb_dirs(corpus_roots, logger)
        if kb_dirs_to_index and config.rag.startup_index_enabled:
            rag_startup_index_task = schedule_rag_startup_index(
                indexer=indexer,
                kb_dirs_to_index=kb_dirs_to_index,
                logger=logger,
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
            config.ollama.embedding_model,
        )

        reranker = None
        if config.rag.rerank_enabled:
            reranker_available = True
            if model_registry is not None:
                try:
                    reranker_available = model_registry.is_available("reranker")
                except Exception as exc:
                    handle_optional_component_failure(
                        config,
                        logger,
                        degraded_components,
                        component="rag_reranker_availability",
                        error=exc,
                    )
                    reranker_available = False
            if reranker_available:
                reranker = RAGReranker(
                    retrieval_proto,
                    config.ollama.reranker_model,
                    config.rag,
                )

        rag_pipeline = RAGPipeline(
            retriever,
            reranker,
            RAGContextBuilder(),
            config.rag,
        )
        logger.info(
            "rag_pipeline_initialized",
            chunks=indexer.chunk_count,
            reranker=reranker is not None,
        )
    except Exception as exc:
        handle_optional_component_failure(
            config,
            logger,
            degraded_components,
            component="rag_pipeline",
            error=exc,
        )
        rag_pipeline = None

    return rag_pipeline, rag_startup_index_task


def normalize_corpus_roots(configured_kb_dirs: list[str]) -> list[str]:
    seen_dirs: set[str] = set()
    corpus_roots: list[str] = []
    for root_dir in configured_kb_dirs:
        path_text = str(root_dir).strip()
        if not path_text or path_text in seen_dirs:
            continue
        seen_dirs.add(path_text)
        corpus_roots.append(path_text)
    return corpus_roots


def collect_existing_kb_dirs(corpus_roots: list[str], logger: Any) -> list[str]:
    kb_dirs_to_index: list[str] = []
    for root_dir in corpus_roots:
        kb_path = Path(root_dir)
        if not kb_path.exists():
            logger.warning("rag_kb_path_not_found", path=root_dir)
            continue
        kb_dirs_to_index.append(root_dir)
    return kb_dirs_to_index


def schedule_rag_startup_index(indexer: Any, kb_dirs_to_index: list[str], logger: Any) -> asyncio.Task[Any]:
    rag_startup_index_timeout = 600.0

    async def _run_rag_startup_index() -> None:
        try:
            result = await asyncio.wait_for(
                indexer.index_corpus(kb_dirs_to_index),
                timeout=rag_startup_index_timeout,
            )
            logger.info(
                "rag_startup_index_completed",
                indexed=result.get("indexed", 0),
                skipped=result.get("skipped", 0),
                removed=result.get("removed", 0),
                total_chunks=result.get("total_chunks", 0),
            )
        except TimeoutError:
            logger.error(
                "rag_startup_index_timeout",
                timeout_seconds=rag_startup_index_timeout,
            )
        except Exception as exc:
            logger.error("rag_startup_index_failed", error=str(exc))

    task = asyncio.create_task(
        _run_rag_startup_index(),
        name="rag_startup_index",
    )
    logger.info(
        "rag_startup_index_started",
        roots=len(kb_dirs_to_index),
    )
    return task


async def initialize_scheduler_stack(
    config: AppSettings,
    logger: Any,
    security: Any,
    engine: Any,
    telegram: Any,
    memory: MemoryManager,
    feedback: FeedbackManager | None,
) -> tuple[AutoScheduler, int, Any]:
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
    if not scheduler.dependencies_ready():
        raise StartupError(
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
    if not telegram.has_scheduler():
        raise StartupError(
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
    return scheduler, auto_count, app
