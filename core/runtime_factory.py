"""런타임 의존성 초기화 팩토리."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import aiosqlite

from core.async_utils import run_in_thread
from core.auto_scheduler import AutoScheduler
from core.automation_callables import register_builtin_callables
from core.config import AppSettings, OllamaConfig
from core.context_compressor import ContextCompressor
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.instant_responder import InstantResponder
from core.lemonade_client import LemonadeClient
from core.llm_protocol import LLMClientProtocol, RetrievalClientProtocol
from core.memory import MemoryManager
from core.ollama_client import OllamaClient
from core.runtime_env import resolve_wsl_loopback_host
from core.security import SecurityManager
from core.skill_manager import SkillManager
from core.telegram_handler import TelegramHandler


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
    semantic_cache: Any = None
    rag_startup_index_task: asyncio.Task[Any] | None = None
    sim_scheduler: Any = None
    degraded_components: list[dict[str, str]] = field(default_factory=list)


def model_for_provider(config: AppSettings) -> str:
    return config.lemonade.default_model


def _create_llm_client(config: AppSettings) -> LLMClientProtocol:
    return LemonadeClient(config.lemonade)


def _create_retrieval_client(config: AppSettings) -> OllamaClient:
    """Ollama 기반 retrieval 전용 클라이언트를 생성한다."""
    retrieval_config = OllamaConfig(
        host=config.ollama.host,
        model=config.ollama.embedding_model,
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


def validate_required_settings(config: AppSettings, logger: Any) -> None:
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


def _record_degraded_component(
    logger: Any,
    degraded_components: list[dict[str, str]],
    *,
    component: str,
    error: str,
) -> None:
    error_text = error.strip() or "unknown_error"
    degraded_components.append({"component": component, "error": error_text})
    logger.warning(
        "startup_component_degraded",
        component=component,
        error=error_text,
    )


async def _init_dft(
    config: AppSettings,
    cleanup_stack: AsyncExitStack,
    logger: Any,
    degraded_components: list[dict[str, str]],
) -> tuple[Any, Any]:
    """DFT 인덱스와 컨텍스트 프로바이더를 초기화한다."""
    if not config.dft.enabled:
        return None, None
    try:
        from core.dft_index import DFTIndex
        from core.dft_query import DFTContextProvider, DFTQueryEngine

        dft_db_path = str(Path(config.data_dir) / "rag_index" / "dft.db")
        dft_index = DFTIndex()
        await dft_index.initialize(dft_db_path)
        cleanup_stack.push_async_callback(dft_index.close)

        dft_query_engine = DFTQueryEngine(dft_index)
        dft_context_provider = DFTContextProvider(dft_query_engine)

        if config.dft.auto_index_on_startup and config.rag.kb_dirs:
            kb_dirs = [d for d in config.rag.kb_dirs if Path(d).is_dir()]
            if kb_dirs:
                result = await dft_index.index_calculations(
                    kb_dirs,
                    max_file_size_mb=config.dft.max_file_size_mb,
                )
                logger.info("dft_startup_index_complete", **result)

        logger.info("dft_index_initialized")
        return dft_index, dft_context_provider
    except Exception as exc:
        handle_optional_component_failure(
            config,
            logger,
            degraded_components,
            component="dft_index",
            error=exc,
        )
        return None, None


def build_dft_completion_hook(dft_index: Any, logger: Any):
    """SimScheduler용 DFT 인덱싱 완료 훅을 생성한다."""

    async def _hook(job: dict) -> None:
        if job.get("tool") not in ("orca_auto",):
            return
        output_file = job.get("output_file")
        if not output_file:
            return
        success = await dft_index.upsert_single(output_file)
        if success:
            logger.info(
                "sim_job_output_indexed",
                job_id=job.get("job_id", ""),
                output_file=output_file,
            )

    return _hook


def handle_optional_component_failure(
    config: AppSettings,
    logger: Any,
    degraded_components: list[dict[str, str]],
    *,
    component: str,
    error: Exception,
) -> None:
    error_text = str(error).strip() or type(error).__name__
    _record_degraded_component(
        logger,
        degraded_components,
        component=component,
        error=error_text,
    )
    if config.strict_startup:
        raise StartupError(
            "오류: strict_startup=true로 설정되어 선택 컴포넌트 초기화 실패 시 시작을 중단합니다.\n"
            f"- component: {component}\n"
            f"- reason: {error_text}"
        )


def log_degraded_startup_summary(
    logger: Any,
    degraded_components: list[dict[str, str]],
) -> None:
    if not degraded_components:
        return
    logger.warning(
        "startup_degraded_summary",
        degraded_count=len(degraded_components),
        degraded_components=degraded_components,
    )


async def build_runtime(
    config: AppSettings,
    logger: Any,
) -> RuntimeState:
    """의존성 순서대로 모듈을 초기화하고 런타임 상태를 반환한다."""
    cleanup_stack = AsyncExitStack()
    degraded_components: list[dict[str, str]] = []
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

        config.lemonade.host = resolve_wsl_loopback_host(
            url=config.lemonade.host,
            service_name="lemonade",
            logger=logger,
        )
        config.ollama.host = resolve_wsl_loopback_host(
            url=config.ollama.host,
            service_name="ollama",
            logger=logger,
        )

        llm_provider = "lemonade"
        llm = _create_llm_client(config)
        llm.default_model = config.lemonade.default_model
        llm_host = getattr(llm, "host", config.lemonade.host)
        try:
            await llm.initialize()
        except Exception as exc:
            logger.error("llm_init_failed", provider=llm_provider, error=str(exc))
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
            logger.info(
                "instant_responder_initialized",
                rules=instant_responder.rules_count,
            )

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
                logger.info(
                    "semantic_cache_initialized",
                    enabled=semantic_cache.enabled,
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
                handle_optional_component_failure(
                    config,
                    logger,
                    degraded_components,
                    component="intent_router",
                    error=exc,
                )
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

        rag_pipeline = None
        model_registry = None
        retrieval_client: OllamaClient | None = None
        rag_startup_index_task: asyncio.Task[Any] | None = None

        if config.rag.enabled:
            try:
                retrieval_client = _create_retrieval_client(config)
                await retrieval_client.initialize()
                cleanup_stack.push_async_callback(retrieval_client.close)
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
                retrieval_client = None

        if retrieval_client is not None:
            try:
                from core.config import ModelRegistryConfig
                from core.model_registry import ModelRegistry

                retrieval_proto = cast(RetrievalClientProtocol, retrieval_client)
                model_registry = ModelRegistry(
                    ModelRegistryConfig(
                        default_model=config.lemonade.default_model,
                        embedding_model=config.ollama.embedding_model,
                        reranker_model=config.ollama.reranker_model,
                    ),
                    retrieval_proto,
                )
                await model_registry.initialize()
            except Exception as exc:
                handle_optional_component_failure(
                    config,
                    logger,
                    degraded_components,
                    component="model_registry",
                    error=exc,
                )
                model_registry = None

        default_model = config.lemonade.default_model
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

        if config.rag.enabled and retrieval_client is None:
            logger.warning(
                "rag_pipeline_disabled",
                reason="retrieval_client_unavailable",
            )
        if config.rag.enabled and retrieval_client is not None:
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

        dft_index, dft_context_provider = await _init_dft(
            config,
            cleanup_stack,
            logger,
            degraded_components,
        )

        sim_scheduler = None
        if config.sim_queue.enabled:
            try:
                from core.sim_job_store import SimJobStore
                from core.sim_resource_manager import ResourceManager
                from core.sim_scheduler import SimJobScheduler

                sim_db_path = str(Path(config.data_dir) / "sim_queue" / "sim_jobs.db")
                sim_store = SimJobStore()
                await sim_store.initialize(sim_db_path)
                cleanup_stack.push_async_callback(sim_store.close)

                sim_resources = ResourceManager(
                    max_concurrent=config.sim_queue.max_concurrent_jobs,
                )

                sim_scheduler = SimJobScheduler(
                    config=config.sim_queue,
                    store=sim_store,
                    resources=sim_resources,
                )
                sim_scheduler.set_allowed_users(config.security.allowed_users)
                if dft_index is not None:
                    sim_scheduler.add_completion_hook(
                        build_dft_completion_hook(dft_index, logger),
                    )
                logger.info("sim_scheduler_initialized")
            except Exception as exc:
                handle_optional_component_failure(
                    config,
                    logger,
                    degraded_components,
                    component="sim_scheduler",
                    error=exc,
                )
                sim_scheduler = None

        context_providers = [dft_context_provider] if dft_context_provider else []
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
            rag_pipeline=rag_pipeline,
            context_providers=context_providers,
        )

        telegram = TelegramHandler(
            config=config,
            engine=engine,
            security=security,
            feedback=feedback,
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
            dft_index=dft_index,
            kb_dirs=config.rag.kb_dirs if config.rag.enabled else None,
            sim_scheduler=sim_scheduler,
        )
        telegram.set_scheduler(scheduler)
        if not telegram.has_scheduler():
            raise StartupError(
                "Telegram handler must receive scheduler before initialization."
            )

        if sim_scheduler is not None:
            telegram.set_sim_scheduler(sim_scheduler)
            sim_scheduler.set_telegram(telegram)

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
        log_degraded_startup_summary(logger, degraded_components)

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
            semantic_cache=semantic_cache,
            rag_startup_index_task=rag_startup_index_task,
            sim_scheduler=sim_scheduler,
            degraded_components=degraded_components,
        )
    except Exception:
        await cleanup_stack.aclose()
        raise
