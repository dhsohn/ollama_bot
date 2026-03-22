"""Runtime dependency initialization factory."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from core.auto_scheduler import AutoScheduler
from core.config import AppSettings
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.llm_protocol import LLMClientProtocol
from core.memory import MemoryManager
from core.runtime_factory_steps import (
    OptionalRuntimeComponents,
    RetrievalRuntimeComponents,
    initialize_chat_client,
    initialize_memory_stack,
    initialize_optional_components,
    initialize_rag_pipeline,
    initialize_retrieval_components,
    initialize_scheduler_stack,
    initialize_skills,
    preload_default_model,
    rewrite_ollama_host,
)
from core.runtime_factory_support import (
    StartupError,
    _acquire_runtime_lock,
    handle_optional_component_failure,
    log_degraded_startup_summary,
    validate_required_settings,
)
from core.security import SecurityManager
from core.telegram_handler import TelegramHandler


@dataclass
class RuntimeState:
    """Bundle initialized runtime dependencies."""

    config: AppSettings
    logger: Any
    memory: MemoryManager
    llm: LLMClientProtocol
    app: Any
    scheduler: AutoScheduler
    skill_count: int
    auto_count: int
    cleanup_stack: AsyncExitStack
    feedback: FeedbackManager | None = None
    semantic_cache: Any = None
    rag_startup_index_task: asyncio.Task[Any] | None = None
    degraded_components: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RuntimeBootstrapArtifacts:
    """Intermediate runtime dependencies collected before final wiring."""

    security: SecurityManager
    memory: MemoryManager
    feedback: FeedbackManager | None
    llm: LLMClientProtocol
    default_model: str
    skills: Any
    skill_count: int
    optional_components: OptionalRuntimeComponents
    retrieval_components: RetrievalRuntimeComponents
    rag_pipeline: Any
    rag_startup_index_task: asyncio.Task[Any] | None


async def _initialize_runtime_dependencies(
    config: AppSettings,
    logger: Any,
    cleanup_stack: AsyncExitStack,
    degraded_components: list[dict[str, str]],
) -> RuntimeBootstrapArtifacts:
    """Build the dependency graph up to the Engine/Telegram wiring boundary."""
    security = SecurityManager(config.security)
    logger.info(
        "security_initialized",
        allowed_users=len(config.security.allowed_users),
    )

    memory, feedback = await initialize_memory_stack(config, cleanup_stack, logger)
    rewrite_ollama_host(config, logger)
    llm, default_model = await initialize_chat_client(
        config,
        cleanup_stack,
        logger,
    )
    skills, skill_count = await initialize_skills(security, logger)
    optional_components = await initialize_optional_components(
        config,
        logger,
        degraded_components,
        llm=llm,
        memory=memory,
        cleanup_stack=cleanup_stack,
    )
    retrieval_components = await initialize_retrieval_components(
        config,
        logger,
        degraded_components,
        cleanup_stack,
    )
    await preload_default_model(
        llm,
        default_model=default_model,
        config=config,
        logger=logger,
        degraded_components=degraded_components,
    )
    rag_pipeline, rag_startup_index_task = await initialize_rag_pipeline(
        config,
        logger,
        degraded_components,
        cleanup_stack,
        retrieval_client=retrieval_components.retrieval_client,
        model_registry=retrieval_components.model_registry,
    )
    return RuntimeBootstrapArtifacts(
        security=security,
        memory=memory,
        feedback=feedback,
        llm=llm,
        default_model=default_model,
        skills=skills,
        skill_count=skill_count,
        optional_components=optional_components,
        retrieval_components=retrieval_components,
        rag_pipeline=rag_pipeline,
        rag_startup_index_task=rag_startup_index_task,
    )


def _build_engine(
    config: AppSettings,
    artifacts: RuntimeBootstrapArtifacts,
) -> Engine:
    """Construct the Engine from the already-initialized runtime artifacts."""
    return Engine(
        config=config,
        llm_client=artifacts.llm,
        memory=artifacts.memory,
        skills=artifacts.skills,
        feedback_manager=artifacts.feedback,
        instant_responder=artifacts.optional_components.instant_responder,
        semantic_cache=artifacts.optional_components.semantic_cache,
        intent_router=artifacts.optional_components.intent_router,
        context_compressor=artifacts.optional_components.context_compressor,
        rag_pipeline=artifacts.rag_pipeline,
    )


def _build_telegram(
    config: AppSettings,
    artifacts: RuntimeBootstrapArtifacts,
    engine: Engine,
) -> TelegramHandler:
    """Construct the Telegram handler from the initialized runtime artifacts."""
    return TelegramHandler(
        config=config,
        engine=engine,
        security=artifacts.security,
        feedback=artifacts.feedback,
        semantic_cache=artifacts.optional_components.semantic_cache,
    )


async def build_runtime(
    config: AppSettings,
    logger: Any,
) -> RuntimeState:
    """Initialize modules in dependency order and return the runtime state."""
    cleanup_stack = AsyncExitStack()
    degraded_components: list[dict[str, str]] = []
    try:
        _acquire_runtime_lock(config, cleanup_stack, logger)
        artifacts = await _initialize_runtime_dependencies(
            config,
            logger,
            cleanup_stack,
            degraded_components,
        )
        engine = _build_engine(config, artifacts)
        telegram = _build_telegram(config, artifacts, engine)

        scheduler, auto_count, app = await initialize_scheduler_stack(
            config,
            logger,
            artifacts.security,
            engine,
            telegram,
            artifacts.memory,
            artifacts.feedback,
        )
        log_degraded_startup_summary(logger, degraded_components)

        return RuntimeState(
            config=config,
            logger=logger,
            memory=artifacts.memory,
            llm=artifacts.llm,
            app=app,
            scheduler=scheduler,
            skill_count=artifacts.skill_count,
            auto_count=auto_count,
            cleanup_stack=cleanup_stack,
            feedback=artifacts.feedback,
            semantic_cache=artifacts.optional_components.semantic_cache,
            rag_startup_index_task=artifacts.rag_startup_index_task,
            degraded_components=degraded_components,
        )
    except Exception:
        await cleanup_stack.aclose()
        raise
