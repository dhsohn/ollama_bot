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


async def build_runtime(
    config: AppSettings,
    logger: Any,
) -> RuntimeState:
    """Initialize modules in dependency order and return the runtime state."""
    cleanup_stack = AsyncExitStack()
    degraded_components: list[dict[str, str]] = []
    try:
        _acquire_runtime_lock(config, cleanup_stack, logger)

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
        instant_responder = optional_components.instant_responder
        semantic_cache = optional_components.semantic_cache
        intent_router = optional_components.intent_router
        context_compressor = optional_components.context_compressor

        retrieval_components = await initialize_retrieval_components(
            config,
            logger,
            degraded_components,
            cleanup_stack,
        )
        retrieval_client = retrieval_components.retrieval_client
        model_registry = retrieval_components.model_registry

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
            retrieval_client=retrieval_client,
            model_registry=model_registry,
        )

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
        )

        telegram = TelegramHandler(
            config=config,
            engine=engine,
            security=security,
            feedback=feedback,
            semantic_cache=semantic_cache,
        )

        scheduler, auto_count, app = await initialize_scheduler_stack(
            config,
            logger,
            security,
            engine,
            telegram,
            memory,
            feedback,
        )
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
            cleanup_stack=cleanup_stack,
            feedback=feedback,
            semantic_cache=semantic_cache,
            rag_startup_index_task=rag_startup_index_task,
            degraded_components=degraded_components,
        )
    except Exception:
        await cleanup_stack.aclose()
        raise
