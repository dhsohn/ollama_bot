"""runtime_factory 런타임 락 테스트."""

from __future__ import annotations

from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core import runtime_factory
from core.config import AppSettings
from core.runtime_factory import StartupError, _acquire_runtime_lock
from core.runtime_factory_steps import (
    OptionalRuntimeComponents,
    RetrievalRuntimeComponents,
)


@pytest.mark.asyncio
async def test_runtime_lock_blocks_second_instance(tmp_path) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    first_stack = AsyncExitStack()
    second_stack = AsyncExitStack()

    try:
        _acquire_runtime_lock(config, first_stack, MagicMock())

        with pytest.raises(StartupError, match="이미 실행 중인 ollama_bot 인스턴스"):
            _acquire_runtime_lock(config, second_stack, MagicMock())
    finally:
        await second_stack.aclose()
        await first_stack.aclose()


@pytest.mark.asyncio
async def test_runtime_lock_can_be_reacquired_after_cleanup(tmp_path) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    first_stack = AsyncExitStack()
    second_stack = AsyncExitStack()

    try:
        _acquire_runtime_lock(config, first_stack, MagicMock())
        await first_stack.aclose()

        _acquire_runtime_lock(config, second_stack, MagicMock())
    finally:
        await second_stack.aclose()


@pytest.mark.asyncio
async def test_build_runtime_orchestrates_dependencies(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()
    memory = SimpleNamespace()
    feedback = SimpleNamespace()
    llm = SimpleNamespace()
    degraded_sample = {"component": "semantic_cache", "error": "disabled"}
    engine_instances: list[SimpleNamespace] = []
    telegram_instances: list[SimpleNamespace] = []

    class DummyEngine(SimpleNamespace):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            engine_instances.append(self)

    class DummyTelegram(SimpleNamespace):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            telegram_instances.append(self)

    async def fake_initialize_memory_stack(_config, _cleanup_stack, _logger):
        return memory, feedback

    def fake_rewrite_ollama_host(_config, _logger) -> None:
        return None

    async def fake_initialize_chat_client(_config, _cleanup_stack, _logger):
        return llm, "default-model"

    async def fake_initialize_skills(_security, _logger):
        return "skills", 7

    async def fake_initialize_optional_components(
        _config,
        _logger,
        degraded_components,
        *,
        llm,
        memory,
        cleanup_stack,
    ):
        _ = (llm, memory, cleanup_stack)
        degraded_components.append(degraded_sample)
        return OptionalRuntimeComponents(
            instant_responder="instant",
            semantic_cache="cache",
            intent_router="router",
            context_compressor="compressor",
        )

    async def fake_initialize_retrieval_components(
        _config,
        _logger,
        _degraded_components,
        _cleanup_stack,
    ):
        return RetrievalRuntimeComponents(
            retrieval_client="retrieval",
            model_registry="registry",
        )

    async def fake_preload_default_model(
        _llm,
        *,
        default_model: str,
        config,
        logger,
        degraded_components,
    ) -> None:
        _ = (_llm, config, logger, degraded_components)
        assert default_model == "default-model"

    async def fake_initialize_rag_pipeline(
        _config,
        _logger,
        _degraded_components,
        _cleanup_stack,
        *,
        retrieval_client,
        model_registry,
    ):
        assert retrieval_client == "retrieval"
        assert model_registry == "registry"
        return "rag-pipeline", "rag-task"

    async def fake_initialize_scheduler_stack(
        _config,
        _logger,
        security,
        engine,
        telegram,
        memory,
        feedback,
    ):
        assert security == "security"
        assert engine is engine_instances[0]
        assert telegram is telegram_instances[0]
        assert memory is not None
        assert feedback is not None
        return "scheduler", 3, "telegram-app"

    log_summary = MagicMock()

    monkeypatch.setattr(runtime_factory, "_acquire_runtime_lock", lambda *_args: None)
    monkeypatch.setattr(runtime_factory, "SecurityManager", lambda _security: "security")
    monkeypatch.setattr(runtime_factory, "initialize_memory_stack", fake_initialize_memory_stack)
    monkeypatch.setattr(runtime_factory, "rewrite_ollama_host", fake_rewrite_ollama_host)
    monkeypatch.setattr(runtime_factory, "initialize_chat_client", fake_initialize_chat_client)
    monkeypatch.setattr(runtime_factory, "initialize_skills", fake_initialize_skills)
    monkeypatch.setattr(
        runtime_factory,
        "initialize_optional_components",
        fake_initialize_optional_components,
    )
    monkeypatch.setattr(
        runtime_factory,
        "initialize_retrieval_components",
        fake_initialize_retrieval_components,
    )
    monkeypatch.setattr(runtime_factory, "preload_default_model", fake_preload_default_model)
    monkeypatch.setattr(runtime_factory, "initialize_rag_pipeline", fake_initialize_rag_pipeline)
    monkeypatch.setattr(
        runtime_factory,
        "initialize_scheduler_stack",
        fake_initialize_scheduler_stack,
    )
    monkeypatch.setattr(runtime_factory, "Engine", DummyEngine)
    monkeypatch.setattr(runtime_factory, "TelegramHandler", DummyTelegram)
    monkeypatch.setattr(runtime_factory, "log_degraded_startup_summary", log_summary)

    runtime_state = await runtime_factory.build_runtime(config, logger)
    try:
        assert runtime_state.memory is memory
        assert runtime_state.feedback is feedback
        assert runtime_state.llm is llm
        assert runtime_state.scheduler == "scheduler"
        assert runtime_state.app == "telegram-app"
        assert runtime_state.skill_count == 7
        assert runtime_state.auto_count == 3
        assert runtime_state.rag_startup_index_task == "rag-task"
        assert runtime_state.semantic_cache == "cache"
        assert runtime_state.degraded_components == [degraded_sample]
        assert engine_instances[0].rag_pipeline == "rag-pipeline"
        assert telegram_instances[0].semantic_cache == "cache"
        log_summary.assert_called_once_with(logger, [degraded_sample])
    finally:
        await runtime_state.cleanup_stack.aclose()


@pytest.mark.asyncio
async def test_build_runtime_closes_cleanup_stack_on_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()
    cleanup_calls: list[str] = []

    async def fake_cleanup() -> None:
        cleanup_calls.append("closed")

    def fake_acquire_runtime_lock(_config, cleanup_stack: AsyncExitStack, _logger) -> None:
        cleanup_stack.push_async_callback(fake_cleanup)

    async def fake_initialize_memory_stack(_config, _cleanup_stack, _logger):
        return SimpleNamespace(), None

    async def fail_initialize_chat_client(_config, _cleanup_stack, _logger):
        raise RuntimeError("llm init failed")

    monkeypatch.setattr(runtime_factory, "_acquire_runtime_lock", fake_acquire_runtime_lock)
    monkeypatch.setattr(runtime_factory, "SecurityManager", lambda _security: "security")
    monkeypatch.setattr(runtime_factory, "initialize_memory_stack", fake_initialize_memory_stack)
    monkeypatch.setattr(runtime_factory, "rewrite_ollama_host", lambda *_args: None)
    monkeypatch.setattr(runtime_factory, "initialize_chat_client", fail_initialize_chat_client)

    with pytest.raises(RuntimeError, match="llm init failed"):
        await runtime_factory.build_runtime(config, logger)

    assert cleanup_calls == ["closed"]
