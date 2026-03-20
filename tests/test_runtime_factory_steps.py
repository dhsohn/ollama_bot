"""Focused tests for extracted runtime factory helpers."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from core import runtime_factory_steps
from core.config import AppSettings
from core.runtime_factory_support import StartupError


def _install_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attrs,
) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class DummyMemory:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def prune_old_conversations(self) -> int:
        return 3

    async def close(self) -> None:
        self.closed = True


class DummyFeedbackDb:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class DummyFeedbackManager:
    def __init__(self, db: DummyFeedbackDb) -> None:
        self.db = db

    async def initialize_schema(self) -> None:
        return None

    async def prune_old_feedback(self, retention_days: int) -> int:
        assert retention_days == 90
        return 2


class DummyInstantResponder:
    def __init__(self, *, rules_path: str) -> None:
        self.rules_path = rules_path
        self.rules_count = 4


class DummySemanticCache:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.enabled = True
        self.encoder = "shared-encoder"

    async def initialize(self) -> None:
        return None


class FailingSemanticCache(DummySemanticCache):
    async def initialize(self) -> None:
        raise RuntimeError("cache boom")


class DummyIntentRouter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.enabled = True
        self.routes_count = 2


class DummyContextCompressor:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class DummyRetrievalClient:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.closed = False

    async def initialize(self) -> None:
        if self.should_fail:
            raise RuntimeError("retrieval boom")

    async def close(self) -> None:
        self.closed = True


class DummyModelRegistry:
    def __init__(self, config, retrieval_client) -> None:
        self.config = config
        self.retrieval_client = retrieval_client
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    def is_available(self, role: str) -> bool:
        return role == "reranker"


class FailingModelRegistry(DummyModelRegistry):
    async def initialize(self) -> None:
        raise RuntimeError("registry boom")


class DummyIndexer:
    instances: ClassVar[list[DummyIndexer]] = []

    def __init__(self, config, retrieval_client, embedding_model: str) -> None:
        self.config = config
        self.retrieval_client = retrieval_client
        self.embedding_model = embedding_model
        self.chunk_count = 7
        self.closed = False
        self.initialized_path: str | None = None
        DummyIndexer.instances.append(self)

    async def initialize(self, path: str) -> None:
        self.initialized_path = path

    async def close(self) -> None:
        self.closed = True

    async def index_corpus(self, kb_dirs: list[str]) -> dict[str, int]:
        _ = kb_dirs
        return {"indexed": 1, "skipped": 0, "removed": 0, "total_chunks": 7}


class FailingIndexer(DummyIndexer):
    async def initialize(self, path: str) -> None:
        _ = path
        raise RuntimeError("indexer boom")


class DummyRetriever:
    def __init__(self, indexer, retrieval_client, embedding_model: str) -> None:
        self.indexer = indexer
        self.retrieval_client = retrieval_client
        self.embedding_model = embedding_model


class DummyReranker:
    def __init__(self, retrieval_client, reranker_model: str, rag_config) -> None:
        self.retrieval_client = retrieval_client
        self.reranker_model = reranker_model
        self.rag_config = rag_config


class DummyContextBuilder:
    pass


class DummyRAGPipeline:
    def __init__(self, retriever, reranker, context_builder, config) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.config = config


class DummyLLM:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.host = "http://llm"
        self.default_model = ""
        self.closed = False
        self.should_fail = should_fail

    async def initialize(self) -> None:
        if self.should_fail:
            raise RuntimeError("boom")

    async def close(self) -> None:
        self.closed = True


class DummyScheduler:
    def __init__(
        self,
        *,
        config: AppSettings,
        security,
        auto_dir: str,
        ready: bool = True,
        load_error: Exception | None = None,
    ) -> None:
        self.config = config
        self.security = security
        self.auto_dir = auto_dir
        self._ready = ready
        self._load_error = load_error
        self.dependencies: dict[str, object] = {}

    def set_dependencies(self, *, engine, telegram) -> None:
        self.dependencies = {"engine": engine, "telegram": telegram}

    def dependencies_ready(self) -> bool:
        return self._ready

    async def load_automations(self, *, strict: bool) -> int:
        assert strict is True
        if self._load_error is not None:
            raise self._load_error
        return 2

    def get_last_load_errors(self) -> list[str]:
        return ["broken.yaml"]


class DummyTelegram:
    def __init__(self) -> None:
        self.scheduler = None

    def set_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler

    def has_scheduler(self) -> bool:
        return self.scheduler is not None

    async def initialize(self) -> str:
        return "telegram-app"


class BrokenTelegram(DummyTelegram):
    def has_scheduler(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_initialize_memory_stack_without_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    config.feedback.enabled = False
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()

    monkeypatch.setattr(runtime_factory_steps, "MemoryManager", DummyMemory)

    try:
        memory, feedback = await runtime_factory_steps.initialize_memory_stack(
            config,
            cleanup_stack,
            logger,
        )
        assert isinstance(memory, DummyMemory)
        assert feedback is None
        assert memory.kwargs["archive_enabled"] is True
        logger.info.assert_any_call("memory_retention_pruned_on_start", deleted=3)
    finally:
        await cleanup_stack.aclose()

    assert memory.closed is True


@pytest.mark.asyncio
async def test_initialize_memory_stack_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    feedback_db = DummyFeedbackDb()

    async def fake_open_sqlite_db(_path: Path) -> DummyFeedbackDb:
        return feedback_db

    monkeypatch.setattr(runtime_factory_steps, "MemoryManager", DummyMemory)
    monkeypatch.setattr(runtime_factory_steps, "_open_sqlite_db", fake_open_sqlite_db)
    monkeypatch.setattr(
        runtime_factory_steps,
        "FeedbackManager",
        DummyFeedbackManager,
    )

    try:
        _memory, feedback = await runtime_factory_steps.initialize_memory_stack(
            config,
            cleanup_stack,
            logger,
        )
        assert isinstance(feedback, DummyFeedbackManager)
        logger.info.assert_any_call("feedback_pruned", count=2)
    finally:
        await cleanup_stack.aclose()

    assert feedback_db.closed is True


def test_rewrite_ollama_host_rewrites_local_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()

    monkeypatch.setattr(
        runtime_factory_steps,
        "resolve_wsl_loopback_host",
        lambda *, url, service_name, logger: f"{service_name}:{url}",
    )

    runtime_factory_steps.rewrite_ollama_host(config, logger)
    assert config.ollama.host.startswith("ollama:")


@pytest.mark.asyncio
async def test_initialize_chat_client_sets_default_model_and_registers_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    llm = DummyLLM()

    monkeypatch.setattr(runtime_factory_steps, "OllamaClient", lambda _config: llm)
    monkeypatch.setattr(runtime_factory_steps, "get_default_chat_model", lambda _config: "model-x")

    try:
        resolved_llm, default_model = await runtime_factory_steps.initialize_chat_client(
            config,
            cleanup_stack,
            logger,
        )
        assert resolved_llm is llm
        assert default_model == "model-x"
        assert llm.default_model == "model-x"
    finally:
        await cleanup_stack.aclose()

    assert llm.closed is True


@pytest.mark.asyncio
async def test_initialize_chat_client_wraps_startup_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    llm = DummyLLM(should_fail=True)

    monkeypatch.setattr(runtime_factory_steps, "OllamaClient", lambda _config: llm)
    monkeypatch.setattr(runtime_factory_steps, "get_default_chat_model", lambda _config: "model-x")

    with pytest.raises(StartupError, match="failed to initialize Ollama"):
        await runtime_factory_steps.initialize_chat_client(
            config,
            cleanup_stack,
            logger,
        )

    await cleanup_stack.aclose()


@pytest.mark.asyncio
async def test_initialize_skills_logs_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.skill_manager as skill_manager

    class DummySkillManager:
        def __init__(self, *, security, skills_dir: str) -> None:
            self.security = security
            self.skills_dir = skills_dir

        async def load_skills(self, *, strict: bool) -> int:
            assert strict is True
            return 4

        def get_last_load_errors(self) -> list[str]:
            return ["bad-skill.yaml"]

    logger = MagicMock()
    monkeypatch.setattr(skill_manager, "SkillManager", DummySkillManager)

    skills, count = await runtime_factory_steps.initialize_skills(object(), logger)

    assert isinstance(skills, DummySkillManager)
    assert count == 4
    logger.warning.assert_any_call(
        "skills_loaded_with_partial_failures",
        error_count=1,
        sample=["bad-skill.yaml"],
    )


@pytest.mark.asyncio
async def test_initialize_optional_components_initializes_enabled_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    semantic_db = DummyFeedbackDb()

    async def fake_open_sqlite_db(_path: Path) -> DummyFeedbackDb:
        return semantic_db

    async def fake_run_in_thread(cls, **kwargs):
        return cls(**kwargs)

    monkeypatch.setattr(runtime_factory_steps, "_open_sqlite_db", fake_open_sqlite_db)
    monkeypatch.setattr(runtime_factory_steps, "InstantResponder", DummyInstantResponder)
    monkeypatch.setattr(
        runtime_factory_steps,
        "ContextCompressor",
        DummyContextCompressor,
    )
    monkeypatch.setattr(runtime_factory_steps, "run_in_thread", fake_run_in_thread)
    _install_module(monkeypatch, "core.semantic_cache", SemanticCache=DummySemanticCache)
    _install_module(monkeypatch, "core.intent_router", IntentRouter=DummyIntentRouter)

    try:
        components = await runtime_factory_steps.initialize_optional_components(
            config,
            logger,
            degraded_components,
            llm=object(),
            memory=object(),
            cleanup_stack=cleanup_stack,
        )
        assert isinstance(components.instant_responder, DummyInstantResponder)
        assert isinstance(components.semantic_cache, DummySemanticCache)
        assert isinstance(components.intent_router, DummyIntentRouter)
        assert isinstance(components.context_compressor, DummyContextCompressor)
        assert components.intent_router.kwargs["encoder"] == "shared-encoder"
        assert degraded_components == []
    finally:
        await cleanup_stack.aclose()

    assert semantic_db.closed is True
    logger.info.assert_any_call("context_compressor_initialized")


@pytest.mark.asyncio
async def test_initialize_optional_components_records_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    config.instant_responder.enabled = False
    config.context_compressor.enabled = False
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    semantic_db = DummyFeedbackDb()
    recorder = MagicMock()

    async def fake_open_sqlite_db(_path: Path) -> DummyFeedbackDb:
        return semantic_db

    async def fake_run_in_thread(cls, **kwargs):
        return cls(**kwargs)

    class FailingIntentRouter(DummyIntentRouter):
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            raise RuntimeError("router boom")

    monkeypatch.setattr(runtime_factory_steps, "_open_sqlite_db", fake_open_sqlite_db)
    monkeypatch.setattr(runtime_factory_steps, "run_in_thread", fake_run_in_thread)
    monkeypatch.setattr(
        runtime_factory_steps,
        "handle_optional_component_failure",
        recorder,
    )
    _install_module(monkeypatch, "core.semantic_cache", SemanticCache=FailingSemanticCache)
    _install_module(monkeypatch, "core.intent_router", IntentRouter=FailingIntentRouter)

    try:
        components = await runtime_factory_steps.initialize_optional_components(
            config,
            logger,
            degraded_components,
            llm=object(),
            memory=object(),
            cleanup_stack=cleanup_stack,
        )
        assert components.instant_responder is None
        assert components.semantic_cache is None
        assert components.intent_router is None
        assert components.context_compressor is None
    finally:
        await cleanup_stack.aclose()

    assert semantic_db.closed is True
    assert [call.kwargs["component"] for call in recorder.call_args_list] == [
        "semantic_cache",
        "intent_router",
    ]


@pytest.mark.asyncio
async def test_initialize_retrieval_components_initializes_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    retrieval_client = DummyRetrievalClient()

    monkeypatch.setattr(
        runtime_factory_steps,
        "_create_retrieval_client",
        lambda _config: retrieval_client,
    )
    monkeypatch.setattr(runtime_factory_steps, "get_default_chat_model", lambda _config: "chat-model")
    _install_module(monkeypatch, "core.model_registry", ModelRegistry=DummyModelRegistry)

    try:
        components = await runtime_factory_steps.initialize_retrieval_components(
            config,
            logger,
            degraded_components,
            cleanup_stack,
        )
        assert components.retrieval_client is retrieval_client
        assert isinstance(components.model_registry, DummyModelRegistry)
        assert components.model_registry.config.default_model == "chat-model"
        assert components.model_registry.initialized is True
        assert degraded_components == []
    finally:
        await cleanup_stack.aclose()

    assert retrieval_client.closed is True


@pytest.mark.asyncio
async def test_initialize_retrieval_components_records_client_and_registry_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    recorder = MagicMock()

    monkeypatch.setattr(
        runtime_factory_steps,
        "handle_optional_component_failure",
        recorder,
    )
    monkeypatch.setattr(
        runtime_factory_steps,
        "_create_retrieval_client",
        lambda _config: DummyRetrievalClient(should_fail=True),
    )

    cleanup_stack = AsyncExitStack()
    try:
        components = await runtime_factory_steps.initialize_retrieval_components(
            config,
            logger,
            degraded_components,
            cleanup_stack,
        )
        assert components.retrieval_client is None
        assert components.model_registry is None
    finally:
        await cleanup_stack.aclose()

    assert recorder.call_args.kwargs["component"] == "retrieval_client"

    retrieval_client = DummyRetrievalClient()
    recorder.reset_mock()
    monkeypatch.setattr(
        runtime_factory_steps,
        "_create_retrieval_client",
        lambda _config: retrieval_client,
    )
    _install_module(monkeypatch, "core.model_registry", ModelRegistry=FailingModelRegistry)

    cleanup_stack = AsyncExitStack()
    try:
        components = await runtime_factory_steps.initialize_retrieval_components(
            config,
            logger,
            degraded_components,
            cleanup_stack,
        )
        assert components.retrieval_client is retrieval_client
        assert components.model_registry is None
    finally:
        await cleanup_stack.aclose()

    assert recorder.call_args.kwargs["component"] == "model_registry"


@pytest.mark.asyncio
async def test_preload_default_model_reports_optional_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    recorder = MagicMock()

    async def failing_prepare_model(**_kwargs) -> None:
        raise RuntimeError("load failed")

    llm = SimpleNamespace(prepare_model=failing_prepare_model)
    monkeypatch.setattr(
        runtime_factory_steps,
        "handle_optional_component_failure",
        recorder,
    )

    await runtime_factory_steps.preload_default_model(
        llm,
        default_model="model-x",
        config=config,
        logger=logger,
        degraded_components=degraded_components,
    )

    recorder.assert_called_once()
    assert recorder.call_args.kwargs["component"] == "default_model_preload"


def test_normalize_corpus_roots_and_collect_existing_kb_dirs(tmp_path: Path) -> None:
    logger = MagicMock()
    existing = tmp_path / "kb"
    existing.mkdir()
    roots = runtime_factory_steps.normalize_corpus_roots(
        [" ", str(existing), str(existing), str(tmp_path / "missing")]
    )

    assert roots == [str(existing), str(tmp_path / "missing")]
    assert runtime_factory_steps.collect_existing_kb_dirs(roots, logger) == [str(existing)]
    logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_rag_pipeline_builds_pipeline_and_startup_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    config.rag.kb_dirs = [str(tmp_path / "kb")]
    (tmp_path / "kb").mkdir()
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    retrieval_client = object()
    model_registry = MagicMock(is_available=MagicMock(return_value=True))
    startup_task = object()

    DummyIndexer.instances.clear()
    monkeypatch.setattr(
        runtime_factory_steps,
        "schedule_rag_startup_index",
        MagicMock(return_value=startup_task),
    )
    _install_module(monkeypatch, "core.rag.context_builder", RAGContextBuilder=DummyContextBuilder)
    _install_module(monkeypatch, "core.rag.indexer", RAGIndexer=DummyIndexer)
    _install_module(monkeypatch, "core.rag.pipeline", RAGPipeline=DummyRAGPipeline)
    _install_module(monkeypatch, "core.rag.reranker", RAGReranker=DummyReranker)
    _install_module(monkeypatch, "core.rag.retriever", RAGRetriever=DummyRetriever)

    try:
        rag_pipeline, rag_task = await runtime_factory_steps.initialize_rag_pipeline(
            config,
            logger,
            degraded_components,
            cleanup_stack,
            retrieval_client=retrieval_client,
            model_registry=model_registry,
        )
        assert isinstance(rag_pipeline, DummyRAGPipeline)
        assert isinstance(rag_pipeline.retriever, DummyRetriever)
        assert isinstance(rag_pipeline.reranker, DummyReranker)
        assert isinstance(rag_pipeline.context_builder, DummyContextBuilder)
        assert rag_task is startup_task
        assert DummyIndexer.instances[-1].initialized_path is not None
    finally:
        await cleanup_stack.aclose()

    assert DummyIndexer.instances[-1].closed is True
    logger.info.assert_any_call(
        "rag_pipeline_initialized",
        chunks=7,
        reranker=True,
    )


@pytest.mark.asyncio
async def test_initialize_rag_pipeline_handles_disabled_or_failed_reranker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    config.rag.kb_dirs = [str(tmp_path / "kb")]
    config.rag.startup_index_enabled = False
    (tmp_path / "kb").mkdir()
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    retrieval_client = object()
    recorder = MagicMock()

    class BrokenRegistry:
        def is_available(self, role: str) -> bool:
            _ = role
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(
        runtime_factory_steps,
        "handle_optional_component_failure",
        recorder,
    )
    monkeypatch.setattr(
        runtime_factory_steps,
        "schedule_rag_startup_index",
        MagicMock(),
    )
    _install_module(monkeypatch, "core.rag.context_builder", RAGContextBuilder=DummyContextBuilder)
    _install_module(monkeypatch, "core.rag.indexer", RAGIndexer=DummyIndexer)
    _install_module(monkeypatch, "core.rag.pipeline", RAGPipeline=DummyRAGPipeline)
    _install_module(monkeypatch, "core.rag.reranker", RAGReranker=DummyReranker)
    _install_module(monkeypatch, "core.rag.retriever", RAGRetriever=DummyRetriever)

    try:
        rag_pipeline, rag_task = await runtime_factory_steps.initialize_rag_pipeline(
            config,
            logger,
            degraded_components,
            cleanup_stack,
            retrieval_client=retrieval_client,
            model_registry=BrokenRegistry(),
        )
        assert isinstance(rag_pipeline, DummyRAGPipeline)
        assert rag_pipeline.reranker is None
        assert rag_task is None
    finally:
        await cleanup_stack.aclose()

    assert recorder.call_args.kwargs["component"] == "rag_reranker_availability"
    logger.info.assert_any_call("rag_startup_index_skipped", reason="disabled", roots=1)


@pytest.mark.asyncio
async def test_initialize_rag_pipeline_handles_disabled_missing_and_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    cleanup_stack = AsyncExitStack()
    logger = MagicMock()
    degraded_components: list[dict[str, str]] = []
    recorder = MagicMock()

    rag_pipeline, rag_task = await runtime_factory_steps.initialize_rag_pipeline(
        config,
        logger,
        degraded_components,
        cleanup_stack,
        retrieval_client=None,
        model_registry=None,
    )
    assert rag_pipeline is None
    assert rag_task is None
    logger.warning.assert_any_call(
        "rag_pipeline_disabled",
        reason="retrieval_client_unavailable",
    )

    logger.reset_mock()
    config.rag.enabled = False
    rag_pipeline, rag_task = await runtime_factory_steps.initialize_rag_pipeline(
        config,
        logger,
        degraded_components,
        cleanup_stack,
        retrieval_client=object(),
        model_registry=None,
    )
    assert rag_pipeline is None
    assert rag_task is None
    logger.warning.assert_not_called()

    logger.reset_mock()
    config.rag.enabled = True
    config.rag.kb_dirs = [str(tmp_path / "missing")]
    monkeypatch.setattr(
        runtime_factory_steps,
        "handle_optional_component_failure",
        recorder,
    )
    _install_module(monkeypatch, "core.rag.context_builder", RAGContextBuilder=DummyContextBuilder)
    _install_module(monkeypatch, "core.rag.indexer", RAGIndexer=FailingIndexer)
    _install_module(monkeypatch, "core.rag.pipeline", RAGPipeline=DummyRAGPipeline)
    _install_module(monkeypatch, "core.rag.reranker", RAGReranker=DummyReranker)
    _install_module(monkeypatch, "core.rag.retriever", RAGRetriever=DummyRetriever)

    rag_pipeline, rag_task = await runtime_factory_steps.initialize_rag_pipeline(
        config,
        logger,
        degraded_components,
        cleanup_stack,
        retrieval_client=object(),
        model_registry=None,
    )
    assert rag_pipeline is None
    assert rag_task is None
    assert recorder.call_args.kwargs["component"] == "rag_pipeline"

    await cleanup_stack.aclose()


@pytest.mark.asyncio
async def test_schedule_rag_startup_index_logs_completion() -> None:
    logger = MagicMock()
    indexer = SimpleNamespace(
        index_corpus=AsyncMock(
            return_value={"indexed": 1, "skipped": 2, "removed": 0, "total_chunks": 3}
        )
    )

    task = runtime_factory_steps.schedule_rag_startup_index(
        indexer=indexer,
        kb_dirs_to_index=["/tmp/kb"],
        logger=logger,
    )
    await task

    logger.info.assert_any_call("rag_startup_index_started", roots=1)
    logger.info.assert_any_call(
        "rag_startup_index_completed",
        indexed=1,
        skipped=2,
        removed=0,
        total_chunks=3,
    )


@pytest.mark.asyncio
async def test_schedule_rag_startup_index_logs_timeout_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    indexer = SimpleNamespace(index_corpus=AsyncMock())

    async def fake_wait_for_timeout(awaitable, *, timeout: float | None = None):
        _ = timeout
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(runtime_factory_steps.asyncio, "wait_for", fake_wait_for_timeout)
    task = runtime_factory_steps.schedule_rag_startup_index(
        indexer=indexer,
        kb_dirs_to_index=["/tmp/kb"],
        logger=logger,
    )
    await task
    logger.error.assert_any_call("rag_startup_index_timeout", timeout_seconds=600.0)

    async def fake_wait_for_error(awaitable, *, timeout: float | None = None):
        _ = timeout
        awaitable.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_factory_steps.asyncio, "wait_for", fake_wait_for_error)
    task = runtime_factory_steps.schedule_rag_startup_index(
        indexer=indexer,
        kb_dirs_to_index=["/tmp/kb"],
        logger=logger,
    )
    await task
    logger.error.assert_any_call("rag_startup_index_failed", error="boom")


@pytest.mark.asyncio
async def test_initialize_scheduler_stack_wires_scheduler_and_loads_automations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()
    telegram = DummyTelegram()
    register_callables = MagicMock()

    monkeypatch.setattr(runtime_factory_steps, "AutoScheduler", DummyScheduler)
    monkeypatch.setattr(
        runtime_factory_steps,
        "register_builtin_callables",
        register_callables,
    )

    scheduler, auto_count, app = await runtime_factory_steps.initialize_scheduler_stack(
        config,
        logger,
        security=object(),
        engine=object(),
        telegram=telegram,
        memory=object(),
        feedback=None,
    )

    assert isinstance(scheduler, DummyScheduler)
    assert auto_count == 2
    assert app == "telegram-app"
    assert telegram.scheduler is scheduler
    register_callables.assert_called_once()
    logger.info.assert_any_call("automations_loaded", count=2)
    logger.warning.assert_any_call(
        "automations_loaded_with_partial_failures",
        error_count=1,
        sample=["broken.yaml"],
    )


@pytest.mark.asyncio
async def test_initialize_scheduler_stack_wraps_load_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()
    telegram = DummyTelegram()

    monkeypatch.setattr(
        runtime_factory_steps,
        "AutoScheduler",
        lambda **kwargs: DummyScheduler(load_error=RuntimeError("bad yaml"), **kwargs),
    )
    monkeypatch.setattr(
        runtime_factory_steps,
        "register_builtin_callables",
        MagicMock(),
    )

    with pytest.raises(StartupError, match="failed to load automations"):
        await runtime_factory_steps.initialize_scheduler_stack(
            config,
            logger,
            security=object(),
            engine=object(),
            telegram=telegram,
            memory=object(),
            feedback=None,
        )


@pytest.mark.asyncio
async def test_initialize_scheduler_stack_wraps_init_and_wiring_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = AppSettings(data_dir=str(tmp_path))
    logger = MagicMock()

    def raise_scheduler_init(**kwargs):
        _ = kwargs
        raise RuntimeError("bad timezone")

    monkeypatch.setattr(runtime_factory_steps, "AutoScheduler", raise_scheduler_init)
    with pytest.raises(
        StartupError,
        match="failed to initialize the automation scheduler",
    ):
        await runtime_factory_steps.initialize_scheduler_stack(
            config,
            logger,
            security=object(),
            engine=object(),
            telegram=DummyTelegram(),
            memory=object(),
            feedback=None,
        )

    monkeypatch.setattr(
        runtime_factory_steps,
        "AutoScheduler",
        lambda **kwargs: DummyScheduler(ready=False, **kwargs),
    )
    monkeypatch.setattr(
        runtime_factory_steps,
        "register_builtin_callables",
        MagicMock(),
    )
    with pytest.raises(StartupError, match="Scheduler dependencies must be wired"):
        await runtime_factory_steps.initialize_scheduler_stack(
            config,
            logger,
            security=object(),
            engine=object(),
            telegram=DummyTelegram(),
            memory=object(),
            feedback=None,
        )

    monkeypatch.setattr(runtime_factory_steps, "AutoScheduler", DummyScheduler)
    with pytest.raises(StartupError, match="Telegram handler must receive scheduler"):
        await runtime_factory_steps.initialize_scheduler_stack(
            config,
            logger,
            security=object(),
            engine=object(),
            telegram=BrokenTelegram(),
            memory=object(),
            feedback=None,
        )
