"""engine_management 모듈 추가 커버리지 테스트."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine_management import (
    change_model,
    classify_intent,
    clear_conversation,
    execute_skill,
    export_conversation_markdown,
    generate,
    get_current_model,
    get_last_skill_load_errors,
    get_memory_stats,
    list_models,
    list_skills,
    process_prompt,
    reload_skills,
    retrieve,
    rollback_last_turn,
    route_request,
)
from core.llm_types import ChatResponse
from core.rag.types import Chunk, ChunkMetadata, RAGResult, RAGTrace, RetrievedItem
from core.skill_manager import SkillDefinition


def _make_engine(**overrides):
    engine = MagicMock()
    engine._llm_client = AsyncMock()
    engine._llm_client.default_model = "test-model"
    engine._llm_client.system_prompt = "sys"
    engine._llm_client.chat = AsyncMock(return_value=ChatResponse(content="response"))
    engine._llm_client.list_models = AsyncMock(return_value=[{"name": "test-model", "size": 1024}])
    engine._llm_client.health_check = AsyncMock(return_value={"status": "ok"})
    engine._memory = AsyncMock()
    engine._skills = MagicMock()
    engine._rag_pipeline = None
    engine._semantic_cache = None
    engine._config = MagicMock()
    engine._config.bot.response_timeout = 60
    engine._config.lemonade.default_model = "test-model"
    engine._system_prompt = "sys"
    engine._inject_language_policy = MagicMock(side_effect=lambda x: x)
    engine._resolve_inference_timeout = MagicMock(return_value=60)
    engine._prepare_target_model = AsyncMock(return_value=("test-model", None))
    engine._resolve_model_for_role = MagicMock(return_value=None)
    engine._logger = MagicMock()
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


class TestRollbackLastTurn:
    @pytest.mark.asyncio
    async def test_delegates_to_memory(self) -> None:
        engine = _make_engine()
        engine._memory.delete_last_turn = AsyncMock(return_value=2)
        result = await rollback_last_turn(engine, 111)
        assert result == 2


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_returns_intent(self) -> None:
        engine = _make_engine()
        engine._classify_route = AsyncMock(return_value=SimpleNamespace(intent="greeting"))
        result = await classify_intent(engine, "hello")
        assert result == "greeting"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_route(self) -> None:
        engine = _make_engine()
        engine._classify_route = AsyncMock(return_value=None)
        result = await classify_intent(engine, "hello")
        assert result is None


class TestRouteRequest:
    @pytest.mark.asyncio
    async def test_returns_routing_dict(self) -> None:
        engine = _make_engine()
        result = await route_request(engine, "test")
        assert result["selected_model"] == "test-model"
        assert result["trigger"] == "single_model"


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_disabled_rag(self) -> None:
        engine = _make_engine()
        result = await retrieve(engine, "query")
        assert result["candidates"] == []
        assert result["rag_trace_partial"]["rag_used"] is False

    @pytest.mark.asyncio
    async def test_with_rag_pipeline(self) -> None:
        engine = _make_engine()
        chunk = Chunk(
            text="relevant text",
            metadata=ChunkMetadata(
                doc_id="doc1",
                source_path="test.md",
                chunk_id=0,
                section_title="Title",
                tokens_estimate=10,
            ),
        )
        candidate = RetrievedItem(chunk=chunk, retrieval_score=0.9, rerank_score=0.8)
        trace = RAGTrace(
            rag_used=True,
            retrieve_k0=1,
            rerank_k=1,
        )
        rag_result = RAGResult(
            candidates=[candidate],
            contexts=["context text"],
            trace=trace,
        )
        engine._rag_pipeline = AsyncMock()
        engine._rag_pipeline.execute = AsyncMock(return_value=rag_result)
        result = await retrieve(engine, "query")
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["chunk_text"] == "relevant text"


class TestGenerate:
    @pytest.mark.asyncio
    async def test_basic_generation(self) -> None:
        engine = _make_engine()
        engine.route_request = AsyncMock(return_value={
            "selected_model": "test-model",
            "trigger": "single_model",
        })
        result = await generate(engine, "hello")
        assert result["answer"] == "response"
        assert "routing_decision" in result

    @pytest.mark.asyncio
    async def test_image_only_uses_default_text(self) -> None:
        engine = _make_engine()
        engine.route_request = AsyncMock(return_value={"selected_model": "m"})
        result = await generate(engine, "", images=[b"img"])
        assert result["answer"] == "response"

    @pytest.mark.asyncio
    async def test_with_rag(self) -> None:
        engine = _make_engine()
        engine._rag_pipeline = MagicMock()
        engine._rag_pipeline.should_trigger_rag = MagicMock(return_value=True)
        trace = RAGTrace(rag_used=True, retrieve_k0=1, rerank_k=0)
        rag_result = RAGResult(candidates=[], contexts=["ctx"], trace=trace)
        engine._rag_pipeline.execute = AsyncMock(return_value=rag_result)
        engine._inject_rag_context = MagicMock(side_effect=lambda m, r: m)
        engine.route_request = AsyncMock(return_value={"selected_model": "m"})
        result = await generate(engine, "query")
        assert result["rag_trace"]["rag_used"] is True


class TestExecuteSkill:
    @pytest.mark.asyncio
    async def test_skill_not_found(self) -> None:
        engine = _make_engine()
        engine._skills.get_skill = MagicMock(return_value=None)
        result = await execute_skill(engine, "missing", {})
        assert "찾을 수 없습니다" in result

    @pytest.mark.asyncio
    async def test_skill_executed(self) -> None:
        engine = _make_engine()
        skill = SkillDefinition(
            name="test",
            description="test",
            triggers=["/test"],
            system_prompt="sys",
        )
        engine._skills.get_skill = MagicMock(return_value=skill)
        engine._run_skill_chat = AsyncMock(return_value=("skill result", None, "model"))
        result = await execute_skill(engine, "test", {"input_text": "hello"}, chat_id=111)
        assert result == "skill result"
        engine._memory.add_message.assert_called_once()


class TestProcessPrompt:
    @pytest.mark.asyncio
    async def test_basic(self) -> None:
        engine = _make_engine()
        result = await process_prompt(engine, "test prompt")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_with_overrides(self) -> None:
        engine = _make_engine()
        result = await process_prompt(
            engine, "test",
            model_override="custom",
            model_role="reasoning",
            max_tokens=500,
            temperature=0.5,
            timeout=120,
            system_prompt_override="custom sys",
        )
        assert result == "response"


class TestChangeModel:
    @pytest.mark.asyncio
    async def test_model_not_available(self) -> None:
        engine = _make_engine()
        result = await change_model(engine, "nonexistent")
        assert result["success"] is False
        assert "찾을 수 없습니다" in result["error"]

    @pytest.mark.asyncio
    async def test_model_changed(self) -> None:
        engine = _make_engine()
        result = await change_model(engine, "test-model")
        assert result["success"] is True
        assert result["new_model"] == "test-model"

    @pytest.mark.asyncio
    async def test_invalidates_cache(self) -> None:
        engine = _make_engine()
        engine._semantic_cache = AsyncMock()
        result = await change_model(engine, "test-model")
        assert result["success"] is True
        engine._semantic_cache.invalidate.assert_called_once()


class TestSimpleDelegates:
    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        engine = _make_engine()
        result = await list_models(engine)
        assert result == [{"name": "test-model", "size": 1024}]

    def test_get_current_model(self) -> None:
        engine = _make_engine()
        assert get_current_model(engine) == "test-model"

    @pytest.mark.asyncio
    async def test_reload_skills(self) -> None:
        engine = _make_engine()
        engine._skills.reload_skills = AsyncMock(return_value=5)
        result = await reload_skills(engine)
        assert result == 5

    def test_list_skills(self) -> None:
        engine = _make_engine()
        engine._skills.list_skills = MagicMock(return_value=[{"name": "a"}])
        assert list_skills(engine) == [{"name": "a"}]

    def test_get_last_skill_load_errors(self) -> None:
        engine = _make_engine()
        engine._skills.get_last_load_errors = MagicMock(return_value=["err"])
        assert get_last_skill_load_errors(engine) == ["err"]

    @pytest.mark.asyncio
    async def test_get_memory_stats(self) -> None:
        engine = _make_engine()
        engine._memory.get_memory_stats = AsyncMock(return_value={"count": 10})
        result = await get_memory_stats(engine, 111)
        assert result == {"count": 10}

    @pytest.mark.asyncio
    async def test_clear_conversation(self) -> None:
        engine = _make_engine()
        engine._memory.clear_conversation = AsyncMock(return_value=5)
        result = await clear_conversation(engine, 111)
        assert result == 5

    @pytest.mark.asyncio
    async def test_clear_conversation_with_cache(self) -> None:
        engine = _make_engine()
        engine._memory.clear_conversation = AsyncMock(return_value=3)
        engine._semantic_cache = AsyncMock()
        result = await clear_conversation(engine, 111)
        assert result == 3
        engine._semantic_cache.invalidate.assert_called_once_with(chat_id=111)

    @pytest.mark.asyncio
    async def test_export_conversation_markdown(self, tmp_path: Path) -> None:
        engine = _make_engine()
        expected_path = tmp_path / "export.md"
        engine._memory.export_conversation_markdown = AsyncMock(return_value=expected_path)
        result = await export_conversation_markdown(engine, 111, tmp_path)
        assert result == expected_path
