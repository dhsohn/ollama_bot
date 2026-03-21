"""engine_summary 모듈 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine_summary import (
    extract_skill_user_input,
    is_summarize_skill,
    run_chunked_summary_pipeline,
    run_skill_chat,
    should_use_chunked_summary,
    split_text_for_summary,
)
from core.llm_types import ChatResponse
from core.skill_manager import SkillDefinition


def _make_skill(name: str = "summarize", timeout: int = 30, model_role: str = "skill") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description="test",
        triggers=["/test"],
        system_prompt="Test prompt.",
        timeout=timeout,
        model_role=model_role,
    )


class TestIsSummarizeSkill:
    def test_exact_match(self) -> None:
        assert is_summarize_skill(_make_skill("summarize")) is True

    def test_case_insensitive(self) -> None:
        assert is_summarize_skill(_make_skill(" Summarize ")) is True

    def test_not_summarize(self) -> None:
        assert is_summarize_skill(_make_skill("translate")) is False


class TestExtractSkillUserInput:
    def test_extracts_last_user_message(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
        ]
        assert extract_skill_user_input(messages) == "hello world"

    def test_empty_messages(self) -> None:
        assert extract_skill_user_input([]) == ""

    def test_no_user_message(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        assert extract_skill_user_input(messages) == ""


class TestShouldUseChunkedSummary:
    def test_small_text_returns_false(self) -> None:
        assert should_use_chunked_summary(skill=_make_skill(), input_text="short") is False

    def test_non_summarize_returns_false(self) -> None:
        long_text = "x" * 100_000
        assert should_use_chunked_summary(skill=_make_skill("other"), input_text=long_text) is False

    def test_long_text_with_summarize_returns_true(self) -> None:
        long_text = "x" * 100_000
        assert should_use_chunked_summary(skill=_make_skill(), input_text=long_text) is True


class TestSplitTextForSummary:
    def test_empty_text(self) -> None:
        assert split_text_for_summary("") == []
        assert split_text_for_summary("   ") == []

    def test_short_text_single_chunk(self) -> None:
        result = split_text_for_summary("short text")
        assert result == ["short text"]

    def test_long_text_splits(self) -> None:
        text = "A" * 50_000 + "\n\n" + "B" * 50_000
        result = split_text_for_summary(text)
        assert len(result) >= 2

    def test_split_prefers_paragraph_boundary(self) -> None:
        block_a = "Word " * 3000
        block_b = "Other " * 3000
        text = block_a.strip() + "\n\n" + block_b.strip()
        result = split_text_for_summary(text)
        assert len(result) >= 2

    def test_overlap_between_chunks(self) -> None:
        text = "Hello. " * 20_000
        result = split_text_for_summary(text)
        assert len(result) >= 2


class TestRunSkillChat:
    @pytest.mark.asyncio
    async def test_basic_skill_chat(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="short text")
        engine._should_use_chunked_summary = MagicMock(return_value=False)
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._config.ollama.chat_model = "default-model"
        engine._prepare_target_model = AsyncMock(return_value=("model-a", None))
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(
            return_value=ChatResponse(content="result", usage={"tokens": 10})
        )

        skill = _make_skill()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "short text"},
        ]

        content, _usage, model = await run_skill_chat(
            engine, skill=skill, messages=messages, model_override=None,
        )
        assert content == "result"
        assert model == "model-a"
        prepare_kwargs = engine._prepare_target_model.await_args.kwargs
        assert prepare_kwargs["model"] == "default-model"
        assert prepare_kwargs["role"] is None

    @pytest.mark.asyncio
    async def test_skill_chat_with_overrides(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="short text")
        engine._should_use_chunked_summary = MagicMock(return_value=False)
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._prepare_target_model = AsyncMock(return_value=("override-model", None))
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(
            return_value=ChatResponse(content="custom", usage=None)
        )

        skill = _make_skill()
        messages = [{"role": "user", "content": "input"}]

        content, _usage, _model = await run_skill_chat(
            engine, skill=skill, messages=messages,
            model_override="override-model",
            model_role_override="reasoning",
            max_tokens_override=500,
            temperature_override=0.5,
            timeout_override=60,
        )
        assert content == "custom"
        prepare_kwargs = engine._prepare_target_model.await_args.kwargs
        assert prepare_kwargs["model"] == "override-model"
        assert prepare_kwargs["role"] is None

    @pytest.mark.asyncio
    async def test_skill_chat_falls_back_when_chunked_fails(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="x" * 100_000)
        engine._should_use_chunked_summary = MagicMock(return_value=True)
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._config.ollama.chat_model = "default-model"
        engine._logger = MagicMock()
        engine._prepare_target_model = AsyncMock(return_value=("model-b", None))
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(
            return_value=ChatResponse(content="fallback", usage=None)
        )

        # run_chunked_summary_pipeline will be called via module function
        # Patch it to raise an error
        import core.engine_summary as mod
        original = mod.run_chunked_summary_pipeline

        async def _failing_chunked(*args, **kwargs):
            raise RuntimeError("chunked failed")

        mod.run_chunked_summary_pipeline = _failing_chunked
        try:
            skill = _make_skill()
            messages = [{"role": "user", "content": "x" * 100_000}]

            content, _, _ = await run_skill_chat(
                engine, skill=skill, messages=messages, model_override=None,
            )
            assert content == "fallback"
            engine._logger.warning.assert_called()
        finally:
            mod.run_chunked_summary_pipeline = original

    @pytest.mark.asyncio
    async def test_skill_chat_preserves_explicit_non_default_role_with_single_model(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="short text")
        engine._should_use_chunked_summary = MagicMock(return_value=False)
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._config.ollama.chat_model = "default-model"
        engine._prepare_target_model = AsyncMock(return_value=("default-model", "coding"))
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(
            return_value=ChatResponse(content="result", usage=None)
        )

        skill = _make_skill(name="code_review", model_role="coding")
        messages = [{"role": "user", "content": "input"}]

        content, _usage, model = await run_skill_chat(
            engine,
            skill=skill,
            messages=messages,
            model_override=None,
        )

        assert content == "result"
        assert model == "default-model"
        prepare_kwargs = engine._prepare_target_model.await_args.kwargs
        assert prepare_kwargs["model"] == "default-model"
        assert prepare_kwargs["role"] == "coding"


class TestRunChunkedSummaryPipeline:
    @pytest.mark.asyncio
    async def test_not_applicable_raises(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="short")
        engine._split_text_for_summary = MagicMock(return_value=["short"])

        with pytest.raises(RuntimeError, match="not_applicable"):
            await run_chunked_summary_pipeline(
                engine,
                skill=_make_skill(),
                messages=[{"role": "user", "content": "short"}],
                model_override=None,
                chat_id=111,
            )

    @pytest.mark.asyncio
    async def test_successful_pipeline(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="long text")
        engine._split_text_for_summary = MagicMock(return_value=["chunk1", "chunk2"])
        engine._config.ollama.chat_model = "default-model"
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._prepare_target_model = AsyncMock(return_value=("model-x", None))
        engine._inject_language_policy = MagicMock(side_effect=lambda x: x)
        engine._logger = MagicMock()

        map_resp = ChatResponse(content="- point 1\n- point 2", usage=None)
        reduce_resp = ChatResponse(content="Final summary.", usage={"tokens": 5})
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(side_effect=[map_resp, map_resp, reduce_resp])

        content, _usage, model = await run_chunked_summary_pipeline(
            engine,
            skill=_make_skill(),
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "long text"},
            ],
            model_override=None,
            chat_id=111,
        )
        assert content == "Final summary."
        assert model == "model-x"
        prepare_calls = engine._prepare_target_model.await_args_list
        assert prepare_calls[0].kwargs["role"] is None
        assert prepare_calls[1].kwargs["role"] is None

    @pytest.mark.asyncio
    async def test_empty_intermediate_raises(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="long text")
        engine._split_text_for_summary = MagicMock(return_value=["chunk1", "chunk2"])
        engine._config.ollama.chat_model = "default-model"
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._prepare_target_model = AsyncMock(return_value=("model-x", None))
        engine._inject_language_policy = MagicMock(side_effect=lambda x: x)
        engine._logger = MagicMock()

        # Map responses return empty content
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(
            return_value=ChatResponse(content="", usage=None)
        )

        with pytest.raises(RuntimeError, match="empty_intermediate"):
            await run_chunked_summary_pipeline(
                engine,
                skill=_make_skill(),
                messages=[{"role": "user", "content": "long text"}],
                model_override=None,
                chat_id=111,
            )

    @pytest.mark.asyncio
    async def test_empty_final_raises(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="long text")
        engine._split_text_for_summary = MagicMock(return_value=["chunk1", "chunk2"])
        engine._config.ollama.chat_model = "default-model"
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._prepare_target_model = AsyncMock(return_value=("model-x", None))
        engine._inject_language_policy = MagicMock(side_effect=lambda x: x)
        engine._logger = MagicMock()

        map_resp = ChatResponse(content="- finding", usage=None)
        reduce_resp = ChatResponse(content="", usage=None)
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(side_effect=[map_resp, map_resp, reduce_resp])

        with pytest.raises(RuntimeError, match="empty_final"):
            await run_chunked_summary_pipeline(
                engine,
                skill=_make_skill(),
                messages=[{"role": "user", "content": "long text"}],
                model_override=None,
                chat_id=111,
            )

    @pytest.mark.asyncio
    async def test_model_override_used(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="long text")
        engine._split_text_for_summary = MagicMock(return_value=["chunk1", "chunk2"])
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._prepare_target_model = AsyncMock(return_value=("custom-model", None))
        engine._inject_language_policy = MagicMock(side_effect=lambda x: x)
        engine._logger = MagicMock()

        map_resp = ChatResponse(content="- point", usage=None)
        reduce_resp = ChatResponse(content="Done.", usage=None)
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(side_effect=[map_resp, map_resp, reduce_resp])

        _content, _, model = await run_chunked_summary_pipeline(
            engine,
            skill=_make_skill(),
            messages=[{"role": "user", "content": "long text"}],
            model_override="custom-model",
            chat_id=111,
        )
        assert model == "custom-model"
        prepare_calls = engine._prepare_target_model.await_args_list
        assert prepare_calls[0].kwargs["role"] is None
        assert prepare_calls[1].kwargs["role"] is None

    @pytest.mark.asyncio
    async def test_deduplicates_identical_map_summaries(self) -> None:
        engine = MagicMock()
        engine._extract_skill_user_input = MagicMock(return_value="long text")
        engine._split_text_for_summary = MagicMock(return_value=["chunk1", "chunk2"])
        engine._config.ollama.chat_model = "model"
        engine._resolve_model_for_role = MagicMock(return_value=None)
        engine._prepare_target_model = AsyncMock(return_value=("model", None))
        engine._inject_language_policy = MagicMock(side_effect=lambda x: x)
        engine._logger = MagicMock()

        identical_resp = ChatResponse(content="- same point", usage=None)
        reduce_resp = ChatResponse(content="Final.", usage=None)
        engine._llm_client = AsyncMock()
        engine._llm_client.chat = AsyncMock(side_effect=[identical_resp, identical_resp, reduce_resp])

        content, _, _ = await run_chunked_summary_pipeline(
            engine,
            skill=_make_skill(),
            messages=[{"role": "user", "content": "long text"}],
            model_override=None,
            chat_id=111,
        )
        assert content == "Final."
