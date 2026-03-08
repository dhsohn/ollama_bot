"""ContextCompressor 추가 커버리지 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.context_compressor import ContextCompressor
from core.llm_types import ChatResponse


def _make_compressor(
    *,
    llm_client=None,
    memory=None,
    recent_keep: int = 5,
    refresh_interval: int = 5,
) -> ContextCompressor:
    return ContextCompressor(
        llm_client=llm_client or AsyncMock(),
        memory=memory or AsyncMock(),
        recent_keep=recent_keep,
        summary_refresh_interval=refresh_interval,
    )


class TestBuildCompressedHistory:
    @pytest.mark.asyncio
    async def test_short_history_returned_as_is(self) -> None:
        memory = AsyncMock()
        memory.get_conversation = AsyncMock(return_value=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        compressor = _make_compressor(memory=memory, recent_keep=5)
        result = await compressor.build_compressed_history(111)
        assert len(result) == 2
        memory.get_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_with_cached_summary(self) -> None:
        recent = [{"role": "user", "content": f"msg-{i}"} for i in range(5)]
        memory = AsyncMock()
        memory.get_conversation = AsyncMock(return_value=recent)
        memory.get_summary = AsyncMock(return_value={
            "summary": "Previous topics discussed.",
            "last_archive_id": 100,
            "message_count": 100,
        })
        compressor = _make_compressor(memory=memory, recent_keep=5)
        result = await compressor.build_compressed_history(111)
        assert len(result) == 6  # 1 summary + 5 recent
        assert result[0]["role"] == "system"
        assert "이전 대화 요약" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_no_summary_returns_recent_only(self) -> None:
        recent = [{"role": "user", "content": f"msg-{i}"} for i in range(5)]
        memory = AsyncMock()
        memory.get_conversation = AsyncMock(return_value=recent)
        memory.get_summary = AsyncMock(return_value=None)
        compressor = _make_compressor(memory=memory, recent_keep=5)
        result = await compressor.build_compressed_history(111)
        assert len(result) == 5


class TestMaybeRefreshSummary:
    @pytest.mark.asyncio
    async def test_skips_when_semaphore_locked(self) -> None:
        memory = AsyncMock()
        memory.get_summary = AsyncMock(return_value=None)
        memory.get_archived_messages = AsyncMock(return_value=[
            {"id": i, "role": "user", "content": f"m-{i}", "timestamp": "t"}
            for i in range(10)
        ])

        compressor = _make_compressor(memory=memory, refresh_interval=5)
        # Lock the semaphore
        await compressor._summarize_sem.acquire()
        try:
            result = await compressor.maybe_refresh_summary(111)
            assert result is False
        finally:
            compressor._summarize_sem.release()

    @pytest.mark.asyncio
    async def test_no_new_archived_after_recheck(self) -> None:
        memory = AsyncMock()
        memory.get_summary = AsyncMock(side_effect=[
            {"summary": "old", "last_archive_id": 100, "message_count": 100},
            {"summary": "old", "last_archive_id": 100, "message_count": 100},
        ])
        archived = [{"id": i, "role": "user", "content": "m", "timestamp": "t"} for i in range(101, 111)]
        memory.get_archived_messages = AsyncMock(side_effect=[archived, []])

        compressor = _make_compressor(memory=memory, refresh_interval=5)
        result = await compressor.maybe_refresh_summary(111)
        assert result is False

    @pytest.mark.asyncio
    async def test_generation_failure_returns_false(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM error"))

        memory = AsyncMock()
        memory.get_summary = AsyncMock(side_effect=[None, None])
        archived = [{"id": i, "role": "user", "content": "m", "timestamp": "t"} for i in range(10)]
        memory.get_archived_messages = AsyncMock(side_effect=[archived, archived])

        compressor = _make_compressor(llm_client=llm, memory=memory, refresh_interval=5)
        result = await compressor.maybe_refresh_summary(111)
        assert result is False


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_with_previous_summary(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=ChatResponse(content="updated summary"))
        compressor = _make_compressor(llm_client=llm)

        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(5)]
        result = await compressor._generate_summary(messages, previous_summary="old summary")

        assert result == "updated summary"
        call_args = llm.chat.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "기존 요약" in user_content

    @pytest.mark.asyncio
    async def test_without_previous_summary(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=ChatResponse(content="new summary"))
        compressor = _make_compressor(llm_client=llm)

        messages = [{"role": "user", "content": "hello"}]
        result = await compressor._generate_summary(messages)

        assert result == "new summary"

    @pytest.mark.asyncio
    async def test_truncates_to_50_messages(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=ChatResponse(content="summary"))
        compressor = _make_compressor(llm_client=llm)

        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(100)]
        await compressor._generate_summary(messages)

        call_args = llm.chat.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        # Should have at most 50 lines
        lines = [line for line in user_content.split("\n") if line.strip()]
        assert len(lines) <= 50
