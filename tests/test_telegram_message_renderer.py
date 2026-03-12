"""텔레그램 메시지 렌더러 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.telegram_message_renderer import (
    StreamResult,
    split_message,
    stream_and_render,
)


class TestSplitMessage:
    def test_split_message_respects_max_length(self) -> None:
        text = "A" * 9000
        parts = split_message(text, max_length=4096)
        assert len(parts) >= 3
        assert all(len(part) <= 4096 for part in parts)


class TestStreamAndRender:
    @pytest.mark.asyncio
    async def test_skip_oversized_intermediate_edit(self) -> None:
        async def _stream():
            yield "A" * 5000

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        reply_msg = AsyncMock()
        reply_text = AsyncMock(return_value=reply_msg)

        result = await stream_and_render(
            stream=_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda _: ["part-1", "part-2"],
            edit_interval=0.0,
            edit_char_threshold=1,
            max_edit_length=4096,
        )

        assert isinstance(result, StreamResult)
        assert len(result.full_response) == 5000
        # 중간 편집은 길이 제한으로 건너뛰고 최종 편집/분할만 수행한다.
        sent_message.edit_text.assert_awaited_once_with("part-1")
        reply_text.assert_awaited_once_with("part-2")
        # 분할 메시지 시 마지막 메시지가 reply_text의 반환값이어야 한다
        assert result.last_message is reply_msg

    @pytest.mark.asyncio
    async def test_intermediate_cursor_edit_when_within_limit(self) -> None:
        async def _stream():
            yield "hello"
            yield " world"

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        result = await stream_and_render(
            stream=_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda text: [text],
            edit_interval=0.0,
            edit_char_threshold=1,
            max_edit_length=4096,
        )

        assert isinstance(result, StreamResult)
        assert result.full_response == "hello world"
        edit_text_calls = [call.args[0] for call in sent_message.edit_text.await_args_list]
        assert any(text.endswith(" ▌") for text in edit_text_calls)
        assert edit_text_calls[-1] == "hello world"
        reply_text.assert_not_awaited()
        # 단일 메시지 시 last_message가 sent_message와 동일
        assert result.last_message is sent_message

    @pytest.mark.asyncio
    async def test_sanitizes_internal_analysis_channel_format(self) -> None:
        async def _stream():
            yield (
                "Great.. ...\n\n"
                "<|start|>assistant<|channel|>analysis<|message|>"
                "Need to respond in Korean."
                "<|end|>"
                "<|start|>assistant<|channel|>final<|message|>"
                "한국어 최종 답변입니다."
                "<|end|>"
            )

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        result = await stream_and_render(
            stream=_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda text: [text],
            edit_interval=0.0,
            edit_char_threshold=10000,  # 중간 편집 비활성화
            max_edit_length=4096,
        )

        assert result.full_response == "한국어 최종 답변입니다."
        sent_message.edit_text.assert_awaited_once_with("한국어 최종 답변입니다.")
        reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_defers_intermediate_internal_reasoning_until_final_answer(self) -> None:
        async def _stream():
            yield "We need to respond in Korean. "
            yield "assistantanalysis to=final code최종 답변입니다."

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        result = await stream_and_render(
            stream=_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda text: [text],
            edit_interval=0.0,
            edit_char_threshold=1,
            max_edit_length=4096,
        )

        assert result.full_response == "최종 답변입니다."
        edit_text_calls = [call.args[0] for call in sent_message.edit_text.await_args_list]
        assert edit_text_calls == ["최종 답변입니다. ▌", "최종 답변입니다."]
        reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stops_when_chunk_timeout_exceeded(self) -> None:
        async def _slow_stream():
            await asyncio.sleep(0.05)
            yield "late"

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        result = await stream_and_render(
            stream=_slow_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda text: [text],
            chunk_timeout_seconds=0.01,
            max_stream_seconds=1.0,
        )

        assert result.stop_reason == "chunk_timeout"
        assert "중단" in result.full_response
        sent_message.edit_text.assert_awaited_once()
        assert "중단" in sent_message.edit_text.await_args.args[0]
        reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_longer_timeout_for_first_chunk_only(self) -> None:
        async def _slow_then_stall_stream():
            await asyncio.sleep(0.03)
            yield "first"
            await asyncio.sleep(0.03)
            yield "second"

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        result = await stream_and_render(
            stream=_slow_then_stall_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda text: [text],
            first_chunk_timeout_seconds=0.05,
            chunk_timeout_seconds=0.01,
            max_stream_seconds=1.0,
            edit_char_threshold=10_000,
        )

        assert "first" in result.full_response
        assert "중단" in result.full_response
        sent_message.edit_text.assert_awaited_once()
        assert "중단" in sent_message.edit_text.await_args.args[0]
        reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stops_when_repeated_chunks_detected(self) -> None:
        async def _stream():
            for _ in range(10):
                yield "가"

        sent_message = AsyncMock()
        sent_message.edit_text = AsyncMock()
        reply_text = AsyncMock()

        result = await stream_and_render(
            stream=_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda text: [text],
            max_repeated_chunks=3,
            edit_char_threshold=10_000,
        )

        assert result.stop_reason == "repeated_chunks"
        assert "반복 출력이 감지" in result.full_response
        assert result.full_response.split("\n\n")[0] == "가"
        sent_message.edit_text.assert_awaited_once()
        assert "반복 출력이 감지" in sent_message.edit_text.await_args.args[0]
