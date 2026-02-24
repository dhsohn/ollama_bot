"""텔레그램 메시지 렌더러 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.telegram_message_renderer import split_message, stream_and_render


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

        result = await stream_and_render(
            stream=_stream(),
            sent_message=sent_message,
            reply_text=reply_text,
            split_message_fn=lambda _: ["part-1", "part-2"],
            edit_interval=0.0,
            edit_char_threshold=1,
            max_edit_length=4096,
        )

        assert len(result) == 5000
        # 중간 편집은 길이 제한으로 건너뛰고 최종 편집/분할만 수행한다.
        sent_message.edit_text.assert_awaited_once_with("part-1")
        reply_text.assert_awaited_once_with("part-2")

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

        assert result == "hello world"
        edit_text_calls = [call.args[0] for call in sent_message.edit_text.await_args_list]
        assert any(text.endswith(" ▌") for text in edit_text_calls)
        assert edit_text_calls[-1] == "hello world"
        reply_text.assert_not_awaited()
