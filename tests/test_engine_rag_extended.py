"""engine_rag 모듈 추가 커버리지 테스트."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine_rag import (
    build_full_scan_segments,
    emit_full_scan_progress,
    extract_json_payload,
    inject_extra_context,
    inject_rag_context,
    pack_blocks_for_reduction,
)


class TestEmitFullScanProgress:
    @pytest.mark.asyncio
    async def test_none_callback_noop(self) -> None:
        await emit_full_scan_progress(None, {"phase": "test"})

    @pytest.mark.asyncio
    async def test_sync_callback(self) -> None:
        results = []
        def cb(payload):
            results.append(payload)
        await emit_full_scan_progress(cb, {"phase": "map"})
        assert results == [{"phase": "map"}]

    @pytest.mark.asyncio
    async def test_async_callback(self) -> None:
        results = []
        async def cb(payload):
            results.append(payload)
        await emit_full_scan_progress(cb, {"phase": "reduce"})
        assert results == [{"phase": "reduce"}]

    @pytest.mark.asyncio
    async def test_callback_exception_suppressed(self) -> None:
        def cb(payload):
            raise ValueError("fail")
        await emit_full_scan_progress(cb, {"phase": "test"})  # No exception raised


class TestBuildFullScanSegments:
    def test_empty_chunks(self) -> None:
        assert build_full_scan_segments([], max_chars=1000, segment_factory=dict) == []

    def test_single_chunk(self) -> None:
        chunk = SimpleNamespace(
            text="hello world",
            metadata=SimpleNamespace(source_path="doc.md", chunk_id=0),
        )
        result = build_full_scan_segments([chunk], max_chars=10000, segment_factory=lambda **kw: kw)
        assert len(result) == 1
        assert result[0]["source_path"] == "doc.md"

    def test_multiple_chunks_same_source(self) -> None:
        chunks = [
            SimpleNamespace(
                text=f"content-{i}",
                metadata=SimpleNamespace(source_path="doc.md", chunk_id=i),
            )
            for i in range(3)
        ]
        result = build_full_scan_segments(chunks, max_chars=100000, segment_factory=lambda **kw: kw)
        assert len(result) == 1
        assert result[0]["start_chunk_id"] == 0
        assert result[0]["end_chunk_id"] == 2

    def test_different_sources_split(self) -> None:
        chunks = [
            SimpleNamespace(text="a", metadata=SimpleNamespace(source_path="a.md", chunk_id=0)),
            SimpleNamespace(text="b", metadata=SimpleNamespace(source_path="b.md", chunk_id=0)),
        ]
        result = build_full_scan_segments(chunks, max_chars=100000, segment_factory=lambda **kw: kw)
        assert len(result) == 2

    def test_max_chars_causes_split(self) -> None:
        chunks = [
            SimpleNamespace(
                text="x" * 100,
                metadata=SimpleNamespace(source_path="doc.md", chunk_id=i),
            )
            for i in range(5)
        ]
        result = build_full_scan_segments(chunks, max_chars=200, segment_factory=lambda **kw: kw)
        assert len(result) >= 2

    def test_empty_text_chunks_skipped(self) -> None:
        chunks = [
            SimpleNamespace(text="", metadata=SimpleNamespace(source_path="doc.md", chunk_id=0)),
            SimpleNamespace(text="real", metadata=SimpleNamespace(source_path="doc.md", chunk_id=1)),
        ]
        result = build_full_scan_segments(chunks, max_chars=100000, segment_factory=lambda **kw: kw)
        assert len(result) == 1

    def test_empty_source_path_skipped(self) -> None:
        chunks = [
            SimpleNamespace(text="text", metadata=SimpleNamespace(source_path="", chunk_id=0)),
        ]
        result = build_full_scan_segments(chunks, max_chars=100000, segment_factory=lambda **kw: kw)
        assert len(result) == 0


class TestPackBlocksForReduction:
    def test_empty_blocks(self) -> None:
        assert pack_blocks_for_reduction([], max_chars=1000) == []

    def test_single_block(self) -> None:
        result = pack_blocks_for_reduction(["- point 1\n- point 2"], max_chars=1000)
        assert len(result) == 1

    def test_blocks_split_at_max_chars(self) -> None:
        blocks = ["line " * 50 for _ in range(10)]
        result = pack_blocks_for_reduction(blocks, max_chars=200)
        assert len(result) >= 2

    def test_oversized_single_line(self) -> None:
        blocks = ["x" * 500]
        result = pack_blocks_for_reduction(blocks, max_chars=100)
        assert len(result) >= 2

    def test_empty_blocks_ignored(self) -> None:
        blocks = ["", "  ", "real content"]
        result = pack_blocks_for_reduction(blocks, max_chars=1000)
        assert len(result) == 1
        assert "real content" in result[0]


class TestInjectRagContext:
    def test_no_rag_result(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        assert inject_rag_context(messages, None) == messages

    def test_empty_contexts(self) -> None:
        rag_result = SimpleNamespace(contexts=[])
        messages = [{"role": "system", "content": "sys"}]
        assert inject_rag_context(messages, rag_result) == messages

    def test_injects_into_existing_system(self) -> None:
        rag_result = SimpleNamespace(contexts=["context text"])
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = inject_rag_context(messages, rag_result)
        assert "sys" in result[0]["content"]
        assert len(result) == 2

    def test_inserts_system_when_missing(self) -> None:
        rag_result = SimpleNamespace(contexts=["context text"])
        messages = [{"role": "user", "content": "hi"}]
        result = inject_rag_context(messages, rag_result)
        assert result[0]["role"] == "system"


class TestInjectExtraContext:
    def test_empty_context_noop(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        assert inject_extra_context(messages, "") == messages

    def test_appends_to_system(self) -> None:
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = inject_extra_context(messages, "extra info")
        assert "Additional Context" in result[0]["content"]
        assert "extra info" in result[0]["content"]

    def test_inserts_system_when_missing(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        result = inject_extra_context(messages, "extra info")
        assert result[0]["role"] == "system"
        assert "Additional Context" in result[0]["content"]


class TestExtractJsonPayload:
    def test_empty_string(self) -> None:
        assert extract_json_payload("") is None
        assert extract_json_payload("   ") is None

    def test_valid_json_dict(self) -> None:
        assert extract_json_payload('{"key": "value"}') == {"key": "value"}

    def test_json_array_returns_none(self) -> None:
        assert extract_json_payload('[1, 2, 3]') is None

    def test_json_embedded_in_text(self) -> None:
        text = 'Here is the result: {"relevant": true, "findings": ["a"]} done.'
        result = extract_json_payload(text)
        assert result is not None
        assert result["relevant"] is True

    def test_invalid_json(self) -> None:
        assert extract_json_payload("not json at all") is None

    def test_no_braces(self) -> None:
        assert extract_json_payload("just text") is None
