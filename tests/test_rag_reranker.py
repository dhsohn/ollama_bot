"""RAG reranker tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import RAGConfig
from core.rag.reranker import RAGReranker
from core.rag.types import Chunk, ChunkMetadata, RetrievedItem


def _candidate(index: int, *, text: str = "doc", retrieval_score: float = 0.5) -> RetrievedItem:
    return RetrievedItem(
        chunk=Chunk(
            text=f"{text}-{index}",
            metadata=ChunkMetadata(
                doc_id=f"doc-{index}",
                source_path=f"kb/doc-{index}.md",
                chunk_id=index,
            ),
        ),
        retrieval_score=retrieval_score,
    )


@pytest.mark.asyncio
async def test_rerank_returns_empty_when_no_candidates() -> None:
    client = SimpleNamespace(rerank=AsyncMock())
    reranker = RAGReranker(client, "reranker-model", RAGConfig())

    result = await reranker.rerank("query", [], k=3)

    assert result == []
    client.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_maps_scores_sorts_and_limits_top_k() -> None:
    client = SimpleNamespace(
        rerank=AsyncMock(
            return_value=[
                {"index": 1, "score": 0.4},
                {"index": 0, "score": 0.9},
                {"index": 2, "score": 0.7},
            ],
        ),
    )
    reranker = RAGReranker(client, "reranker-model", RAGConfig(rerank_budget_ms=1200))
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    result = await reranker.rerank("query", candidates, k=2)

    assert [item.chunk.metadata.chunk_id for item in result] == [0, 2]
    assert [item.rerank_score for item in result] == [0.9, 0.7]
    client.rerank.assert_awaited_once_with(
        query="query",
        documents=["doc-0", "doc-1", "doc-2"],
        model="reranker-model",
        top_n=2,
        timeout=5,
    )


@pytest.mark.asyncio
async def test_rerank_skips_malformed_and_out_of_range_items() -> None:
    client = SimpleNamespace(
        rerank=AsyncMock(
            return_value=[
                {"index": None, "score": 0.9},
                {"index": 99, "score": 0.8},
                {"index": 0, "score": None},
                {"index": 1, "score": 0.6},
            ],
        ),
    )
    reranker = RAGReranker(client, "reranker-model", RAGConfig())
    reranker._logger = MagicMock()

    result = await reranker.rerank("query", [_candidate(0), _candidate(1)], k=5)

    assert len(result) == 1
    assert result[0].chunk.metadata.chunk_id == 1
    assert result[0].rerank_score == 0.6
    assert reranker._logger.warning.call_count == 2
