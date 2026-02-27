"""RAG 리랭커 — bge-reranker-v2-m3 기반 크로스인코더 리랭크."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import RAGConfig
from core.logging_setup import get_logger
from core.rag.types import RetrievedItem

if TYPE_CHECKING:
    from core.llm_protocol import RetrievalClientProtocol


class RAGReranker:
    """bge-reranker-v2-m3 기반 리랭크."""

    def __init__(
        self,
        client: RetrievalClientProtocol,
        reranker_model: str,
        config: RAGConfig,
    ) -> None:
        self._client = client
        self._model = reranker_model
        self._config = config
        self._logger = get_logger("rag_reranker")

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedItem],
        k: int = 8,
    ) -> list[RetrievedItem]:
        """후보를 리랭크하여 top-k를 반환한다."""
        if not candidates:
            return []

        documents = [c.chunk.text for c in candidates]

        timeout_ms = self._config.rerank_budget_ms
        timeout_s = max(5, timeout_ms // 1000)

        scored = await self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_n=k,
            timeout=timeout_s,
        )

        # 결과를 원래 candidates에 매핑
        reranked: list[RetrievedItem] = []
        for item in scored:
            idx = item["index"]
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                reranked.append(
                    RetrievedItem(
                        chunk=candidate.chunk,
                        retrieval_score=candidate.retrieval_score,
                        rerank_score=item["score"],
                    )
                )

        # rerank_score 내림차순 정렬 후 top-k
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        result = reranked[:k]

        self._logger.debug(
            "rerank_done",
            input_count=len(candidates),
            output_count=len(result),
            top_score=result[0].rerank_score if result else None,
        )
        return result
