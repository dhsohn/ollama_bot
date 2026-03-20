"""RAG reranker built on the bge-reranker-v2-m3 cross-encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import RAGConfig
from core.logging_setup import get_logger
from core.rag.types import RetrievedItem

if TYPE_CHECKING:
    from core.llm_protocol import RetrievalClientProtocol


class RAGReranker:
    """Rerank candidates with bge-reranker-v2-m3."""

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
        """Rerank candidates and return the top-k results."""
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

        # Map reranker output back to the original candidates.
        reranked: list[RetrievedItem] = []
        for item in scored:
            idx = item.get("index")
            score = item.get("score")
            if idx is None or score is None:
                self._logger.warning("reranker_malformed_result", item=item)
                continue
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                reranked.append(
                    RetrievedItem(
                        chunk=candidate.chunk,
                        retrieval_score=candidate.retrieval_score,
                        rerank_score=score,
                    )
                )

        # Sort by descending rerank score, then keep top-k.
        reranked.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        result = reranked[:k]

        self._logger.debug(
            "rerank_done",
            input_count=len(candidates),
            output_count=len(result),
            top_score=result[0].rerank_score if result else None,
        )
        return result
