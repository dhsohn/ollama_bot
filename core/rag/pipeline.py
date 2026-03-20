"""Integrated RAG orchestration for retrieval, reranking, and context building."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from core.config import RAGConfig
from core.logging_setup import get_logger
from core.rag.context_builder import RAGContextBuilder
from core.rag.types import RAGResult, RAGTrace

if TYPE_CHECKING:
    from core.rag.reranker import RAGReranker
    from core.rag.retriever import RAGRetriever
    from core.rag.types import Chunk


class RAGPipeline:
    """Integrated pipeline for retrieval, reranking, and context building."""

    def __init__(
        self,
        retriever: RAGRetriever,
        reranker: RAGReranker | None,
        context_builder: RAGContextBuilder,
        config: RAGConfig,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._config = config
        self._logger = get_logger("rag_pipeline")

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None and self._config.rerank_enabled

    @property
    def chunk_count(self) -> int:
        return self._retriever.chunk_count

    async def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk in the index for full-scan workflows."""
        return await self._retriever.get_all_chunks()

    async def reindex_corpus(self, kb_paths: str | list[str]) -> dict[str, Any]:
        """Incrementally reindex the specified paths."""
        return await self._retriever.reindex(kb_paths)

    def should_trigger_rag(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> bool:
        """Check whether the request should trigger RAG."""
        if metadata and metadata.get("use_rag"):
            return True
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self._config.trigger_keywords)

    async def execute(
        self,
        query: str,
        metadata: dict | None = None,
    ) -> RAGResult:
        """Run the full RAG pipeline."""
        t0 = time.monotonic()
        trace = RAGTrace(rag_used=True, retrieve_k0=self._config.retrieve_k0)

        try:
            # 1) Retrieval
            t_ret = time.monotonic()
            candidates = await self._retriever.retrieve(
                query,
                k0=self._config.retrieve_k0,
                score_floor=self._config.retrieval_score_floor,
            )
            trace.retrieval_latency_ms = (time.monotonic() - t_ret) * 1000

            if not candidates:
                trace.rag_used = False
                trace.total_latency_ms = (time.monotonic() - t0) * 1000
                self._logger.debug("rag_no_candidates", query_len=len(query))
                return RAGResult(contexts=[], candidates=[], trace=trace)

            # 2) Rerank
            if self._reranker and self._config.rerank_enabled:
                t_rerank = time.monotonic()
                try:
                    candidates = await self._reranker.rerank(
                        query, candidates, k=self._config.rerank_topk,
                    )
                    trace.rerank_used = True
                    trace.rerank_k = self._config.rerank_topk
                except Exception as exc:
                    self._logger.warning("rerank_failed", error=str(exc))
                    candidates = candidates[: self._config.rerank_topk]
                    trace.error = f"rerank_failed: {exc}"
                trace.rerank_latency_ms = (time.monotonic() - t_rerank) * 1000
            else:
                candidates = candidates[: self._config.rerank_topk]

            # 3) Context packing
            context_text, citation_map = self._context_builder.build_context(
                candidates,
            )
            trace.context_tokens_estimate = max(1, len(context_text) // 3)
            trace.retrieved_items = [
                {
                    "doc_id": c.chunk.metadata.doc_id,
                    "source_path": c.chunk.metadata.source_path,
                    "chunk_id": c.chunk.metadata.chunk_id,
                    "retrieval_score": round(c.retrieval_score, 4),
                    "rerank_score": (
                        round(c.rerank_score, 4) if c.rerank_score is not None else None
                    ),
                }
                for c in candidates
            ]
            trace.citations_keys_used = list(citation_map.keys())
            trace.total_latency_ms = (time.monotonic() - t0) * 1000

            self._logger.info(
                "rag_executed",
                candidates=len(candidates),
                context_tokens=trace.context_tokens_estimate,
                rerank_used=trace.rerank_used,
                latency_ms=round(trace.total_latency_ms, 1),
            )

            return RAGResult(
                contexts=[context_text] if context_text else [],
                candidates=candidates,
                trace=trace,
                citation_map=citation_map,
            )
        except Exception as exc:
            trace.error = str(exc)
            trace.total_latency_ms = (time.monotonic() - t0) * 1000
            self._logger.error("rag_pipeline_failed", error=str(exc))
            return RAGResult(contexts=[], candidates=[], trace=trace)
