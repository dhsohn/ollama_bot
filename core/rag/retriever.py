"""RAG retriever with vector search and duplicate suppression."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from core.logging_setup import get_logger
from core.rag.types import RetrievedItem

if TYPE_CHECKING:
    from core.llm_protocol import RetrievalClientProtocol
    from core.rag.indexer import RAGIndexer
    from core.rag.types import Chunk

_MAX_CHUNKS_PER_DOC = 2


class RAGRetriever:
    """Vector retrieval with adjacent-chunk deduplication."""

    def __init__(
        self,
        indexer: RAGIndexer,
        client: RetrievalClientProtocol,
        embedding_model: str,
    ) -> None:
        self._indexer = indexer
        self._client = client
        self._embedding_model = embedding_model
        self._logger = get_logger("rag_retriever")

    @property
    def chunk_count(self) -> int:
        return self._indexer.chunk_count

    async def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk in the index for full-scan workflows."""
        return await self._indexer.get_all_chunks()

    async def reindex(self, kb_paths: str | list[str]) -> dict[str, Any]:
        """Incrementally reindex the specified paths."""
        return await self._indexer.index_corpus(kb_paths)

    async def retrieve(
        self,
        query: str,
        k0: int = 40,
        score_floor: float = 0.0,
    ) -> list[RetrievedItem]:
        """Return top-k0 candidates for the query.

        Adjacent chunks from the same document are treated as duplicates.
        """
        if self._indexer.chunk_count == 0:
            return []

        # 1) Embed the query.
        embeddings = await self._client.embed(
            [query], model=self._embedding_model,
        )
        query_emb = np.array(embeddings[0], dtype=np.float32)

        # 2) Run vector search.
        search_results = await self._indexer.search(query_emb, k=k0 * 2)
        if score_floor > 0:
            search_results = [
                (rid, score) for rid, score in search_results
                if score >= score_floor
            ]

        if not search_results:
            return []

        # 3) Fetch chunks in bulk for an exact rid -> chunk mapping.
        row_ids = [rid for rid, _ in search_results]
        chunk_by_rid: dict[int, Chunk] = await self._indexer.get_chunks_map_by_ids(row_ids)

        # 4) Remove adjacent same-document chunks in score order.
        doc_selected: defaultdict[str, list[int]] = defaultdict(list)
        items: list[RetrievedItem] = []

        for rid, score in search_results:
            if rid not in chunk_by_rid:
                continue
            chunk = chunk_by_rid[rid]

            doc_id = chunk.metadata.doc_id
            chunk_id = chunk.metadata.chunk_id
            selected = doc_selected[doc_id]

            if len(selected) >= _MAX_CHUNKS_PER_DOC:
                continue

            # Treat adjacent chunks from the same document (±1) as duplicates.
            if selected and any(abs(chunk_id - sid) <= 1 for sid in selected):
                continue

            selected.append(chunk_id)
            items.append(RetrievedItem(chunk=chunk, retrieval_score=score))

            if len(items) >= k0:
                break

        self._logger.debug(
            "retrieval_done",
            query_len=len(query),
            candidates=len(items),
            total_searched=len(search_results),
        )
        return items
