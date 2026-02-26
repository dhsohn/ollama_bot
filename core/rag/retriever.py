"""RAG 검색기 — 벡터 검색 + 중복 제거."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from core.logging_setup import get_logger
from core.rag.types import RetrievedItem

if TYPE_CHECKING:
    from core.lemonade_client import LemonadeClient
    from core.rag.indexer import RAGIndexer
    from core.rag.types import Chunk

_MAX_CHUNKS_PER_DOC = 2


class RAGRetriever:
    """벡터 검색 + 인접 chunk 중복 제거."""

    def __init__(
        self,
        indexer: RAGIndexer,
        client: LemonadeClient,
        embedding_model: str,
    ) -> None:
        self._indexer = indexer
        self._client = client
        self._embedding_model = embedding_model
        self._logger = get_logger("rag_retriever")

    async def retrieve(
        self,
        query: str,
        k0: int = 40,
        score_floor: float = 0.0,
    ) -> list[RetrievedItem]:
        """쿼리에 대한 top-k0 후보를 반환한다.

        동일 문서 내 인접 chunk 중복 제거를 적용한다.
        """
        if self._indexer.chunk_count == 0:
            return []

        # 1) 쿼리 임베딩
        embeddings = await self._client.embed(
            [query], model=self._embedding_model,
        )
        query_emb = np.array(embeddings[0], dtype=np.float32)

        # 2) 벡터 검색
        search_results = await self._indexer.search(query_emb, k=k0 * 2)
        if score_floor > 0:
            search_results = [
                (rid, score) for rid, score in search_results
                if score >= score_floor
            ]

        if not search_results:
            return []

        # 3) 청크 일괄 조회
        row_ids = [rid for rid, _ in search_results]
        score_map = dict(search_results)
        chunks = await self._indexer.get_chunks_by_ids(row_ids)

        # row_id → chunk 매핑 (순서 보존)
        chunk_by_rid: dict[int, Chunk] = {}
        for rid, chunk in zip(row_ids, chunks):
            chunk_by_rid[rid] = chunk

        # 4) 동일 문서 인접 chunk 중복 제거 (score 순으로 처리)
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

            # 동일 문서 내 인접 chunk(±1)는 중복으로 간주해 스킵
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
