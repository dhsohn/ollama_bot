"""RAG 인덱서 — SQLite + numpy 기반 벡터 인덱스.

코퍼스 파일을 청킹/임베딩하여 SQLite에 저장하고,
인메모리 numpy 행렬로 빠른 코사인 유사도 검색을 제공한다.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import numpy as np

from core.config import RAGConfig
from core.logging_setup import get_logger
from core.rag.chunker import DocumentChunker
from core.rag.types import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from core.llm_protocol import RetrievalClientProtocol

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    chunk_id        INTEGER NOT NULL,
    section_title   TEXT,
    text            TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    content_hash    TEXT NOT NULL,
    mtime           REAL NOT NULL,
    tokens_estimate INTEGER NOT NULL,
    file_type       TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source_path, content_hash);
CREATE INDEX IF NOT EXISTS idx_rag_doc ON rag_chunks(doc_id, chunk_id);
"""

_EMBED_BATCH_SIZE = 32


class RAGIndexer:
    """코퍼스 인덱싱 및 벡터 검색."""

    def __init__(
        self,
        config: RAGConfig,
        client: RetrievalClientProtocol,
        embedding_model: str,
    ) -> None:
        self._config = config
        self._client = client
        self._embedding_model = embedding_model
        self._chunker = DocumentChunker(config)
        self._db: aiosqlite.Connection | None = None
        self._logger = get_logger("rag_indexer")

        # 인메모리 벡터 인덱스
        self._row_ids: list[int] = []
        self._embeddings: np.ndarray | None = None  # shape: (n, dim)
        self._chunk_meta: list[dict[str, Any]] = []

    async def initialize(self, db_path: str) -> None:
        """DB 스키마 생성 + 인메모리 인덱스 로드."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._load_index_to_memory()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def chunk_count(self) -> int:
        return len(self._row_ids)

    async def index_corpus(self, kb_dir: str) -> dict[str, Any]:
        """코퍼스를 인덱싱한다. 증분 지원."""
        assert self._db is not None
        t0 = time.monotonic()
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            self._logger.warning("kb_dir_not_found", path=kb_dir)
            return {"indexed": 0, "skipped": 0, "removed": 0}

        # 1) 지원 확장자 파일 목록
        files: list[str] = []
        for ext in self._config.supported_extensions:
            files.extend(str(p) for p in kb_path.rglob(f"*{ext}"))

        # 2) 기존 인덱스의 source_path → content_hash 매핑
        existing: dict[str, str] = {}
        async with self._db.execute(
            "SELECT DISTINCT source_path, content_hash FROM rag_chunks"
        ) as cursor:
            async for row in cursor:
                existing[row["source_path"]] = row["content_hash"]

        # 3) 변경분 검출
        to_index: list[str] = []
        to_remove: set[str] = set(existing.keys())

        for fpath in files:
            to_remove.discard(fpath)
            current_hash = DocumentChunker.content_hash(fpath)
            if existing.get(fpath) == current_hash:
                continue
            to_index.append(fpath)

        # 4) 삭제된 파일의 청크 제거
        removed = 0
        for path in to_remove:
            await self._db.execute(
                "DELETE FROM rag_chunks WHERE source_path = ?", (path,),
            )
            removed += 1

        # 5) 변경/신규 파일 재인덱싱
        indexed = 0
        for fpath in to_index:
            # 기존 청크 삭제
            await self._db.execute(
                "DELETE FROM rag_chunks WHERE source_path = ?", (fpath,),
            )
            chunks = self._chunker.chunk_file(fpath)
            if not chunks:
                continue

            # 배치 임베딩
            texts = [c.text for c in chunks]
            embeddings = await self._batch_embed(texts)

            for chunk, emb in zip(chunks, embeddings):
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
                await self._db.execute(
                    """INSERT INTO rag_chunks
                    (doc_id, source_path, chunk_id, section_title, text,
                     embedding, content_hash, mtime, tokens_estimate, file_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.metadata.doc_id,
                        chunk.metadata.source_path,
                        chunk.metadata.chunk_id,
                        chunk.metadata.section_title,
                        chunk.text,
                        emb_blob,
                        chunk.metadata.content_hash,
                        chunk.metadata.mtime,
                        chunk.metadata.tokens_estimate,
                        chunk.metadata.file_type,
                    ),
                )
            indexed += 1

        await self._db.commit()
        await self._load_index_to_memory()

        elapsed = (time.monotonic() - t0) * 1000
        self._logger.info(
            "corpus_indexed",
            indexed=indexed,
            skipped=len(files) - len(to_index),
            removed=removed,
            total_chunks=self.chunk_count,
            latency_ms=round(elapsed, 1),
        )
        return {
            "indexed": indexed,
            "skipped": len(files) - len(to_index),
            "removed": removed,
            "total_chunks": self.chunk_count,
        }

    async def search(
        self, query_embedding: np.ndarray, k: int = 40,
    ) -> list[tuple[int, float]]:
        """코사인 유사도 기반 top-k 검색.

        Returns:
            [(row_id, score), ...] 내림차순.
        """
        if self._embeddings is None or len(self._row_ids) == 0:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normalized = self._embeddings / norms
        scores = normalized @ query_norm

        actual_k = min(k, len(scores))
        if actual_k >= len(scores):
            top_indices = np.argsort(-scores)[:actual_k]
        else:
            top_indices = np.argpartition(-scores, actual_k)[:actual_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            (self._row_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]

    async def get_chunk_by_id(self, row_id: int) -> Chunk | None:
        """row_id로 청크를 조회한다."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM rag_chunks WHERE id = ?", (row_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    async def get_chunks_by_ids(self, row_ids: list[int]) -> list[Chunk]:
        """여러 row_id로 청크를 조회한다."""
        if not row_ids:
            return []
        assert self._db is not None
        placeholders = ",".join("?" for _ in row_ids)
        async with self._db.execute(
            f"SELECT * FROM rag_chunks WHERE id IN ({placeholders})",
            row_ids,
        ) as cursor:
            rows = await cursor.fetchall()

        chunk_map = {row["id"]: self._row_to_chunk(row) for row in rows}
        return [chunk_map[rid] for rid in row_ids if rid in chunk_map]

    async def _load_index_to_memory(self) -> None:
        """DB에서 인메모리 벡터 인덱스를 로드한다."""
        assert self._db is not None
        self._row_ids = []
        embeddings_list: list[np.ndarray] = []
        self._chunk_meta = []

        async with self._db.execute(
            "SELECT id, doc_id, source_path, chunk_id, section_title, "
            "tokens_estimate, file_type, embedding FROM rag_chunks ORDER BY id"
        ) as cursor:
            async for row in cursor:
                self._row_ids.append(row["id"])
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                embeddings_list.append(emb)
                self._chunk_meta.append({
                    "doc_id": row["doc_id"],
                    "source_path": row["source_path"],
                    "chunk_id": row["chunk_id"],
                    "section_title": row["section_title"],
                })

        if embeddings_list:
            self._embeddings = np.stack(embeddings_list)
        else:
            self._embeddings = None

        self._logger.debug(
            "index_loaded_to_memory", chunks=len(self._row_ids),
        )

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """배치 단위로 임베딩을 생성한다."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            embeddings = await self._client.embed(
                batch, model=self._embedding_model,
            )
            all_embeddings.extend(embeddings)
        return all_embeddings

    @staticmethod
    def _row_to_chunk(row: Any) -> Chunk:
        meta = ChunkMetadata(
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            chunk_id=row["chunk_id"],
            section_title=row["section_title"],
            content_hash=row["content_hash"],
            mtime=row["mtime"],
            tokens_estimate=row["tokens_estimate"],
            file_type=row["file_type"],
        )
        return Chunk(text=row["text"], metadata=meta)
