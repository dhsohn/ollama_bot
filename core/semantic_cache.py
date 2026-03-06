"""시맨틱 캐싱 엔진.

사용자 질문의 임베딩 유사도를 기반으로 캐시 히트를 판정한다.
캐시는 SQLite에 영구 저장하고, 인메모리 인덱스로 빠른 검색을 제공한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from fastembed import TextEmbedding

from core.async_utils import run_in_thread
from core.db_migrations import MigrationRunner, MigrationStep
from core.embedding_utils import embed_texts
from core.logging_setup import get_logger

if TYPE_CHECKING:
    import aiosqlite


@dataclass(frozen=True)
class CacheContext:
    """캐시 키 구성 정보."""

    model: str
    prompt_ver: str
    intent: str | None = None
    scope: str = "user"  # "global" | "user"
    chat_id: int | None = None


@dataclass(frozen=True)
class CacheResult:
    """캐시 히트 결과."""

    response: str
    similarity: float
    cached_query: str
    cache_id: int


_SEMANTIC_CACHE_SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT NOT NULL DEFAULT 'user',
    chat_id     INTEGER,
    query       TEXT NOT NULL,
    response    TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    model       TEXT NOT NULL,
    prompt_ver  TEXT NOT NULL DEFAULT 'v1',
    intent      TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sc_scope_chat ON semantic_cache(scope, chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sc_model ON semantic_cache(scope, model, prompt_ver, intent);
"""

_SEMANTIC_CACHE_SCHEMA_V2_SQL = """
CREATE TABLE IF NOT EXISTS semantic_cache_feedback_links (
    chat_id        INTEGER NOT NULL,
    bot_message_id INTEGER NOT NULL,
    cache_id       INTEGER NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
DELETE FROM semantic_cache_feedback_links
WHERE rowid NOT IN (
    SELECT MAX(rowid)
    FROM semantic_cache_feedback_links
    GROUP BY chat_id, bot_message_id
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_scfl_unique
    ON semantic_cache_feedback_links(chat_id, bot_message_id);
CREATE INDEX IF NOT EXISTS idx_scfl_lookup ON semantic_cache_feedback_links(chat_id, bot_message_id);
"""


async def _apply_semantic_cache_v1(db: aiosqlite.Connection) -> None:
    await db.executescript(_SEMANTIC_CACHE_SCHEMA_V1_SQL)


async def _apply_semantic_cache_v2(db: aiosqlite.Connection) -> None:
    await db.executescript(_SEMANTIC_CACHE_SCHEMA_V2_SQL)


class SemanticCache:
    """임베딩 기반 시맨틱 캐시."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        *,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.92,
        max_entries: int = 5000,
        ttl_hours: int = 168,
        min_query_chars: int = 4,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._db = db
        self._model_name = model_name
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._ttl_hours = ttl_hours
        self._min_query_chars = min_query_chars
        self._exclude_res = [
            re.compile(p) for p in (exclude_patterns or [])
        ]
        self._logger = get_logger("semantic_cache")
        self._encoder: Any = None
        self._enabled = False

        # 인메모리 인덱스
        self._ids: list[int] = []
        self._embeddings_by_id: dict[int, Any] = {}
        self._meta_by_id: dict[int, dict[str, Any]] = {}
        self._context_key_by_id: dict[int, tuple[str, int | None, str, str, str | None]] = {}
        self._cache_ids_by_context: dict[tuple[str, int | None, str, str, str | None], list[int]] = {}
        self._eviction_batch_size = max(1, min(64, max(1, self._max_entries // 20)))

        # 통계
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def encoder(self) -> Any:
        return self._encoder

    async def initialize(self) -> None:
        """테이블 생성 + 인코더 로드 + 인메모리 인덱스 빌드."""
        runner = MigrationRunner(self._db, self._logger, db_label="semantic_cache")
        await runner.run(
            [
                MigrationStep(1, _apply_semantic_cache_v1, "create_semantic_cache_tables"),
                MigrationStep(2, _apply_semantic_cache_v2, "create_cache_feedback_links"),
            ],
            backup_tables={"semantic_cache", "semantic_cache_feedback_links"},
        )

        try:
            self._encoder = TextEmbedding(
                model_name=self._model_name,
                providers=["CPUExecutionProvider"],
            )
            self._enabled = True
            self._logger.info("semantic_cache_encoder_loaded", model=self._model_name)
        except Exception as exc:
            self._logger.warning("semantic_cache_encoder_failed", error=str(exc))
            return

        await self._rebuild_index()

    async def _rebuild_index(self) -> None:
        """DB에서 전체 임베딩을 메모리로 로드한다."""
        cursor = await self._db.execute(
            "SELECT id, scope, chat_id, embedding, model, prompt_ver, intent "
            "FROM semantic_cache ORDER BY id"
        )
        rows = await cursor.fetchall()

        self._clear_index()

        for row in rows:
            rid, scope, chat_id, emb_blob, model, prompt_ver, intent = row
            self._append_index_entry(
                cache_id=int(rid),
                embedding=np.frombuffer(emb_blob, dtype=np.float32),
                scope=scope,
                chat_id=chat_id,
                model=model,
                prompt_ver=prompt_ver,
                intent=intent,
            )

        self._logger.info("semantic_cache_index_built", entries=len(self._ids))

    @staticmethod
    def _make_context_key(
        *,
        scope: str,
        chat_id: int | None,
        model: str,
        prompt_ver: str,
        intent: str | None,
    ) -> tuple[str, int | None, str, str, str | None]:
        return (scope, chat_id, model, prompt_ver, intent)

    def _rebuild_context_lookup(self) -> None:
        lookup: dict[tuple[str, int | None, str, str, str | None], list[int]] = {}
        for cache_id in self._ids:
            key = self._context_key_by_id.get(cache_id)
            if key is None:
                continue
            lookup.setdefault(key, []).append(cache_id)
        self._cache_ids_by_context = lookup

    def _select_candidate_positions(self, context: CacheContext) -> list[int]:
        lookup_chat_id = context.chat_id if context.scope == "user" else None
        key = self._make_context_key(
            scope=context.scope,
            chat_id=lookup_chat_id,
            model=context.model,
            prompt_ver=context.prompt_ver,
            intent=context.intent,
        )
        return self._cache_ids_by_context.get(key, [])

    def _append_index_entry(
        self,
        *,
        cache_id: int,
        embedding: Any,
        scope: str,
        chat_id: int | None,
        model: str,
        prompt_ver: str,
        intent: str | None,
    ) -> None:
        effective_chat_id = chat_id if scope == "user" else None
        key = self._make_context_key(
            scope=scope,
            chat_id=effective_chat_id,
            model=model,
            prompt_ver=prompt_ver,
            intent=intent,
        )
        self._ids.append(cache_id)
        self._embeddings_by_id[cache_id] = np.asarray(embedding, dtype=np.float32)
        self._meta_by_id[cache_id] = {
            "scope": scope,
            "chat_id": chat_id,
            "model": model,
            "prompt_ver": prompt_ver,
            "intent": intent,
        }
        self._context_key_by_id[cache_id] = key
        self._cache_ids_by_context.setdefault(key, []).append(cache_id)

    def _drop_index_entry(self, cache_id: int, *, remove_from_order: bool) -> bool:
        removed = False
        if remove_from_order:
            try:
                self._ids.remove(cache_id)
                removed = True
            except ValueError:
                pass

        if self._embeddings_by_id.pop(cache_id, None) is not None:
            removed = True
        if self._meta_by_id.pop(cache_id, None) is not None:
            removed = True

        key = self._context_key_by_id.pop(cache_id, None)
        if key is not None:
            bucket = self._cache_ids_by_context.get(key)
            if bucket is not None:
                try:
                    bucket.remove(cache_id)
                    removed = True
                except ValueError:
                    pass
                if not bucket:
                    self._cache_ids_by_context.pop(key, None)
        return removed

    def _clear_index(self) -> None:
        self._ids.clear()
        self._embeddings_by_id.clear()
        self._meta_by_id.clear()
        self._context_key_by_id.clear()
        self._cache_ids_by_context.clear()

    def is_cacheable(self, query: str) -> bool:
        """캐시 대상 여부를 판단한다."""
        if not self._enabled:
            return False
        if len(query.strip()) < self._min_query_chars:
            return False
        return all(not pattern.search(query) for pattern in self._exclude_res)

    async def get(self, query: str, context: CacheContext) -> CacheResult | None:
        """유사한 캐시 항목을 검색한다."""
        if not self._enabled or len(self._ids) == 0:
            self._misses += 1
            return None

        candidate_cache_ids = [
            cache_id
            for cache_id in self._select_candidate_positions(context)
            if cache_id in self._embeddings_by_id
        ]
        if not candidate_cache_ids:
            self._misses += 1
            return None

        query_vec = (
            await run_in_thread(embed_texts, self._encoder, [query], normalize=True)
        )[0]

        # 코사인 유사도 (정규화된 벡터 → 내적), 컨텍스트 후보군만 탐색
        candidate_embeddings = np.stack(
            [self._embeddings_by_id[cache_id] for cache_id in candidate_cache_ids]
        )
        similarities = candidate_embeddings @ query_vec
        best_local_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_local_idx])

        if best_sim < self._threshold:
            self._misses += 1
            return None

        cache_id = candidate_cache_ids[best_local_idx]

        # DB에서 응답 조회 + last_hit_at 갱신
        cursor = await self._db.execute(
            "SELECT query, response FROM semantic_cache WHERE id = ?", (cache_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            self._misses += 1
            return None

        await self._db.execute(
            "UPDATE semantic_cache SET last_hit_at = CURRENT_TIMESTAMP WHERE id = ?",
            (cache_id,),
        )
        await self._db.commit()

        self._hits += 1
        return CacheResult(
            response=row[1],
            similarity=best_sim,
            cached_query=row[0],
            cache_id=cache_id,
        )

    async def put(self, query: str, response: str, context: CacheContext) -> int:
        """질문-응답 쌍을 캐시에 저장하고 cache_id를 반환한다."""
        if not self._enabled:
            return -1

        query_emb = (
            await run_in_thread(embed_texts, self._encoder, [query], normalize=True)
        )[0]
        emb_blob = np.asarray(query_emb, dtype=np.float32).tobytes()

        cursor = await self._db.execute(
            "INSERT INTO semantic_cache (scope, chat_id, query, response, embedding, model, prompt_ver, intent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                context.scope,
                context.chat_id,
                query,
                response,
                emb_blob,
                context.model,
                context.prompt_ver,
                context.intent,
            ),
        )
        await self._db.commit()
        cache_id_raw = cursor.lastrowid
        if cache_id_raw is None:
            raise RuntimeError("semantic_cache_insert_failed: missing lastrowid")
        cache_id = int(cache_id_raw)

        # 인메모리 인덱스 갱신
        self._append_index_entry(
            cache_id=cache_id,
            embedding=query_emb,
            scope=context.scope,
            chat_id=context.chat_id,
            model=context.model,
            prompt_ver=context.prompt_ver,
            intent=context.intent,
        )

        # 최대 항목 수 초과 시 배치 단위 제거(저수위까지)로 O(n) 재배열 빈도를 줄인다.
        if len(self._ids) > self._max_entries:
            low_watermark = max(self._max_entries - self._eviction_batch_size + 1, 0)
            evict_count = max(1, len(self._ids) - low_watermark)
            await self._evict_oldest(evict_count)

        return cache_id

    async def _evict_oldest(self, count: int = 1) -> None:
        """가장 오래된 항목을 count개 제거한다."""
        if not self._ids:
            return
        evict_count = min(max(1, count), len(self._ids))
        oldest_ids = self._ids[:evict_count]

        placeholders = ",".join("?" for _ in oldest_ids)
        await self._db.execute(
            f"DELETE FROM semantic_cache WHERE id IN ({placeholders})",
            tuple(oldest_ids),
        )
        await self._db.execute(
            f"DELETE FROM semantic_cache_feedback_links WHERE cache_id IN ({placeholders})",
            tuple(oldest_ids),
        )
        await self._db.commit()

        self._ids = self._ids[evict_count:]
        for cache_id in oldest_ids:
            self._drop_index_entry(cache_id, remove_from_order=False)

    async def invalidate(self, chat_id: int | None = None) -> int:
        """캐시를 무효화한다."""
        if chat_id is not None:
            cursor = await self._db.execute(
                "DELETE FROM semantic_cache WHERE chat_id = ?", (chat_id,)
            )
            await self._db.execute(
                "DELETE FROM semantic_cache_feedback_links WHERE chat_id = ?", (chat_id,)
            )
        else:
            cursor = await self._db.execute("DELETE FROM semantic_cache")
            await self._db.execute("DELETE FROM semantic_cache_feedback_links")
        await self._db.commit()
        deleted = max(0, int(cursor.rowcount or 0))

        if chat_id is None:
            self._clear_index()
            return deleted

        target_ids = [
            cache_id
            for cache_id in self._ids
            if self._meta_by_id.get(cache_id, {}).get("chat_id") == chat_id
        ]
        if target_ids:
            removed_set = set(target_ids)
            self._ids = [cache_id for cache_id in self._ids if cache_id not in removed_set]
            for cache_id in target_ids:
                self._drop_index_entry(cache_id, remove_from_order=False)
        return deleted

    async def invalidate_by_id(self, cache_id: int) -> bool:
        """단일 캐시 항목을 삭제한다."""
        cursor = await self._db.execute(
            "DELETE FROM semantic_cache WHERE id = ?", (cache_id,)
        )
        await self._db.execute(
            "DELETE FROM semantic_cache_feedback_links WHERE cache_id = ?",
            (cache_id,),
        )
        await self._db.commit()
        if cursor.rowcount > 0:
            return self._remove_from_index(cache_id)
        return False

    def _remove_from_index(self, cache_id: int) -> bool:
        """인메모리 인덱스에서 단일 cache_id를 제거한다."""
        return self._drop_index_entry(cache_id, remove_from_order=True)

    async def link_feedback_target(
        self, chat_id: int, bot_message_id: int, cache_id: int
    ) -> None:
        """텔레그램 메시지와 캐시 항목을 연결한다."""
        await self._db.execute(
            "INSERT INTO semantic_cache_feedback_links "
            "(chat_id, bot_message_id, cache_id) VALUES (?, ?, ?) "
            "ON CONFLICT(chat_id, bot_message_id) DO UPDATE SET "
            "cache_id = excluded.cache_id, "
            "created_at = CURRENT_TIMESTAMP",
            (chat_id, bot_message_id, cache_id),
        )
        await self._db.commit()

    async def get_feedback_cache_id(self, chat_id: int, bot_message_id: int) -> int | None:
        """피드백에 연결된 cache_id를 조회한다."""
        cursor = await self._db.execute(
            "SELECT cache_id FROM semantic_cache_feedback_links "
            "WHERE chat_id = ? AND bot_message_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (chat_id, bot_message_id),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def prune_expired(self) -> int:
        """TTL 만료 항목을 삭제한다."""
        expired_cursor = await self._db.execute(
            "SELECT id FROM semantic_cache WHERE created_at < datetime('now', ?)",
            (f"-{self._ttl_hours} hours",),
        )
        expired_rows = await expired_cursor.fetchall()
        expired_ids = [int(row[0]) for row in expired_rows]

        cursor = await self._db.execute(
            "DELETE FROM semantic_cache WHERE created_at < datetime('now', ?)",
            (f"-{self._ttl_hours} hours",),
        )
        if expired_ids:
            placeholders = ",".join("?" for _ in expired_ids)
            await self._db.execute(
                f"DELETE FROM semantic_cache_feedback_links WHERE cache_id IN ({placeholders})",
                tuple(expired_ids),
            )
        await self._db.commit()
        deleted = max(0, int(cursor.rowcount or 0))
        if expired_ids:
            removed_set = set(expired_ids)
            self._ids = [cache_id for cache_id in self._ids if cache_id not in removed_set]
            for cache_id in expired_ids:
                self._drop_index_entry(cache_id, remove_from_order=False)
        return deleted if deleted > 0 else len(expired_ids)

    async def get_stats(self) -> dict:
        """캐시 통계."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries": len(self._ids),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "enabled": self._enabled,
        }

    async def close(self) -> None:
        """리소스 정리."""
        self._encoder = None
        self._clear_index()
