"""시맨틱 캐싱 엔진.

사용자 질문의 임베딩 유사도를 기반으로 캐시 히트를 판정한다.
캐시는 SQLite에 영구 저장하고, 인메모리 인덱스로 빠른 검색을 제공한다.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from core.logging_setup import get_logger

if TYPE_CHECKING:
    import aiosqlite

try:
    import numpy as np
except ImportError:
    np = cast(Any, None)

try:
    import sentence_transformers as sentence_transformers_module

    _HAS_ENCODER = True
except ImportError:
    _HAS_ENCODER = False
    sentence_transformers_module = cast(Any, None)


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


_SCHEMA_SQL = """
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


class SemanticCache:
    """임베딩 기반 시맨틱 캐시."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        *,
        model_name: str = "intfloat/multilingual-e5-small",
        embedding_device: str = "cpu",
        similarity_threshold: float = 0.92,
        max_entries: int = 5000,
        ttl_hours: int = 168,
        min_query_chars: int = 4,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._db = db
        self._model_name = model_name
        self._device = embedding_device
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
        self._embeddings: Any = None  # numpy array (N, dim)
        self._meta: list[dict] = []  # scope, chat_id, model, prompt_ver, intent
        self._positions_by_context: dict[tuple[str, int | None, str, str, str | None], list[int]] = {}

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
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

        if not _HAS_ENCODER:
            self._logger.warning("semantic_cache_no_encoder", reason="sentence-transformers not installed")
            return

        try:
            self._encoder = sentence_transformers_module.SentenceTransformer(
                self._model_name, device=self._device
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

        self._ids = []
        embeddings_list = []
        self._meta = []

        for row in rows:
            rid, scope, chat_id, emb_blob, model, prompt_ver, intent = row
            self._ids.append(rid)
            embeddings_list.append(np.frombuffer(emb_blob, dtype=np.float32))
            self._meta.append({
                "scope": scope,
                "chat_id": chat_id,
                "model": model,
                "prompt_ver": prompt_ver,
                "intent": intent,
            })

        if embeddings_list:
            self._embeddings = np.stack(embeddings_list)
        else:
            self._embeddings = None
        self._rebuild_context_lookup()

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
        for idx, meta in enumerate(self._meta):
            key = self._make_context_key(
                scope=meta["scope"],
                chat_id=meta["chat_id"],
                model=meta["model"],
                prompt_ver=meta["prompt_ver"],
                intent=meta["intent"],
            )
            lookup.setdefault(key, []).append(idx)
        self._positions_by_context = lookup

    def _select_candidate_positions(self, context: CacheContext) -> list[int]:
        lookup_chat_id = context.chat_id if context.scope == "user" else None
        key = self._make_context_key(
            scope=context.scope,
            chat_id=lookup_chat_id,
            model=context.model,
            prompt_ver=context.prompt_ver,
            intent=context.intent,
        )
        return self._positions_by_context.get(key, [])

    def is_cacheable(self, query: str) -> bool:
        """캐시 대상 여부를 판단한다."""
        if not self._enabled:
            return False
        if len(query.strip()) < self._min_query_chars:
            return False
        for pattern in self._exclude_res:
            if pattern.search(query):
                return False
        return True

    async def get(self, query: str, context: CacheContext) -> CacheResult | None:
        """유사한 캐시 항목을 검색한다."""
        if not self._enabled or self._embeddings is None or len(self._ids) == 0:
            self._misses += 1
            return None

        candidate_positions = self._select_candidate_positions(context)
        if not candidate_positions:
            self._misses += 1
            return None

        query_emb = await asyncio.to_thread(
            self._encoder.encode, query, normalize_embeddings=True
        )
        query_vec = np.array(query_emb, dtype=np.float32)

        # 코사인 유사도 (정규화된 벡터 → 내적), 컨텍스트 후보군만 탐색
        candidate_embeddings = self._embeddings[candidate_positions]
        similarities = candidate_embeddings @ query_vec
        best_local_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_local_idx])

        if best_sim < self._threshold:
            self._misses += 1
            return None

        best_idx = candidate_positions[best_local_idx]
        cache_id = self._ids[best_idx]

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

        query_emb = await asyncio.to_thread(
            self._encoder.encode, query, normalize_embeddings=True
        )
        emb_blob = np.array(query_emb, dtype=np.float32).tobytes()

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
        self._ids.append(cache_id)
        emb_vec = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = emb_vec
        else:
            self._embeddings = np.vstack([self._embeddings, emb_vec])
        self._meta.append({
            "scope": context.scope,
            "chat_id": context.chat_id,
            "model": context.model,
            "prompt_ver": context.prompt_ver,
            "intent": context.intent,
        })
        key = self._make_context_key(
            scope=context.scope,
            chat_id=context.chat_id if context.scope == "user" else None,
            model=context.model,
            prompt_ver=context.prompt_ver,
            intent=context.intent,
        )
        self._positions_by_context.setdefault(key, []).append(len(self._ids) - 1)

        # 최대 항목 수 초과 시 LRU 제거
        if len(self._ids) > self._max_entries:
            await self._evict_oldest()

        return cache_id

    async def _evict_oldest(self) -> None:
        """가장 오래된 항목을 제거한다."""
        if not self._ids:
            return
        oldest_id = self._ids[0]
        await self._db.execute("DELETE FROM semantic_cache WHERE id = ?", (oldest_id,))
        await self._db.commit()

        self._ids.pop(0)
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = self._embeddings[1:]
        self._meta.pop(0)
        self._rebuild_context_lookup()

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
        deleted = cursor.rowcount
        await self._rebuild_index()
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
        try:
            idx = self._ids.index(cache_id)
        except ValueError:
            return False

        self._ids.pop(idx)
        self._meta.pop(idx)
        if self._embeddings is not None and len(self._embeddings) > idx:
            self._embeddings = np.delete(self._embeddings, idx, axis=0)
            if len(self._embeddings) == 0:
                self._embeddings = None
        self._rebuild_context_lookup()
        return True

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
        cursor = await self._db.execute(
            "DELETE FROM semantic_cache WHERE created_at < datetime('now', ?)",
            (f"-{self._ttl_hours} hours",),
        )
        await self._db.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            await self._rebuild_index()
        return deleted

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
        self._embeddings = None
        self._ids.clear()
        self._meta.clear()
        self._positions_by_context.clear()
