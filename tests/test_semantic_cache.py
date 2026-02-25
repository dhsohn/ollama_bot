"""시맨틱 캐시 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiosqlite
import pytest
import pytest_asyncio

np = pytest.importorskip("numpy")

import core.semantic_cache as semantic_cache_module
from core.semantic_cache import CacheContext, SemanticCache


class _FakeEncoder:
    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, list):
            return [self._encode_one(text, normalize_embeddings) for text in texts]
        return self._encode_one(texts, normalize_embeddings)

    def _encode_one(self, text: str, normalize_embeddings: bool) -> np.ndarray:
        seed = float(sum(ord(ch) for ch in text))
        vec = np.array(
            [
                (seed % 97.0) + 1.0,
                (len(text) % 31) + 1.0,
                ((seed / max(len(text), 1)) % 53.0) + 1.0,
            ],
            dtype=np.float32,
        )
        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec


@pytest_asyncio.fixture
async def db(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "cache.db"))
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def semantic_cache(db, monkeypatch) -> SemanticCache:
    monkeypatch.setattr(semantic_cache_module, "_HAS_ENCODER", True)
    monkeypatch.setattr(semantic_cache_module, "np", np)
    monkeypatch.setattr(semantic_cache_module, "SentenceTransformer", lambda *args, **kwargs: _FakeEncoder())

    cache = SemanticCache(
        db=db,
        similarity_threshold=0.8,
        max_entries=100,
    )
    await cache.initialize()
    try:
        yield cache
    finally:
        await cache.close()


class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_invalidate_by_id_removes_index_without_full_rebuild(
        self, semantic_cache: SemanticCache,
    ) -> None:
        ctx = CacheContext(
            model="test-model",
            prompt_ver="v1",
            intent="simple_qa",
            scope="user",
            chat_id=111,
        )
        first_id = await semantic_cache.put("첫 질문", "첫 응답", context=ctx)
        second_id = await semantic_cache.put("둘째 질문", "둘째 응답", context=ctx)

        rebuild_spy = AsyncMock(wraps=semantic_cache._rebuild_index)
        semantic_cache._rebuild_index = rebuild_spy

        deleted = await semantic_cache.invalidate_by_id(first_id)

        assert deleted is True
        rebuild_spy.assert_not_awaited()
        assert first_id not in semantic_cache._ids
        assert second_id in semantic_cache._ids

    @pytest.mark.asyncio
    async def test_get_uses_context_bucket(
        self, semantic_cache: SemanticCache,
    ) -> None:
        global_ctx = CacheContext(
            model="test-model",
            prompt_ver="v1",
            intent="simple_qa",
            scope="global",
            chat_id=None,
        )
        user_ctx = CacheContext(
            model="test-model",
            prompt_ver="v1",
            intent="simple_qa",
            scope="user",
            chat_id=222,
        )
        await semantic_cache.put("파이썬", "전역 응답", context=global_ctx)
        await semantic_cache.put("파이썬", "사용자 응답", context=user_ctx)

        assert len(semantic_cache._select_candidate_positions(global_ctx)) == 1
        assert len(semantic_cache._select_candidate_positions(user_ctx)) == 1

        user_hit = await semantic_cache.get("파이썬", context=user_ctx)
        global_hit = await semantic_cache.get("파이썬", context=global_ctx)

        assert user_hit is not None
        assert global_hit is not None
        assert user_hit.response == "사용자 응답"
        assert global_hit.response == "전역 응답"
