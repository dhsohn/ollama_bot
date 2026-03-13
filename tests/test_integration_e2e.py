"""Engine 통합/E2E 테스트.

4-tier 라우팅 흐름, 스트림 에러 폴백, 시맨틱 캐시 수명주기,
RAG 컨텍스트 주입을 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
import pytest_asyncio

np = pytest.importorskip("numpy")

import core.semantic_cache as semantic_cache_module
from core.config import (
    AppSettings,
    BotConfig,
    MemoryConfig,
    RetrievalProviderConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.engine import Engine
from core.enums import RoutingTier
from core.instant_responder import InstantResponder
from core.intent_router import ContextStrategy, RouteResult
from core.llm_types import ChatResponse, ChatStreamState
from core.memory import MemoryManager
from core.rag.types import Chunk, ChunkMetadata, RAGResult, RAGTrace
from core.semantic_cache import CacheContext, SemanticCache
from core.skill_manager import SkillDefinition, SkillManager

# ── 테스트 헬퍼 ──


class _FakeEncoder:
    """결정론적 가짜 임베딩 인코더."""

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, list):
            return [self._encode_one(t, normalize_embeddings) for t in texts]
        return self._encode_one(texts, normalize_embeddings)

    def _encode_one(self, text: str, normalize_embeddings: bool):
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


class _FakeOllama:
    """LLMClientProtocol을 구현하는 가짜 클라이언트."""

    def __init__(self) -> None:
        self.default_model = "test-model"
        self.system_prompt = "You are a test bot."
        self.calls: list[dict] = []
        self.stream_chunks: list[str] | None = None
        self.stream_error_after: int | None = None

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def prepare_model(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        pass

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        response_format: str | dict | None = None,
        **kwargs,
    ) -> ChatResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "timeout": timeout,
                "max_tokens": max_tokens,
            }
        )
        return ChatResponse(content=f"LLM:{messages[-1]['content']}")

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        stream_state: ChatStreamState | None = None,
    ):
        chunks = self.stream_chunks or ["Hello", " world", "!"]
        for i, chunk in enumerate(chunks):
            if self.stream_error_after is not None and i >= self.stream_error_after:
                raise ConnectionError("stream_interrupted")
            yield chunk

    async def list_models(self) -> list[dict]:
        return [{"name": "test-model", "size": 1024}]

    async def health_check(self, *, attempt_recovery: bool = False) -> dict:
        return {"status": "ok"}

    async def recover_connection(self, *, force: bool = False) -> bool:
        return True


class _NoRouteRouter:
    """항상 None을 반환하는 라우터 (full LLM으로 전달)."""

    enabled = True
    routes_count = 0

    def classify(self, text: str) -> RouteResult | None:
        return None


class _SkillTriggerRouter:
    """스킬 트리거를 반환하는 스킬 매니저 대용."""

    def __init__(self, skill: SkillDefinition | None = None) -> None:
        self._skill = skill

    def match_trigger(self, text: str) -> SkillDefinition | None:
        if self._skill and text.startswith(self._skill.triggers[0]):
            return self._skill
        return None


# ── 픽스처 ──


@pytest_asyncio.fixture
async def e2e_runtime(tmp_path: Path, monkeypatch):
    """4-tier 라우팅에 필요한 모든 컴포넌트를 구성한 Engine 런타임."""
    monkeypatch.setattr(semantic_cache_module, "np", np)
    monkeypatch.setattr(
        semantic_cache_module,
        "TextEmbedding",
        lambda *args, **kwargs: _FakeEncoder(),
    )

    config = AppSettings(
        data_dir=str(tmp_path),
        bot=BotConfig(max_conversation_length=10, response_timeout=60),
        ollama=RetrievalProviderConfig(
            chat_model="test-model",
            chat_system_prompt="You are a test bot.",
        ),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(bot_token="test-token"),
    )

    memory = MemoryManager(config.memory, str(tmp_path), max_conversation_length=10)
    await memory.initialize()

    cache_db = await aiosqlite.connect(str(tmp_path / "cache.db"))
    semantic_cache = SemanticCache(
        db=cache_db,
        similarity_threshold=0.8,
        max_entries=100,
    )
    await semantic_cache.initialize()

    rules_path = tmp_path / "instant_rules.yaml"
    rules_path.write_text(
        "\n".join(
            [
                "rules:",
                "  - name: ping",
                "    patterns:",
                '      - "^ping$"',
                "    responses:",
                '      - "pong"',
                "  - name: hello",
                "    patterns:",
                '      - "^hello$"',
                "    responses:",
                '      - "hi there"',
            ]
        ),
        encoding="utf-8",
    )
    instant_responder = InstantResponder(rules_path=str(rules_path))

    test_skill = SkillDefinition(
        name="summarize",
        description="요약 스킬",
        triggers=["/summarize"],
        system_prompt="You are a summarizer.",
        timeout=30,
    )

    skills = MagicMock(spec=SkillManager)
    skills.match_trigger = MagicMock(
        side_effect=lambda text: test_skill if text.startswith("/summarize") else None,
    )
    skills.skill_count = 1
    skills.get_skill = MagicMock(return_value=test_skill)

    ollama = _FakeOllama()
    engine = Engine(
        config=config,
        llm_client=ollama,
        memory=memory,
        skills=skills,
        instant_responder=instant_responder,
        semantic_cache=semantic_cache,
        intent_router=_NoRouteRouter(),
    )

    runtime = SimpleNamespace(
        engine=engine,
        memory=memory,
        llm=ollama,
        semantic_cache=semantic_cache,
        cache_db=cache_db,
        skills=skills,
        config=config,
    )
    try:
        yield runtime
    finally:
        await semantic_cache.close()
        await cache_db.close()
        await memory.close()


# ── 테스트 1: Full 4-tier routing flow ──


class TestFourTierRoutingFlow:
    """스킬 -> 즉시 응답 -> 캐시 -> Full LLM 4-tier 라우팅 흐름을 검증한다."""

    @pytest.mark.asyncio
    async def test_skill_tier_takes_priority(self, e2e_runtime) -> None:
        """스킬 트리거 매칭 시 RoutingTier.SKILL로 처리된다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        result = await engine.process_message(111, "/summarize 이 텍스트를 요약해줘")

        assert "LLM:" in result
        assert len(llm.calls) == 1
        # 스킬 시스템 프롬프트가 사용되었는지 확인
        system_msgs = [
            m for m in llm.calls[0]["messages"] if m["role"] == "system"
        ]
        assert any("summarizer" in m["content"].lower() for m in system_msgs)

    @pytest.mark.asyncio
    async def test_instant_tier_bypasses_llm(self, e2e_runtime) -> None:
        """즉시 응답 규칙 매칭 시 LLM을 호출하지 않는다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        result = await engine.process_message(111, "ping")

        assert result == "pong"
        assert len(llm.calls) == 0

    @pytest.mark.asyncio
    async def test_cache_tier_reuses_previous_response(self, e2e_runtime) -> None:
        """동일 질문 시 캐시에서 응답을 재사용한다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        # 첫 번째: Full LLM
        first = await engine.process_message(111, "캐시 테스트 질문입니다")
        assert len(llm.calls) == 1

        # 두 번째: 캐시 히트
        second = await engine.process_message(111, "캐시 테스트 질문입니다")
        assert second == first
        assert len(llm.calls) == 1  # LLM 추가 호출 없음

    @pytest.mark.asyncio
    async def test_full_tier_fallthrough(self, e2e_runtime) -> None:
        """스킬/즉시/캐시 모두 미스 시 Full LLM으로 처리된다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        result = await engine.process_message(111, "이것은 새로운 질문입니다")

        assert "LLM:" in result
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_tier_priority_order(self, e2e_runtime) -> None:
        """4-tier 우선순위: skill > instant > cache > full 순서로 처리된다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm
        memory = e2e_runtime.memory

        # 1. Skill
        r1 = await engine.process_message(111, "/summarize hello world")
        assert "LLM:" in r1
        skill_call_count = len(llm.calls)
        assert skill_call_count == 1

        # 2. Instant
        r2 = await engine.process_message(111, "hello")
        assert r2 == "hi there"
        assert len(llm.calls) == skill_call_count  # LLM 호출 없음

        # 3. Full (first time, will be cached)
        r3 = await engine.process_message(111, "새로운 일반 대화 질문")
        assert "LLM:" in r3
        full_call_count = len(llm.calls)
        assert full_call_count == skill_call_count + 1

        # 4. Cache (same question)
        r4 = await engine.process_message(111, "새로운 일반 대화 질문")
        assert r4 == r3
        assert len(llm.calls) == full_call_count  # LLM 추가 호출 없음

        # 메모리에 모든 턴이 기록되었는지 확인
        history = await memory.get_conversation(111)
        assert len(history) == 8  # 4턴 x (user + assistant)


# ── 테스트 2: Stream + error fallback ──


class TestStreamErrorFallback:
    """스트리밍 실패 시 chat 폴백으로 응답을 반환하는지 검증한다."""

    @pytest.mark.asyncio
    async def test_stream_error_falls_back_to_chat(self, e2e_runtime) -> None:
        """스트림이 청크 전달 전에 실패하면 chat 폴백으로 응답한다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        # 첫 번째 청크에서 즉시 실패하도록 설정
        llm.stream_error_after = 0

        chunks = []
        async for chunk in engine.process_message_stream(111, "스트림 테스트"):
            chunks.append(chunk)

        # chat 폴백으로 응답이 반환되어야 함
        full_response = "".join(chunks)
        assert full_response
        assert "LLM:" in full_response
        # chat 폴백 호출 확인
        assert len(llm.calls) >= 1

    @pytest.mark.asyncio
    async def test_stream_partial_then_error_keeps_partial(self, e2e_runtime) -> None:
        """스트림이 일부 청크 전달 후 실패하면 부분 응답을 유지한다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        # 청크 2개 후 에러
        llm.stream_chunks = ["Part1", "Part2", "Part3", "Part4"]
        llm.stream_error_after = 2

        chunks = []
        async for chunk in engine.process_message_stream(111, "부분 스트림 테스트"):
            chunks.append(chunk)

        full_response = "".join(chunks)
        # 처음 2개 청크는 성공적으로 전달됨
        assert "Part1" in full_response
        assert "Part2" in full_response

    @pytest.mark.asyncio
    async def test_stream_success_persists_turn(self, e2e_runtime) -> None:
        """스트리밍 성공 시 대화 기록이 저장된다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm
        memory = e2e_runtime.memory

        llm.stream_chunks = ["응답", " 내용"]

        chunks = []
        async for chunk in engine.process_message_stream(111, "스트림 저장 테스트"):
            chunks.append(chunk)

        history = await memory.get_conversation(111)
        assert len(history) == 2  # user + assistant
        assert history[0]["content"] == "스트림 저장 테스트"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_instant_response_via_stream(self, e2e_runtime) -> None:
        """스트리밍 모드에서도 즉시 응답이 정상 작동한다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm

        chunks = []
        async for chunk in engine.process_message_stream(111, "ping"):
            chunks.append(chunk)

        assert "".join(chunks) == "pong"
        assert len(llm.calls) == 0


# ── 테스트 3: Semantic cache lifecycle ──


class TestSemanticCacheLifecycle:
    """캐시 미스 -> LLM -> 캐시 히트 -> 무효화 -> 캐시 미스 수명주기를 검증한다."""

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit_then_invalidate(self, e2e_runtime) -> None:
        """query -> miss -> LLM -> same query -> hit -> invalidate -> miss."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm
        cache = e2e_runtime.semantic_cache

        query = "시맨틱 캐시 라이프사이클 테스트 질문"

        # 1단계: 캐시 미스 -> Full LLM
        result1 = await engine.process_message(111, query)
        assert "LLM:" in result1
        call_count_after_first = len(llm.calls)
        assert call_count_after_first == 1

        # 캐시 통계 확인: 1개 항목 저장됨
        stats = await cache.get_stats()
        assert stats["entries"] >= 1

        # 2단계: 동일 질의 -> 캐시 히트
        result2 = await engine.process_message(111, query)
        assert result2 == result1
        assert len(llm.calls) == call_count_after_first  # LLM 미호출

        # 캐시 히트 수 증가 확인
        stats_after_hit = await cache.get_stats()
        assert stats_after_hit["hits"] >= 1

        # 3단계: 캐시 무효화
        deleted = await cache.invalidate(chat_id=111)
        assert deleted >= 1

        # 4단계: 무효화 후 동일 질의 -> 캐시 미스 -> Full LLM
        result3 = await engine.process_message(111, query)
        assert "LLM:" in result3
        assert len(llm.calls) == call_count_after_first + 1  # LLM 재호출

    @pytest.mark.asyncio
    async def test_global_invalidate_clears_all(self, e2e_runtime) -> None:
        """전역 무효화 시 모든 캐시가 삭제된다."""
        engine = e2e_runtime.engine
        cache = e2e_runtime.semantic_cache

        # 충분히 다른 질문으로 캐시 채우기
        await engine.process_message(111, "양자역학에서 슈뢰딩거 방정식의 의미는 무엇인가요?")
        await engine.process_message(111, "오늘 서울 날씨를 알려주세요")

        stats = await cache.get_stats()
        assert stats["entries"] >= 2

        # 전역 무효화
        await cache.invalidate()
        stats_after = await cache.get_stats()
        assert stats_after["entries"] == 0

    @pytest.mark.asyncio
    async def test_cache_not_used_for_short_queries(self, e2e_runtime) -> None:
        """짧은 질의는 캐시 대상에서 제외된다."""
        engine = e2e_runtime.engine
        llm = e2e_runtime.llm
        cache = e2e_runtime.semantic_cache

        # 짧은 질의 (min_query_chars 미만)
        await engine.process_message(111, "ab")
        entries_before = (await cache.get_stats())["entries"]

        # 같은 짧은 질의 반복
        await engine.process_message(111, "ab")

        # 캐시에 저장되지 않아야 함
        entries_after = (await cache.get_stats())["entries"]
        assert entries_after == entries_before
        # 두 번 모두 LLM 호출
        assert len(llm.calls) == 2


# ── 테스트 4: RAG integration ──


class TestRAGIntegration:
    """RAG 파이프라인의 컨텍스트 주입을 검증한다."""

    @pytest_asyncio.fixture
    async def rag_runtime(self, tmp_path: Path, monkeypatch):
        """RAG 파이프라인이 주입된 Engine 런타임."""
        config = AppSettings(
            data_dir=str(tmp_path),
            bot=BotConfig(max_conversation_length=10, response_timeout=60),
            ollama=RetrievalProviderConfig(
                chat_model="test-model",
                chat_system_prompt="You are a test bot.",
            ),
            security=SecurityConfig(allowed_users=[111]),
            memory=MemoryConfig(),
            telegram=TelegramConfig(bot_token="test-token"),
        )

        memory = MemoryManager(config.memory, str(tmp_path), max_conversation_length=10)
        await memory.initialize()

        skills = MagicMock(spec=SkillManager)
        skills.match_trigger = MagicMock(return_value=None)
        skills.skill_count = 0

        ollama = _FakeOllama()

        # Mock RAG 파이프라인
        rag_pipeline = MagicMock()
        rag_pipeline.should_trigger_rag = MagicMock(return_value=True)
        rag_pipeline.execute = AsyncMock(
            return_value=RAGResult(
                contexts=["[#1] 관련 문서 내용입니다. 파이썬은 프로그래밍 언어입니다."],
                candidates=[],
                trace=RAGTrace(
                    rag_used=True,
                    rerank_used=True,
                    retrieve_k0=10,
                    rerank_k=3,
                    context_tokens_estimate=50,
                    total_latency_ms=15.0,
                ),
            ),
        )

        engine = Engine(
            config=config,
            llm_client=ollama,
            memory=memory,
            skills=skills,
            rag_pipeline=rag_pipeline,
        )

        runtime = SimpleNamespace(
            engine=engine,
            memory=memory,
            llm=ollama,
            rag_pipeline=rag_pipeline,
        )
        try:
            yield runtime
        finally:
            await memory.close()

    @pytest.mark.asyncio
    async def test_rag_context_injected_into_messages(self, rag_runtime) -> None:
        """RAG 컨텍스트가 LLM 호출 메시지에 주입된다."""
        engine = rag_runtime.engine
        llm = rag_runtime.llm
        rag = rag_runtime.rag_pipeline

        result = await engine.process_message(111, "파이썬이란 무엇인가요?")

        # RAG 파이프라인이 실행됨
        rag.should_trigger_rag.assert_called_once()
        rag.execute.assert_awaited_once()

        # LLM에 전달된 메시지에 RAG 컨텍스트가 포함됨
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        system_content = " ".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        assert "참고 문서" in system_content
        assert "관련 문서 내용입니다" in system_content

    @pytest.mark.asyncio
    async def test_rag_not_triggered_when_condition_false(self, rag_runtime) -> None:
        """should_trigger_rag가 False이면 RAG를 실행하지 않는다."""
        engine = rag_runtime.engine
        rag = rag_runtime.rag_pipeline
        llm = rag_runtime.llm

        rag.should_trigger_rag.return_value = False

        await engine.process_message(111, "안녕하세요")

        rag.should_trigger_rag.assert_called()
        rag.execute.assert_not_awaited()

        # RAG 없이 일반 LLM 호출
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        system_content = " ".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        assert "참고 문서" not in system_content

    @pytest.mark.asyncio
    async def test_rag_with_empty_contexts_skips_injection(self, rag_runtime) -> None:
        """RAG 결과에 contexts가 비어있으면 주입을 건너뛴다."""
        engine = rag_runtime.engine
        rag = rag_runtime.rag_pipeline
        llm = rag_runtime.llm

        rag.execute = AsyncMock(
            return_value=RAGResult(
                contexts=[],
                candidates=[],
                trace=RAGTrace(rag_used=True, total_latency_ms=5.0),
            ),
        )

        await engine.process_message(111, "문서가 없는 질문")

        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        system_content = " ".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        assert "참고 문서" not in system_content

    @pytest.mark.asyncio
    async def test_rag_result_persisted_in_memory(self, rag_runtime) -> None:
        """RAG로 생성된 응답도 대화 기록에 정상 저장된다."""
        engine = rag_runtime.engine
        memory = rag_runtime.memory

        result = await engine.process_message(111, "RAG 메모리 테스트 질문")

        history = await memory.get_conversation(111)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "RAG 메모리 테스트 질문"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == result
