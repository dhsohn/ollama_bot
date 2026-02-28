"""엔진 라우팅 통합 테스트."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiosqlite
import pytest
import pytest_asyncio

np = pytest.importorskip("numpy")

import core.semantic_cache as semantic_cache_module
from core.config import AppSettings, BotConfig, MemoryConfig, OllamaConfig, SecurityConfig, TelegramConfig
from core.engine import Engine
from core.instant_responder import InstantResponder
from core.intent_router import ContextStrategy, RouteResult
from core.memory import MemoryManager
from core.llm_types import ChatResponse
from core.semantic_cache import SemanticCache


class _FakeEncoder:
    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, list):
            return [self._encode_one(text, normalize_embeddings) for text in texts]
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
    def __init__(self) -> None:
        self.default_model = "test-model"
        self.calls: list[dict] = []

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        timeout: int = 60,
        max_tokens: int | None = None,
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

    async def health_check(self):
        return {"status": "ok"}


class _SimpleIntentRouter:
    enabled = True
    routes_count = 1

    def classify(self, text: str) -> RouteResult | None:
        if "간단" not in text:
            return None
        return RouteResult(
            intent="simple_qa",
            confidence=0.99,
            context_strategy=ContextStrategy(
                max_history=3,
                include_dicl=False,
                include_preferences=False,
                max_tokens=128,
            ),
        )


@pytest_asyncio.fixture
async def integration_runtime(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(semantic_cache_module, "np", np)
    monkeypatch.setattr(semantic_cache_module, "TextEmbedding", lambda *args, **kwargs: _FakeEncoder())

    config = AppSettings(
        telegram_bot_token="test-token",
        data_dir=str(tmp_path),
        bot=BotConfig(max_conversation_length=10, response_timeout=60),
        ollama=OllamaConfig(model="test-model", system_prompt="integration prompt"),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(),
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
                "      - \"^ping$\"",
                "    responses:",
                "      - \"pong\"",
            ]
        ),
        encoding="utf-8",
    )
    instant_responder = InstantResponder(rules_path=str(rules_path))

    skills = MagicMock()
    skills.match_trigger.return_value = None
    skills.skill_count = 0

    ollama = _FakeOllama()
    engine = Engine(
        config=config,
        llm_client=ollama,
        memory=memory,
        skills=skills,
        instant_responder=instant_responder,
        semantic_cache=semantic_cache,
        intent_router=_SimpleIntentRouter(),
    )

    runtime = SimpleNamespace(
        engine=engine,
        memory=memory,
        ollama=ollama,
        semantic_cache=semantic_cache,
        cache_db=cache_db,
    )
    try:
        yield runtime
    finally:
        await semantic_cache.close()
        await cache_db.close()
        await memory.close()


class TestRoutingIntegration:
    @pytest.mark.asyncio
    async def test_full_then_cache_then_instant(self, integration_runtime) -> None:
        engine = integration_runtime.engine
        ollama = integration_runtime.ollama
        memory = integration_runtime.memory

        first = await engine.process_message(111, "간단 질문")
        assert first == "LLM:간단 질문"
        assert len(ollama.calls) == 1

        second = await engine.process_message(111, "간단 질문")
        assert second == first
        assert len(ollama.calls) == 1  # 캐시 히트

        instant = await engine.process_message(111, "ping")
        assert instant == "pong"
        assert len(ollama.calls) == 1  # 즉시 응답은 LLM 미호출

        history = await memory.get_conversation(111)
        assert len(history) == 6  # 3턴(user+assistant)

    @pytest.mark.asyncio
    async def test_intent_strategy_max_tokens_applied(self, integration_runtime) -> None:
        engine = integration_runtime.engine
        ollama = integration_runtime.ollama

        await engine.process_message(111, "간단 토큰 테스트")

        assert ollama.calls
        assert ollama.calls[-1]["max_tokens"] == 128
