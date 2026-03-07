"""Edge-case and failure-scenario tests for Engine, SecurityManager, and MemoryManager."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from core.config import (
    AppSettings,
    BotConfig,
    LemonadeConfig,
    MemoryConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.engine import Engine
from core.enums import RoutingTier
from core.llm_types import ChatResponse, ChatStreamState
from core.memory import MemoryManager
from core.security import GlobalConcurrencyError, SecurityManager
from core.skill_manager import SkillManager

# ── fixtures ──


@pytest.fixture
def security_config() -> SecurityConfig:
    return SecurityConfig(
        allowed_users=[111, 222],
        rate_limit=10,
        max_file_size=10_485_760,
        blocked_paths=["/etc/*", "/proc/*", "/sys/*"],
        max_concurrent_requests=2,
        max_input_length=100,
    )


@pytest.fixture
def app_settings(security_config: SecurityConfig) -> AppSettings:
    return AppSettings(
        telegram_bot_token="test",
        bot=BotConfig(max_conversation_length=10),
        lemonade=LemonadeConfig(
            default_model="test-model",
            system_prompt="You are a test bot.",
        ),
        security=security_config,
        memory=MemoryConfig(),
        telegram=TelegramConfig(),
    )


@pytest.fixture
def mock_ollama() -> AsyncMock:
    client = AsyncMock()
    client.default_model = "test-model"
    client.system_prompt = "You are a test bot."
    client.chat = AsyncMock(return_value=ChatResponse(content="OK"))
    client.health_check = AsyncMock(return_value={"status": "ok"})
    client.list_models = AsyncMock(return_value=[{"name": "test-model", "size": 1024}])
    return client


@pytest.fixture
def mock_skills() -> MagicMock:
    skills = MagicMock(spec=SkillManager)
    skills.match_trigger = MagicMock(return_value=None)
    skills.get_skill = MagicMock(return_value=None)
    skills.skill_count = 0
    return skills


@pytest_asyncio.fixture
async def memory(tmp_path: Path) -> MemoryManager:
    mm = MemoryManager(MemoryConfig(), str(tmp_path), max_conversation_length=10)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
def engine(app_settings, mock_ollama, memory, mock_skills) -> Engine:
    return Engine(
        config=app_settings,
        llm_client=mock_ollama,
        memory=memory,
        skills=mock_skills,
    )


@pytest.fixture
def security_manager(security_config: SecurityConfig) -> SecurityManager:
    return SecurityManager(security_config)


# ── 1. LLM timeout propagates ──


class TestLLMTimeout:
    @pytest.mark.asyncio
    async def test_timeout_propagates(self, engine: Engine, mock_ollama) -> None:
        """TimeoutError from the LLM client must propagate through process_message."""
        mock_ollama.chat = AsyncMock(side_effect=TimeoutError("LLM timed out"))

        with pytest.raises(TimeoutError, match="LLM timed out"):
            await engine.process_message(111, "hello")


# ── 2. Empty LLM response ──


class TestEmptyLLMResponse:
    @pytest.mark.asyncio
    async def test_empty_string_raises_runtime_error(
        self, engine: Engine, mock_ollama
    ) -> None:
        """LLM returning an empty string must raise RuntimeError('empty_response_from_llm')."""
        mock_ollama.chat = AsyncMock(return_value=ChatResponse(content=""))

        with pytest.raises(RuntimeError, match="empty_response_from_llm"):
            await engine.process_message(111, "hello")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises_runtime_error(
        self, engine: Engine, mock_ollama
    ) -> None:
        """LLM returning only whitespace must also raise RuntimeError."""
        mock_ollama.chat = AsyncMock(return_value=ChatResponse(content="   \n\t  "))

        with pytest.raises(RuntimeError, match="empty_response_from_llm"):
            await engine.process_message(111, "hello")


# ── 3. Concurrent request limit ──


class TestConcurrentRequestLimit:
    @pytest.mark.asyncio
    async def test_acquire_blocks_when_slots_exhausted(
        self, security_manager: SecurityManager
    ) -> None:
        """When all global slots are taken, the next acquire must raise GlobalConcurrencyError."""
        max_concurrent = security_manager._max_concurrent_requests  # 2

        # Exhaust all slots.
        for _ in range(max_concurrent):
            await security_manager.acquire_global_slot(chat_id=111)

        with pytest.raises(GlobalConcurrencyError):
            await security_manager.acquire_global_slot(chat_id=222)

    @pytest.mark.asyncio
    async def test_release_then_acquire_succeeds(
        self, security_manager: SecurityManager
    ) -> None:
        """After releasing a slot, a new acquire must succeed."""
        max_concurrent = security_manager._max_concurrent_requests

        for _ in range(max_concurrent):
            await security_manager.acquire_global_slot(chat_id=111)

        # Release one slot.
        security_manager.release_global_slot()

        # Should succeed now.
        await security_manager.acquire_global_slot(chat_id=222)

    @pytest.mark.asyncio
    async def test_global_slot_context_manager(
        self, security_manager: SecurityManager
    ) -> None:
        """The global_slot context manager must auto-release on exit."""
        max_concurrent = security_manager._max_concurrent_requests

        # Use all but one slot via context manager (exits immediately).
        async with security_manager.global_slot(chat_id=111):
            pass  # slot acquired then released

        # Should be able to acquire max_concurrent times now.
        for _ in range(max_concurrent):
            await security_manager.acquire_global_slot(chat_id=111)


# ── 4. Memory retention overflow ──


class TestMemoryRetentionOverflow:
    @pytest_asyncio.fixture
    async def small_memory(self, tmp_path: Path) -> MemoryManager:
        config = MemoryConfig(max_long_term_entries=3)
        mm = MemoryManager(config=config, data_dir=str(tmp_path), max_conversation_length=10)
        await mm.initialize()
        yield mm
        await mm.close()

    @pytest.mark.asyncio
    async def test_oldest_entry_evicted_at_capacity(
        self, small_memory: MemoryManager
    ) -> None:
        """When max_long_term_entries is reached, the oldest entry is evicted on insert."""
        chat_id = 999

        # Fill to capacity.
        await small_memory.store_memory(chat_id, "key_a", "val_a")
        await small_memory.store_memory(chat_id, "key_b", "val_b")
        await small_memory.store_memory(chat_id, "key_c", "val_c")

        # Insert one more -- key_a (oldest) should be evicted.
        await small_memory.store_memory(chat_id, "key_d", "val_d")

        memories = await small_memory.recall_memory(chat_id)
        keys = {m["key"] for m in memories}
        assert "key_a" not in keys, "oldest entry should have been evicted"
        assert keys == {"key_b", "key_c", "key_d"}

    @pytest.mark.asyncio
    async def test_upsert_does_not_evict(self, small_memory: MemoryManager) -> None:
        """Updating an existing key should not trigger eviction."""
        chat_id = 999

        await small_memory.store_memory(chat_id, "key_a", "val_a")
        await small_memory.store_memory(chat_id, "key_b", "val_b")
        await small_memory.store_memory(chat_id, "key_c", "val_c")

        # Upsert existing key -- no eviction expected.
        await small_memory.store_memory(chat_id, "key_b", "updated_b")

        memories = await small_memory.recall_memory(chat_id)
        keys = {m["key"] for m in memories}
        assert keys == {"key_a", "key_b", "key_c"}
        values = {m["key"]: m["value"] for m in memories}
        assert values["key_b"] == "updated_b"


# ── 5. Malformed Unicode input ──


class TestMalformedUnicodeInput:
    def test_null_bytes_removed(self, security_manager: SecurityManager) -> None:
        """Null bytes must be stripped from input."""
        result = security_manager.sanitize_input("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_ansi_escapes_removed(self, security_manager: SecurityManager) -> None:
        """ANSI escape sequences must be stripped."""
        text = "\x1b[31mred text\x1b[0m"
        result = security_manager.sanitize_input(text)
        assert "\x1b" not in result
        assert "red text" in result

    def test_multiple_ansi_sequences(self, security_manager: SecurityManager) -> None:
        """Multiple and nested ANSI sequences must all be removed."""
        text = "\x1b[1m\x1b[4mBold underline\x1b[0m normal \x1b[32mgreen\x1b[0m"
        result = security_manager.sanitize_input(text)
        assert "\x1b" not in result
        assert "Bold underline" in result
        assert "normal" in result

    def test_unicode_nfc_normalization(self, security_manager: SecurityManager) -> None:
        """Combining characters must be NFC-normalized."""
        # e + combining acute accent (NFD) -> precomposed e-acute (NFC)
        nfd_input = "e\u0301"  # e + combining acute
        result = security_manager.sanitize_input(nfd_input)
        assert result == "\u00e9"  # precomposed e-acute

    def test_emoji_preserved(self, security_manager: SecurityManager) -> None:
        """Emoji and supplementary-plane characters must survive sanitization."""
        text = "Hello \U0001f600 world \U0001f4a9"
        result = security_manager.sanitize_input(text)
        assert "\U0001f600" in result
        assert "\U0001f4a9" in result

    def test_input_truncation(self, security_manager: SecurityManager) -> None:
        """Input exceeding max_input_length must be truncated (max is 100 in fixture)."""
        text = "A" * 200
        result = security_manager.sanitize_input(text)
        assert len(result) == 100

    def test_mixed_edge_cases(self, security_manager: SecurityManager) -> None:
        """Null bytes, ANSI, and NFD combining in a single input."""
        text = "\x00\x1b[31me\u0301\x00\x1b[0m"
        result = security_manager.sanitize_input(text)
        assert "\x00" not in result
        assert "\x1b" not in result
        assert result == "\u00e9"

    def test_empty_input(self, security_manager: SecurityManager) -> None:
        """Empty string must pass through without error."""
        result = security_manager.sanitize_input("")
        assert result == ""


# ── 6. Stream error with partial response ──


class TestStreamErrorPartialResponse:
    @pytest.mark.asyncio
    async def test_partial_chunks_preserved_after_stream_error(
        self, engine: Engine, mock_ollama, mock_skills
    ) -> None:
        """When a stream yields chunks then raises, the partial response is persisted."""
        from core.skill_manager import SkillDefinition

        skill = SkillDefinition(
            name="test_stream",
            description="test",
            triggers=["/stream"],
            system_prompt="Stream test.",
            timeout=10,
            streaming=True,
        )
        mock_skills.match_trigger.return_value = skill

        async def _fake_stream(**kwargs):
            yield "chunk1"
            yield "chunk2"
            raise ConnectionError("stream interrupted")

        mock_ollama.chat_stream = _fake_stream

        collected: list[str] = []
        async for chunk in engine.process_message_stream(111, "/stream test"):
            collected.append(chunk)

        # The partial chunks should have been yielded.
        assert "chunk1" in collected
        assert "chunk2" in collected

    @pytest.mark.asyncio
    async def test_empty_stream_then_error_falls_back(
        self, engine: Engine, mock_ollama, mock_skills
    ) -> None:
        """When stream fails immediately with no chunks, fallback to non-streaming chat."""
        from core.skill_manager import SkillDefinition

        skill = SkillDefinition(
            name="test_fallback",
            description="test",
            triggers=["/fallback"],
            system_prompt="Fallback test.",
            timeout=10,
            streaming=True,
        )
        mock_skills.match_trigger.return_value = skill

        async def _empty_stream(**kwargs):
            raise ConnectionError("stream dead on arrival")
            yield

        mock_ollama.chat_stream = _empty_stream
        mock_ollama.chat = AsyncMock(
            return_value=ChatResponse(content="fallback response")
        )

        collected: list[str] = []
        async for chunk in engine.process_message_stream(111, "/fallback test"):
            collected.append(chunk)

        full = "".join(collected)
        assert "fallback response" in full


# ── 7. Routing decision with missing skill ──


class TestRoutingMissingSkill:
    @pytest.mark.asyncio
    async def test_skill_tier_with_none_skill_raises(
        self, engine: Engine, mock_skills
    ) -> None:
        """When routing yields tier=SKILL but skill=None, RuntimeError must be raised."""
        from core.engine import _RoutingDecision

        bad_decision = _RoutingDecision(tier=RoutingTier.SKILL, skill=None)

        with (
            patch.object(
                engine, "_decide_routing", new_callable=AsyncMock, return_value=bad_decision
            ),
            pytest.raises(RuntimeError, match="routing_decision_invalid.*missing skill"),
        ):
            await engine.process_message(111, "hello")

    @pytest.mark.asyncio
    async def test_skill_tier_with_none_skill_raises_stream(
        self, engine: Engine, mock_skills
    ) -> None:
        """Same scenario under streaming path must also raise RuntimeError."""
        from core.engine import _RoutingDecision

        bad_decision = _RoutingDecision(tier=RoutingTier.SKILL, skill=None)

        with (
            patch.object(
                engine, "_decide_routing", new_callable=AsyncMock, return_value=bad_decision
            ),
            pytest.raises(RuntimeError, match="routing_decision_invalid.*missing skill"),
        ):
            async for _ in engine.process_message_stream(111, "hello"):
                pass
