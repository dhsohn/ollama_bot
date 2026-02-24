"""엔진 모듈 테스트."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from core.config import AppSettings, BotConfig, OllamaConfig, SecurityConfig, MemoryConfig, TelegramConfig
from core.engine import Engine
from core.memory import MemoryManager
from core.skill_manager import SkillDefinition, SecurityLevel, SkillManager


@pytest.fixture
def app_settings() -> AppSettings:
    return AppSettings(
        telegram_bot_token="test",
        bot=BotConfig(max_conversation_length=10),
        ollama=OllamaConfig(
            system_prompt="You are a test bot.",
            model="test-model",
        ),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(),
    )


@pytest.fixture
def mock_ollama() -> AsyncMock:
    client = AsyncMock()
    client.default_model = "test-model"
    client.system_prompt = "You are a test bot."
    client.chat = AsyncMock(return_value="LLM response")
    client.health_check = AsyncMock(return_value={"status": "ok"})
    client.list_models = AsyncMock(return_value=[{"name": "test-model", "size": 1024}])
    return client


@pytest.fixture
def mock_skills() -> MagicMock:
    skills = MagicMock(spec=SkillManager)
    skills.match_trigger = MagicMock(return_value=None)
    skills.get_skill = MagicMock(return_value=None)
    skills.skill_count = 3
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
        ollama=mock_ollama,
        memory=memory,
        skills=mock_skills,
    )


class TestProcessMessage:
    @pytest.mark.asyncio
    async def test_free_conversation(self, engine: Engine, mock_ollama) -> None:
        result = await engine.process_message(111, "안녕하세요")
        assert result == "LLM response"
        mock_ollama.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_stored_in_memory(self, engine: Engine, memory: MemoryManager) -> None:
        await engine.process_message(111, "테스트 메시지")
        history = await memory.get_conversation(111)
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "테스트 메시지"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "LLM response"

    @pytest.mark.asyncio
    async def test_skill_triggered(self, engine: Engine, mock_skills, mock_ollama) -> None:
        skill = SkillDefinition(
            name="summarize",
            description="요약",
            triggers=["/summarize"],
            system_prompt="You are a summarizer.",
            timeout=30,
        )
        mock_skills.match_trigger.return_value = skill

        result = await engine.process_message(111, "/summarize 이 텍스트를 요약해줘")
        assert result == "LLM response"

        # 시스템 프롬프트가 스킬의 것으로 변경되었는지 확인
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = [m for m in messages if m["role"] == "system"]
        assert system_msg[0]["content"] == "You are a summarizer."

    @pytest.mark.asyncio
    async def test_context_includes_history(self, engine: Engine, memory: MemoryManager, mock_ollama) -> None:
        # 이전 대화 기록 추가
        await memory.add_message(111, "user", "이전 질문")
        await memory.add_message(111, "assistant", "이전 답변")

        await engine.process_message(111, "새 질문")

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # system + 이전 user + 이전 assistant + 현재 user = 4
        assert len(messages) >= 4

    @pytest.mark.asyncio
    async def test_model_override(self, engine: Engine, mock_ollama) -> None:
        await engine.process_message(111, "test", model_override="other-model")
        call_args = mock_ollama.chat.call_args
        assert call_args.kwargs.get("model") == "other-model"

    @pytest.mark.asyncio
    async def test_stream_uses_default_timeout(
        self, engine: Engine, mock_ollama, app_settings: AppSettings
    ) -> None:
        async def _stream():
            yield "chunk"

        mock_ollama.chat_stream = MagicMock(return_value=_stream())

        chunks = []
        async for chunk in engine.process_message_stream(111, "hello"):
            chunks.append(chunk)

        assert chunks == ["chunk"]
        call_args = mock_ollama.chat_stream.call_args
        assert call_args.kwargs.get("timeout") == app_settings.bot.response_timeout

    @pytest.mark.asyncio
    async def test_stream_uses_skill_timeout(self, engine: Engine, mock_skills, mock_ollama) -> None:
        async def _stream():
            yield "chunk"

        skill = SkillDefinition(
            name="summarize",
            description="요약",
            triggers=["/summarize"],
            system_prompt="You are a summarizer.",
            timeout=12,
        )
        mock_skills.match_trigger.return_value = skill
        mock_ollama.chat_stream = MagicMock(return_value=_stream())

        async for _ in engine.process_message_stream(111, "/summarize test"):
            pass

        call_args = mock_ollama.chat_stream.call_args
        assert call_args.kwargs.get("timeout") == 12


class TestExecuteSkill:
    @pytest.mark.asyncio
    async def test_execute_skill_programmatic(self, engine: Engine, mock_skills, mock_ollama) -> None:
        skill = SkillDefinition(
            name="summarize",
            description="요약",
            triggers=["/summarize"],
            system_prompt="You are a summarizer.",
            timeout=30,
        )
        mock_skills.get_skill.return_value = skill

        result = await engine.execute_skill("summarize", {"input_text": "테스트"})
        assert result == "LLM response"

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self, engine: Engine, mock_skills) -> None:
        mock_skills.get_skill.return_value = None
        result = await engine.execute_skill("nonexistent", {})
        assert "찾을 수 없습니다" in result


class TestChangeModel:
    @pytest.mark.asyncio
    async def test_change_model_success(self, engine: Engine, mock_ollama) -> None:
        result = await engine.change_model("test-model")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_change_model_not_found(self, engine: Engine) -> None:
        result = await engine.change_model("nonexistent-model")
        assert result["success"] is False


class TestPreferenceInjection:
    @pytest.mark.asyncio
    async def test_preferences_injected_into_system_prompt(
        self, engine: Engine, memory: MemoryManager, mock_ollama,
    ) -> None:
        """장기 메모리에 선호도가 있으면 시스템 프롬프트에 포함된다."""
        await memory.store_memory(111, "preferred_language", "한국어", category="preferences")
        await memory.store_memory(111, "response_style", "간결", category="preferences")

        await engine.process_message(111, "안녕")

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_content = messages[0]["content"]
        assert "사용자 고정 정보 및 선호도" in system_content
        assert "preferred_language" in system_content
        assert "한국어" in system_content
        assert "response_style" in system_content

    @pytest.mark.asyncio
    async def test_no_preferences_no_injection(
        self, engine: Engine, memory: MemoryManager, mock_ollama,
    ) -> None:
        """선호도가 없으면 시스템 프롬프트가 변경되지 않는다."""
        await engine.process_message(111, "안녕")

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_content = messages[0]["content"]
        assert "사용자 고정 정보 및 선호도" not in system_content


class TestProcessPrompt:
    @pytest.mark.asyncio
    async def test_process_prompt_forwards_format_and_options(
        self, engine: Engine, mock_ollama,
    ) -> None:
        """format, max_tokens, temperature가 ollama.chat()에 전달된다."""
        schema = {"type": "object", "properties": {"k": {"type": "string"}}}
        await engine.process_prompt(
            prompt="test",
            format=schema,
            max_tokens=256,
            temperature=0.2,
        )

        call_kwargs = mock_ollama.chat.call_args.kwargs
        assert call_kwargs["format"] is schema
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.2


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_get_status(self, engine: Engine) -> None:
        status = await engine.get_status()
        assert "uptime_seconds" in status
        assert "ollama" in status
        assert status["ollama"]["status"] == "ok"
        assert status["skills_loaded"] == 3
