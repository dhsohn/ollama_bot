"""내장 자동화 callable 테스트."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.auto_scheduler import AutoAction, AutoDefinition, AutoScheduler
from core.automation_callables import register_builtin_callables
from core.config import (
    AppSettings,
    BotConfig,
    MemoryConfig,
    OllamaConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.memory import MemoryManager
from core.security import SecurityManager


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        telegram_bot_token="test_token",
        data_dir=str(tmp_path),
        bot=BotConfig(),
        ollama=OllamaConfig(),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(),
    )


@pytest.fixture
def security_manager() -> SecurityManager:
    return SecurityManager(SecurityConfig(allowed_users=[111], rate_limit=30))


@pytest_asyncio.fixture
async def memory_manager(tmp_path: Path) -> MemoryManager:
    mm = MemoryManager(MemoryConfig(), str(tmp_path), max_conversation_length=20)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
def scheduler(
    app_settings: AppSettings,
    security_manager: SecurityManager,
    tmp_path: Path,
) -> AutoScheduler:
    auto_dir = tmp_path / "auto"
    (auto_dir / "_builtin").mkdir(parents=True)
    (auto_dir / "custom").mkdir(parents=True)
    return AutoScheduler(
        config=app_settings,
        security=security_manager,
        auto_dir=str(auto_dir),
    )


class TestDailySummaryCallable:
    @pytest.mark.asyncio
    async def test_daily_summary_uses_previous_day_history(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="요약 결과")
        scheduler.set_dependencies(engine=engine, telegram=AsyncMock())
        register_builtin_callables(
            scheduler=scheduler,
            engine=engine,
            memory=memory_manager,
            allowed_users=app_settings.security.allowed_users,
            data_dir=app_settings.data_dir,
        )

        await memory_manager.add_message(111, "user", "어제-사용자-메시지")
        await memory_manager.add_message(111, "assistant", "어제-봇-메시지")
        await memory_manager.add_message(111, "user", "오늘-메시지")

        assert memory_manager._db is not None
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
            (yesterday.replace(hour=9, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S"), 111, "어제-사용자-메시지"),
        )
        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
            (yesterday.replace(hour=18, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S"), 111, "어제-봇-메시지"),
        )
        await memory_manager._db.execute(
            "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
            (now.replace(hour=9, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S"), 111, "오늘-메시지"),
        )
        await memory_manager._db.commit()

        auto = AutoDefinition(
            name="daily_summary",
            description="일일 요약",
            schedule="0 9 * * *",
            action=AutoAction(
                type="callable",
                target="daily_summary",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        result = await scheduler._run_action(auto)

        assert "요약 결과" in result
        engine.process_prompt.assert_awaited_once()
        prompt = engine.process_prompt.await_args.kwargs["prompt"]
        assert "어제-사용자-메시지" in prompt
        assert "어제-봇-메시지" in prompt
        assert "오늘-메시지" not in prompt

    @pytest.mark.asyncio
    async def test_daily_summary_without_history_returns_empty_message(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="요약 결과")
        scheduler.set_dependencies(engine=engine, telegram=AsyncMock())
        register_builtin_callables(
            scheduler=scheduler,
            engine=engine,
            memory=memory_manager,
            allowed_users=app_settings.security.allowed_users,
            data_dir=app_settings.data_dir,
        )

        auto = AutoDefinition(
            name="daily_summary",
            description="일일 요약",
            schedule="0 9 * * *",
            action=AutoAction(
                type="callable",
                target="daily_summary",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        result = await scheduler._run_action(auto)

        assert "요약할 대화 기록이 없습니다" in result
        engine.process_prompt.assert_not_awaited()


def _setup_callables(scheduler, engine, memory_manager, app_settings):
    """공통 callable 등록 헬퍼."""
    scheduler.set_dependencies(engine=engine, telegram=AsyncMock())
    register_builtin_callables(
        scheduler=scheduler,
        engine=engine,
        memory=memory_manager,
        allowed_users=app_settings.security.allowed_users,
        data_dir=app_settings.data_dir,
    )


async def _insert_yesterday_messages(memory_manager, chat_id=111):
    """어제 날짜의 테스트 메시지를 삽입한다."""
    await memory_manager.add_message(chat_id, "user", "한국어로 대답해줘")
    await memory_manager.add_message(chat_id, "assistant", "네, 한국어로 답변하겠습니다")

    assert memory_manager._db is not None
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    await memory_manager._db.execute(
        "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
        (yesterday.replace(hour=10, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S"), chat_id, "한국어로 대답해줘"),
    )
    await memory_manager._db.execute(
        "UPDATE conversations SET timestamp = ? WHERE chat_id = ? AND content = ?",
        (yesterday.replace(hour=10, minute=1, second=0).strftime("%Y-%m-%d %H:%M:%S"), chat_id, "네, 한국어로 답변하겠습니다"),
    )
    await memory_manager._db.commit()


class TestExtractPreferencesCallable:
    @pytest.mark.asyncio
    async def test_extracts_preferences_from_conversations(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 유효 JSON 반환 시 선호도가 장기 메모리에 저장된다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value='[{"key": "preferred_language", "value": "한국어"}]',
        )
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

        auto = AutoDefinition(
            name="preference_extraction",
            description="선호도 추출",
            schedule="0 0 * * *",
            action=AutoAction(
                type="callable",
                target="extract_preferences",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        result = await scheduler._run_action(auto)

        assert "추출 완료" in result
        prefs = await memory_manager.recall_memory(111, category="preferences")
        assert len(prefs) == 1
        assert prefs[0]["key"] == "preferred_language"
        assert prefs[0]["value"] == "한국어"

    @pytest.mark.asyncio
    async def test_handles_empty_conversations(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """대화 기록이 없으면 LLM 호출 안 함."""
        engine = AsyncMock()
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="preference_extraction",
            description="선호도 추출",
            schedule="0 0 * * *",
            action=AutoAction(
                type="callable",
                target="extract_preferences",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        result = await scheduler._run_action(auto)

        assert "대화 기록이 없습니다" in result
        engine.process_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_json(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 잘못된 JSON을 반환해도 에러 없이 건너뛴다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="이것은 JSON이 아닙니다")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

        auto = AutoDefinition(
            name="preference_extraction",
            description="선호도 추출",
            schedule="0 0 * * *",
            action=AutoAction(
                type="callable",
                target="extract_preferences",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        result = await scheduler._run_action(auto)

        prefs = await memory_manager.recall_memory(111, category="preferences")
        assert len(prefs) == 0
        assert "JSON 파싱 실패" in result

    @pytest.mark.asyncio
    async def test_upserts_existing_preferences(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """이미 존재하는 선호도 키를 업데이트한다."""
        await memory_manager.store_memory(111, "preferred_language", "영어", category="preferences")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value='[{"key": "preferred_language", "value": "한국어"}]',
        )
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

        auto = AutoDefinition(
            name="preference_extraction",
            description="선호도 추출",
            schedule="0 0 * * *",
            action=AutoAction(
                type="callable",
                target="extract_preferences",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        await scheduler._run_action(auto)

        prefs = await memory_manager.recall_memory(111, key="preferred_language", category="preferences")
        assert len(prefs) == 1
        assert prefs[0]["value"] == "한국어"

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 코드 펜스로 감싼 JSON도 파싱한다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value='```json\n[{"key": "occupation", "value": "개발자"}]\n```',
        )
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

        auto = AutoDefinition(
            name="preference_extraction",
            description="선호도 추출",
            schedule="0 0 * * *",
            action=AutoAction(
                type="callable",
                target="extract_preferences",
                parameters={"days_ago": 1, "timezone_name": "UTC"},
            ),
        )
        await scheduler._run_action(auto)

        prefs = await memory_manager.recall_memory(111, category="preferences")
        assert len(prefs) == 1
        assert prefs[0]["key"] == "occupation"


class TestErrorLogTriageCallable:
    @pytest.mark.asyncio
    async def test_analyzes_error_logs(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """에러 로그를 읽어 LLM 분석 결과를 반환한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        log_entry = json.dumps({
            "event": "ollama_retry",
            "log_level": "error",
            "timestamp": now.isoformat(),
            "error": "connection refused",
        })
        (log_dir / "app.log").write_text(log_entry + "\n")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="🔴 긴급: Ollama 연결 실패")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 24, "max_errors": 50},
            ),
        )
        result = await scheduler._run_action(auto)

        assert "오류 로그 분석" in result
        engine.process_prompt.assert_awaited_once()
        prompt = engine.process_prompt.await_args.kwargs.get("prompt") or engine.process_prompt.await_args[0][0]
        assert "ollama_retry" in prompt

    @pytest.mark.asyncio
    async def test_no_errors_returns_empty(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """에러가 없으면 빈 문자열을 반환한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        log_entry = json.dumps({
            "event": "bot_running",
            "log_level": "info",
            "timestamp": now.isoformat(),
        })
        (log_dir / "app.log").write_text(log_entry + "\n")

        engine = AsyncMock()
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 24, "max_errors": 50},
            ),
        )
        result = await scheduler._run_action(auto)

        assert result == ""
        engine.process_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_log_dir_returns_empty(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """로그 디렉토리가 없으면 빈 문자열을 반환한다."""
        engine = AsyncMock()
        scheduler.set_dependencies(engine=engine, telegram=AsyncMock())
        register_builtin_callables(
            scheduler=scheduler,
            engine=engine,
            memory=memory_manager,
            allowed_users=app_settings.security.allowed_users,
            data_dir=str(Path(app_settings.data_dir) / "nonexistent"),
        )

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 6, "max_errors": 50},
            ),
        )
        result = await scheduler._run_action(auto)
        assert result == ""

    @pytest.mark.asyncio
    async def test_handles_malformed_log_lines(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """잘못된 JSON 라인은 건너뛴다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        content = (
            "not valid json\n"
            + json.dumps({
                "event": "test_error",
                "log_level": "error",
                "timestamp": now.isoformat(),
            }) + "\n"
        )
        (log_dir / "app.log").write_text(content)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="분석 결과")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 24, "max_errors": 50},
            ),
        )
        result = await scheduler._run_action(auto)

        assert "오류 로그 분석" in result
        engine.process_prompt.assert_awaited_once()
