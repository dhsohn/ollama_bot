"""내장 자동화 callable 테스트."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from core.auto_scheduler import AutoAction, AutoDefinition, AutoScheduler
from core.automation_callables import (
    _DAILY_SUMMARY_SCHEMA,
    _MEMORY_HYGIENE_SCHEMA,
    _PREFERENCES_SCHEMA,
    _STALE_EVALUATION_SCHEMA,
    _TRIAGE_SCHEMA,
    register_builtin_callables,
)
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
        json_response = json.dumps(
            {"topics": ["테스트 주제"], "decisions": [], "todos": [], "notes": None},
            ensure_ascii=False,
        )
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=json_response)
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

        # 포매팅된 결과 검증
        assert "핵심 주제" in result
        assert "테스트 주제" in result
        assert "특이사항: 없음" in result

        engine.process_prompt.assert_awaited_once()
        call_kwargs = engine.process_prompt.await_args.kwargs
        prompt = call_kwargs["prompt"]
        assert "어제-사용자-메시지" in prompt
        assert "어제-봇-메시지" in prompt
        assert "오늘-메시지" not in prompt
        # JSON Schema format 전달 검증
        assert call_kwargs["format"] is _DAILY_SUMMARY_SCHEMA

    @pytest.mark.asyncio
    async def test_daily_summary_without_history_returns_empty_message(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value='{"topics":[],"decisions":[],"todos":[],"notes":null}')
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

    @pytest.mark.asyncio
    async def test_daily_summary_json_parse_fallback(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """JSON 파싱 실패 시 원본 텍스트를 사용한다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="이것은 JSON이 아닙니다")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

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

        assert "이것은 JSON이 아닙니다" in result

    @pytest.mark.asyncio
    async def test_daily_summary_unexpected_type_fallback(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """json.loads 결과가 dict가 아닐 때 원본을 사용한다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value='["이것은 리스트"]')
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

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

        assert '["이것은 리스트"]' in result

    @pytest.mark.asyncio
    async def test_daily_summary_invalid_field_types_fallback(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """필드 타입이 스키마와 다르면 원본 텍스트를 사용한다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value='{"topics":"문자열","decisions":[],"todos":[],"notes":null}',
        )
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

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

        assert '"topics":"문자열"' in result

    @pytest.mark.asyncio
    async def test_daily_summary_transcript_no_timestamp(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """transcript에 타임스탬프가 포함되지 않는다."""
        json_response = json.dumps(
            {"topics": ["주제"], "decisions": [], "todos": [], "notes": None},
            ensure_ascii=False,
        )
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=json_response)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        await _insert_yesterday_messages(memory_manager)

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
        await scheduler._run_action(auto)

        prompt = engine.process_prompt.await_args.kwargs["prompt"]
        # 타임스탬프 패턴 [20xx- 가 프롬프트에 없어야 한다
        assert "[20" not in prompt


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

    @pytest.mark.asyncio
    async def test_extract_preferences_limits_to_ten_items(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 10개 초과를 반환해도 상위 10개만 저장한다."""
        payload = [
            {"key": f"key_{idx}", "value": f"value_{idx}"}
            for idx in range(12)
        ]
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value=json.dumps(payload, ensure_ascii=False),
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
        assert len(prefs) == 10
        assert {item["key"] for item in prefs} == {f"key_{idx}" for idx in range(10)}


class TestErrorLogTriageCallable:
    @pytest.mark.asyncio
    async def test_analyzes_error_logs(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """에러 로그를 읽어 JSON 분석 결과를 포매팅하여 반환한다."""
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

        triage_json = json.dumps([{
            "event": "ollama_retry",
            "severity": "urgent",
            "cause": "연결 거부",
            "action": "Ollama 서비스 재시작",
            "recurring": True,
        }], ensure_ascii=False)
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=triage_json)
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
        assert "ollama_retry" in result
        assert "연결 거부" in result
        assert "반복 패턴: 예" in result
        engine.process_prompt.assert_awaited_once()
        call_kwargs = engine.process_prompt.await_args.kwargs
        assert "ollama_retry" in call_kwargs["prompt"]
        assert call_kwargs["format"] is _TRIAGE_SCHEMA

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

    @pytest.mark.asyncio
    async def test_error_log_triage_json_parse_fallback(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """JSON 파싱 실패 시 원본 텍스트를 사용한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        log_entry = json.dumps({
            "event": "test_error",
            "log_level": "error",
            "timestamp": now.isoformat(),
        })
        (log_dir / "app.log").write_text(log_entry + "\n")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="이것은 JSON이 아닙니다")
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

        assert "이것은 JSON이 아닙니다" in result

    @pytest.mark.asyncio
    async def test_error_log_triage_unexpected_type_fallback(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """json.loads 결과가 list/dict가 아닐 때 원본을 사용한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        log_entry = json.dumps({
            "event": "test_error",
            "log_level": "error",
            "timestamp": now.isoformat(),
        })
        (log_dir / "app.log").write_text(log_entry + "\n")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value='"단순 문자열"')
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

        assert '"단순 문자열"' in result

    @pytest.mark.asyncio
    async def test_error_log_triage_non_boolean_recurring_unknown(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """recurring이 bool이 아니면 '?'로 표시한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        log_entry = json.dumps({
            "event": "test_error",
            "log_level": "error",
            "timestamp": now.isoformat(),
        })
        (log_dir / "app.log").write_text(log_entry + "\n")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value=json.dumps([{
                "event": "test_error",
                "severity": "warning",
                "cause": "원인",
                "action": "조치",
                "recurring": "false",
            }], ensure_ascii=False),
        )
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

        assert "반복 패턴: ?" in result

    @pytest.mark.asyncio
    async def test_error_log_triage_uses_latest_sample_when_limited(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """max_errors 제한 시 최신 샘플부터 분석한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        old_entry = json.dumps({
            "event": "old_event",
            "log_level": "error",
            "timestamp": (now - timedelta(minutes=10)).isoformat(),
        })
        new_entry = json.dumps({
            "event": "new_event",
            "log_level": "error",
            "timestamp": now.isoformat(),
        })
        (log_dir / "app.log").write_text(old_entry + "\n" + new_entry + "\n")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="[]")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 24, "max_errors": 1},
            ),
        )
        result = await scheduler._run_action(auto)

        call_kwargs = engine.process_prompt.await_args.kwargs
        assert "new_event" in call_kwargs["prompt"]
        assert "old_event" not in call_kwargs["prompt"]
        assert "분석 샘플: 1건" in result

    @pytest.mark.asyncio
    async def test_error_log_triage_invalid_hours_back_raises(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """hours_back <= 0이면 ValueError를 발생시킨다."""
        engine = AsyncMock()
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 0, "max_errors": 50},
            ),
        )
        with pytest.raises(ValueError, match="hours_back must be > 0"):
            await scheduler._run_action(auto)
        engine.process_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_log_triage_invalid_max_errors_raises(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """max_errors <= 0이면 ValueError를 발생시킨다."""
        engine = AsyncMock()
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        auto = AutoDefinition(
            name="error_log_triage",
            description="오류 분석",
            schedule="0 */6 * * *",
            action=AutoAction(
                type="callable",
                target="error_log_triage",
                parameters={"hours_back": 6, "max_errors": 0},
            ),
        )
        with pytest.raises(ValueError, match="max_errors must be > 0"):
            await scheduler._run_action(auto)
        engine.process_prompt.assert_not_awaited()


class TestExtractPreferencesSchemaFormat:
    @pytest.mark.asyncio
    async def test_extract_preferences_uses_schema_format(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """format에 _PREFERENCES_SCHEMA dict가 전달된다."""
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value='[{"key": "lang", "value": "ko"}]',
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

        call_kwargs = engine.process_prompt.await_args.kwargs
        assert call_kwargs["format"] is _PREFERENCES_SCHEMA
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temperature"] == 0.3
        assert _PREFERENCES_SCHEMA["maxItems"] == 10


# ── health_check 테스트 ──


def _health_check_auto(**overrides) -> AutoDefinition:
    """health_check AutoDefinition 생성 헬퍼."""
    params = {"disk_warn_pct": 85, "error_hours_back": 1}
    params.update(overrides)
    return AutoDefinition(
        name="health_check",
        description="시스템 점검",
        schedule="*/30 * * * *",
        action=AutoAction(
            type="callable",
            target="health_check",
            parameters=params,
        ),
    )


def _ok_status() -> dict:
    """정상 상태 engine.get_status() 반환값."""
    return {
        "uptime_seconds": 3600,
        "uptime_human": "1시간 0분 0초",
        "ollama": {
            "status": "ok",
            "host": "http://localhost:11434",
            "models_count": 3,
            "models": ["m1", "m2", "m3"],
            "default_model": "m1",
            "default_model_available": True,
        },
        "skills_loaded": 4,
        "current_model": "m1",
    }


class TestHealthCheckCallable:
    @pytest.mark.asyncio
    async def test_all_ok_returns_empty(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """모든 점검 통과 시 빈 문자열을 반환한다."""
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=_ok_status())
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_health_check_auto())

        assert result == ""

    @pytest.mark.asyncio
    async def test_ollama_error(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """Ollama 상태 error 시 보고서에 포함된다."""
        status = _ok_status()
        status["ollama"] = {"status": "error", "error": "connection refused"}
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=status)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_health_check_auto())

        assert "Ollama" in result
        assert "오류" in result
        assert "connection refused" in result

    @pytest.mark.asyncio
    async def test_default_model_unavailable(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """기본 모델 사용 불가 시 경고가 포함된다."""
        status = _ok_status()
        status["ollama"]["default_model_available"] = False
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=status)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_health_check_auto())

        assert "기본 모델 사용 불가" in result

    @pytest.mark.asyncio
    async def test_db_failure(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """DB ping 실패 시 오류가 보고된다."""
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=_ok_status())
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        # DB 연결을 닫아 ping 실패 유도
        await memory_manager.close()

        result = await scheduler._run_action(_health_check_auto())

        assert "데이터베이스" in result
        assert "오류" in result

    @pytest.mark.asyncio
    async def test_disk_warning(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """디스크 사용률이 임계값 초과 시 경고가 포함된다."""
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=_ok_status())
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        from collections import namedtuple
        DiskUsage = namedtuple("usage", ["total", "used", "free"])
        fake_usage = DiskUsage(
            total=100_000_000, used=90_000_000, free=10_000_000,
        )
        with patch("shutil.disk_usage", return_value=fake_usage):
            result = await scheduler._run_action(_health_check_auto())

        assert "디스크" in result
        assert "90%" in result

    @pytest.mark.asyncio
    async def test_error_logs_found(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """에러 로그가 있으면 건수가 보고된다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        for i in range(3):
            entry = json.dumps({
                "event": f"test_error_{i}",
                "log_level": "error",
                "timestamp": now.isoformat(),
            })
            with open(log_dir / "app.log", "a") as f:
                f.write(entry + "\n")

        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=_ok_status())
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_health_check_auto())

        assert "오류 로그" in result
        assert "3건" in result

    @pytest.mark.asyncio
    async def test_multiple_issues(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """복합 장애 시 모든 항목이 보고서에 포함된다."""
        status = _ok_status()
        status["ollama"] = {"status": "error", "error": "timeout"}
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=status)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        # 에러 로그도 추가
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        entry = json.dumps({
            "event": "test_error",
            "log_level": "error",
            "timestamp": now.isoformat(),
        })
        (log_dir / "app.log").write_text(entry + "\n")

        result = await scheduler._run_action(_health_check_auto())

        assert "Ollama" in result
        assert "오류 로그" in result
        assert "시스템 상태 점검" in result

    @pytest.mark.asyncio
    async def test_no_log_dir(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """로그 디렉토리가 없어도 크래시 없이 0건으로 처리한다."""
        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=_ok_status())
        # data_dir을 존재하지 않는 경로로 등록
        scheduler.set_dependencies(engine=engine, telegram=AsyncMock())
        register_builtin_callables(
            scheduler=scheduler,
            engine=engine,
            memory=memory_manager,
            allowed_users=app_settings.security.allowed_users,
            data_dir=str(Path(app_settings.data_dir) / "nonexistent"),
        )

        result = await scheduler._run_action(_health_check_auto())

        # 로그 디렉토리 없음은 이상이 아니라 0건 처리
        # 다른 항목이 모두 정상이면 빈 문자열
        assert result == "" or "오류 0건" in result

    @pytest.mark.asyncio
    async def test_malformed_log_lines(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """파손된 JSON 라인은 건너뛰고 유효 에러만 카운트한다."""
        log_dir = Path(app_settings.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        content = (
            "not valid json\n"
            "another broken line { incomplete\n"
            + json.dumps({
                "event": "real_error",
                "log_level": "error",
                "timestamp": now.isoformat(),
            }) + "\n"
        )
        (log_dir / "app.log").write_text(content)

        engine = AsyncMock()
        engine.get_status = AsyncMock(return_value=_ok_status())
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_health_check_auto())

        assert "오류 로그" in result
        assert "1건" in result


# ── memory_hygiene 테스트 ──


def _memory_hygiene_auto(**overrides) -> AutoDefinition:
    """memory_hygiene AutoDefinition 생성 헬퍼."""
    params = {"stale_days": 90, "max_llm_calls": 3}
    params.update(overrides)
    return AutoDefinition(
        name="memory_hygiene",
        description="메모리 정리",
        schedule="30 3 * * *",
        action=AutoAction(
            type="callable",
            target="memory_hygiene",
            parameters=params,
        ),
    )


class TestMemoryHygieneCallable:
    @pytest.mark.asyncio
    async def test_no_memories_returns_empty(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """메모리가 없으면 빈 문자열을 반환한다."""
        engine = AsyncMock()
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert result == ""
        engine.process_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exact_duplicates_removed(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """value가 동일한 항목 중 최신만 유지한다."""
        await memory_manager.store_memory(111, "food_kr", "피자", "preferences")
        await memory_manager.store_memory(111, "food_en", "피자", "preferences")

        # food_kr을 더 오래된 것으로 백데이트
        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE long_term_memory SET updated_at = '2020-01-01 00:00:00' "
            "WHERE chat_id = 111 AND key = 'food_kr'",
        )
        await memory_manager._db.commit()

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="[]")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert "중복 제거: 1건" in result
        prefs = await memory_manager.recall_memory(111)
        keys = {p["key"] for p in prefs}
        assert "food_en" in keys
        assert "food_kr" not in keys

    @pytest.mark.asyncio
    async def test_semantic_duplicates_via_llm(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 의미 중복을 판별하면 삭제한다."""
        await memory_manager.store_memory(111, "favorite_food", "피자", "preferences")
        await memory_manager.store_memory(111, "preferred_cuisine", "이탈리안", "preferences")

        llm_response = json.dumps([{
            "keep_key": "favorite_food",
            "delete_key": "preferred_cuisine",
            "reason": "duplicate",
        }], ensure_ascii=False)
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert "중복 제거: 1건" in result
        prefs = await memory_manager.recall_memory(111)
        keys = {p["key"] for p in prefs}
        assert "favorite_food" in keys
        assert "preferred_cuisine" not in keys

    @pytest.mark.asyncio
    async def test_conflict_detection(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 충돌을 판별하면 삭제하고 충돌 해소로 카운트한다."""
        await memory_manager.store_memory(111, "fav_color", "빨강", "preferences")
        await memory_manager.store_memory(111, "fav_colour", "파랑", "preferences")

        llm_response = json.dumps([{
            "keep_key": "fav_colour",
            "delete_key": "fav_color",
            "reason": "conflict",
        }], ensure_ascii=False)
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert "충돌 해소: 1건" in result
        prefs = await memory_manager.recall_memory(111)
        keys = {p["key"] for p in prefs}
        assert "fav_colour" in keys
        assert "fav_color" not in keys

    @pytest.mark.asyncio
    async def test_stale_cleanup(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """오래된 항목을 LLM이 stale 판정하면 삭제한다."""
        await memory_manager.store_memory(111, "old_pref", "값", "preferences")
        await memory_manager.store_memory(111, "recent_pref", "최근값", "preferences")

        # old_pref만 100일 전으로 백데이트
        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE long_term_memory SET updated_at = '2020-01-01 00:00:00' "
            "WHERE chat_id = 111 AND key = 'old_pref'",
        )
        await memory_manager._db.commit()

        stale_response = json.dumps([{
            "key": "old_pref",
            "stale": True,
            "reason": "일시적 선호",
        }], ensure_ascii=False)
        engine = AsyncMock()
        # Phase 2 (semantic) → 빈 배열, Phase 3 (stale) → stale 판정
        engine.process_prompt = AsyncMock(
            side_effect=["[]", stale_response],
        )
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert "오래된 항목 정리: 1건" in result
        prefs = await memory_manager.recall_memory(111)
        keys = {p["key"] for p in prefs}
        assert "recent_pref" in keys
        assert "old_pref" not in keys

    @pytest.mark.asyncio
    async def test_stale_retained(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 stale=false로 판정하면 유지한다."""
        await memory_manager.store_memory(111, "name", "홍길동", "preferences")

        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE long_term_memory SET updated_at = '2020-01-01 00:00:00' "
            "WHERE chat_id = 111 AND key = 'name'",
        )
        await memory_manager._db.commit()

        stale_response = json.dumps([{
            "key": "name",
            "stale": False,
            "reason": "이름은 영구 정보",
        }], ensure_ascii=False)
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            side_effect=["[]", stale_response],
        )
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        # stale 제거 0건, 변경 없으므로 빈 문자열
        assert result == ""
        prefs = await memory_manager.recall_memory(111)
        assert len(prefs) == 1
        assert prefs[0]["key"] == "name"

    @pytest.mark.asyncio
    async def test_respects_llm_budget_zero(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """max_llm_calls=0이면 LLM 호출 없이 정확 중복만 처리한다."""
        await memory_manager.store_memory(111, "a", "같은값", "preferences")
        await memory_manager.store_memory(111, "b", "같은값", "preferences")

        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE long_term_memory SET updated_at = '2020-01-01 00:00:00' "
            "WHERE chat_id = 111 AND key = 'a'",
        )
        await memory_manager._db.commit()

        engine = AsyncMock()
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(
            _memory_hygiene_auto(max_llm_calls=0),
        )

        assert "중복 제거: 1건" in result
        engine.process_prompt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_json_failure(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM JSON 파싱 실패 시 정확 중복만 처리하고 에러 없다."""
        await memory_manager.store_memory(111, "x", "동일", "preferences")
        await memory_manager.store_memory(111, "y", "동일", "preferences")

        assert memory_manager._db is not None
        await memory_manager._db.execute(
            "UPDATE long_term_memory SET updated_at = '2020-01-01 00:00:00' "
            "WHERE chat_id = 111 AND key = 'x'",
        )
        await memory_manager._db.commit()

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="이것은 JSON이 아닙니다")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert "중복 제거: 1건" in result

    @pytest.mark.asyncio
    async def test_no_changes_returns_empty(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """중복/충돌/노후 없으면 빈 문자열을 반환한다."""
        await memory_manager.store_memory(111, "unique_key", "고유값", "preferences")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="[]")
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        assert result == ""

    @pytest.mark.asyncio
    async def test_multi_user_budget_exhaustion(
        self,
        tmp_path: Path,
        security_manager: SecurityManager,
    ) -> None:
        """user 2명 + max_llm_calls=1 → user A Phase 2 소진, user B는 Phase 1만."""
        # 두 사용자용 설정
        settings = AppSettings(
            telegram_bot_token="test_token",
            data_dir=str(tmp_path),
            bot=BotConfig(),
            ollama=OllamaConfig(),
            security=SecurityConfig(allowed_users=[111, 222]),
            memory=MemoryConfig(),
            telegram=TelegramConfig(),
        )
        auto_dir = tmp_path / "auto"
        (auto_dir / "_builtin").mkdir(parents=True)
        (auto_dir / "custom").mkdir(parents=True)
        sched = AutoScheduler(
            config=settings, security=security_manager, auto_dir=str(auto_dir),
        )

        mm = MemoryManager(MemoryConfig(), str(tmp_path), max_conversation_length=20)
        await mm.initialize()
        try:
            # 각 사용자에 의미 중복 후보 저장
            await mm.store_memory(111, "a", "val_a", "preferences")
            await mm.store_memory(111, "b", "val_b", "preferences")
            await mm.store_memory(222, "c", "val_c", "preferences")
            await mm.store_memory(222, "d", "val_d", "preferences")

            llm_response = json.dumps([{
                "keep_key": "a",
                "delete_key": "b",
                "reason": "duplicate",
            }], ensure_ascii=False)
            engine = AsyncMock()
            engine.process_prompt = AsyncMock(return_value=llm_response)
            sched.set_dependencies(engine=engine, telegram=AsyncMock())
            register_builtin_callables(
                scheduler=sched,
                engine=engine,
                memory=mm,
                allowed_users=settings.security.allowed_users,
                data_dir=settings.data_dir,
            )

            result = await sched._run_action(
                _memory_hygiene_auto(max_llm_calls=1),
            )

            # LLM은 1번만 호출 (user 111의 Phase 2)
            assert engine.process_prompt.await_count == 1
            # user 111의 중복은 LLM으로 처리됨
            assert "중복 제거: 1건" in result
            # user 222의 메모리는 그대로 (LLM 예산 소진)
            prefs_222 = await mm.recall_memory(222)
            assert len(prefs_222) == 2
        finally:
            await mm.close()

    @pytest.mark.asyncio
    async def test_invalid_delete_key_ignored(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
        memory_manager: MemoryManager,
    ) -> None:
        """LLM이 존재하지 않는 키를 반환하면 무시한다."""
        await memory_manager.store_memory(111, "real_key", "값", "preferences")

        llm_response = json.dumps([{
            "keep_key": "real_key",
            "delete_key": "nonexistent_key",
            "reason": "duplicate",
        }], ensure_ascii=False)
        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup_callables(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_memory_hygiene_auto())

        # 삭제 대상이 없으므로 변경 없음 → 빈 문자열
        assert result == ""
        prefs = await memory_manager.recall_memory(111)
        assert len(prefs) == 1
