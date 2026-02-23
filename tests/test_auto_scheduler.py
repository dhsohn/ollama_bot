"""자동화 스케줄러 테스트."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import textwrap
from unittest.mock import AsyncMock

import pytest

from core.auto_scheduler import AutoAction, AutoDefinition, AutoScheduler
from core.config import AppSettings, BotConfig, MemoryConfig, OllamaConfig, SecurityConfig, TelegramConfig
from core.security import SecurityManager


def _write_auto_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


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


@pytest.fixture
def auto_dir(tmp_path: Path) -> Path:
    (tmp_path / "_builtin").mkdir()
    (tmp_path / "custom").mkdir()
    return tmp_path


@pytest.fixture
def scheduler(
    app_settings: AppSettings,
    security_manager: SecurityManager,
    auto_dir: Path,
) -> AutoScheduler:
    instance = AutoScheduler(
        config=app_settings,
        security=security_manager,
        auto_dir=str(auto_dir),
    )
    engine = AsyncMock()
    engine.process_prompt = AsyncMock(return_value="자동화 결과")
    engine.execute_skill = AsyncMock(return_value="스킬 결과")
    telegram = AsyncMock()
    telegram.send_message = AsyncMock()
    instance.set_dependencies(engine=engine, telegram=telegram)
    return instance


class TestAutoScheduler:
    @pytest.mark.asyncio
    async def test_load_and_list_automations(self, scheduler: AutoScheduler, auto_dir: Path) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "daily.yaml",
            """
            name: "daily"
            description: "일일 보고"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "오늘 상태 요약"
            output:
              send_to_telegram: false
            """,
        )

        count = await scheduler.load_automations()
        autos = scheduler.list_automations()

        assert count == 1
        assert len(autos) == 1
        assert autos[0]["name"] == "daily"
        assert autos[0]["enabled"] is True

    @pytest.mark.asyncio
    async def test_execute_prompt_automation_delivers_outputs(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "report.yaml",
            """
            name: "report"
            description: "상태 리포트"
            enabled: true
            schedule: "*/30 * * * *"
            action:
              type: "prompt"
              target: "상태 점검"
            output:
              send_to_telegram: true
              save_to_file: "reports/report_{date}.md"
            retry:
              max_attempts: 1
              delay_seconds: 1
            timeout: 30
            """,
        )

        await scheduler.load_automations()
        await scheduler._execute_automation("report")

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_file = auto_dir / "reports" / f"report_{today}.md"
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == "자동화 결과"

        scheduler._engine.process_prompt.assert_awaited_once()  # type: ignore[attr-defined]
        scheduler._telegram.send_message.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_command_action_is_disabled(self, scheduler: AutoScheduler) -> None:
        auto = AutoDefinition(
            name="cmd_test",
            description="명령 실행 테스트",
            schedule="* * * * *",
            action=AutoAction(type="command", target="echo hello"),
        )
        result = await scheduler._run_action(auto)
        assert "비활성화" in result

    @pytest.mark.asyncio
    async def test_register_and_execute_async_callable(
        self, scheduler: AutoScheduler, auto_dir: Path
    ) -> None:
        async def my_func(greeting: str = "hello") -> str:
            return f"result: {greeting}"

        scheduler.register_callable("test_func", my_func)

        _write_auto_yaml(
            auto_dir / "custom" / "call_test.yaml",
            """
            name: "call_test"
            description: "callable 테스트"
            enabled: true
            schedule: "0 12 * * *"
            action:
              type: "callable"
              target: "test_func"
              parameters:
                greeting: "world"
            output:
              send_to_telegram: true
            retry:
              max_attempts: 1
              delay_seconds: 1
            timeout: 10
            """,
        )

        await scheduler.load_automations()
        await scheduler._execute_automation("call_test")

        scheduler._telegram.send_message.assert_awaited_once()
        call_args = scheduler._telegram.send_message.call_args
        assert "result: world" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_register_and_execute_sync_callable(self, scheduler: AutoScheduler) -> None:
        def sync_func(x: str = "a") -> str:
            return f"sync: {x}"

        scheduler.register_callable("sync_fn", sync_func)
        auto = AutoDefinition(
            name="sync_test",
            description="sync callable",
            schedule="* * * * *",
            action=AutoAction(type="callable", target="sync_fn", parameters={"x": "b"}),
        )
        result = await scheduler._run_action(auto)
        assert result == "sync: b"

    @pytest.mark.asyncio
    async def test_callable_not_registered_raises(self, scheduler: AutoScheduler) -> None:
        auto = AutoDefinition(
            name="missing_test",
            description="missing callable",
            schedule="* * * * *",
            action=AutoAction(type="callable", target="nonexistent"),
        )
        with pytest.raises(ValueError, match="not registered"):
            await scheduler._run_action(auto)

    def test_register_callable_non_callable_raises(self, scheduler: AutoScheduler) -> None:
        with pytest.raises(TypeError, match="expects a callable"):
            scheduler.register_callable("bad", "not a function")  # type: ignore[arg-type]
