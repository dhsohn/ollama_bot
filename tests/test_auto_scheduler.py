"""자동화 스케줄러 테스트."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import textwrap
from unittest.mock import AsyncMock, patch

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
    def test_scheduler_uses_configured_timezone(
        self,
        scheduler: AutoScheduler,
        app_settings: AppSettings,
    ) -> None:
        assert str(scheduler._scheduler.timezone) == app_settings.scheduler.timezone

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

        today = datetime.now(scheduler._timezone).strftime("%Y%m%d")
        output_file = auto_dir / "reports" / f"report_{today}.md"
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == "자동화 결과"

        scheduler._engine.process_prompt.assert_awaited_once()  # type: ignore[attr-defined]
        scheduler._telegram.send_message.assert_awaited_once()  # type: ignore[attr-defined]
        send_text = scheduler._telegram.send_message.call_args[0][1]  # type: ignore[attr-defined]
        assert send_text.startswith("⏰ 자동화: report")
        assert "*자동화:" not in send_text

    @pytest.mark.asyncio
    async def test_prompt_action_uses_configured_action_model_role(
        self,
        scheduler: AutoScheduler,
    ) -> None:
        auto = AutoDefinition(
            name="prompt_model_role",
            description="prompt role routing",
            schedule="* * * * *",
            action=AutoAction(
                type="prompt",
                target="상태 점검",
                model_role="reasoning",
                temperature=0.1,
                max_tokens=321,
            ),
        )

        await scheduler._run_action(auto)

        scheduler._engine.process_prompt.assert_awaited_once()  # type: ignore[attr-defined]
        call_kwargs = scheduler._engine.process_prompt.await_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["model_role"] == "reasoning"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 321

    @pytest.mark.asyncio
    async def test_callable_action_injects_model_kwargs_if_supported(
        self,
        scheduler: AutoScheduler,
    ) -> None:
        captured: dict[str, object] = {}

        async def my_func(
            greeting: str = "hello",
            model: str | None = None,
            model_role: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> str:
            captured["greeting"] = greeting
            captured["model"] = model
            captured["model_role"] = model_role
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            return "ok"

        scheduler.register_callable("with_model_args", my_func)
        auto = AutoDefinition(
            name="callable_model_args",
            description="callable kwargs injection",
            schedule="* * * * *",
            action=AutoAction(
                type="callable",
                target="with_model_args",
                parameters={"greeting": "world"},
                model="my-model",
                model_role="low_cost",
                temperature=0.4,
                max_tokens=700,
            ),
        )

        result = await scheduler._run_action(auto)
        assert result == "ok"
        assert captured["greeting"] == "world"
        assert captured["model"] == "my-model"
        assert captured["model_role"] == "low_cost"
        assert captured["temperature"] == 0.4
        assert captured["max_tokens"] == 700

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

    @pytest.mark.asyncio
    async def test_duplicate_automation_name_raises(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "_builtin" / "same_name.yaml",
            """
            name: "dup_auto"
            description: "builtin"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "A"
            """,
        )
        _write_auto_yaml(
            auto_dir / "custom" / "same_name.yaml",
            """
            name: "dup_auto"
            description: "custom"
            enabled: true
            schedule: "0 10 * * *"
            action:
              type: "prompt"
              target: "B"
            """,
        )

        with pytest.raises(ValueError, match="Duplicate automation name"):
            await scheduler.load_automations()

    @pytest.mark.asyncio
    async def test_invalid_cron_is_reported_and_skipped(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "ok.yaml",
            """
            name: "ok_auto"
            description: "정상 자동화"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "ok"
            """,
        )
        _write_auto_yaml(
            auto_dir / "custom" / "bad.yaml",
            """
            name: "bad_auto"
            description: "잘못된 cron"
            enabled: true
            schedule: "61 9 * * *"
            action:
              type: "prompt"
              target: "bad"
            """,
        )

        count = await scheduler.load_automations()
        assert count == 1
        errors = scheduler.get_last_load_errors()
        assert any("bad.yaml" in item for item in errors)

    @pytest.mark.asyncio
    async def test_invalid_cron_strict_mode_raises_and_keeps_previous_state(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "base.yaml",
            """
            name: "base_auto"
            description: "기준 자동화"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "base"
            """,
        )
        await scheduler.load_automations()
        assert scheduler._scheduler.get_job("auto_base_auto") is not None

        _write_auto_yaml(
            auto_dir / "custom" / "bad.yaml",
            """
            name: "bad_auto"
            description: "잘못된 cron"
            enabled: true
            schedule: "99 9 * * *"
            action:
              type: "prompt"
              target: "bad"
            """,
        )

        with pytest.raises(ValueError, match="strict mode"):
            await scheduler.load_automations(strict=True)

        autos = scheduler.list_automations()
        assert len(autos) == 1
        assert autos[0]["name"] == "base_auto"
        assert scheduler._scheduler.get_job("auto_base_auto") is not None

    @pytest.mark.asyncio
    async def test_load_automations_rolls_back_on_register_failure(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "base.yaml",
            """
            name: "base_auto"
            description: "기준 자동화"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "base"
            """,
        )
        await scheduler.load_automations()
        assert scheduler._scheduler.get_job("auto_base_auto") is not None

        (auto_dir / "custom" / "base.yaml").unlink()
        _write_auto_yaml(
            auto_dir / "custom" / "new.yaml",
            """
            name: "new_auto"
            description: "새 자동화"
            enabled: true
            schedule: "5 10 * * *"
            action:
              type: "prompt"
              target: "new"
            """,
        )

        original_register = scheduler._register_job

        def flaky_register(auto, trigger=None):
            if auto.name == "new_auto":
                raise RuntimeError("register boom")
            return original_register(auto, trigger=trigger)

        with patch.object(scheduler, "_register_job", side_effect=flaky_register):
            with pytest.raises(RuntimeError, match="register boom"):
                await scheduler.load_automations()

        autos = scheduler.list_automations()
        assert len(autos) == 1
        assert autos[0]["name"] == "base_auto"
        assert scheduler._scheduler.get_job("auto_base_auto") is not None
        assert scheduler._scheduler.get_job("auto_new_auto") is None

    @pytest.mark.asyncio
    async def test_empty_result_is_success_without_retry(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        scheduler._engine.process_prompt = AsyncMock(return_value="")  # type: ignore[attr-defined]

        _write_auto_yaml(
            auto_dir / "custom" / "empty_result.yaml",
            """
            name: "empty_result"
            description: "빈 결과 테스트"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "빈 문자열을 반환"
            output:
              send_to_telegram: true
            retry:
              max_attempts: 3
              delay_seconds: 1
            timeout: 30
            """,
        )

        await scheduler.load_automations()
        await scheduler._execute_automation("empty_result")

        scheduler._engine.process_prompt.assert_awaited_once()  # type: ignore[attr-defined]
        scheduler._telegram.send_message.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_save_to_file_uses_scheduler_timezone_for_date(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "tz_date.yaml",
            """
            name: "tz_date"
            description: "타임존 날짜 테스트"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "테스트"
            output:
              send_to_telegram: false
              save_to_file: "reports/tz_{date}.txt"
            retry:
              max_attempts: 1
              delay_seconds: 1
            timeout: 30
            """,
        )

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
                if tz is None:
                    return base.replace(tzinfo=None)
                return base.astimezone(tz)

        await scheduler.load_automations()
        with patch("core.auto_scheduler.datetime", _FixedDateTime):
            await scheduler._execute_automation("tz_date")

        # Asia/Seoul 기준이면 UTC 23:30은 다음날(2026-01-02)
        output_file = auto_dir / "reports" / "tz_20260102.txt"
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_failed_automation_sends_failure_notice(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        scheduler._engine.process_prompt = AsyncMock(side_effect=TimeoutError())  # type: ignore[attr-defined]

        _write_auto_yaml(
            auto_dir / "custom" / "fail_auto.yaml",
            """
            name: "fail_auto"
            description: "실패 알림 테스트"
            enabled: true
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "실패 유도"
            output:
              send_to_telegram: true
            retry:
              max_attempts: 2
              delay_seconds: 0
            timeout: 1
            """,
        )

        await scheduler.load_automations()
        ok = await scheduler._execute_automation("fail_auto")

        assert ok is False
        assert scheduler._engine.process_prompt.await_count == 2  # type: ignore[attr-defined]
        scheduler._telegram.send_message.assert_awaited_once()  # type: ignore[attr-defined]
        send_text = scheduler._telegram.send_message.call_args[0][1]  # type: ignore[attr-defined]
        assert "자동화 실패: fail_auto" in send_text
        assert "TimeoutError" in send_text

    @pytest.mark.asyncio
    async def test_run_automation_once_handles_missing_and_disabled(
        self,
        scheduler: AutoScheduler,
        auto_dir: Path,
    ) -> None:
        _write_auto_yaml(
            auto_dir / "custom" / "disabled.yaml",
            """
            name: "disabled_auto"
            description: "비활성 테스트"
            enabled: false
            schedule: "0 9 * * *"
            action:
              type: "prompt"
              target: "실행 안됨"
            """,
        )
        await scheduler.load_automations()

        assert await scheduler.run_automation_once("not_found") is False
        assert await scheduler.run_automation_once("disabled_auto") is False
