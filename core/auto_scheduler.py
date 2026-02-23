"""자동화 스케줄러 — YAML 기반 cron 작업 관리.

auto/_builtin/ 및 auto/custom/ 디렉토리의 YAML 파일을 로드하여
APScheduler cron 작업으로 등록하고 실행한다.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, Field, field_validator

from core.config import AppSettings
from core.logging_setup import get_logger
from core.security import SecurityManager


class AutoAction(BaseModel):
    type: str  # "skill" | "command" | "prompt"
    target: str
    parameters: dict = Field(default_factory=dict)


class AutoOutput(BaseModel):
    send_to_telegram: bool = True
    save_to_file: str | None = None


class AutoRetry(BaseModel):
    max_attempts: int = 3
    delay_seconds: int = 60


class AutoDefinition(BaseModel):
    """자동화 YAML 정의를 검증하는 모델."""

    name: str
    description: str
    version: str = "1.0"
    enabled: bool = True
    schedule: str
    action: AutoAction
    output: AutoOutput = Field(default_factory=AutoOutput)
    retry: AutoRetry = Field(default_factory=AutoRetry)
    timeout: int = 120

    @field_validator("schedule")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression (need 5 fields): {v}")
        return v


class AutoScheduler:
    """자동화 작업을 스케줄링하고 실행한다."""

    def __init__(
        self,
        config: AppSettings,
        security: SecurityManager,
        auto_dir: str = "auto",
    ) -> None:
        self._config = config
        self._security = security
        self._auto_dir = Path(auto_dir)
        self._scheduler = AsyncIOScheduler()
        self._automations: dict[str, AutoDefinition] = {}
        self._logger = get_logger("auto_scheduler")
        # engine과 telegram은 순환 의존 방지를 위해 나중에 주입
        self._engine = None
        self._telegram = None
        self._callables: dict[str, Callable[..., Any]] = {}

    def set_dependencies(self, engine, telegram) -> None:
        """engine과 telegram 참조를 주입한다."""
        self._engine = engine
        self._telegram = telegram

    def register_callable(self, name: str, func: Callable[..., Any]) -> None:
        """외부 callable을 이름으로 등록한다. YAML의 callable 액션에서 참조."""
        if not callable(func):
            raise TypeError(f"register_callable expects a callable, got {type(func)}")
        self._callables[name] = func
        self._logger.info("callable_registered", name=name)

    async def load_automations(self) -> int:
        """_builtin/ 및 custom/ 디렉토리에서 자동화 YAML을 로드한다."""
        self._automations.clear()
        loaded = 0

        for sub_dir in ["_builtin", "custom"]:
            auto_path = self._auto_dir / sub_dir
            if not auto_path.exists():
                continue

            for yaml_file in sorted(auto_path.glob("*.yaml")):
                try:
                    auto = self._load_auto_file(yaml_file)
                    if auto:
                        self._automations[auto.name] = auto
                        if auto.enabled:
                            self._register_job(auto)
                        loaded += 1
                        self._logger.info(
                            "automation_loaded",
                            name=auto.name,
                            enabled=auto.enabled,
                            schedule=auto.schedule,
                        )
                except Exception as exc:
                    self._logger.error(
                        "automation_load_failed",
                        file=str(yaml_file),
                        error=str(exc),
                    )

        self._logger.info("automations_loaded_total", count=loaded)
        return loaded

    def _load_auto_file(self, path: Path) -> AutoDefinition | None:
        """단일 YAML 파일에서 자동화를 로드한다."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        return AutoDefinition(**data)

    def _register_job(self, auto: AutoDefinition) -> None:
        """자동화를 APScheduler cron 작업으로 등록한다."""
        parts = auto.schedule.strip().split()
        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

        self._scheduler.add_job(
            self._execute_automation,
            trigger=trigger,
            args=[auto.name],
            id=f"auto_{auto.name}",
            replace_existing=True,
            misfire_grace_time=300,
        )

    async def _execute_automation(self, auto_name: str) -> None:
        """자동화 작업을 실행한다. 재시도 로직 포함."""
        auto = self._automations.get(auto_name)
        if not auto or not auto.enabled:
            return

        self._logger.info("automation_executing", name=auto_name)

        last_error: Exception | None = None
        result: str | None = None

        for attempt in range(auto.retry.max_attempts):
            try:
                result = await asyncio.wait_for(
                    self._run_action(auto),
                    timeout=auto.timeout,
                )
                break
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "automation_attempt_failed",
                    name=auto_name,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < auto.retry.max_attempts - 1:
                    await asyncio.sleep(auto.retry.delay_seconds)

        if result:
            self._logger.info("automation_completed", name=auto_name)
            await self._deliver_output(auto, result)
        else:
            self._logger.error(
                "automation_failed",
                name=auto_name,
                error=str(last_error),
            )

    async def _run_action(self, auto: AutoDefinition) -> str:
        """액션 타입에 따라 적절한 처리를 수행한다."""
        action = auto.action

        if action.type == "skill":
            if self._engine is None:
                raise RuntimeError("Engine not set")
            return await self._engine.execute_skill(
                skill_name=action.target,
                parameters=action.parameters,
            )
        elif action.type == "prompt":
            if self._engine is None:
                raise RuntimeError("Engine not set")
            return await self._engine.process_prompt(prompt=action.target)
        elif action.type == "callable":
            func = self._callables.get(action.target)
            if func is None:
                raise ValueError(
                    f"Callable '{action.target}' not registered. "
                    f"Available: {list(self._callables.keys())}"
                )
            if asyncio.iscoroutinefunction(func):
                return await func(**action.parameters)
            return func(**action.parameters)
        elif action.type == "command":
            # v0.1에서는 보안상 시스템 명령 실행 비활성화
            self._logger.warning(
                "command_action_disabled",
                name=auto.name,
                target=action.target,
            )
            return f"[보안 제한] 'command' 타입은 v0.1에서 비활성화되어 있습니다."
        else:
            raise ValueError(f"Unknown action type: {action.type}")

    async def _deliver_output(self, auto: AutoDefinition, result: str) -> None:
        """실행 결과를 텔레그램 전송 및/또는 파일 저장한다."""
        output = auto.output

        # 텔레그램 전송
        if output.send_to_telegram and self._telegram:
            for user_id in self._config.security.allowed_users:
                try:
                    header = f"⏰ *자동화: {auto.name}*\n\n"
                    await self._telegram.send_message(user_id, header + result)
                except Exception as exc:
                    self._logger.error(
                        "auto_telegram_send_failed",
                        user_id=user_id,
                        error=str(exc),
                    )

        # 파일 저장
        if output.save_to_file:
            try:
                # 날짜 플레이스홀더 치환
                now = datetime.now(timezone.utc)
                file_path = output.save_to_file.replace(
                    "{date}", now.strftime("%Y%m%d")
                )

                # 경로 검증
                validated_path = self._security.validate_path(
                    file_path, base_dir=self._config.data_dir
                )
                validated_path.parent.mkdir(parents=True, exist_ok=True)
                validated_path.write_text(result, encoding="utf-8")
                self._logger.info(
                    "auto_output_saved", path=str(validated_path)
                )
            except Exception as exc:
                self._logger.error(
                    "auto_file_save_failed",
                    path=output.save_to_file,
                    error=str(exc),
                )

    def start(self) -> None:
        """스케줄러를 시작한다."""
        if not self._scheduler.running:
            self._scheduler.start()
            self._logger.info("scheduler_started")

    def stop(self) -> None:
        """스케줄러를 종료한다."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            self._logger.info("scheduler_stopped")

    def list_automations(self) -> list[dict]:
        """등록된 자동화 목록을 반환한다."""
        result = []
        for auto in self._automations.values():
            job = self._scheduler.get_job(f"auto_{auto.name}")
            next_run_time = getattr(job, "next_run_time", None) if job else None
            next_run = str(next_run_time) if next_run_time else None
            result.append({
                "name": auto.name,
                "description": auto.description,
                "schedule": auto.schedule,
                "enabled": auto.enabled,
                "action_type": auto.action.type,
                "next_run": next_run,
            })
        return result

    async def enable_automation(self, name: str) -> bool:
        """자동화를 활성화하고 작업을 등록한다."""
        auto = self._automations.get(name)
        if not auto:
            return False

        auto.enabled = True
        self._register_job(auto)
        self._logger.info("automation_enabled", name=name)
        return True

    async def disable_automation(self, name: str) -> bool:
        """자동화를 비활성화하고 작업을 제거한다."""
        auto = self._automations.get(name)
        if not auto:
            return False

        auto.enabled = False
        job_id = f"auto_{name}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        self._logger.info("automation_disabled", name=name)
        return True

    async def reload_automations(self) -> int:
        """모든 작업을 제거하고 다시 로드한다."""
        for auto in self._automations.values():
            job_id = f"auto_{auto.name}"
            if self._scheduler.get_job(job_id):
                self._scheduler.remove_job(job_id)
        return await self.load_automations()
