"""Automation scheduler with YAML-defined cron jobs.

Loads YAML files from `auto/_builtin/` and `auto/custom/`, registers them as
APScheduler cron jobs, and executes them.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from datetime import UTC, datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, Field, field_validator, model_validator

from core.config import AppSettings, get_default_chat_model
from core.engine_context import normalize_language
from core.logging_setup import get_logger
from core.security import SecurityManager


class ActionType(StrEnum):
    SKILL = "skill"
    COMMAND = "command"
    PROMPT = "prompt"
    CALLABLE = "callable"


class AutoAction(BaseModel):
    type: ActionType
    target: str
    parameters: dict = Field(default_factory=dict)
    model: str | None = None
    model_role: str | None = None
    llm_timeout: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    @field_validator("model", "model_role")
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1:
            raise ValueError("max_tokens must be >= 1")
        return value

    @field_validator("llm_timeout")
    @classmethod
    def validate_llm_timeout(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1:
            raise ValueError("llm_timeout must be >= 1")
        return value


class AutoOutput(BaseModel):
    send_to_telegram: bool = True
    save_to_file: str | None = None


class AutoRetry(BaseModel):
    max_attempts: int = 3
    delay_seconds: int = 60

    @field_validator("max_attempts")
    @classmethod
    def validate_max_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_attempts must be >= 1")
        return value

    @field_validator("delay_seconds")
    @classmethod
    def validate_delay_seconds(cls, value: int) -> int:
        if value < 0:
            raise ValueError("delay_seconds must be >= 0")
        return value


class AutoDefinition(BaseModel):
    """Validation model for automation YAML definitions."""

    name: str
    description: str
    description_en: str = ""
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
        normalized = v.strip()
        try:
            # Validate not just field count, but also value range and syntax.
            CronTrigger.from_crontab(normalized, timezone=UTC)
        except ValueError as exc:
            raise ValueError(f"Invalid cron expression: {v}") from exc
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value < 1:
            raise ValueError("timeout must be >= 1")
        return value

    @model_validator(mode="after")
    def validate_timeout_budget_order(self) -> AutoDefinition:
        llm_timeout = self.action.llm_timeout
        if llm_timeout is not None and llm_timeout > self.timeout:
            raise ValueError("action.llm_timeout must be <= timeout")
        return self


class DuplicateAutomationError(ValueError):
    """Raised when two automations share the same name."""


class EngineInterface(Protocol):
    async def execute_skill(
        self,
        skill_name: str,
        parameters: dict,
        chat_id: int | None = None,
        model_override: str | None = None,
        model_role_override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> str: ...

    async def process_prompt(
        self,
        prompt: str,
        chat_id: int | None = None,
        response_format: str | dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model_override: str | None = None,
        model_role: str | None = None,
        timeout: int | None = None,
        timeout_is_hard: bool = False,
    ) -> str: ...


class TelegramInterface(Protocol):
    async def send_message(
        self, chat_id: int, text: str, parse_mode: str | None = None,
    ) -> None: ...


class AutoScheduler:
    """Schedule and execute automation jobs."""

    def __init__(
        self,
        config: AppSettings,
        security: SecurityManager,
        auto_dir: str = "auto",
    ) -> None:
        self._config = config
        self._security = security
        self._auto_dir = Path(auto_dir)
        self._timezone = self._resolve_timezone(config.scheduler.timezone)
        self._scheduler = AsyncIOScheduler(timezone=self._timezone)
        self._automations: dict[str, AutoDefinition] = {}
        self._logger = get_logger("auto_scheduler")
        # Inject engine and telegram later to avoid circular dependencies.
        self._engine: EngineInterface | None = None
        self._telegram: TelegramInterface | None = None
        self._callables: dict[str, Callable[..., Any]] = {}
        self._last_load_errors: list[str] = []
        self._action_handlers: dict[ActionType, Callable[[AutoDefinition], Any]] = {
            ActionType.SKILL: self._run_skill_action,
            ActionType.PROMPT: self._run_prompt_action,
            ActionType.CALLABLE: self._run_callable_action,
            ActionType.COMMAND: self._run_command_action,
        }

    @staticmethod
    def _resolve_timezone(name: str):
        """Return a tzinfo object available in the current runtime environment."""
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid timezone: {name}") from exc

    def set_dependencies(
        self,
        engine: EngineInterface,
        telegram: TelegramInterface,
    ) -> None:
        """Inject engine and telegram references."""
        self._engine = engine
        self._telegram = telegram

    def dependencies_ready(self) -> bool:
        """Return whether runtime dependencies are fully injected."""
        return self._engine is not None and self._telegram is not None

    def register_callable(self, name: str, func: Callable[..., Any]) -> None:
        """Register an external callable by name for YAML callable actions."""
        if not callable(func):
            raise TypeError(f"register_callable expects a callable, got {type(func)}")
        self._callables[name] = func
        self._logger.info("callable_registered", name=name)

    async def load_automations(self, *, strict: bool = False) -> int:
        """Load automation YAML files from `_builtin/` and `custom/`."""
        old_automations = self._automations
        new_automations: dict[str, AutoDefinition] = {}
        new_triggers: dict[str, CronTrigger] = {}
        name_sources: dict[str, Path] = {}
        self._last_load_errors = []
        loaded = 0

        for sub_dir in ["_builtin", "custom"]:
            auto_path = self._auto_dir / sub_dir
            if not auto_path.exists():
                continue

            for yaml_file in sorted(auto_path.glob("*.yaml")):
                try:
                    auto = self._load_auto_file(yaml_file)
                    if auto:
                        existing_source = name_sources.get(auto.name)
                        if existing_source is not None:
                            raise DuplicateAutomationError(
                                f"Duplicate automation name '{auto.name}' "
                                f"({existing_source} vs {yaml_file})"
                            )
                        trigger = self._build_cron_trigger(auto.schedule)
                        name_sources[auto.name] = yaml_file
                        new_automations[auto.name] = auto
                        new_triggers[auto.name] = trigger
                        loaded += 1
                        self._logger.info(
                            "automation_loaded",
                            name=auto.name,
                            enabled=auto.enabled,
                            schedule=auto.schedule,
                        )
                except DuplicateAutomationError:
                    raise
                except Exception as exc:
                    self._last_load_errors.append(f"{yaml_file.name}: {exc}")
                    self._logger.error(
                        "automation_load_failed",
                        file=str(yaml_file),
                        error=str(exc),
                    )

        if strict and self._last_load_errors:
            raise ValueError(self._format_load_error_summary(self._last_load_errors))

        self._swap_automations(
            old_automations=old_automations,
            new_automations=new_automations,
            new_triggers=new_triggers,
        )
        if self._last_load_errors:
            self._logger.warning(
                "automations_loaded_with_errors",
                loaded=loaded,
                error_count=len(self._last_load_errors),
            )

        self._logger.info("automations_loaded_total", count=loaded)
        return loaded

    def _load_auto_file(self, path: Path) -> AutoDefinition | None:
        """Load a single automation from YAML."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        return AutoDefinition(**data)

    def _build_cron_trigger(self, schedule: str) -> CronTrigger:
        """Build and validate a cron trigger in the configured timezone."""
        try:
            return CronTrigger.from_crontab(schedule.strip(), timezone=self._timezone)
        except ValueError as exc:
            raise ValueError(f"Invalid cron expression: {schedule}") from exc

    def _remove_job(self, job_id: str) -> None:
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

    def _swap_automations(
        self,
        *,
        old_automations: dict[str, AutoDefinition],
        new_automations: dict[str, AutoDefinition],
        new_triggers: dict[str, CronTrigger],
    ) -> None:
        """Atomically swap automation state and roll back on failure."""
        old_enabled = [auto for auto in old_automations.values() if auto.enabled]
        added_job_ids: list[str] = []
        try:
            for name in old_automations:
                self._remove_job(f"auto_{name}")

            self._automations = new_automations
            for auto in self._automations.values():
                if not auto.enabled:
                    continue
                trigger = new_triggers.get(auto.name)
                if trigger is None:
                    trigger = self._build_cron_trigger(auto.schedule)
                self._register_job(auto, trigger=trigger)
                added_job_ids.append(f"auto_{auto.name}")
        except Exception as exc:
            for job_id in added_job_ids:
                self._remove_job(job_id)
            self._automations = old_automations
            for auto in old_enabled:
                try:
                    self._register_job(auto)
                except Exception as rollback_exc:
                    self._logger.error(
                        "automation_rollback_register_failed",
                        name=auto.name,
                        error=str(rollback_exc),
                    )
            self._logger.error("automations_swap_failed", error=str(exc))
            raise

    def _register_job(
        self,
        auto: AutoDefinition,
        trigger: CronTrigger | None = None,
    ) -> None:
        """Register an automation as an APScheduler cron job."""
        if trigger is None:
            trigger = self._build_cron_trigger(auto.schedule)

        self._scheduler.add_job(
            self._execute_automation,
            trigger=trigger,
            args=[auto.name],
            id=f"auto_{auto.name}",
            replace_existing=True,
            misfire_grace_time=300,
        )

    async def _execute_automation(self, auto_name: str) -> bool:
        """Execute an automation job, including retry logic."""
        auto = self._automations.get(auto_name)
        if not auto or not auto.enabled:
            return False

        self._logger.info("automation_executing", name=auto_name)

        last_error: Exception | None = None
        result: str | None = None
        succeeded = False

        for attempt in range(auto.retry.max_attempts):
            try:
                result = await asyncio.wait_for(
                    self._run_action(auto),
                    timeout=auto.timeout,
                )
                succeeded = True
                break
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "automation_attempt_failed",
                    name=auto_name,
                    attempt=attempt + 1,
                    error=self._format_exception(exc),
                )
                if attempt < auto.retry.max_attempts - 1:
                    await asyncio.sleep(auto.retry.delay_seconds)

        if succeeded:
            if result:
                self._logger.info("automation_completed", name=auto_name)
                await self._deliver_output(auto, result)
            else:
                self._logger.info("automation_completed_no_output", name=auto_name)
            return True
        else:
            self._logger.error(
                "automation_failed",
                name=auto_name,
                error=self._format_exception(last_error),
            )
            await self._deliver_failure_notice(auto, last_error)
            return False

    async def _run_action(self, auto: AutoDefinition) -> str:
        """Dispatch to the appropriate handler for the action type."""
        action = auto.action
        handler = self._action_handlers.get(action.type)
        if handler is None:
            raise ValueError(f"Unknown action type: {action.type}")
        result = handler(auto)
        if inspect.isawaitable(result):
            return await result
        return str(result)

    async def _run_skill_action(self, auto: AutoDefinition) -> str:
        if self._engine is None:
            raise RuntimeError("Engine not set")
        action = auto.action
        model_override, model_role = self._resolve_action_model(action)
        llm_timeout = action.llm_timeout if action.llm_timeout is not None else auto.timeout
        return await self._engine.execute_skill(
            skill_name=action.target,
            parameters=action.parameters,
            model_override=model_override,
            model_role_override=model_role,
            max_tokens=action.max_tokens,
            temperature=action.temperature,
            timeout=llm_timeout,
        )

    async def _run_prompt_action(self, auto: AutoDefinition) -> str:
        if self._engine is None:
            raise RuntimeError("Engine not set")
        action = auto.action
        model_override, model_role = self._resolve_action_model(action)
        response_format = action.parameters.get("response_format")
        chat_id = action.parameters.get("chat_id")
        max_tokens = (
            action.max_tokens
            if action.max_tokens is not None
            else action.parameters.get("max_tokens")
        )
        temperature = (
            action.temperature
            if action.temperature is not None
            else action.parameters.get("temperature")
        )
        llm_timeout = action.llm_timeout if action.llm_timeout is not None else auto.timeout
        return await self._engine.process_prompt(
            prompt=action.target,
            chat_id=chat_id if isinstance(chat_id, int) else None,
            response_format=response_format,
            max_tokens=max_tokens if isinstance(max_tokens, int) else None,
            temperature=temperature if isinstance(temperature, int | float) else None,
            model_override=model_override,
            model_role=model_role,
            timeout=llm_timeout,
            timeout_is_hard=action.llm_timeout is not None,
        )

    async def _run_callable_action(self, auto: AutoDefinition) -> str:
        action = auto.action
        func = self._callables.get(action.target)
        if func is None:
            raise ValueError(
                f"Callable '{action.target}' not registered. "
                f"Available: {list(self._callables.keys())}"
            )
        call_kwargs = dict(action.parameters)
        model_override, model_role = self._resolve_action_model(action)
        optional_kwargs: dict[str, Any] = {}
        if model_override is not None:
            optional_kwargs["model"] = model_override
        if model_role is not None:
            optional_kwargs["model_role"] = model_role
        if action.temperature is not None:
            optional_kwargs["temperature"] = action.temperature
        if action.max_tokens is not None:
            optional_kwargs["max_tokens"] = action.max_tokens
        optional_kwargs["timeout"] = auto.timeout
        if action.llm_timeout is not None:
            optional_kwargs["llm_timeout"] = action.llm_timeout

        self._inject_callable_kwargs(func, call_kwargs, optional_kwargs)

        output = func(**call_kwargs)
        if inspect.isawaitable(output):
            output = await output
        return "" if output is None else str(output)

    def _resolve_action_model(self, action: AutoAction) -> tuple[str | None, str | None]:
        """Resolve the model and role to use for an automation action.

        The default is the global default model without a role tag.
        """
        model_override = action.model
        model_role = action.model_role
        if model_override is not None or model_role is not None:
            return model_override, model_role
        fallback_model = get_default_chat_model(self._config).strip() or None
        return fallback_model, None

    @staticmethod
    def _inject_callable_kwargs(
        func: Callable[..., Any],
        call_kwargs: dict[str, Any],
        optional_kwargs: dict[str, Any],
    ) -> None:
        """Inject only supported optional kwargs based on the callable signature."""
        if not optional_kwargs:
            return
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return

        accepts_var_kw = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        accepted_names = {
            name
            for name, param in signature.parameters.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        for key, value in optional_kwargs.items():
            if key in call_kwargs:
                continue
            if accepts_var_kw or key in accepted_names:
                call_kwargs[key] = value

    def _run_command_action(self, auto: AutoDefinition) -> str:
        # In v0.1, system-command execution is disabled for security reasons.
        self._logger.warning(
            "command_action_disabled",
            name=auto.name,
            target=auto.action.target,
        )
        return "[보안 제한] 'command' 타입은 v0.1에서 비활성화되어 있습니다."

    async def _deliver_output(self, auto: AutoDefinition, result: str) -> None:
        """Deliver automation output to Telegram and/or save it to a file."""
        output = auto.output

        # Telegram delivery
        if output.send_to_telegram and self._telegram:
            lang = normalize_language(self._config.bot.language)
            # Send HTML-tagged results with HTML parse mode.
            use_html = "<b>" in result or "<pre>" in result
            parse_mode = "HTML" if use_html else None
            header = (
                f"⏰ <b>{'Automation' if lang == 'en' else '자동화'}: {auto.name}</b>\n\n"
                if use_html
                else f"⏰ {'Automation' if lang == 'en' else '자동화'}: {auto.name}\n\n"
            )
            for user_id in self._config.security.allowed_users:
                try:
                    await self._telegram.send_message(
                        user_id, header + result, parse_mode=parse_mode,
                    )
                except Exception as exc:
                    self._logger.error(
                        "auto_telegram_send_failed",
                        user_id=user_id,
                        error=str(exc),
                    )

        # File output
        if output.save_to_file:
            try:
                # Replace date placeholders.
                now = datetime.now(self._timezone)
                file_path = output.save_to_file.replace(
                    "{date}", now.strftime("%Y%m%d")
                )

                # Validate the output path.
                validated_path = self._security.validate_path(
                    file_path, base_dir=self._config.data_dir
                )
                validated_path.parent.mkdir(parents=True, exist_ok=True)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    functools.partial(validated_path.write_text, result, encoding="utf-8"),
                )
                self._logger.info(
                    "auto_output_saved", path=str(validated_path)
                )
            except Exception as exc:
                self._logger.error(
                    "auto_file_save_failed",
                    path=output.save_to_file,
                    error=str(exc),
                )

    async def _deliver_failure_notice(
        self,
        auto: AutoDefinition,
        error: Exception | None,
    ) -> None:
        """Send an automation failure notice to Telegram."""
        if self._telegram is None:
            return

        from html import escape as _h

        now = datetime.now(self._timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
        message = (
            f"⚠️ <b>자동화 실패: {_h(auto.name)}</b>\n"
            f"- 시각: {now}\n"
            f"- 원인: <code>{_h(self._format_exception(error))}</code>\n"
            f"- 재시도: {auto.retry.max_attempts}회 모두 실패"
        )

        for user_id in self._config.security.allowed_users:
            try:
                await self._telegram.send_message(
                    user_id, message, parse_mode="HTML",
                )
            except Exception as exc:
                self._logger.error(
                    "auto_failure_notice_send_failed",
                    user_id=user_id,
                    error=self._format_exception(exc),
                )

    @staticmethod
    def _format_exception(exc: Exception | None) -> str:
        """Normalize an exception into a string that includes its class name."""
        if exc is None:
            return "unknown"
        message = str(exc).strip()
        if message:
            return f"{exc.__class__.__name__}: {message}"
        return exc.__class__.__name__

    def start(self) -> None:
        """Start the scheduler."""
        if not self.dependencies_ready():
            raise RuntimeError(
                "AutoScheduler dependencies are not set. "
                "Call set_dependencies(engine, telegram) before start()."
            )
        if not self._scheduler.running:
            self._scheduler.start()
            self._logger.info("scheduler_started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            self._logger.info("scheduler_stopped")

    def list_automations(self, lang: str = "ko") -> list[dict]:
        """Return a list of registered automations with localized descriptions."""
        result = []
        for auto in self._automations.values():
            job = self._scheduler.get_job(f"auto_{auto.name}")
            next_run_time = getattr(job, "next_run_time", None) if job else None
            next_run = str(next_run_time) if next_run_time else None
            description = (
                auto.description_en
                if lang == "en" and auto.description_en
                else auto.description
            )
            result.append({
                "name": auto.name,
                "description": description,
                "schedule": auto.schedule,
                "enabled": auto.enabled,
                "action_type": auto.action.type,
                "action_model": auto.action.model,
                "action_model_role": auto.action.model_role,
                "action_llm_timeout": auto.action.llm_timeout,
                "next_run": next_run,
            })
        return result

    def get_last_load_errors(self) -> list[str]:
        """Return errors collected during the most recent `load_automations` call."""
        return list(self._last_load_errors)

    @staticmethod
    def _format_load_error_summary(errors: list[str], max_items: int = 3) -> str:
        preview = errors[:max_items]
        message = (
            f"Automation loading failed in strict mode "
            f"({len(errors)} error(s)): {'; '.join(preview)}"
        )
        if len(errors) > max_items:
            message += f"; ... and {len(errors) - max_items} more"
        return message

    async def disable_automation(self, name: str) -> bool:
        """Disable an automation and remove its scheduled job."""
        auto = self._automations.get(name)
        if not auto:
            return False

        auto.enabled = False
        job_id = f"auto_{name}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        self._logger.info("automation_disabled", name=name)
        return True

    async def reload_automations(self, *, strict: bool = False) -> int:
        """Remove all jobs and reload automations."""
        return await self.load_automations(strict=strict)

    async def run_automation_once(self, name: str) -> bool:
        """Run the specified automation immediately once."""
        auto = self._automations.get(name)
        if auto is None or not auto.enabled:
            return False
        return await self._execute_automation(name)
