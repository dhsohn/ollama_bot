"""Action execution helpers for :mod:`core.auto_scheduler`."""

from __future__ import annotations

import inspect
from typing import Any


async def execute_automation(scheduler: Any, auto_name: str) -> bool:
    """Execute a single automation, including retry logic."""
    auto = scheduler._automations.get(auto_name)
    if not auto or not auto.enabled:
        return False

    scheduler._logger.info("automation_executing", name=auto_name)

    last_error: Exception | None = None
    result: str | None = None
    succeeded = False

    for attempt in range(auto.retry.max_attempts):
        try:
            result = await scheduler._run_action(auto)
            succeeded = True
            break
        except Exception as exc:
            last_error = exc
            scheduler._logger.warning(
                "automation_attempt_failed",
                name=auto_name,
                attempt=attempt + 1,
                error=scheduler._format_exception(exc),
            )
            if attempt < auto.retry.max_attempts - 1:
                await scheduler._sleep(auto.retry.delay_seconds)

    if succeeded:
        if result:
            scheduler._logger.info("automation_completed", name=auto_name)
            await scheduler._deliver_output(auto, result)
        else:
            scheduler._logger.info("automation_completed_no_output", name=auto_name)
        return True

    scheduler._logger.error(
        "automation_failed",
        name=auto_name,
        error=scheduler._format_exception(last_error),
    )
    await scheduler._deliver_failure_notice(auto, last_error)
    return False


async def run_action(scheduler: Any, auto: Any) -> str:
    """Dispatch to the appropriate handler for the action type."""
    action = auto.action
    handler = scheduler._action_handlers.get(action.type)
    if handler is None:
        raise ValueError(f"Unknown action type: {action.type}")
    result = handler(auto)
    if inspect.isawaitable(result):
        return await result
    return str(result)


async def run_skill_action(scheduler: Any, auto: Any) -> str:
    """Execute a skill action through the engine dependency."""
    if scheduler._engine is None:
        raise RuntimeError("Engine not set")
    action = auto.action
    model_override, model_role = scheduler._resolve_action_model(action)
    llm_timeout = action.llm_timeout if action.llm_timeout is not None else auto.timeout
    return await scheduler._engine.execute_skill(
        skill_name=action.target,
        parameters=action.parameters,
        model_override=model_override,
        model_role_override=model_role,
        max_tokens=action.max_tokens,
        temperature=action.temperature,
        timeout=llm_timeout,
    )


async def run_prompt_action(scheduler: Any, auto: Any) -> str:
    """Execute a prompt action through the engine dependency."""
    if scheduler._engine is None:
        raise RuntimeError("Engine not set")
    action = auto.action
    model_override, model_role = scheduler._resolve_action_model(action)
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
    return await scheduler._engine.process_prompt(
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


async def run_callable_action(scheduler: Any, auto: Any) -> str:
    """Execute a registered callable action."""
    action = auto.action
    func = scheduler._callables.get(action.target)
    if func is None:
        raise ValueError(
            f"Callable '{action.target}' not registered. "
            f"Available: {list(scheduler._callables.keys())}"
        )
    call_kwargs = dict(action.parameters)
    model_override, model_role = scheduler._resolve_action_model(action)
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

    scheduler._inject_callable_kwargs(func, call_kwargs, optional_kwargs)

    output = func(**call_kwargs)
    if inspect.isawaitable(output):
        output = await output
    return "" if output is None else str(output)


def resolve_action_model(scheduler: Any, action: Any) -> tuple[str | None, str | None]:
    """Resolve the model and role to use for an automation action."""
    model_override = action.model
    model_role = action.model_role
    if model_override is not None or model_role is not None:
        return model_override, model_role
    fallback_model = scheduler._get_default_model().strip() or None
    return fallback_model, None


def inject_callable_kwargs(
    func: Any,
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


def run_command_action(scheduler: Any, auto: Any) -> str:
    """Reject command actions for security reasons."""
    scheduler._logger.warning(
        "command_action_disabled",
        name=auto.name,
        target=auto.action.target,
    )
    return "[보안 제한] 'command' 타입은 v0.1에서 비활성화되어 있습니다."
