"""Shared application runtime bootstrap.

Provides reusable initialization and execution flow for entrypoints such as
`apps/ollama_bot`.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from core.config import load_config
from core.logging_setup import get_logger, setup_logging
from core.runtime_env import (
    can_connect_tcp,
    is_wsl_environment,
    iter_wsl_bridge_candidates,
    normalize_host_token,
)
from core.runtime_env import (
    resolve_wsl_loopback_host as _resolve_wsl_loopback_host_impl,
)
from core.runtime_factory import RuntimeState, StartupError
from core.runtime_factory import build_runtime as _build_runtime
from core.runtime_factory import (
    handle_optional_component_failure as _handle_optional_component_failure,
)
from core.runtime_factory import (
    log_degraded_startup_summary as _log_degraded_startup_summary,
)
from core.runtime_factory import (
    validate_required_settings as _validate_required_settings,
)
from core.runtime_lifecycle import run_runtime
from core.runtime_lifecycle import shutdown_runtime as _shutdown_runtime
from core.runtime_tasks import llm_recovery_loop as _llm_recovery_loop
from core.runtime_tasks import memory_maintenance_loop as _memory_maintenance_loop

_is_wsl_environment = is_wsl_environment
_normalize_host_token = normalize_host_token
_iter_wsl_bridge_candidates = iter_wsl_bridge_candidates
_can_connect_tcp = can_connect_tcp


def _resolve_wsl_loopback_host(
    *,
    url: str,
    service_name: str,
    logger: Any,
) -> str:
    """Expose WSL host resolution from the app_runtime namespace for tests."""
    return _resolve_wsl_loopback_host_impl(
        url=url,
        service_name=service_name,
        logger=logger,
        is_wsl_environment_fn=_is_wsl_environment,
        iter_wsl_bridge_candidates_fn=_iter_wsl_bridge_candidates,
        can_connect_tcp_fn=lambda host, port: _can_connect_tcp(host, port),
    )


async def async_main(
    *,
    app_name: str,
) -> None:
    """Async main loop."""
    try:
        config = load_config()
    except ValueError as exc:
        print(
            f"Error: invalid configuration. {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    log_dir = str(Path(config.data_dir) / "logs")
    setup_logging(config.log_level, log_dir=log_dir)
    logger = get_logger(app_name)
    logger.info("starting", app=app_name, version="0.1.0")
    try:
        _validate_required_settings(config, logger)
        runtime = await _build_runtime(config, logger)
    except StartupError as exc:
        print(exc.message, file=sys.stderr)
        sys.exit(1)

    await run_runtime(runtime)


def run_app(
    *,
    app_name: str,
) -> None:
    """Synchronous entrypoint wrapper."""
    with suppress(KeyboardInterrupt):
        asyncio.run(async_main(app_name=app_name))
