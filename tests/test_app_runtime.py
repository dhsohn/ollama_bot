"""app_runtime startup strict/degraded 처리 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.app_runtime import (
    StartupError,
    _handle_optional_component_failure,
    _log_degraded_startup_summary,
)
from core.config import AppSettings


def test_optional_component_failure_records_in_non_strict_mode() -> None:
    config = AppSettings(strict_startup=False)
    logger = MagicMock()
    degraded: list[dict[str, str]] = []

    _handle_optional_component_failure(
        config,
        logger,
        degraded,
        component="rag_pipeline",
        error=RuntimeError("rag init failed"),
    )

    assert degraded == [{"component": "rag_pipeline", "error": "rag init failed"}]
    logger.warning.assert_called()


def test_optional_component_failure_raises_in_strict_mode() -> None:
    config = AppSettings(strict_startup=True)
    logger = MagicMock()
    degraded: list[dict[str, str]] = []

    with pytest.raises(StartupError, match="strict_startup=true"):
        _handle_optional_component_failure(
            config,
            logger,
            degraded,
            component="sim_scheduler",
            error=RuntimeError("sim init failed"),
        )

    assert degraded == [{"component": "sim_scheduler", "error": "sim init failed"}]


def test_log_degraded_startup_summary_noop_when_empty() -> None:
    logger = MagicMock()
    _log_degraded_startup_summary(logger, [])
    logger.warning.assert_not_called()
