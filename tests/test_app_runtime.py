"""app_runtime startup strict/degraded 처리 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.app_runtime import (
    StartupError,
    _handle_optional_component_failure,
    _log_degraded_startup_summary,
    _resolve_wsl_loopback_host,
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
            component="test_component",
            error=RuntimeError("init failed"),
        )

    assert degraded == [{"component": "test_component", "error": "init failed"}]


def test_log_degraded_startup_summary_noop_when_empty() -> None:
    logger = MagicMock()
    _log_degraded_startup_summary(logger, [])
    logger.warning.assert_not_called()


def test_resolve_wsl_loopback_host_rewrites_when_local_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr("core.app_runtime._is_wsl_environment", lambda: True)
    monkeypatch.setattr(
        "core.app_runtime._iter_wsl_bridge_candidates",
        lambda: ["homelab", "10.255.255.254"],
    )

    def _fake_probe(host: str, _port: int, *, timeout_seconds: float = 0.35) -> bool:
        _ = timeout_seconds
        return host == "homelab"

    monkeypatch.setattr("core.app_runtime._can_connect_tcp", _fake_probe)

    resolved = _resolve_wsl_loopback_host(
        url="http://localhost:11434",
        service_name="ollama",
        logger=logger,
    )

    assert resolved == "http://homelab:11434"
    logger.warning.assert_called()


def test_resolve_wsl_loopback_host_keeps_localhost_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr("core.app_runtime._is_wsl_environment", lambda: True)
    monkeypatch.setattr("core.app_runtime._can_connect_tcp", lambda *_args, **_kwargs: True)

    resolved = _resolve_wsl_loopback_host(
        url="http://localhost:8020",
        service_name="lemonade",
        logger=logger,
    )

    assert resolved == "http://localhost:8020"
