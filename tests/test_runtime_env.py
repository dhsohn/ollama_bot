"""Runtime environment helper tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core import runtime_env


def test_is_wsl_environment_detects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert runtime_env.is_wsl_environment() is True


def test_is_wsl_environment_reads_proc_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)

    def fake_read_text(self: Path, *args, **kwargs) -> str:
        _ = (args, kwargs)
        if str(self) == "/proc/version":
            return "Linux version 5.15.90.1-microsoft-standard-WSL2"
        raise OSError

    monkeypatch.setattr(Path, "read_text", fake_read_text, raising=False)
    assert runtime_env.is_wsl_environment() is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" http://LOCALHOST:8000 ", "localhost"),
        ("[::1]:11434", "::1"),
        ("192.168.0.10:8020", "192.168.0.10"),
        ("example.com", "example.com"),
    ],
)
def test_normalize_host_token(raw: str, expected: str) -> None:
    assert runtime_env.normalize_host_token(raw) == expected


def test_iter_wsl_bridge_candidates_collects_and_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WINDOWS_HOST", "10.0.0.1")
    monkeypatch.setenv("WSL_HOST_IP", "10.0.0.1")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: (
            "nameserver 10.0.0.2\n"
            if str(self) == "/etc/resolv.conf"
            else "10.0.0.3 homelab host.docker.internal\n"
        ),
        raising=False,
    )

    candidates = runtime_env.iter_wsl_bridge_candidates()

    assert candidates[:4] == ["10.0.0.1", "10.0.0.2", "10.0.0.3", "homelab"]
    assert candidates.count("host.docker.internal") == 1


def test_can_connect_tcp_uses_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(runtime_env.socket, "create_connection", lambda *_args, **_kwargs: DummySocket())
    assert runtime_env.can_connect_tcp("localhost", 80) is True


def test_can_connect_tcp_returns_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_oserror(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(runtime_env.socket, "create_connection", raise_oserror)
    assert runtime_env.can_connect_tcp("localhost", 80) is False


def test_resolve_wsl_loopback_host_returns_original_for_non_loopback() -> None:
    logger = MagicMock()
    result = runtime_env.resolve_wsl_loopback_host(
        url="http://homelab:8020",
        service_name="ollama",
        logger=logger,
    )
    assert result == "http://homelab:8020"
    logger.warning.assert_not_called()


def test_resolve_wsl_loopback_host_rewrites_with_credentials_and_ipv6() -> None:
    logger = MagicMock()
    result = runtime_env.resolve_wsl_loopback_host(
        url="http://user:pass@localhost:8000/api",
        service_name="ollama",
        logger=logger,
        is_wsl_environment_fn=lambda: True,
        iter_wsl_bridge_candidates_fn=lambda: ["fe80::1"],
        can_connect_tcp_fn=lambda host, port: host == "fe80::1" and port == 8000,
    )
    assert result == "http://user:pass@[fe80::1]:8000/api"
    logger.warning.assert_called_once()


def test_resolve_wsl_loopback_host_logs_unreachable_candidates() -> None:
    logger = MagicMock()
    result = runtime_env.resolve_wsl_loopback_host(
        url="http://localhost:11434",
        service_name="ollama",
        logger=logger,
        is_wsl_environment_fn=lambda: True,
        iter_wsl_bridge_candidates_fn=lambda: ["10.0.0.1", "homelab"],
        can_connect_tcp_fn=lambda _host, _port: False,
    )
    assert result == "http://localhost:11434"
    logger.warning.assert_called_once()
