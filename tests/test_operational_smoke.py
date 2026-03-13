"""엔트리포인트와 운영 스크립트 smoke tests."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

_SHELL_SCRIPTS = [
    "check_requirements_lock.sh",
    "healthcheck.sh",
    "install_boot_service.sh",
    "run_bot.sh",
    "setup.sh",
    "soak_monitor.sh",
    "update_wsl_hosts.sh",
]

_HELP_SCRIPTS = [
    ("healthcheck.sh", "Usage: bash scripts/healthcheck.sh"),
    ("install_boot_service.sh", "Usage: bash scripts/install_boot_service.sh"),
    ("run_bot.sh", "Usage: ./scripts/run_bot.sh"),
    ("setup.sh", "Usage: bash scripts/setup.sh"),
    ("soak_monitor.sh", "Usage: bash scripts/soak_monitor.sh"),
    ("update_wsl_hosts.sh", "Usage: bash scripts/update_wsl_hosts.sh"),
]


def test_ollama_bot_main_invokes_run_app(monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.ollama_bot.main as ollama_bot_main

    called: list[str] = []
    monkeypatch.setattr(
        ollama_bot_main,
        "run_app",
        lambda *, app_name: called.append(app_name),
    )

    ollama_bot_main.main()

    assert called == ["ollama_bot"]


def test_root_main_entrypoint_invokes_ollama_bot_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.ollama_bot.main as ollama_bot_main

    called: list[str] = []
    monkeypatch.setattr(ollama_bot_main, "main", lambda: called.append("called"))

    runpy.run_module("main", run_name="__main__")

    assert called == ["called"]


def test_cli_module_entrypoint_invokes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.cli.main as cli_main

    called: list[str] = []
    monkeypatch.setattr(cli_main, "main", lambda: called.append("called"))

    runpy.run_module("apps.cli.__main__", run_name="__main__")

    assert called == ["called"]


@pytest.mark.parametrize("script_name", _SHELL_SCRIPTS)
def test_operational_shell_scripts_have_valid_bash_syntax(script_name: str) -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPTS_DIR / script_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(("script_name", "expected"), _HELP_SCRIPTS)
def test_operational_scripts_expose_safe_help_output(
    script_name: str,
    expected: str,
) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPTS_DIR / script_name), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected in completed.stdout
