"""설정 로더 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config


def _write_minimal_yaml(path: Path) -> None:
    path.write_text("{}", encoding="utf-8")


def test_allowed_telegram_users_parsed_from_env(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    _write_minimal_yaml(config_path)

    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=123456789, 987654321",
                "OLLAMA_HOST=http://localhost:11434",
                "OLLAMA_MODEL=gpt-oss:20b",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.security.allowed_users == [123456789, 987654321]


def test_allowed_telegram_users_invalid_value_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    _write_minimal_yaml(config_path)

    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=123,abc,456",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ALLOWED_TELEGRAM_USERS"):
        load_config(config_path=str(config_path), env_file=str(env_path))
