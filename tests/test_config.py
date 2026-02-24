"""설정 로더 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import FeedbackConfig, load_config


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


def test_scheduler_timezone_loaded_from_env(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    _write_minimal_yaml(config_path)

    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=123456789",
                "SCHEDULER_TIMEZONE=Asia/Seoul",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.scheduler.timezone == "Asia/Seoul"


def test_scheduler_timezone_invalid_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    _write_minimal_yaml(config_path)

    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=123456789",
                "SCHEDULER_TIMEZONE=Invalid/Timezone",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid timezone"):
        load_config(config_path=str(config_path), env_file=str(env_path))


def test_yaml_values_preserved_when_env_not_set(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join(
            [
                "ollama:",
                "  host: \"http://yaml-host:11434\"",
                "  model: \"yaml-model\"",
                "scheduler:",
                "  timezone: \"UTC\"",
            ]
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=123456789",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.ollama.host == "http://yaml-host:11434"
    assert settings.ollama.model == "yaml-model"
    assert settings.scheduler.timezone == "UTC"


def test_env_values_override_yaml_when_explicitly_set(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join(
            [
                "ollama:",
                "  host: \"http://yaml-host:11434\"",
                "  model: \"yaml-model\"",
                "scheduler:",
                "  timezone: \"UTC\"",
            ]
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=123456789",
                "OLLAMA_HOST=http://env-host:11434",
                "OLLAMA_MODEL=env-model",
                "SCHEDULER_TIMEZONE=Asia/Seoul",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.ollama.host == "http://env-host:11434"
    assert settings.ollama.model == "env-model"
    assert settings.scheduler.timezone == "Asia/Seoul"


def test_feedback_default_values() -> None:
    """FeedbackConfig 기본값이 올바르게 설정된다."""
    cfg = FeedbackConfig()
    assert cfg.enabled is True
    assert cfg.show_buttons is True
    assert cfg.min_feedback_for_analysis == 5
    assert cfg.max_guidelines == 5
    assert cfg.preview_max_chars == 300
    assert cfg.preview_cache_max_size == 500
    assert cfg.preview_cache_ttl_hours == 24
    assert cfg.retention_days == 90


def test_security_invalid_numeric_value_raises(tmp_path: Path) -> None:
    """security 숫자 설정이 1 미만이면 예외가 발생한다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "security:",
            "  max_concurrent_requests: 0",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="security numeric settings must be >= 1"):
        load_config(config_path=str(config_path), env_file=str(env_path))


def test_feedback_section_loaded_from_yaml(tmp_path: Path) -> None:
    """YAML의 feedback 섹션이 올바르게 로드된다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "feedback:",
            "  enabled: false",
            "  show_buttons: false",
            "  min_feedback_for_analysis: 10",
            "  max_guidelines: 3",
            "  retention_days: 30",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.feedback.enabled is False
    assert settings.feedback.show_buttons is False
    assert settings.feedback.min_feedback_for_analysis == 10
    assert settings.feedback.max_guidelines == 3
    assert settings.feedback.retention_days == 30


def test_feedback_invalid_numeric_value_raises(tmp_path: Path) -> None:
    """feedback 숫자 설정이 1 미만이면 예외가 발생한다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "feedback:",
            "  preview_cache_max_size: 0",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feedback numeric settings must be >= 1"):
        load_config(config_path=str(config_path), env_file=str(env_path))
