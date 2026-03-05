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


def test_allowed_telegram_users_negative_ids_parsed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    _write_minimal_yaml(config_path)

    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=-100123, 456",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.security.allowed_users == [-100123, 456]


def test_scheduler_timezone_env_ignored_and_yaml_used(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join(
            [
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
                "SCHEDULER_TIMEZONE=Asia/Seoul",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.scheduler.timezone == "UTC"


def test_scheduler_timezone_invalid_env_ignored(tmp_path: Path) -> None:
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

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.scheduler.timezone == "Asia/Seoul"


def test_yaml_values_preserved_when_env_not_set(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join(
            [
                "lemonade:",
                "  host: \"http://yaml-host:8000\"",
                "  default_model: \"yaml-model\"",
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
    assert settings.lemonade.host == "http://yaml-host:8000"
    assert settings.lemonade.default_model == "yaml-model"
    assert settings.scheduler.timezone == "UTC"


def test_env_values_do_not_override_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join(
            [
                "lemonade:",
                "  host: \"http://yaml-host:8000\"",
                "  default_model: \"yaml-model\"",
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
                "LEMONADE_HOST=http://env-host:8000",
                "LEMONADE_DEFAULT_MODEL=env-model",
                "SCHEDULER_TIMEZONE=Asia/Seoul",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.lemonade.host == "http://yaml-host:8000"
    assert settings.lemonade.default_model == "yaml-model"
    assert settings.scheduler.timezone == "UTC"


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


def test_runtime_maintenance_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "runtime_maintenance:",
            "  memory_maintenance_interval_seconds: 300",
            "  llm_recovery_interval_seconds: 15",
            "  memory_maintenance_jitter_ratio: 0.2",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.runtime_maintenance.memory_maintenance_interval_seconds == 300
    assert settings.runtime_maintenance.llm_recovery_interval_seconds == 15
    assert settings.runtime_maintenance.memory_maintenance_jitter_ratio == 0.2


def test_runtime_maintenance_invalid_jitter_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "runtime_maintenance:",
            "  memory_maintenance_jitter_ratio: 1.5",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="memory_maintenance_jitter_ratio"):
        load_config(config_path=str(config_path), env_file=str(env_path))


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


def test_partial_lemonade_section_preserves_defaults(tmp_path: Path) -> None:
    """lemonade 섹션 일부만 지정해도 누락 필드는 기본값을 유지한다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "lemonade:",
            "  default_model: \"custom-model\"",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.lemonade.default_model == "custom-model"
    assert settings.lemonade.host == "http://localhost:8000"
    assert settings.lemonade.temperature == 0.7
    assert settings.lemonade.max_tokens == 4096


def test_partial_feedback_section_preserves_defaults(tmp_path: Path) -> None:
    """feedback 섹션 일부 오버라이드 시 나머지 기본값이 유지된다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join([
            "feedback:",
            "  retention_days: 14",
            "  collect_reason: false",
        ]),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.feedback.retention_days == 14
    assert settings.feedback.collect_reason is False
    # 누락 필드 기본값 유지 여부 검증
    assert settings.feedback.preview_cache_max_size == 500
    assert settings.feedback.preview_cache_ttl_hours == 24


def test_lemonade_env_values_ignored(tmp_path: Path) -> None:
    """lemonade 관련 .env 값은 반영되지 않는다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    _write_minimal_yaml(config_path)
    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=111",
                "LEMONADE_HOST=http://localhost:8000",
                "LEMONADE_DEFAULT_MODEL=llama-3.1-8b",
                "LEMONADE_API_KEY=secret",
                "LEMONADE_BASE_PATH=/api/v1",
                "LEMONADE_TIMEOUT_SECONDS=45",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.lemonade.host == "http://localhost:8000"
    assert settings.lemonade.default_model == "gpt-oss-20b-NPU"
    assert settings.lemonade.api_key == ""
    assert settings.lemonade.base_path == "/api/v1"
    assert settings.lemonade.timeout_seconds == 60


def test_unknown_top_level_key_ignored(tmp_path: Path) -> None:
    """알 수 없는 최상위 키는 무시되고 lemonade 기본값이 유지된다."""
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "unknown_key: unsupported",
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=test_token",
                "ALLOWED_TELEGRAM_USERS=111",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.lemonade.default_model == "gpt-oss-20b-NPU"


def test_rag_kb_dirs_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        "\n".join(
            [
                "rag:",
                "  enabled: true",
                "  kb_dirs:",
                "    - \"/app/orca_runs\"",
                "    - \"/app/orca_outputs\"",
            ]
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=test_token\nALLOWED_TELEGRAM_USERS=111",
        encoding="utf-8",
    )

    settings = load_config(config_path=str(config_path), env_file=str(env_path))
    assert settings.rag.enabled is True
    assert settings.rag.kb_dirs == ["/app/orca_runs", "/app/orca_outputs"]
