"""설정 로더 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import FeedbackConfig, load_config


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _minimal_yaml(extra: str = "") -> str:
    """telegram 시크릿이 포함된 최소 YAML."""
    base = (
        "telegram:\n"
        "  bot_token: \"test_token\"\n"
        "  allowed_users: \"111\"\n"
    )
    if extra:
        base += extra
    return base


def test_allowed_telegram_users_parsed_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, (
        "telegram:\n"
        "  bot_token: \"test_token\"\n"
        "  allowed_users: \"123456789, 987654321\"\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.security.allowed_users == [123456789, 987654321]


def test_allowed_telegram_users_invalid_value_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, (
        "telegram:\n"
        "  bot_token: \"test_token\"\n"
        "  allowed_users: \"123,abc,456\"\n"
    ))

    with pytest.raises(ValueError, match=r"telegram\.allowed_users"):
        load_config(config_path=str(config_path))


def test_allowed_telegram_users_negative_ids_parsed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, (
        "telegram:\n"
        "  bot_token: \"test_token\"\n"
        "  allowed_users: \"-100123, 456\"\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.security.allowed_users == [-100123, 456]


def test_scheduler_timezone_yaml_used(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, (
        "telegram:\n"
        "  bot_token: \"test_token\"\n"
        "  allowed_users: \"123456789\"\n"
        "scheduler:\n"
        "  timezone: \"UTC\"\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.scheduler.timezone == "UTC"


def test_scheduler_timezone_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml())

    settings = load_config(config_path=str(config_path))
    assert settings.scheduler.timezone == "Asia/Seoul"


def test_yaml_values_preserved(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, (
        "telegram:\n"
        "  bot_token: \"test_token\"\n"
        "  allowed_users: \"123456789\"\n"
        "lemonade:\n"
        "  host: \"http://yaml-host:8000\"\n"
        "  default_model: \"yaml-model\"\n"
        "scheduler:\n"
        "  timezone: \"UTC\"\n"
    ))

    settings = load_config(config_path=str(config_path))
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
    _write_yaml(config_path, _minimal_yaml(
        "runtime_maintenance:\n"
        "  memory_maintenance_interval_seconds: 300\n"
        "  llm_recovery_interval_seconds: 15\n"
        "  memory_maintenance_jitter_ratio: 0.2\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.runtime_maintenance.memory_maintenance_interval_seconds == 300
    assert settings.runtime_maintenance.llm_recovery_interval_seconds == 15
    assert settings.runtime_maintenance.memory_maintenance_jitter_ratio == 0.2


def test_runtime_maintenance_invalid_jitter_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "runtime_maintenance:\n"
        "  memory_maintenance_jitter_ratio: 1.5\n"
    ))

    with pytest.raises(ValueError, match="memory_maintenance_jitter_ratio"):
        load_config(config_path=str(config_path))


def test_security_invalid_numeric_value_raises(tmp_path: Path) -> None:
    """security 숫자 설정이 1 미만이면 예외가 발생한다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "security:\n"
        "  max_concurrent_requests: 0\n"
    ))

    with pytest.raises(ValueError, match="security numeric settings must be >= 1"):
        load_config(config_path=str(config_path))


def test_feedback_section_loaded_from_yaml(tmp_path: Path) -> None:
    """YAML의 feedback 섹션이 올바르게 로드된다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "feedback:\n"
        "  enabled: false\n"
        "  show_buttons: false\n"
        "  min_feedback_for_analysis: 10\n"
        "  max_guidelines: 3\n"
        "  retention_days: 30\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.feedback.enabled is False
    assert settings.feedback.show_buttons is False
    assert settings.feedback.min_feedback_for_analysis == 10
    assert settings.feedback.max_guidelines == 3
    assert settings.feedback.retention_days == 30


def test_feedback_invalid_numeric_value_raises(tmp_path: Path) -> None:
    """feedback 숫자 설정이 1 미만이면 예외가 발생한다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "feedback:\n"
        "  preview_cache_max_size: 0\n"
    ))

    with pytest.raises(ValueError, match="feedback numeric settings must be >= 1"):
        load_config(config_path=str(config_path))


def test_response_planner_section_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "response_planner:\n"
        "  enabled: true\n"
        "  min_input_chars: 120\n"
        "  trigger_intents:\n"
        "    - complex\n"
        "    - code\n"
        "  force_for_rag: true\n"
        "  max_plan_tokens: 320\n"
        "  timeout_seconds: 30\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.response_planner.enabled is True
    assert settings.response_planner.min_input_chars == 120
    assert settings.response_planner.trigger_intents == ["complex", "code"]
    assert settings.response_planner.force_for_rag is True
    assert settings.response_planner.max_plan_tokens == 320
    assert settings.response_planner.timeout_seconds == 30


def test_response_reviewer_section_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "response_reviewer:\n"
        "  enabled: true\n"
        "  only_when_planner_used: false\n"
        "  max_review_tokens: 512\n"
        "  timeout_seconds: 25\n"
        "  stream_buffering: false\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.response_reviewer.enabled is True
    assert settings.response_reviewer.only_when_planner_used is False
    assert settings.response_reviewer.max_review_tokens == 512
    assert settings.response_reviewer.timeout_seconds == 25
    assert settings.response_reviewer.stream_buffering is False


def test_partial_lemonade_section_preserves_defaults(tmp_path: Path) -> None:
    """lemonade 섹션 일부만 지정해도 누락 필드는 기본값을 유지한다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "lemonade:\n"
        "  default_model: \"custom-model\"\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.lemonade.default_model == "custom-model"
    assert settings.lemonade.host == "http://localhost:8000"
    assert settings.lemonade.temperature == 0.7
    assert settings.lemonade.max_tokens == 4096


def test_partial_feedback_section_preserves_defaults(tmp_path: Path) -> None:
    """feedback 섹션 일부 오버라이드 시 나머지 기본값이 유지된다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "feedback:\n"
        "  retention_days: 14\n"
        "  collect_reason: false\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.feedback.retention_days == 14
    assert settings.feedback.collect_reason is False
    assert settings.feedback.preview_cache_max_size == 500
    assert settings.feedback.preview_cache_ttl_hours == 24


def test_lemonade_values_from_yaml_only(tmp_path: Path) -> None:
    """lemonade 설정은 YAML에서만 로드된다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml())

    settings = load_config(config_path=str(config_path))
    assert settings.lemonade.host == "http://localhost:8000"
    assert settings.lemonade.default_model == "gpt-oss-20b-NPU"
    assert settings.lemonade.api_key == ""
    assert settings.lemonade.base_path == "/api/v1"
    assert settings.lemonade.timeout_seconds == 60


def test_unknown_top_level_key_ignored(tmp_path: Path) -> None:
    """알 수 없는 최상위 키는 무시되고 기본값이 유지된다."""
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "unknown_key: unsupported\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.lemonade.default_model == "gpt-oss-20b-NPU"


def test_rag_kb_dirs_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "rag:\n"
        "  enabled: true\n"
        "  kb_dirs:\n"
        "    - \"./kb\"\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.rag.enabled is True
    assert settings.rag.kb_dirs == ["./kb"]


def test_strict_startup_default_false(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml())

    settings = load_config(config_path=str(config_path))
    assert settings.strict_startup is False


def test_strict_startup_loaded_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _minimal_yaml(
        "strict_startup: true\n"
    ))

    settings = load_config(config_path=str(config_path))
    assert settings.strict_startup is True
