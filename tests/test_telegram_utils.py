"""telegram_utils 및 telegram_decorators 모듈 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.telegram_utils import (
    coerce_error_list,
    format_memory_gb,
    format_reload_warnings,
    get_auto_reload_errors,
    get_skill_reload_errors,
)


class TestFormatMemoryGb:
    def test_zero(self) -> None:
        assert format_memory_gb(0) == "0GB"

    def test_exact_gb(self) -> None:
        assert format_memory_gb(1024) == "1GB"
        assert format_memory_gb(2048) == "2GB"

    def test_fractional_gb_small(self) -> None:
        result = format_memory_gb(512)
        assert result == "0.50GB"

    def test_fractional_gb_large(self) -> None:
        result = format_memory_gb(10752)  # 10.5 GB
        assert result == "10.5GB"

    def test_string_input(self) -> None:
        assert format_memory_gb("1024") == "1GB"
        assert format_memory_gb("  2048  ") == "2GB"

    def test_invalid_string(self) -> None:
        assert format_memory_gb("not_a_number") == "0GB"

    def test_bool_treated_as_zero(self) -> None:
        assert format_memory_gb(True) == "0GB"
        assert format_memory_gb(False) == "0GB"

    def test_negative_clamped_to_zero(self) -> None:
        assert format_memory_gb(-100) == "0GB"

    def test_none_type(self) -> None:
        assert format_memory_gb(None) == "0GB"


class TestCoerceErrorList:
    def test_list_of_strings(self) -> None:
        assert coerce_error_list(["a", "b"]) == ["a", "b"]

    def test_list_of_mixed(self) -> None:
        assert coerce_error_list([1, None, "x"]) == ["1", "None", "x"]

    def test_non_list(self) -> None:
        assert coerce_error_list("string") == []
        assert coerce_error_list(42) == []
        assert coerce_error_list(None) == []


class TestGetSkillReloadErrors:
    def test_no_getter(self) -> None:
        engine = MagicMock(spec=[])
        assert get_skill_reload_errors(engine) == []

    def test_non_callable_getter(self) -> None:
        engine = MagicMock()
        engine.get_last_skill_load_errors = "not_callable"
        assert get_skill_reload_errors(engine) == []

    def test_getter_raises(self) -> None:
        engine = MagicMock()
        engine.get_last_skill_load_errors = MagicMock(side_effect=RuntimeError("fail"))
        assert get_skill_reload_errors(engine) == []

    def test_getter_returns_list(self) -> None:
        engine = MagicMock()
        engine.get_last_skill_load_errors = MagicMock(return_value=["err1", "err2"])
        assert get_skill_reload_errors(engine) == ["err1", "err2"]

    def test_getter_returns_awaitable(self) -> None:
        engine = MagicMock()
        coro = AsyncMock(return_value=["err"])()
        engine.get_last_skill_load_errors = MagicMock(return_value=coro)
        result = get_skill_reload_errors(engine)
        assert result == []


class TestGetAutoReloadErrors:
    def test_none_scheduler(self) -> None:
        assert get_auto_reload_errors(None) == []

    def test_no_getter(self) -> None:
        scheduler = MagicMock(spec=[])
        assert get_auto_reload_errors(scheduler) == []

    def test_non_callable_getter(self) -> None:
        scheduler = MagicMock()
        scheduler.get_last_load_errors = "not_callable"
        assert get_auto_reload_errors(scheduler) == []

    def test_getter_raises(self) -> None:
        scheduler = MagicMock()
        scheduler.get_last_load_errors = MagicMock(side_effect=RuntimeError("fail"))
        assert get_auto_reload_errors(scheduler) == []

    def test_getter_returns_list(self) -> None:
        scheduler = MagicMock()
        scheduler.get_last_load_errors = MagicMock(return_value=["auto_err"])
        assert get_auto_reload_errors(scheduler) == ["auto_err"]

    def test_getter_returns_awaitable(self) -> None:
        scheduler = MagicMock()
        coro = AsyncMock(return_value=["err"])()
        scheduler.get_last_load_errors = MagicMock(return_value=coro)
        result = get_auto_reload_errors(scheduler)
        assert result == []


class TestFormatReloadWarnings:
    def test_single_error(self) -> None:
        result = format_reload_warnings(["err1"])
        assert "err1" in result

    def test_truncates_at_max_items(self) -> None:
        errors = [f"err{i}" for i in range(10)]
        result = format_reload_warnings(errors, max_items=3)
        assert "err0" in result
        assert "err2" in result
        assert "err3" not in result
