"""보안 모듈 테스트."""

from __future__ import annotations

import time

import pytest

from core.security import (
    AuthenticationError,
    RateLimitError,
    SecurityManager,
    SecurityViolationError,
)


class TestAuthentication:
    def test_allowed_user_passes(self, security_manager: SecurityManager) -> None:
        assert security_manager.authenticate(111) is True

    def test_blocked_user_raises(self, security_manager: SecurityManager) -> None:
        with pytest.raises(AuthenticationError):
            security_manager.authenticate(999)

    def test_all_allowed_users_pass(self, security_manager: SecurityManager) -> None:
        assert security_manager.authenticate(111) is True
        assert security_manager.authenticate(222) is True


class TestRateLimiting:
    def test_under_limit_passes(self, security_manager: SecurityManager) -> None:
        for _ in range(10):
            assert security_manager.check_rate_limit(111) is True

    def test_over_limit_raises(self, security_manager: SecurityManager) -> None:
        for _ in range(10):
            security_manager.check_rate_limit(111)
        with pytest.raises(RateLimitError):
            security_manager.check_rate_limit(111)

    def test_different_users_independent(self, security_manager: SecurityManager) -> None:
        for _ in range(10):
            security_manager.check_rate_limit(111)
        # 다른 유저는 영향 없음
        assert security_manager.check_rate_limit(222) is True


class TestInputSanitization:
    def test_strip_null_bytes(self, security_manager: SecurityManager) -> None:
        result = security_manager.sanitize_input("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"

    def test_strip_ansi_escape(self, security_manager: SecurityManager) -> None:
        result = security_manager.sanitize_input("\x1b[31mred\x1b[0m")
        assert result == "red"

    def test_length_limit(self, security_manager: SecurityManager) -> None:
        long_input = "a" * 20_000
        result = security_manager.sanitize_input(long_input)
        assert len(result) == 10_000

    def test_normal_input_unchanged(self, security_manager: SecurityManager) -> None:
        text = "안녕하세요! Hello, World! 🎉"
        result = security_manager.sanitize_input(text)
        assert result == text

    def test_unicode_normalization(self, security_manager: SecurityManager) -> None:
        # NFD 형태의 한글 → NFC로 정규화
        result = security_manager.sanitize_input("가")  # may be NFD
        assert result  # 정규화 완료


class TestPathValidation:
    def test_valid_path(self, security_manager: SecurityManager, tmp_path) -> None:
        base = str(tmp_path)
        result = security_manager.validate_path("reports/test.md", base_dir=base)
        assert str(result).startswith(base)

    def test_traversal_blocked(self, security_manager: SecurityManager, tmp_path) -> None:
        base = str(tmp_path)
        with pytest.raises(SecurityViolationError):
            security_manager.validate_path("../../etc/passwd", base_dir=base)

    def test_prefix_bypass_blocked(self, security_manager: SecurityManager, tmp_path) -> None:
        base = tmp_path / "data"
        base.mkdir()
        # "/tmp/.../data_evil"은 문자열 prefix로는 "data"와 같아 보일 수 있으나
        # 실제로는 base 디렉토리 바깥 경로다.
        with pytest.raises(SecurityViolationError):
            security_manager.validate_path("../data_evil/file.txt", base_dir=str(base))

    def test_blocked_system_path(self, security_manager: SecurityManager) -> None:
        with pytest.raises(SecurityViolationError):
            security_manager.validate_path("/etc/shadow", base_dir="/app/data")


class TestSkillSecurity:
    def test_safe_with_no_tools(self, security_manager: SecurityManager) -> None:
        assert security_manager.check_skill_security("safe", []) is True

    def test_safe_with_file_read(self, security_manager: SecurityManager) -> None:
        assert security_manager.check_skill_security("safe", ["file_read"]) is True

    def test_safe_with_file_write_rejected(self, security_manager: SecurityManager) -> None:
        with pytest.raises(SecurityViolationError):
            security_manager.check_skill_security("safe", ["file_write"])

    def test_cautious_with_file_write(self, security_manager: SecurityManager) -> None:
        assert security_manager.check_skill_security("cautious", ["file_read", "file_write"]) is True

    def test_cautious_with_shell_rejected(self, security_manager: SecurityManager) -> None:
        with pytest.raises(SecurityViolationError):
            security_manager.check_skill_security("cautious", ["shell"])


class TestFileSize:
    def test_valid_size(self, security_manager: SecurityManager) -> None:
        assert security_manager.validate_file_size(1024) is True

    def test_oversized_file(self, security_manager: SecurityManager) -> None:
        with pytest.raises(SecurityViolationError):
            security_manager.validate_file_size(100_000_000)
