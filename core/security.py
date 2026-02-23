"""보안 모듈 — 인증, 레이트리밋, 입력 검증, 경로 격리.

모든 보안 검사를 한 곳에서 관리한다.
다른 모듈은 직접 보안 로직을 구현하지 않고 SecurityManager를 호출한다.
"""

from __future__ import annotations

import fnmatch
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from core.config import SecurityConfig
from core.logging_setup import get_logger


class AuthenticationError(Exception):
    """미인증 사용자 접근 시 발생."""


class RateLimitError(Exception):
    """레이트리밋 초과 시 발생."""


class SecurityViolationError(Exception):
    """경로 탐색, 차단 입력 등 보안 위반 시 발생."""


# ANSI 이스케이프 시퀀스 패턴
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# 보안 등급별 허용 도구
_TOOLS_BY_LEVEL: dict[str, set[str]] = {
    "safe": {"file_read"},
    "cautious": {"file_read", "file_write"},
    "restricted": {"file_read", "file_write", "shell", "network"},
}


class SecurityManager:
    """보안 관리자. 인증, 레이트리밋, 입력 검증, 경로 격리를 담당한다."""

    def __init__(self, config: SecurityConfig) -> None:
        self._allowed_users: set[int] = set(config.allowed_users)
        self._rate_limit: int = config.rate_limit
        self._max_file_size: int = config.max_file_size
        self._blocked_paths: list[str] = config.blocked_paths
        self._request_log: dict[int, list[float]] = defaultdict(list)
        self._logger = get_logger("security")

    # ── 인증 ──

    def authenticate(self, chat_id: int) -> bool:
        """Chat ID가 화이트리스트에 있는지 확인한다.

        Returns:
            True if authenticated.

        Raises:
            AuthenticationError: 미인증 사용자인 경우.
        """
        if chat_id in self._allowed_users:
            self._logger.debug("auth_success", chat_id=chat_id)
            return True
        self._logger.warning("auth_failure", chat_id=chat_id)
        raise AuthenticationError(f"Unauthorized chat_id: {chat_id}")

    # ── 레이트리밋 ──

    def check_rate_limit(self, chat_id: int) -> bool:
        """슬라이딩 윈도우 레이트리밋을 검사한다.

        60초 내 요청 수가 rate_limit 이상이면 차단한다.

        Raises:
            RateLimitError: 초과 시.
        """
        now = time.monotonic()
        window = self._request_log[chat_id]

        # 60초 이전 항목 제거
        self._request_log[chat_id] = [t for t in window if now - t < 60.0]
        window = self._request_log[chat_id]

        if len(window) >= self._rate_limit:
            self._logger.warning(
                "rate_limit_exceeded", chat_id=chat_id, count=len(window)
            )
            raise RateLimitError(
                f"Rate limit exceeded for chat_id {chat_id}: "
                f"{len(window)}/{self._rate_limit} per minute"
            )

        window.append(now)
        return True

    # ── 입력 검증 ──

    def sanitize_input(self, text: str) -> str:
        """사용자 입력을 정제한다.

        - Null 바이트 제거
        - ANSI 이스케이프 제거
        - 길이 제한 (10,000자)
        - Unicode NFC 정규화
        """
        # Null 바이트 제거
        text = text.replace("\x00", "")

        # ANSI 이스케이프 제거
        text = _ANSI_RE.sub("", text)

        # Unicode 정규화
        text = unicodedata.normalize("NFC", text)

        # 길이 제한
        if len(text) > 10_000:
            self._logger.warning("input_truncated", original_length=len(text))
            text = text[:10_000]

        return text

    # ── 경로 격리 ──

    def validate_path(self, path: str, base_dir: str = "/app/data") -> Path:
        """파일 경로가 허용된 디렉토리 내에 있는지 검증한다.

        Raises:
            SecurityViolationError: 경로 탐색 또는 차단 경로 접근 시.
        """
        resolved = Path(base_dir).joinpath(path).resolve()
        base_resolved = Path(base_dir).resolve()

        # base_dir 바깥으로 나가는지 확인
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            self._logger.warning(
                "path_traversal_blocked",
                requested=path,
                resolved=str(resolved),
            )
            raise SecurityViolationError(
                f"Path traversal detected: {path}"
            )

        # 차단 경로 패턴 체크
        for pattern in self._blocked_paths:
            if fnmatch.fnmatch(str(resolved), pattern):
                self._logger.warning(
                    "blocked_path_access",
                    requested=path,
                    pattern=pattern,
                )
                raise SecurityViolationError(
                    f"Access to blocked path: {path}"
                )

        return resolved

    # ── 스킬 보안 ──

    def check_skill_security(
        self, security_level: str, allowed_tools: list[str]
    ) -> bool:
        """스킬의 보안 등급과 요청 도구가 호환되는지 검증한다.

        Returns:
            True if valid.

        Raises:
            SecurityViolationError: 보안 등급에 맞지 않는 도구를 요청한 경우.
        """
        permitted = _TOOLS_BY_LEVEL.get(security_level, set())
        requested = set(allowed_tools)
        unauthorized = requested - permitted

        if unauthorized:
            self._logger.warning(
                "skill_security_violation",
                security_level=security_level,
                unauthorized_tools=list(unauthorized),
            )
            raise SecurityViolationError(
                f"Tools {unauthorized} not allowed for "
                f"security level '{security_level}'"
            )
        return True

    # ── 파일 크기 검증 ──

    def validate_file_size(self, size_bytes: int) -> bool:
        """파일 크기가 허용 범위 내인지 확인한다.

        Raises:
            SecurityViolationError: 초과 시.
        """
        if size_bytes > self._max_file_size:
            raise SecurityViolationError(
                f"File size {size_bytes} exceeds limit {self._max_file_size}"
            )
        return True
