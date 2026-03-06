"""보안 모듈 — 인증, 레이트리밋, 입력 검증, 경로 격리.

모든 보안 검사를 한 곳에서 관리한다.
다른 모듈은 직접 보안 로직을 구현하지 않고 SecurityManager를 호출한다.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import time
import unicodedata
from collections.abc import AsyncGenerator
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from core.config import SecurityConfig
from core.logging_setup import get_logger


class AuthenticationError(Exception):
    """미인증 사용자 접근 시 발생."""


class RateLimitError(Exception):
    """레이트리밋 초과 시 발생."""


class SecurityViolationError(Exception):
    """경로 탐색, 차단 입력 등 보안 위반 시 발생."""


class GlobalConcurrencyError(Exception):
    """전역 동시 처리 한도 초과 시 발생."""


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
        self._max_concurrent_requests: int = config.max_concurrent_requests
        self._max_input_length: int = config.max_input_length
        self._max_file_size: int = config.max_file_size
        self._blocked_paths: list[str] = config.blocked_paths
        self._request_log: dict[int, deque[float]] = defaultdict(deque)
        self._rate_limit_window_seconds = 60.0
        self._request_log_ttl_seconds = 600.0
        self._request_log_cleanup_every_calls = 256
        self._rate_limit_checks_since_cleanup = 0
        self._global_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        self._global_in_flight = 0
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
        self._rate_limit_checks_since_cleanup += 1
        if self._rate_limit_checks_since_cleanup >= self._request_log_cleanup_every_calls:
            self._cleanup_request_log(now)
            self._rate_limit_checks_since_cleanup = 0

        # defaultdict의 암시적 키 생성을 피해서, 불필요한 chat_id 항목 잔류를 줄인다.
        window = self._request_log.get(chat_id)
        if window is None:
            window = deque()

        # 60초 이전 항목 제거
        while window and now - window[0] >= self._rate_limit_window_seconds:
            window.popleft()
        if not window:
            self._request_log.pop(chat_id, None)

        if len(window) >= self._rate_limit:
            self._logger.warning(
                "rate_limit_exceeded", chat_id=chat_id, count=len(window)
            )
            raise RateLimitError(
                f"Rate limit exceeded for chat_id {chat_id}: "
                f"{len(window)}/{self._rate_limit} per minute"
            )

        window.append(now)
        self._request_log[chat_id] = window
        return True

    def _cleanup_request_log(self, now: float) -> None:
        """오랫동안 요청이 없던 레이트리밋 윈도우를 제거한다."""
        stale_chat_ids = [
            chat_id
            for chat_id, window in self._request_log.items()
            if (not window) or (now - window[-1] >= self._request_log_ttl_seconds)
        ]
        for chat_id in stale_chat_ids:
            self._request_log.pop(chat_id, None)

    async def acquire_global_slot(self, chat_id: int) -> None:
        """전역 동시 요청 슬롯을 획득한다.

        Raises:
            GlobalConcurrencyError: 현재 전역 동시 처리량이 한도에 도달한 경우.
        """
        try:
            await asyncio.wait_for(self._global_semaphore.acquire(), timeout=0.001)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            self._logger.warning(
                "global_concurrency_exceeded",
                chat_id=chat_id,
                in_flight=self._global_in_flight,
                limit=self._max_concurrent_requests,
            )
            raise GlobalConcurrencyError(
                "Too many concurrent requests globally"
            ) from exc
        except asyncio.CancelledError:
            # 세마포어 acquire 후 취소되면 슬롯이 영구 소실되므로 즉시 반환
            self._global_semaphore.release()
            raise

        self._global_in_flight += 1

    def release_global_slot(self) -> None:
        """전역 동시 요청 슬롯을 반환한다."""
        if self._global_in_flight <= 0:
            return
        self._global_in_flight -= 1
        self._global_semaphore.release()

    @asynccontextmanager
    async def global_slot(self, chat_id: int) -> AsyncGenerator[None, None]:
        """전역 동시 요청 슬롯을 요청 범위에 묶어 안전하게 관리한다."""
        await self.acquire_global_slot(chat_id)
        try:
            yield
        finally:
            self.release_global_slot()

    # ── 입력 검증 ──

    def sanitize_input(self, text: str) -> str:
        """사용자 입력을 정제한다.

        - Null 바이트 제거
        - ANSI 이스케이프 제거
        - 길이 제한 (security.max_input_length)
        - Unicode NFC 정규화
        """
        # Null 바이트 제거
        text = text.replace("\x00", "")

        # ANSI 이스케이프 제거
        text = _ANSI_RE.sub("", text)

        # Unicode 정규화
        text = unicodedata.normalize("NFC", text)

        # 길이 제한
        if len(text) > self._max_input_length:
            self._logger.warning("input_truncated", original_length=len(text))
            text = text[:self._max_input_length]

        return text

    # ── 경로 격리 ──

    def validate_path(self, path: str, base_dir: str = "data") -> Path:
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
