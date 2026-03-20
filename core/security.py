"""Security module for auth, rate limiting, input validation, and path isolation.

Centralizes security checks so other modules call `SecurityManager` instead of
implementing security logic themselves.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import time
import unicodedata
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from core.config import SecurityConfig
from core.logging_setup import get_logger


class AuthenticationError(Exception):
    """Raised when an unauthenticated user attempts access."""


class RateLimitError(Exception):
    """Raised when the rate limit is exceeded."""


class SecurityViolationError(Exception):
    """Raised on security violations such as path traversal or blocked input."""


class GlobalConcurrencyError(Exception):
    """Raised when the global concurrency limit is exceeded."""


# ANSI escape-sequence pattern
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Allowed tools per security level
_TOOLS_BY_LEVEL: dict[str, set[str]] = {
    "safe": {"file_read"},
    "cautious": {"file_read", "file_write"},
    "restricted": {"file_read", "file_write", "shell", "network"},
}


class SecurityManager:
    """Security manager for auth, rate limiting, input validation, and path isolation."""

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

    # Authentication

    def authenticate(self, chat_id: int) -> bool:
        """Check whether the chat ID is present in the whitelist.

        Returns:
            True if authenticated.

        Raises:
            AuthenticationError: Raised for unauthenticated users.
        """
        if chat_id in self._allowed_users:
            self._logger.debug("auth_success", chat_id=chat_id)
            return True
        self._logger.warning("auth_failure", chat_id=chat_id)
        raise AuthenticationError(f"Unauthorized chat_id: {chat_id}")

    # Rate limiting

    def check_rate_limit(self, chat_id: int) -> bool:
        """Check the sliding-window rate limit.

        Requests are blocked when they reach `rate_limit` within 60 seconds.

        Raises:
            RateLimitError: Raised when the limit is exceeded.
        """
        now = time.monotonic()
        self._rate_limit_checks_since_cleanup += 1
        if self._rate_limit_checks_since_cleanup >= self._request_log_cleanup_every_calls:
            self._cleanup_request_log(now)
            self._rate_limit_checks_since_cleanup = 0

        # Avoid implicit defaultdict key creation to prevent stale chat_id entries.
        window = self._request_log.get(chat_id)
        if window is None:
            window = deque()

        # Drop entries older than 60 seconds.
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
        """Remove stale rate-limit windows for inactive chats."""
        stale_chat_ids = [
            chat_id
            for chat_id, window in self._request_log.items()
            if (not window) or (now - window[-1] >= self._request_log_ttl_seconds)
        ]
        for chat_id in stale_chat_ids:
            self._request_log.pop(chat_id, None)

    async def acquire_global_slot(self, chat_id: int) -> None:
        """Acquire a global concurrency slot.

        Raises:
            GlobalConcurrencyError: Raised when the global limit has been reached.
        """
        try:
            await asyncio.wait_for(self._global_semaphore.acquire(), timeout=0.001)
        except TimeoutError as exc:
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
            # If cancelled after acquire, release immediately to avoid leaking the slot.
            self._global_semaphore.release()
            raise

        self._global_in_flight += 1

    def release_global_slot(self) -> None:
        """Release a global concurrency slot."""
        if self._global_in_flight <= 0:
            return
        self._global_in_flight -= 1
        self._global_semaphore.release()

    @asynccontextmanager
    async def global_slot(self, chat_id: int) -> AsyncGenerator[None, None]:
        """Safely bind a global slot to a request scope."""
        await self.acquire_global_slot(chat_id)
        try:
            yield
        finally:
            self.release_global_slot()

    # Input validation

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input.

        - Remove null bytes
        - Remove ANSI escapes
        - Enforce the `security.max_input_length` limit
        - Normalize to Unicode NFC
        """
        # Remove null bytes.
        text = text.replace("\x00", "")

        # Remove ANSI escapes.
        text = _ANSI_RE.sub("", text)

        # Normalize Unicode.
        text = unicodedata.normalize("NFC", text)

        # Enforce the maximum length.
        if len(text) > self._max_input_length:
            self._logger.warning("input_truncated", original_length=len(text))
            text = text[:self._max_input_length]

        return text

    # Path isolation

    def validate_path(self, path: str, base_dir: str = "data") -> Path:
        """Validate that a file path stays within the allowed base directory.

        Raises:
            SecurityViolationError: Raised on path traversal or blocked paths.
        """
        resolved = Path(base_dir).joinpath(path).resolve()
        base_resolved = Path(base_dir).resolve()

        # Ensure the resolved path does not escape base_dir.
        try:
            resolved.relative_to(base_resolved)
        except ValueError as exc:
            self._logger.warning(
                "path_traversal_blocked",
                requested=path,
                resolved=str(resolved),
            )
            raise SecurityViolationError(
                f"Path traversal detected: {path}"
            ) from exc

        # Check blocked-path patterns.
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

    # Skill security

    def check_skill_security(
        self, security_level: str, allowed_tools: list[str]
    ) -> bool:
        """Validate that a skill's security level matches its requested tools.

        Returns:
            True if valid.

        Raises:
            SecurityViolationError: Raised for tools not allowed at that security level.
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

    # File-size validation

    def validate_file_size(self, size_bytes: int) -> bool:
        """Validate that the file size is within the allowed limit.

        Raises:
            SecurityViolationError: Raised when the limit is exceeded.
        """
        if size_bytes > self._max_file_size:
            raise SecurityViolationError(
                f"File size {size_bytes} exceeds limit {self._max_file_size}"
            )
        return True
