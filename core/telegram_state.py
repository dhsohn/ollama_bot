"""State containers for Telegram interaction helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ContinuationStore:
    """Manage pending continuation prompts with expiration."""

    def __init__(self) -> None:
        self.entries: dict[int, dict[str, Any]] = {}

    def cleanup(
        self,
        *,
        monotonic_fn: Callable[[], float],
    ) -> None:
        if not self.entries:
            return
        now = monotonic_fn()
        expired_chat_ids = [
            chat_id
            for chat_id, pending in self.entries.items()
            if now > float(pending.get("expires", 0.0))
        ]
        for chat_id in expired_chat_ids:
            del self.entries[chat_id]

    def take(
        self,
        chat_id: int,
        *,
        monotonic_fn: Callable[[], float],
    ) -> dict[str, Any] | None:
        self.cleanup(monotonic_fn=monotonic_fn)
        pending = self.entries.get(chat_id)
        if pending is None:
            return None
        del self.entries[chat_id]
        return pending

    def set(
        self,
        chat_id: int,
        *,
        root_query: str,
        turn: int,
        ttl_seconds: float,
        monotonic_fn: Callable[[], float],
    ) -> None:
        self.cleanup(monotonic_fn=monotonic_fn)
        self.entries[chat_id] = {
            "root_query": root_query,
            "turn": max(1, turn),
            "expires": monotonic_fn() + ttl_seconds,
        }

    def discard(self, chat_id: int) -> None:
        self.entries.pop(chat_id, None)


class PreviewCacheStore:
    """Manage cached feedback previews with TTL and max-size pruning."""

    def __init__(self) -> None:
        self.entries: dict[tuple[int, int], dict[str, Any]] = {}

    def prune(
        self,
        *,
        max_size: int,
        ttl_hours: int,
        monotonic_fn: Callable[[], float],
    ) -> None:
        if max_size <= 0 or ttl_hours <= 0:
            self.entries.clear()
            return

        now = monotonic_fn()
        ttl_seconds = ttl_hours * 3600
        expired_keys = [
            key
            for key, value in self.entries.items()
            if now - float(value.get("ts", 0.0)) > ttl_seconds
        ]
        for key in expired_keys:
            del self.entries[key]

        while len(self.entries) > max_size:
            oldest_key = min(
                self.entries,
                key=lambda key: float(self.entries[key].get("ts", 0.0)),
            )
            del self.entries[oldest_key]

    def cache(
        self,
        *,
        chat_id: int,
        bot_message_id: int,
        user_text: str,
        bot_text: str,
        max_chars: int,
        max_size: int,
        ttl_hours: int,
        monotonic_fn: Callable[[], float],
    ) -> None:
        if max_chars <= 0 or max_size <= 0 or ttl_hours <= 0:
            return

        self.prune(
            max_size=max_size,
            ttl_hours=ttl_hours,
            monotonic_fn=monotonic_fn,
        )
        while len(self.entries) >= max_size:
            oldest_key = min(
                self.entries,
                key=lambda key: float(self.entries[key].get("ts", 0.0)),
            )
            del self.entries[oldest_key]

        self.entries[(chat_id, bot_message_id)] = {
            "user": user_text[:max_chars],
            "bot": bot_text[:max_chars],
            "ts": monotonic_fn(),
        }

    def get(self, chat_id: int, bot_message_id: int) -> dict[str, Any]:
        return self.entries.get((chat_id, bot_message_id), {})


class PendingReasonStore:
    """Manage pending feedback reasons with expiration."""

    def __init__(self) -> None:
        self.entries: dict[int, dict[str, Any]] = {}

    def cleanup(
        self,
        *,
        monotonic_fn: Callable[[], float],
    ) -> None:
        if not self.entries:
            return
        now = monotonic_fn()
        expired_chat_ids = [
            chat_id
            for chat_id, pending in self.entries.items()
            if now > float(pending.get("expires", 0.0))
        ]
        for chat_id in expired_chat_ids:
            del self.entries[chat_id]

    def get(self, chat_id: int) -> dict[str, Any] | None:
        return self.entries.get(chat_id)

    def set(
        self,
        chat_id: int,
        **payload: Any,
    ) -> None:
        self.entries[chat_id] = dict(payload)

    def pop(self, chat_id: int) -> dict[str, Any] | None:
        return self.entries.pop(chat_id, None)

    def discard(self, chat_id: int) -> None:
        self.entries.pop(chat_id, None)


@dataclass
class TelegramInteractionState:
    """Aggregate state buckets used by the Telegram flow helpers."""

    previews: PreviewCacheStore = field(default_factory=PreviewCacheStore)
    pending_reasons: PendingReasonStore = field(default_factory=PendingReasonStore)
    continuations: ContinuationStore = field(default_factory=ContinuationStore)
