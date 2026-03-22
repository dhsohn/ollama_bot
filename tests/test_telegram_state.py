from __future__ import annotations

from core.telegram_state import (
    ContinuationStore,
    PendingReasonStore,
    PreviewCacheStore,
)


def test_continuation_store_set_take_and_cleanup() -> None:
    store = ContinuationStore()
    now = 100.0

    store.set(
        111,
        root_query="long answer",
        turn=2,
        ttl_seconds=30.0,
        monotonic_fn=lambda: now,
    )

    assert store.entries[111]["root_query"] == "long answer"
    assert store.entries[111]["turn"] == 2

    pending = store.take(111, monotonic_fn=lambda: now + 10.0)
    assert pending is not None
    assert pending["turn"] == 2
    assert 111 not in store.entries


def test_continuation_store_cleanup_removes_expired_entries() -> None:
    store = ContinuationStore()
    store.entries[111] = {"root_query": "q", "turn": 1, "expires": 10.0}
    store.entries[222] = {"root_query": "q2", "turn": 1, "expires": 50.0}

    store.cleanup(monotonic_fn=lambda: 20.0)

    assert 111 not in store.entries
    assert 222 in store.entries


def test_pending_reason_store_cleanup_and_discard() -> None:
    store = PendingReasonStore()
    store.set(111, bot_message_id=42, expires=5.0)
    store.set(222, bot_message_id=43, expires=50.0)

    store.cleanup(monotonic_fn=lambda: 10.0)
    assert store.get(111) is None
    assert store.get(222) is not None

    store.discard(222)
    assert store.get(222) is None


def test_preview_cache_store_prunes_by_ttl() -> None:
    store = PreviewCacheStore()
    store.entries[(111, 1)] = {"user": "q", "bot": "a", "ts": 0.0}
    store.entries[(111, 2)] = {"user": "q2", "bot": "a2", "ts": 7200.0}

    store.prune(
        max_size=10,
        ttl_hours=1,
        monotonic_fn=lambda: 7201.0,
    )

    assert (111, 1) not in store.entries
    assert (111, 2) in store.entries


def test_preview_cache_store_enforces_max_size_when_caching() -> None:
    store = PreviewCacheStore()
    current = 0.0

    def monotonic() -> float:
        return current

    store.cache(
        chat_id=111,
        bot_message_id=1,
        user_text="u1",
        bot_text="b1",
        max_chars=10,
        max_size=2,
        ttl_hours=1,
        monotonic_fn=monotonic,
    )
    current = 1.0
    store.cache(
        chat_id=111,
        bot_message_id=2,
        user_text="u2",
        bot_text="b2",
        max_chars=10,
        max_size=2,
        ttl_hours=1,
        monotonic_fn=monotonic,
    )
    current = 2.0
    store.cache(
        chat_id=111,
        bot_message_id=3,
        user_text="u3",
        bot_text="b3",
        max_chars=10,
        max_size=2,
        ttl_hours=1,
        monotonic_fn=monotonic,
    )

    assert (111, 1) not in store.entries
    assert (111, 2) in store.entries
    assert (111, 3) in store.entries


def test_preview_cache_store_get_returns_empty_mapping_for_missing_key() -> None:
    store = PreviewCacheStore()

    assert store.get(111, 42) == {}
