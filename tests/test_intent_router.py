"""intent_router 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import core.intent_router as intent_router_module
from core.intent_router import IntentRouter


def _write_routes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "min_confidence: 0.75",
                "routes:",
                "  - name: code",
                "    utterances:",
                "      - \"코드 짜줘\"",
                "    strategy: {}",
                "  - name: simple_qa",
                "    utterances:",
                "      - \"수학과 커리큘럼 알려줘\"",
                "    strategy: {}",
            ]
        ),
        encoding="utf-8",
    )


def test_code_guard_rejects_non_code_query_even_when_code_score_is_higher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes_path = tmp_path / "intent_routes.yaml"
    _write_routes(routes_path)

    def _fake_embed_texts(_encoder, texts, normalize: bool = True):
        vectors: list[np.ndarray] = []
        for text in texts:
            if "코드 짜줘" in text:
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
            elif "수학과 커리큘럼 알려줘" in text:
                vectors.append(np.array([0.95, 0.0], dtype=np.float32))
            else:
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
        return vectors

    monkeypatch.setattr(intent_router_module, "embed_texts", _fake_embed_texts)

    router = IntentRouter(
        routes_path=str(routes_path),
        encoder=object(),
        min_confidence=0.75,
    )

    result = router.classify("수학과 커리큘럼 알려줘")

    assert result is not None
    assert result.intent == "simple_qa"


def test_code_guard_keeps_code_intent_when_code_signal_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    routes_path = tmp_path / "intent_routes.yaml"
    _write_routes(routes_path)

    def _fake_embed_texts(_encoder, texts, normalize: bool = True):
        vectors: list[np.ndarray] = []
        for text in texts:
            if "코드 짜줘" in text:
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
            elif "수학과 커리큘럼 알려줘" in text:
                vectors.append(np.array([0.95, 0.0], dtype=np.float32))
            else:
                vectors.append(np.array([1.0, 0.0], dtype=np.float32))
        return vectors

    monkeypatch.setattr(intent_router_module, "embed_texts", _fake_embed_texts)

    router = IntentRouter(
        routes_path=str(routes_path),
        encoder=object(),
        min_confidence=0.75,
    )

    result = router.classify("파이썬 코드 짜줘")

    assert result is not None
    assert result.intent == "code"
