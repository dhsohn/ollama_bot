"""Lemonade 다중 인스턴스 라우팅 테스트."""

from __future__ import annotations

from core.config import LemonadeConfig, LemonadeInstanceConfig
from core.lemonade_multi_client import LemonadeMultiClient


def test_embedding_and_reranker_are_pinned_to_primary_by_default() -> None:
    config = LemonadeConfig(
        host="http://primary:11434",
        instances=[
            LemonadeInstanceConfig(
                name="coder",
                host="http://coder:31434",
                model="Qwen3-Coder-Next-GGUF",
                route_models=[
                    "Qwen3-Embedding-0.6B-GGUF",
                    "bge-reranker-v2-m3-GGUF",
                ],
            )
        ],
    )
    client = LemonadeMultiClient(config)

    assert (
        client._resolve_route_key(
            "Qwen3-Embedding-0.6B-GGUF",
            "embedding",
        )
        == "primary"
    )
    assert (
        client._resolve_route_key(
            "bge-reranker-v2-m3-GGUF",
            "reranker",
        )
        == "primary"
    )
    # role 정보가 없으면 모델 기준 라우팅을 그대로 따른다.
    assert client._resolve_route_key("Qwen3-Embedding-0.6B-GGUF", None) == "coder"


def test_route_roles_can_explicitly_override_embedding_and_reranker() -> None:
    config = LemonadeConfig(
        host="http://primary:11434",
        instances=[
            LemonadeInstanceConfig(
                name="vector",
                host="http://vector:41434",
                route_roles=["embedding", "reranker"],
            )
        ],
    )
    client = LemonadeMultiClient(config)

    assert client._resolve_route_key("Qwen3-Embedding-0.6B-GGUF", "embedding") == "vector"
    assert client._resolve_route_key("bge-reranker-v2-m3-GGUF", "reranker") == "vector"
