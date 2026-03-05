#!/bin/bash
# 헬스체크 스크립트
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

"${PROJECT_ROOT}/.venv/bin/python" - <<'PY'
import asyncio
import sys

from core.config import OllamaConfig, load_config
from core.lemonade_client import LemonadeClient
from core.ollama_client import OllamaClient


async def check() -> None:
    try:
        config = load_config()
    except Exception as exc:
        print(f"FAIL: invalid config: {exc}")
        sys.exit(1)

    chat_client = LemonadeClient(config.lemonade)
    chat_client.default_model = config.lemonade.default_model
    retrieval_client = OllamaClient(
        config=OllamaConfig(
            host=config.ollama.host,
            model=config.ollama.embedding_model,
        )
    )
    try:
        await chat_client.initialize()
        chat_health = await chat_client.health_check()
        if chat_health.get("status") != "ok":
            print(f"FAIL: lemonade unhealthy: {chat_health}")
            sys.exit(1)
        if not chat_health.get("default_model_available", False):
            print(f"FAIL: default model unavailable: {config.lemonade.default_model}")
            sys.exit(1)

        await retrieval_client.initialize()
        availability = await retrieval_client.check_model_availability(
            [
                config.ollama.embedding_model,
                config.ollama.reranker_model,
            ]
        )
        missing = [
            model
            for model, available in availability.items()
            if not available
        ]
        if missing:
            print(f"FAIL: retrieval models unavailable: {', '.join(missing)}")
            sys.exit(1)
        print("OK")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
    finally:
        await chat_client.close()
        await retrieval_client.close()


asyncio.run(check())
PY
