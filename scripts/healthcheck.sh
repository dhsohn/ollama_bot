#!/bin/bash
# Docker 컨테이너 헬스체크 스크립트
set -euo pipefail

python - <<'PY'
import asyncio
import sys

from core.config import load_config
from core.lemonade_client import LemonadeClient
from core.ollama_client import OllamaClient


async def check() -> None:
    try:
        config = load_config()
    except Exception as exc:
        print(f"FAIL: invalid config: {exc}")
        sys.exit(1)

    provider = str(getattr(config, "llm_provider", "ollama")).strip().lower()
    if provider == "lemonade":
        client = LemonadeClient(config.lemonade, fallback_ollama=config.ollama)
        expected_model = config.lemonade.model or config.ollama.model
    else:
        client = OllamaClient(config.ollama)
        expected_model = config.ollama.model
    try:
        await client.initialize()
        health = await client.health_check()
        if health.get("status") != "ok":
            print(f"FAIL: {provider} unhealthy: {health}")
            sys.exit(1)
        if not health.get("default_model_available", False):
            print(f"FAIL: default model unavailable: {expected_model}")
            sys.exit(1)
        print("OK")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
    finally:
        await client.close()


asyncio.run(check())
PY
