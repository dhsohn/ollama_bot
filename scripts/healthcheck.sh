#!/bin/bash
# Docker 컨테이너 헬스체크 스크립트
set -euo pipefail

python - <<'PY'
import asyncio
import sys

from core.config import load_config
from core.ollama_client import OllamaClient


async def check() -> None:
    try:
        config = load_config()
    except Exception as exc:
        print(f"FAIL: invalid config: {exc}")
        sys.exit(1)

    client = OllamaClient(config.ollama)
    try:
        await client.initialize()
        health = await client.health_check()
        if health.get("status") != "ok":
            print(f"FAIL: ollama unhealthy: {health}")
            sys.exit(1)
        if not health.get("default_model_available", False):
            print(f"FAIL: default model unavailable: {config.ollama.model}")
            sys.exit(1)
        print("OK")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
    finally:
        await client.close()


asyncio.run(check())
PY
