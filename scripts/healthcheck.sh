#!/usr/bin/env bash
# 헬스체크 스크립트
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: bash scripts/healthcheck.sh

Checks:
  - config/config.yaml 로드 가능 여부
  - Ollama 서버 연결 가능 여부
  - chat / embedding / reranker 모델 가용 여부
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "[healthcheck.sh] 알 수 없는 옵션: $1" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[healthcheck.sh] ERROR: ${PYTHON_BIN} 가 없습니다." >&2
  echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

"${PYTHON_BIN}" - <<'PY'
import asyncio
import sys

from core.config import OllamaConfig, load_config
from core.ollama_client import OllamaClient


async def check() -> None:
    try:
        config = load_config()
    except Exception as exc:
        print(f"FAIL: invalid config: {exc}")
        sys.exit(1)

    chat_model = config.ollama.chat_model.strip()
    if not chat_model:
        print("FAIL: ollama.chat_model is not configured")
        sys.exit(1)

    model_names = list(
        dict.fromkeys(
            name.strip()
            for name in (
                config.ollama.chat_model,
                config.ollama.embedding_model,
                config.ollama.reranker_model,
            )
            if name.strip()
        )
    )

    client = OllamaClient(
        config=OllamaConfig(
            host=config.ollama.host,
            model=chat_model,
        )
    )
    try:
        await client.initialize()
        health = await client.health_check()
        if health.get("status") != "ok":
            print(f"FAIL: ollama unhealthy: {health}")
            sys.exit(1)
        if not health.get("default_model_available", False):
            print(f"FAIL: default model unavailable: {chat_model}")
            sys.exit(1)

        availability = await client.check_model_availability(model_names)
        missing = [
            model
            for model, available in availability.items()
            if not available
        ]
        if missing:
            print(f"FAIL: missing Ollama models: {', '.join(missing)}")
            sys.exit(1)
        print(
            "OK: "
            f"host={config.ollama.host} "
            f"default_model={chat_model} "
            f"models_checked={len(model_names)}"
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
    finally:
        await client.close()


asyncio.run(check())
PY
