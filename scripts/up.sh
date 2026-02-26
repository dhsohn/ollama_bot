#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[up.sh] ERROR: ${ENV_FILE} 파일이 없습니다. 먼저 scripts/setup.sh를 실행하세요."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[up.sh] ERROR: Docker daemon에 연결할 수 없습니다."
  exit 1
fi

mkdir -p \
  "${PROJECT_ROOT}/data/conversations" \
  "${PROJECT_ROOT}/data/memory" \
  "${PROJECT_ROOT}/data/logs" \
  "${PROJECT_ROOT}/data/reports"

cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--build" ]]; then
  echo "[up.sh] docker compose -f docker-compose.yml up -d --build"
  docker compose -f docker-compose.yml up -d --build
else
  echo "[up.sh] docker compose -f docker-compose.yml up -d"
  docker compose -f docker-compose.yml up -d
fi
