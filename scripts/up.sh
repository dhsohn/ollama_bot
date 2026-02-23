#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[up.sh] ERROR: ${ENV_FILE} 파일이 없습니다. 먼저 .env를 생성하세요."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[up.sh] ERROR: Docker daemon에 연결할 수 없습니다."
  exit 1
fi

# WSL 환경에서는 Windows 게이트웨이 IP를 OLLAMA_HOST로 자동 반영한다.
WIN_IP=""
if grep -qi microsoft /proc/version 2>/dev/null || grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  WIN_IP="$(awk '/nameserver/{print $2; exit}' /etc/resolv.conf || true)"
fi

if [[ -n "${WIN_IP}" ]]; then
  NEW_HOST="http://${WIN_IP}:11434"
  if grep -q '^OLLAMA_HOST=' "${ENV_FILE}"; then
    sed -i "s|^OLLAMA_HOST=.*|OLLAMA_HOST=${NEW_HOST}|" "${ENV_FILE}"
  else
    printf '\nOLLAMA_HOST=%s\n' "${NEW_HOST}" >> "${ENV_FILE}"
  fi
  echo "[up.sh] OLLAMA_HOST=${NEW_HOST}"
fi

mkdir -p \
  "${PROJECT_ROOT}/data/conversations" \
  "${PROJECT_ROOT}/data/memory" \
  "${PROJECT_ROOT}/data/logs" \
  "${PROJECT_ROOT}/data/reports"

cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--build" ]]; then
  echo "[up.sh] docker compose up -d --build"
  docker compose up -d --build
else
  echo "[up.sh] docker compose up -d"
  docker compose up -d
fi
