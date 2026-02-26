#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[up.sh] ERROR: ${ENV_FILE} 파일이 없습니다. 먼저 scripts/setup.sh를 실행하세요."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[up.sh] ERROR: Docker daemon에 연결할 수 없습니다."
  exit 1
fi

# WSL 환경에서는 Windows 호스트 IP(기본 게이트웨이)가 재부팅마다 바뀔 수 있다.
# lemonade provider 사용 시 Docker extra_hosts에 현재 IP를 주입한다.
if [[ -f "${CONFIG_FILE}" ]] && grep -Eq '^llm_provider:[[:space:]]*"lemonade"' "${CONFIG_FILE}"; then
  WINDOWS_HOST_IP="${WINDOWS_HOST_IP:-$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf)}"
  if [[ -n "${WINDOWS_HOST_IP}" ]]; then
    export WINDOWS_HOST_IP
    echo "[up.sh] WINDOWS_HOST_IP=${WINDOWS_HOST_IP} (for docker compose extra_hosts)"
  else
    echo "[up.sh] WARN: Windows host IP를 찾지 못했습니다. windows-host 매핑이 실패할 수 있습니다."
  fi
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
