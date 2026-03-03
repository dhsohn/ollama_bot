#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[run_bot.sh] ERROR: ${ENV_FILE} 파일이 없습니다. 먼저 scripts/setup.sh를 실행하세요."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[run_bot.sh] ERROR: Docker daemon에 연결할 수 없습니다."
  exit 1
fi

# WSL 환경에서는 Windows 호스트 IP(기본 게이트웨이)가 재부팅마다 바뀔 수 있다.
# Docker extra_hosts에 현재 IP를 주입한다.
if [[ -f "${CONFIG_FILE}" ]]; then
  WINDOWS_HOST_IP="${WINDOWS_HOST_IP:-$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf)}"
  if [[ -n "${WINDOWS_HOST_IP}" ]]; then
    export WINDOWS_HOST_IP
    echo "[run_bot.sh] WINDOWS_HOST_IP=${WINDOWS_HOST_IP} (for docker compose extra_hosts)"
  else
    echo "[run_bot.sh] WARN: Windows host IP를 찾지 못했습니다. windows-host 매핑이 실패할 수 있습니다."
  fi

  # lemonade 포트 연결 프리체크
  if [[ -n "${WINDOWS_HOST_IP:-}" ]]; then
    LEMONADE_PORT=$(grep -A8 '^lemonade:' "${CONFIG_FILE}" | grep 'host:' | grep -oP ':\K[0-9]+' || echo "8000")
    echo "[run_bot.sh] lemonade 연결 확인 중: ${WINDOWS_HOST_IP}:${LEMONADE_PORT} ..."
    if timeout 5 bash -c "echo >/dev/tcp/${WINDOWS_HOST_IP}/${LEMONADE_PORT}" 2>/dev/null; then
      echo "[run_bot.sh] OK: lemonade-server 응답 확인"
    else
      LEMONADE_APP_PARAMS=""
      if command -v powershell.exe >/dev/null 2>&1; then
        LEMONADE_APP_PARAMS="$(
          powershell.exe -NoProfile -Command '(Get-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Services\\LemonadeServer\\Parameters" -Name AppParameters -ErrorAction SilentlyContinue).AppParameters' 2>/dev/null \
            | tr -d '\r' \
            | sed -n '1p'
        )"
      fi

      echo "[run_bot.sh] WARN: ${WINDOWS_HOST_IP}:${LEMONADE_PORT} 연결 실패."
      if [[ -n "${LEMONADE_APP_PARAMS}" ]]; then
        echo "[run_bot.sh] 감지된 LemonadeServer AppParameters: ${LEMONADE_APP_PARAMS}"
      fi
      echo "  가능한 원인:"
      LOCALHOST_BINDING_DETECTED=0
      if [[ "${LEMONADE_APP_PARAMS}" == *"--host localhost"* ]] \
        || [[ "${LEMONADE_APP_PARAMS}" == *"--host 127.0.0.1"* ]] \
        || [[ "${LEMONADE_APP_PARAMS}" == *"--host ::1"* ]]; then
        LOCALHOST_BINDING_DETECTED=1
        echo "    1) LemonadeServer 서비스가 localhost에만 바인딩됨"
        echo "       → PowerShell(관리자):"
        echo "         Stop-Service LemonadeServer"
        echo "         Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Services\\LemonadeServer\\Parameters' -Name AppParameters -Value 'serve --port ${LEMONADE_PORT} --host 0.0.0.0 --no-tray'"
        echo "         Start-Service LemonadeServer"
      else
        echo "    1) lemonade-server가 0.0.0.0 대신 127.0.0.1에만 바인딩됨"
        echo "       → --host 0.0.0.0 옵션으로 서버를 재시작하세요"
      fi
      echo "    2) Windows 방화벽이 WSL 서브넷(172.x.x.x)을 차단 중"
      echo "       → PowerShell(관리자):"
      if [[ "${LOCALHOST_BINDING_DETECTED}" == "1" ]]; then
        echo "         New-NetFirewallRule -DisplayName 'Lemonade WSL' -Direction Inbound -LocalPort ${LEMONADE_PORT} -Protocol TCP -Action Allow -Profile Any"
      else
        echo "         netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=${LEMONADE_PORT} connectaddress=127.0.0.1 connectport=${LEMONADE_PORT}"
        echo "         New-NetFirewallRule -DisplayName 'Lemonade WSL' -Direction Inbound -LocalPort ${LEMONADE_PORT} -Protocol TCP -Action Allow -Profile Any"
      fi
      echo "         Get-NetTCPConnection -State Listen -LocalPort ${LEMONADE_PORT}"
      echo "  컨테이너는 시작하지만 lemonade 연결이 실패할 수 있습니다."
    fi
  fi
fi

mkdir -p \
  "${PROJECT_ROOT}/data/conversations" \
  "${PROJECT_ROOT}/data/memory" \
  "${PROJECT_ROOT}/data/logs" \
  "${PROJECT_ROOT}/data/reports" \
  "${PROJECT_ROOT}/kb/orca_runs" \
  "${PROJECT_ROOT}/kb/orca_outputs"

cd "${PROJECT_ROOT}"

if [[ "${1:-}" == "--build" ]]; then
  echo "[run_bot.sh] docker compose -f docker-compose.yml up -d --build"
  docker compose -f docker-compose.yml up -d --build
else
  echo "[run_bot.sh] docker compose -f docker-compose.yml up -d"
  docker compose -f docker-compose.yml up -d
fi
