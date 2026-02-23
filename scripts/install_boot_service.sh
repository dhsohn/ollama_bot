#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "[install_boot_service.sh] sudo로 실행하세요."
  echo "예: sudo bash scripts/install_boot_service.sh"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="ollama_bot.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
TARGET_USER="${SUDO_USER:-$(id -un)}"

cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=ollama_bot auto-start
Wants=network-online.target docker.service
After=network-online.target docker.service

[Service]
Type=oneshot
User=${TARGET_USER}
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${PROJECT_ROOT}/scripts/up.sh
RemainAfterExit=yes
Restart=on-failure
RestartSec=10
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "[install_boot_service.sh] 설치 완료: ${SERVICE_NAME}"
systemctl status --no-pager --lines=20 "${SERVICE_NAME}" || true
