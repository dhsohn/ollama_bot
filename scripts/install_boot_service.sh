#!/usr/bin/env bash
# systemd 서비스 설치 스크립트.
# - update-wsl-hosts.service: system-level (sudo 필요, /etc/hosts 별칭 갱신)
# - ollama-bot.service: user-level (sudo 불필요)
#
# 사용법:
#   bash scripts/install_boot_service.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"

mkdir -p "${UNIT_DIR}"

# ── update-wsl-hosts.service (system-level, 부팅 시 별칭 IP 갱신) ──
SYSTEM_UNIT="/etc/systemd/system/update-wsl-hosts.service"
echo "[install] update-wsl-hosts.service 설치 (sudo 필요)"
sudo tee "${SYSTEM_UNIT}" > /dev/null <<EOF
[Unit]
Description=Update /etc/hosts with WSL gateway IP
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=${PROJECT_ROOT}/scripts/update_wsl_hosts.sh

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable update-wsl-hosts.service
echo "[install] update-wsl-hosts.service 생성 완료"

# ── ollama-bot.service (user-level) ──
cat > "${UNIT_DIR}/ollama-bot.service" <<EOF
[Unit]
Description=ollama_bot Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
WorkingDirectory=${PROJECT_ROOT}
Environment=HF_HOME=${PROJECT_ROOT}/data/hf_cache
Environment=FASTEMBED_CACHE_PATH=${PROJECT_ROOT}/data/hf_cache/fastembed
ExecStart=${PROJECT_ROOT}/.venv/bin/python -m apps.ollama_bot.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

echo "[install] ollama-bot.service 생성 완료"

# ── 활성화 ──
systemctl --user daemon-reload
systemctl --user enable ollama-bot.service

# 부팅 시 로그인 없이도 서비스 시작
loginctl enable-linger "$(whoami)" 2>/dev/null || true

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "서비스 시작:"
echo "  sudo systemctl start update-wsl-hosts   # 즉시 hosts 갱신"
echo "  systemctl --user start ollama-bot"
echo ""
echo "로그 확인:"
echo "  journalctl -u update-wsl-hosts           # hosts 갱신 로그"
echo "  journalctl --user -u ollama-bot -f        # 봇 로그"
echo ""
echo "상태 확인:"
echo "  systemctl --user status ollama-bot"
