#!/usr/bin/env bash
# systemd user service 설치 스크립트.
# sudo 불필요 — ~/.config/systemd/user/ 에 설치한다.
#
# 사용법:
#   bash scripts/install_boot_service.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"

mkdir -p "${UNIT_DIR}"

# ── ollama-bot.service ──
cat > "${UNIT_DIR}/ollama-bot.service" <<EOF
[Unit]
Description=ollama_bot Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=${PROJECT_ROOT}/.env
Environment=HF_HOME=${PROJECT_ROOT}/data/hf_cache
Environment=FASTEMBED_CACHE_PATH=${PROJECT_ROOT}/data/hf_cache/fastembed
ExecStart=${PROJECT_ROOT}/.venv/bin/python -m apps.ollama_bot.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

echo "[install] ollama-bot.service 생성 완료"

# ── sim-host-agent.service ──
cat > "${UNIT_DIR}/sim-host-agent.service" <<EOF
[Unit]
Description=Simulation Host Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
WorkingDirectory=${PROJECT_ROOT}
EnvironmentFile=${PROJECT_ROOT}/.env
ExecStart=${PROJECT_ROOT}/scripts/run_host_agent.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

echo "[install] sim-host-agent.service 생성 완료"

# ── 활성화 ──
systemctl --user daemon-reload
systemctl --user enable ollama-bot.service sim-host-agent.service

# 부팅 시 로그인 없이도 서비스 시작
loginctl enable-linger "$(whoami)" 2>/dev/null || true

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "서비스 시작:"
echo "  systemctl --user start ollama-bot"
echo "  systemctl --user start sim-host-agent"
echo ""
echo "로그 확인:"
echo "  journalctl --user -u ollama-bot -f"
echo "  journalctl --user -u sim-host-agent -f"
echo ""
echo "상태 확인:"
echo "  systemctl --user status ollama-bot sim-host-agent"
