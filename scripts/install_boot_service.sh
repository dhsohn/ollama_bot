#!/usr/bin/env bash
# systemd service installation script.
# - update-wsl-hosts.service: system-level (requires sudo, refreshes the /etc/hosts alias)
# - ollama-bot.service: user-level (no sudo required)
#
# Usage:
#   bash scripts/install_boot_service.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_boot_service.sh

Installs:
  - system-level `update-wsl-hosts.service`
  - user-level `ollama-bot.service`
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
    echo "[install_boot_service.sh] unknown option: $1" >&2
    usage
    exit 1
    ;;
esac

mkdir -p "${UNIT_DIR}"

# update-wsl-hosts.service (system-level, refresh alias IP during boot)
SYSTEM_UNIT="/etc/systemd/system/update-wsl-hosts.service"
echo "[install] Installing update-wsl-hosts.service (sudo required)"
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
echo "[install] Created update-wsl-hosts.service"

# ollama-bot.service (user-level)
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

echo "[install] Created ollama-bot.service"

# Enable services
systemctl --user daemon-reload
systemctl --user enable ollama-bot.service

# Start services at boot even without an interactive login
loginctl enable-linger "$(whoami)" 2>/dev/null || true

echo ""
echo "=== Installation complete ==="
echo ""
echo "Start services:"
echo "  sudo systemctl start update-wsl-hosts   # refresh hosts immediately"
echo "  systemctl --user start ollama-bot"
echo ""
echo "Inspect logs:"
echo "  journalctl -u update-wsl-hosts           # hosts refresh log"
echo "  journalctl --user -u ollama-bot -f        # bot log"
echo ""
echo "Check status:"
echo "  systemctl --user status ollama-bot"
