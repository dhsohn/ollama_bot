#!/usr/bin/env bash
# update_wsl_hosts.sh — WSL 부팅 시 Windows 게이트웨이 IP로 /etc/hosts의 homelab 엔트리 갱신
set -euo pipefail

HOSTNAME_ALIAS="homelab"
HOSTS_FILE="/etc/hosts"

GATEWAY_IP=$(ip route show default | awk '{print $3; exit}')

if [[ -z "$GATEWAY_IP" ]]; then
    echo "[update_wsl_hosts] ERROR: 게이트웨이 IP를 감지할 수 없음" >&2
    exit 1
fi

# 기존 homelab 줄이 있으면 교체, 없으면 추가
if grep -qE "\\s${HOSTNAME_ALIAS}(\\s|$)" "$HOSTS_FILE"; then
    sed -i "s/^.*\\s${HOSTNAME_ALIAS}\\.localdomain\\s.*$/${GATEWAY_IP}\t${HOSTNAME_ALIAS}.localdomain\t${HOSTNAME_ALIAS}/" "$HOSTS_FILE"
else
    echo -e "${GATEWAY_IP}\t${HOSTNAME_ALIAS}.localdomain\t${HOSTNAME_ALIAS}" >> "$HOSTS_FILE"
fi

echo "[update_wsl_hosts] ${HOSTNAME_ALIAS} -> ${GATEWAY_IP}"
