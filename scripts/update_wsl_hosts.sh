#!/usr/bin/env bash
# update_wsl_hosts.sh - refresh the /etc/hosts alias entry with the Windows gateway IP on WSL boot
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/update_wsl_hosts.sh

Environment:
  HOSTNAME_ALIAS  alias name to refresh in hosts (default: homelab)
  HOSTS_FILE      target hosts file path (default: /etc/hosts)
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
        echo "[update_wsl_hosts] ERROR: unknown option: $1" >&2
        usage
        exit 1
        ;;
esac

HOSTNAME_ALIAS="${HOSTNAME_ALIAS:-homelab}"
HOSTS_FILE="${HOSTS_FILE:-/etc/hosts}"

if [[ -z "${HOSTNAME_ALIAS}" ]]; then
    echo "[update_wsl_hosts] ERROR: HOSTNAME_ALIAS is empty" >&2
    exit 1
fi
if [[ ! "${HOSTNAME_ALIAS}" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "[update_wsl_hosts] ERROR: invalid HOSTNAME_ALIAS format: ${HOSTNAME_ALIAS}" >&2
    exit 1
fi
if [[ ! -f "${HOSTS_FILE}" ]]; then
    echo "[update_wsl_hosts] ERROR: hosts file not found: ${HOSTS_FILE}" >&2
    exit 1
fi

GATEWAY_IP="$(ip route show default | awk '{print $3; exit}')"
if [[ -z "${GATEWAY_IP}" ]]; then
    echo "[update_wsl_hosts] ERROR: failed to detect the gateway IP" >&2
    exit 1
fi

tmp_file="$(mktemp "${TMPDIR:-/tmp}/hosts_update.XXXXXX")"
trap 'rm -f "${tmp_file}"' EXIT

awk -v alias="${HOSTNAME_ALIAS}" -v ip="${GATEWAY_IP}" '
    BEGIN { updated = 0 }
    {
        matched = 0
        for (i = 2; i <= NF; i++) {
            if ($i == alias || $i == alias ".localdomain") {
                matched = 1
                break
            }
        }
        if (matched) {
            if (!updated) {
                print ip "\t" alias ".localdomain\t" alias
                updated = 1
            }
            next
        }
        print $0
    }
    END {
        if (!updated) {
            print ip "\t" alias ".localdomain\t" alias
        }
    }
' "${HOSTS_FILE}" > "${tmp_file}"

cat "${tmp_file}" > "${HOSTS_FILE}"
echo "[update_wsl_hosts] ${HOSTNAME_ALIAS} -> ${GATEWAY_IP} (${HOSTS_FILE})"
