#!/usr/bin/env bash
# update_wsl_hosts.sh — WSL 부팅 시 Windows 게이트웨이 IP로 /etc/hosts의 별칭 엔트리 갱신
set -euo pipefail

HOSTNAME_ALIAS="${HOSTNAME_ALIAS:-homelab}"
HOSTS_FILE="${HOSTS_FILE:-/etc/hosts}"

if [[ -z "${HOSTNAME_ALIAS}" ]]; then
    echo "[update_wsl_hosts] ERROR: HOSTNAME_ALIAS가 비어 있음" >&2
    exit 1
fi
if [[ ! "${HOSTNAME_ALIAS}" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "[update_wsl_hosts] ERROR: HOSTNAME_ALIAS 형식이 유효하지 않음: ${HOSTNAME_ALIAS}" >&2
    exit 1
fi
if [[ ! -f "${HOSTS_FILE}" ]]; then
    echo "[update_wsl_hosts] ERROR: hosts 파일이 없음: ${HOSTS_FILE}" >&2
    exit 1
fi

GATEWAY_IP="$(ip route show default | awk '{print $3; exit}')"
if [[ -z "${GATEWAY_IP}" ]]; then
    echo "[update_wsl_hosts] ERROR: 게이트웨이 IP를 감지할 수 없음" >&2
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
