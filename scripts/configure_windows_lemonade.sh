#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"

RULE_NAME="${RULE_NAME:-Lemonade WSL}"
CONNECT_ADDRESS="${CONNECT_ADDRESS:-127.0.0.1}"
LEMONADE_HOST_OVERRIDE="${LEMONADE_HOST_OVERRIDE:-}"
WSL_REMOTE_CIDR_OVERRIDE="${WSL_REMOTE_CIDR_OVERRIDE:-}"
REMOVE_RULE_NAMES_CSV_OVERRIDE="${REMOVE_RULE_NAMES_CSV_OVERRIDE:-}"
NO_PORT_PROXY=0

is_ipv4() {
  [[ "$1" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

is_ipv4_or_cidr() {
  [[ "$1" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}(/([0-9]|[1-2][0-9]|3[0-2]))?$ ]]
}

usage() {
  cat <<'EOF'
Usage: bash scripts/configure_windows_lemonade.sh [--no-port-proxy]

Windows 방화벽/portproxy를 자동 설정한다.
- 관리자 권한이 아니면 UAC를 통해 자동 상승을 요청한다.
- LEMONADE_HOST_OVERRIDE 환경변수로 대상 host를 강제할 수 있다.
- 기본값은 WSL 전용으로 제한한다(remote는 WSL subnet CIDR).
- REMOVE_RULE_NAMES_CSV_OVERRIDE로 레거시 방화벽 규칙(DisplayName)을 제거할 수 있다.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-port-proxy)
      NO_PORT_PROXY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[configure_windows_lemonade.sh] 알 수 없는 옵션: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[configure_windows_lemonade.sh] config 파일이 없습니다: ${CONFIG_FILE}" >&2
  exit 1
fi

if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "[configure_windows_lemonade.sh] powershell.exe를 찾을 수 없어 건너뜁니다."
  exit 0
fi

lemonade_host="$(
  awk '
    /^lemonade:[[:space:]]*$/ {in_block=1; next}
    in_block && /^[[:alnum:]_]+:[[:space:]]*/ {in_block=0}
    in_block && /^[[:space:]]*host:[[:space:]]*/ {
      line=$0
      sub(/^[[:space:]]*host:[[:space:]]*/, "", line)
      gsub(/"/, "", line)
      gsub(/[[:space:]]/, "", line)
      print line
      exit
    }
  ' "${CONFIG_FILE}"
)"

if [[ -n "${LEMONADE_HOST_OVERRIDE}" ]]; then
  lemonade_host="${LEMONADE_HOST_OVERRIDE}"
fi

if [[ -z "${lemonade_host}" ]]; then
  lemonade_host="http://localhost:8020"
fi

lemonade_port="8020"
if [[ "${lemonade_host}" =~ :([0-9]+)$ ]]; then
  lemonade_port="${BASH_REMATCH[1]}"
elif [[ "${lemonade_host}" =~ :([0-9]+)/ ]]; then
  lemonade_port="${BASH_REMATCH[1]}"
fi

windows_host_ip="${WINDOWS_HOST_IP:-$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf)}"
if ! is_ipv4 "${windows_host_ip}"; then
  windows_host_ip=""
fi

hosts_homelab_ip="$(
  awk '($2=="homelab.localdomain" || $2=="homelab" || $3=="homelab") {print $1; exit}' /etc/hosts
)"
if ! is_ipv4 "${hosts_homelab_ip}"; then
  hosts_homelab_ip=""
fi

listen_address="${WSL_PORTPROXY_LISTEN_ADDRESS:-}"
if [[ -z "${listen_address}" ]]; then
  if [[ -n "${hosts_homelab_ip}" ]]; then
    listen_address="${hosts_homelab_ip}"
  elif [[ -n "${windows_host_ip}" ]]; then
    listen_address="${windows_host_ip}"
  fi
fi
if ! is_ipv4 "${listen_address}"; then
  echo "[configure_windows_lemonade.sh] WSL 전용 listen address를 결정할 수 없습니다." >&2
  echo "[configure_windows_lemonade.sh] WSL_PORTPROXY_LISTEN_ADDRESS=<windows_wsl_gateway_ip> 로 지정하세요." >&2
  exit 1
fi

wsl_client_ip="${WSL_CLIENT_IP:-$(hostname -I | tr ' ' '\n' | awk '/^[0-9]+(\.[0-9]+){3}$/ {print; exit}')}"
if ! is_ipv4 "${wsl_client_ip}"; then
  echo "[configure_windows_lemonade.sh] WSL client IP를 확인할 수 없습니다." >&2
  echo "[configure_windows_lemonade.sh] WSL_CLIENT_IP=<wsl_ip> 로 지정하세요." >&2
  exit 1
fi

wsl_route_cidr="$(
  ip -4 route show | awk -v listen="${listen_address}" '$0 ~ ("dev " ) && $1 ~ /\/[0-9]+$/ && $0 ~ ("via " listen " ") {print $1; exit}'
)"
if [[ -z "${wsl_route_cidr}" ]]; then
  wsl_route_cidr="$(ip -4 route show | awk '$1 ~ /\/[0-9]+$/ && $0 ~ /dev eth0/ {print $1; exit}')"
fi

remote_address="${WSL_REMOTE_CIDR_OVERRIDE}"
if [[ -z "${remote_address}" ]]; then
  if [[ -n "${wsl_route_cidr}" ]]; then
    remote_address="${wsl_route_cidr}"
  else
    remote_address="${wsl_client_ip}/32"
  fi
fi
if ! is_ipv4_or_cidr "${remote_address}"; then
  echo "[configure_windows_lemonade.sh] WSL remote CIDR 형식이 잘못되었습니다: ${remote_address}" >&2
  echo "[configure_windows_lemonade.sh] WSL_REMOTE_CIDR_OVERRIDE=<x.x.x.x/nn> 로 지정하세요." >&2
  exit 1
fi

cleanup_listen_addresses=("0.0.0.0" "${listen_address}")
if [[ -n "${windows_host_ip}" ]]; then
  cleanup_listen_addresses+=("${windows_host_ip}")
fi
if [[ -n "${hosts_homelab_ip}" ]]; then
  cleanup_listen_addresses+=("${hosts_homelab_ip}")
fi
cleanup_listen_addresses_csv="$(
  printf '%s\n' "${cleanup_listen_addresses[@]}" | awk 'NF && !seen[$0]++' | paste -sd, -
)"

ps_file="$(mktemp "${TMPDIR:-/tmp}/lemonade_wsl_fix.XXXXXX.ps1")"
trap 'rm -f "${ps_file}"' EXIT

{
  # Add UTF-8 BOM so Windows PowerShell parses the script consistently.
  printf '\xEF\xBB\xBF'
  cat <<'POWERSHELL'
param(
  [int]$Port = 8020,
  [string]$RuleName = "Lemonade WSL",
  [string]$ListenAddress = "127.0.0.1",
  [string]$ConnectAddress = "127.0.0.1",
  [string]$RemoteAddress = "LocalSubnet",
  [string]$CleanupListenAddressesCsv = "0.0.0.0",
  [string]$RemoveRuleNamesCsv = "",
  [switch]$NoPortProxy
)

$ErrorActionPreference = "Stop"

function Write-Log([string]$Message) {
  Write-Output "[windows-net] $Message"
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
  Write-Log "Requesting administrator privileges (UAC)."
  $args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$PSCommandPath`"",
    "-Port", "$Port",
    "-RuleName", "`"$RuleName`"",
    "-ListenAddress", "$ListenAddress",
    "-ConnectAddress", "$ConnectAddress",
    "-RemoteAddress", "$RemoteAddress",
    "-CleanupListenAddressesCsv", "`"$CleanupListenAddressesCsv`"",
    "-RemoveRuleNamesCsv", "`"$RemoveRuleNamesCsv`""
  )
  if ($NoPortProxy) {
    $args += "-NoPortProxy"
  }
  $p = Start-Process -FilePath "powershell.exe" -Verb RunAs -PassThru -Wait -ArgumentList $args
  exit $p.ExitCode
}

Write-Log "Applying firewall rule: $RuleName (TCP/$Port)"
$existingRules = @(Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)
if ($existingRules.Count -gt 0) {
  $existingRules | Remove-NetFirewallRule | Out-Null
}
if ($RemoveRuleNamesCsv) {
  $legacyRuleNames = @($RemoveRuleNamesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique)
  foreach ($legacyRuleName in $legacyRuleNames) {
    $legacyRules = @(Get-NetFirewallRule -DisplayName $legacyRuleName -ErrorAction SilentlyContinue)
    if ($legacyRules.Count -gt 0) {
      Write-Log "Removing legacy firewall rule(s): $legacyRuleName"
      $legacyRules | Remove-NetFirewallRule | Out-Null
    }
  }
}
New-NetFirewallRule `
  -DisplayName $RuleName `
  -Direction Inbound `
  -Protocol TCP `
  -LocalAddress $ListenAddress `
  -RemoteAddress $RemoteAddress `
  -LocalPort $Port `
  -Action Allow `
  -Profile Any | Out-Null

$cleanupListenAddresses = @()
if ($CleanupListenAddressesCsv) {
  $cleanupListenAddresses += $CleanupListenAddressesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
$cleanupListenAddresses += @("0.0.0.0", $ListenAddress)
$cleanupListenAddresses = $cleanupListenAddresses | Select-Object -Unique
foreach ($cleanupAddress in $cleanupListenAddresses) {
  & netsh interface portproxy delete v4tov4 listenaddress=$cleanupAddress listenport=$Port | Out-Null
}

$needsPortProxy = $false
if (-not $NoPortProxy) {
  $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
  if ($listeners.Count -eq 0) {
    $needsPortProxy = $true
    Write-Log "No listener detected: configuring portproxy."
  } else {
    $nonLoopback = @($listeners | Where-Object { $_.LocalAddress -notin @("127.0.0.1", "::1") })
    if ($nonLoopback.Count -eq 0) {
      $needsPortProxy = $true
      Write-Log "Loopback listener only: configuring portproxy."
    } else {
      Write-Log "Non-loopback listener detected: skipping portproxy."
    }
  }

  if ($needsPortProxy) {
    & netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$Port connectaddress=$ConnectAddress connectport=$Port | Out-Null
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to configure portproxy (exit=$LASTEXITCODE)"
    }
  }
}

Write-Log "Done: Firewall=OK PortProxy=$needsPortProxy ListenAddress=$ListenAddress RemoteAddress=$RemoteAddress Port=$Port"
POWERSHELL
} > "${ps_file}"

ps_file_win="$(wslpath -w "${ps_file}")"

echo "[configure_windows_lemonade.sh] provider=lemonade, host=${lemonade_host}, port=${lemonade_port}"
echo "[configure_windows_lemonade.sh] wsl_only listen=${listen_address} remote=${remote_address}"

ps_args=(
  -NoProfile
  -ExecutionPolicy Bypass
  -File "${ps_file_win}"
  -Port "${lemonade_port}"
  -RuleName "${RULE_NAME}"
  -ListenAddress "${listen_address}"
  -ConnectAddress "${CONNECT_ADDRESS}"
  -RemoteAddress "${remote_address}"
  -CleanupListenAddressesCsv "${cleanup_listen_addresses_csv}"
  -RemoveRuleNamesCsv "${REMOVE_RULE_NAMES_CSV_OVERRIDE}"
)

if [[ "${NO_PORT_PROXY}" == "1" ]]; then
  ps_args+=(-NoPortProxy)
fi

powershell.exe "${ps_args[@]}" | tr -d '\r'

lemonade_target_host="${lemonade_host#*://}"
lemonade_target_host="${lemonade_target_host%%/*}"
lemonade_target_host="${lemonade_target_host%%:*}"
if [[ -n "${lemonade_target_host}" ]]; then
  if timeout 5 bash -c "echo >/dev/tcp/${lemonade_target_host}/${lemonade_port}" 2>/dev/null; then
    echo "[configure_windows_lemonade.sh] 연결 확인 성공: ${lemonade_target_host}:${lemonade_port}"
  elif timeout 5 bash -c "echo >/dev/tcp/${listen_address}/${lemonade_port}" 2>/dev/null; then
    echo "[configure_windows_lemonade.sh] 연결 확인 성공(대체): ${listen_address}:${lemonade_port}"
  else
    echo "[configure_windows_lemonade.sh] 연결 확인 실패: ${lemonade_target_host}:${lemonade_port}" >&2
    echo "[configure_windows_lemonade.sh] lemonade-server 실행/바인딩 상태를 확인하세요." >&2
  fi
fi
