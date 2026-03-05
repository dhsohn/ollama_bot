#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"

RULE_NAME="${RULE_NAME:-Lemonade WSL}"
CONNECT_ADDRESS="${CONNECT_ADDRESS:-127.0.0.1}"
LEMONADE_HOST_OVERRIDE="${LEMONADE_HOST_OVERRIDE:-}"
NO_PORT_PROXY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/configure_windows_lemonade.sh [--no-port-proxy]

Windows 방화벽/portproxy를 자동 설정한다.
- 관리자 권한이 아니면 UAC를 통해 자동 상승을 요청한다.
- LEMONADE_HOST_OVERRIDE 환경변수로 대상 host를 강제할 수 있다.
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
  lemonade_host="http://localhost:8000"
fi

lemonade_port="8000"
if [[ "${lemonade_host}" =~ :([0-9]+)$ ]]; then
  lemonade_port="${BASH_REMATCH[1]}"
elif [[ "${lemonade_host}" =~ :([0-9]+)/ ]]; then
  lemonade_port="${BASH_REMATCH[1]}"
fi

ps_file="$(mktemp "${TMPDIR:-/tmp}/lemonade_wsl_fix.XXXXXX.ps1")"
trap 'rm -f "${ps_file}"' EXIT

{
  # Add UTF-8 BOM so Windows PowerShell parses the script consistently.
  printf '\xEF\xBB\xBF'
  cat <<'POWERSHELL'
param(
  [int]$Port = 11434,
  [string]$RuleName = "Lemonade WSL",
  [string]$ListenAddress = "0.0.0.0",
  [string]$ConnectAddress = "127.0.0.1",
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
    "-ConnectAddress", "$ConnectAddress"
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
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Any | Out-Null

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
    & netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$Port | Out-Null
    & netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$Port connectaddress=$ConnectAddress connectport=$Port | Out-Null
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to configure portproxy (exit=$LASTEXITCODE)"
    }
  }
}

Write-Log "Done: Firewall=OK PortProxy=$needsPortProxy Port=$Port"
POWERSHELL
} > "${ps_file}"

ps_file_win="$(wslpath -w "${ps_file}")"

echo "[configure_windows_lemonade.sh] provider=lemonade, host=${lemonade_host}, port=${lemonade_port}"

ps_args=(
  -NoProfile
  -ExecutionPolicy Bypass
  -File "${ps_file_win}"
  -Port "${lemonade_port}"
  -RuleName "${RULE_NAME}"
  -ConnectAddress "${CONNECT_ADDRESS}"
)

if [[ "${NO_PORT_PROXY}" == "1" ]]; then
  ps_args+=(-NoPortProxy)
fi

powershell.exe "${ps_args[@]}" | tr -d '\r'

windows_host_ip="${WINDOWS_HOST_IP:-$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf)}"
if [[ -n "${windows_host_ip}" ]]; then
  if timeout 5 bash -c "echo >/dev/tcp/${windows_host_ip}/${lemonade_port}" 2>/dev/null; then
    echo "[configure_windows_lemonade.sh] 연결 확인 성공: ${windows_host_ip}:${lemonade_port}"
  else
    echo "[configure_windows_lemonade.sh] 연결 확인 실패: ${windows_host_ip}:${lemonade_port}" >&2
    echo "[configure_windows_lemonade.sh] lemonade-server 실행/바인딩 상태를 확인하세요." >&2
  fi
fi
