#!/usr/bin/env bash
# Long-running stability monitor for a systemd user service
set -euo pipefail

SERVICE="ollama-bot"
MINUTES=180
INTERVAL_SECONDS=60
MAX_RESTARTS=0
MAX_ERROR_LINES=0

usage() {
  cat <<EOF
Usage: bash scripts/soak_monitor.sh [options]

Options:
  --service NAME           systemd user service name (default: ${SERVICE})
  --minutes N              monitor duration in minutes (default: ${MINUTES})
  --interval-seconds N     poll interval in seconds (default: ${INTERVAL_SECONDS})
  --max-restarts N         allowed restart delta (default: ${MAX_RESTARTS})
  --max-error-lines N      allowed error-line count per interval (default: ${MAX_ERROR_LINES})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service)          SERVICE="$2";           shift 2 ;;
    --minutes)          MINUTES="$2";           shift 2 ;;
    --interval-seconds) INTERVAL_SECONDS="$2";  shift 2 ;;
    --max-restarts)     MAX_RESTARTS="$2";      shift 2 ;;
    --max-error-lines)  MAX_ERROR_LINES="$2";   shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! systemctl --user is-active --quiet "${SERVICE}"; then
  echo "FAIL: service '${SERVICE}' is not active" >&2
  exit 1
fi

# Baseline NRestarts
baseline_restarts="$(systemctl --user show "${SERVICE}" -p NRestarts --value)"
end_epoch=$(( "$(date +%s)" + MINUTES * 60 ))

echo "Soak monitor start: service=${SERVICE}, minutes=${MINUTES}, interval=${INTERVAL_SECONDS}s"
echo "Baseline restarts: ${baseline_restarts}"

while [[ "$(date +%s)" -lt "${end_epoch}" ]]; do
  # Service activity check
  if ! systemctl --user is-active --quiet "${SERVICE}"; then
    echo "FAIL: service '${SERVICE}' is not running"
    exit 1
  fi

  # Restart count
  restarts="$(systemctl --user show "${SERVICE}" -p NRestarts --value)"
  restart_delta=$(( restarts - baseline_restarts ))
  if (( restart_delta > MAX_RESTARTS )); then
    echo "FAIL: restart delta ${restart_delta} exceeded max ${MAX_RESTARTS}"
    exit 1
  fi

  # Recent interval error logs
  interval_logs="$(journalctl --user -u "${SERVICE}" --since "${INTERVAL_SECONDS}s ago" --no-pager 2>/dev/null || true)"
  error_lines="$(printf '%s\n' "${interval_logs}" | grep -Eic '"log_level": ?"error"|automation_failed|telegram_error|ollama_connection_failed' || true)"
  if (( error_lines > MAX_ERROR_LINES )); then
    echo "FAIL: error lines ${error_lines} exceeded max ${MAX_ERROR_LINES} in last ${INTERVAL_SECONDS}s"
    exit 1
  fi

  echo "OK: active=true restart_delta=${restart_delta} error_lines=${error_lines}"
  sleep "${INTERVAL_SECONDS}"
done

echo "PASS: soak monitor completed without threshold violations"
