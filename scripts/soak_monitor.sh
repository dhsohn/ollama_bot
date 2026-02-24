#!/usr/bin/env bash
set -euo pipefail

SERVICE="ollama_bot"
MINUTES=180
INTERVAL_SECONDS=60
MAX_RESTARTS=0
MAX_ERROR_LINES=0

usage() {
  cat <<EOF
Usage: bash scripts/soak_monitor.sh [options]

Options:
  --service NAME           docker compose service name (default: ${SERVICE})
  --minutes N              monitor duration in minutes (default: ${MINUTES})
  --interval-seconds N     poll interval in seconds (default: ${INTERVAL_SECONDS})
  --max-restarts N         allowed restart delta (default: ${MAX_RESTARTS})
  --max-error-lines N      allowed error-line count per interval (default: ${MAX_ERROR_LINES})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service)
      SERVICE="$2"
      shift 2
      ;;
    --minutes)
      MINUTES="$2"
      shift 2
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --max-restarts)
      MAX_RESTARTS="$2"
      shift 2
      ;;
    --max-error-lines)
      MAX_ERROR_LINES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

if ! docker compose ps >/dev/null 2>&1; then
  echo "docker compose is not available in current directory" >&2
  exit 1
fi

container_id="$(docker compose ps -q "${SERVICE}")"
if [[ -z "${container_id}" ]]; then
  echo "service '${SERVICE}' is not running" >&2
  exit 1
fi

baseline_restarts="$(docker inspect -f '{{.RestartCount}}' "${container_id}")"
end_epoch=$(( "$(date +%s)" + MINUTES * 60 ))

echo "Soak monitor start: service=${SERVICE}, minutes=${MINUTES}, interval=${INTERVAL_SECONDS}s"
echo "Baseline restarts: ${baseline_restarts}"

while [[ "$(date +%s)" -lt "${end_epoch}" ]]; do
  container_id="$(docker compose ps -q "${SERVICE}")"
  if [[ -z "${container_id}" ]]; then
    echo "FAIL: service '${SERVICE}' is not running"
    exit 1
  fi

  running="$(docker inspect -f '{{.State.Running}}' "${container_id}")"
  if [[ "${running}" != "true" ]]; then
    echo "FAIL: container is not running"
    exit 1
  fi

  restarts="$(docker inspect -f '{{.RestartCount}}' "${container_id}")"
  restart_delta=$(( restarts - baseline_restarts ))
  if (( restart_delta > MAX_RESTARTS )); then
    echo "FAIL: restart delta ${restart_delta} exceeded max ${MAX_RESTARTS}"
    exit 1
  fi

  health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "unhealthy" ]]; then
    echo "FAIL: container health is unhealthy"
    exit 1
  fi

  interval_logs="$(docker compose logs --since "${INTERVAL_SECONDS}s" "${SERVICE}" 2>/dev/null || true)"
  error_lines="$(printf '%s\n' "${interval_logs}" | grep -Eic '"log_level": ?"error"|automation_failed|telegram_error|ollama_connection_failed' || true)"
  if (( error_lines > MAX_ERROR_LINES )); then
    echo "FAIL: error lines ${error_lines} exceeded max ${MAX_ERROR_LINES} in last ${INTERVAL_SECONDS}s"
    exit 1
  fi

  echo "OK: running=true health=${health} restart_delta=${restart_delta} error_lines=${error_lines}"
  sleep "${INTERVAL_SECONDS}"
done

echo "PASS: soak monitor completed without threshold violations"
