#!/usr/bin/env bash
# Run or restart ollama_bot directly from WSL.
#
# Usage:
#   ./scripts/run_bot.sh
#   ./scripts/run_bot.sh --restart
#   ./scripts/run_bot.sh --restart --pull
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"
VENV_DIR="${PROJECT_ROOT}/.venv"
LOCK_FILE="${PROJECT_ROOT}/data/ollama_bot.runtime.lock"
RESTART_LOG="${PROJECT_ROOT}/data/logs/manual_restart.log"
KILL_TIMEOUT_SECONDS=20

MODE="foreground"
CHILD_MODE=0
PULL_LATEST=0

usage() {
    cat <<'EOF'
Usage: ./scripts/run_bot.sh [options]

Default behavior:
  - Without options, start the bot in the foreground.

Options:
  --restart   stop the existing bot and start a new one in the background
  --pull      fast-forward pull the latest commit for the current branch, then restart
  -h, --help  show help
EOF
}

ensure_prereqs() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[run_bot.sh] ERROR: ${CONFIG_FILE} does not exist." >&2
        echo "  Copy config.yaml.example into place: cp config/config.yaml.example config/config.yaml" >&2
        exit 1
    fi

    if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
        echo "[run_bot.sh] ERROR: .venv is missing." >&2
        echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock" >&2
        exit 1
    fi
}

load_env() {
    # The app loads settings from config.yaml, so no extra env sourcing is needed.
    # Add runtime environment variables here if needed.
    :
}

ensure_runtime_dirs() {
    mkdir -p \
        "${PROJECT_ROOT}/data/conversations" \
        "${PROJECT_ROOT}/data/memory" \
        "${PROJECT_ROOT}/data/logs" \
        "${PROJECT_ROOT}/data/reports" \
        "${PROJECT_ROOT}/data/hf_cache/fastembed" \
        "${PROJECT_ROOT}/kb"
}

prepare_runtime_environment() {
    ensure_prereqs
    load_env
    ensure_runtime_dirs

    export PATH="${HOME}/.local/bin:${PATH}"
    export HF_HOME="${PROJECT_ROOT}/data/hf_cache"
    export FASTEMBED_CACHE_PATH="${PROJECT_ROOT}/data/hf_cache/fastembed"

    cd "${PROJECT_ROOT}"
}

git_pull_latest() {
    if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]]; then
        echo "[run_bot.sh] ERROR: the working tree is not clean, so --pull cannot proceed." >&2
        echo "  Commit or stash your changes, or use --restart without --pull." >&2
        exit 1
    fi

    local branch
    branch="$(git -C "${PROJECT_ROOT}" branch --show-current)"
    if [[ -z "${branch}" ]]; then
        echo "[run_bot.sh] ERROR: failed to determine the current git branch." >&2
        exit 1
    fi

    echo "[run_bot.sh] git pull --ff-only origin ${branch}"
    git -C "${PROJECT_ROOT}" pull --ff-only origin "${branch}"
}

is_systemd_service_active() {
    if ! command -v systemctl >/dev/null 2>&1; then
        return 1
    fi
    # Detect activating and reloading as well as active.
    local state
    state="$(systemctl --user show -P ActiveState ollama-bot 2>/dev/null || true)"
    [[ "${state}" == "active" || "${state}" == "activating" || "${state}" == "reloading" ]]
}

is_systemd_service_enabled() {
    if ! command -v systemctl >/dev/null 2>&1; then
        return 1
    fi
    systemctl --user is-enabled --quiet ollama-bot 2>/dev/null
}

stop_systemd_service_if_running() {
    if ! is_systemd_service_active; then
        return
    fi
    echo "[run_bot.sh] systemd service ollama-bot is running; stopping it first."
    systemctl --user stop ollama-bot
}

add_pid() {
    local pid="$1"
    if [[ -z "${pid}" || ! "${pid}" =~ ^[0-9]+$ ]]; then
        return
    fi
    if [[ "${pid}" == "$$" ]]; then
        return
    fi
    if kill -0 "${pid}" 2>/dev/null; then
        COLLECTED_PIDS["${pid}"]=1
    fi
}

collect_existing_bot_pids() {
    declare -gA COLLECTED_PIDS=()

    if [[ -f "${LOCK_FILE}" ]]; then
        add_pid "$(tr -cd '0-9' < "${LOCK_FILE}")"
    fi

    while read -r pid; do
        add_pid "${pid}"
    done < <(
        ps -eo pid=,args= | awk -v needle="${PROJECT_ROOT}/.venv/bin/python -m apps.ollama_bot.main" '
            index($0, needle) {print $1}
        '
    )
}

stop_existing_bot() {
    # Stop via systemd first if it owns the bot process.
    # Killing it directly may trigger Restart=on-failure.
    stop_systemd_service_if_running

    collect_existing_bot_pids
    if [[ "${#COLLECTED_PIDS[@]}" -eq 0 ]]; then
        echo "[run_bot.sh] No existing bot process is running."
        rm -f "${LOCK_FILE}"
        return
    fi

    local -a pids=("${!COLLECTED_PIDS[@]}")
    echo "[run_bot.sh] Stopping existing bot processes: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true

    local remaining=("${pids[@]}")
    local deadline=$((SECONDS + KILL_TIMEOUT_SECONDS))
    while [[ "${#remaining[@]}" -gt 0 && "${SECONDS}" -lt "${deadline}" ]]; do
        sleep 1
        local -a alive=()
        local pid=""
        for pid in "${remaining[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive+=("${pid}")
            fi
        done
        remaining=("${alive[@]}")
    done

    if [[ "${#remaining[@]}" -gt 0 ]]; then
        echo "[run_bot.sh] Shutdown timed out; sending SIGKILL: ${remaining[*]}"
        kill -9 "${remaining[@]}" 2>/dev/null || true
        sleep 1
    fi

    # Re-check for stragglers after shutdown.
    collect_existing_bot_pids
    if [[ "${#COLLECTED_PIDS[@]}" -gt 0 ]]; then
        local -a stragglers=("${!COLLECTED_PIDS[@]}")
        echo "[run_bot.sh] Force-killing remaining processes: ${stragglers[*]}"
        kill -9 "${stragglers[@]}" 2>/dev/null || true
        sleep 1
    fi

    # Remove the lock file so the new bot starts from a clean state.
    rm -f "${LOCK_FILE}"
}

start_background_bot() {
    echo "[run_bot.sh] Starting a new bot in the background."
    local launch_cmd
    printf -v launch_cmd 'cd %q && exec ./scripts/run_bot.sh --child >> %q 2>&1' \
        "${PROJECT_ROOT}" \
        "${RESTART_LOG}"
    setsid -f bash -lc "${launch_cmd}"

    local new_pid=""
    local attempt=0
    for attempt in $(seq 1 20); do
        if [[ -f "${LOCK_FILE}" ]]; then
            new_pid="$(tr -cd '0-9' < "${LOCK_FILE}")"
        fi
        if [[ -n "${new_pid}" ]] && kill -0 "${new_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if [[ -z "${new_pid}" ]] || ! kill -0 "${new_pid}" 2>/dev/null; then
        echo "[run_bot.sh] ERROR: the new bot exited immediately after startup." >&2
        echo "[run_bot.sh] Recent logs:" >&2
        tail -n 40 "${RESTART_LOG}" >&2 || true
        exit 1
    fi

    echo "[run_bot.sh] New bot PID: ${new_pid}"
    echo "[run_bot.sh] Log: ${RESTART_LOG}"
}

restart_via_systemd() {
    echo "[run_bot.sh] systemd user service (ollama-bot) is enabled; restarting via systemctl."
    systemctl --user restart ollama-bot
    echo "[run_bot.sh] systemctl --user restart ollama-bot completed"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restart)
            MODE="restart"
            shift
            ;;
        --pull)
            PULL_LATEST=1
            shift
            ;;
        --child)
            CHILD_MODE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[run_bot.sh] unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${PULL_LATEST}" == "1" && "${MODE}" != "restart" ]]; then
    echo "[run_bot.sh] ERROR: --pull must be used together with --restart." >&2
    exit 1
fi

prepare_runtime_environment

if [[ "${CHILD_MODE}" == "1" ]]; then
    exec "${VENV_DIR}/bin/python" -m apps.ollama_bot.main
fi

if [[ "${PULL_LATEST}" == "1" ]]; then
    git_pull_latest
fi

if [[ "${MODE}" == "restart" ]]; then
    if is_systemd_service_enabled; then
        # If the service is enabled, always restart through systemd.
        # Starting manually in that state can produce duplicate bot processes.
        # Even when systemd is active, clean up any manual processes first.
        stop_existing_bot
        restart_via_systemd
        exit 0
    fi
    stop_existing_bot
    start_background_bot
    exit 0
fi

exec "${VENV_DIR}/bin/python" -m apps.ollama_bot.main
