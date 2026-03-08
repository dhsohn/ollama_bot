#!/usr/bin/env bash
# WSL에서 ollama_bot을 직접 실행/재시작하는 스크립트.
#
# 사용법:
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

기본 동작:
  - 옵션 없이 실행하면 봇을 포그라운드로 시작한다.

Options:
  --restart   기존 봇을 종료하고 새 봇을 백그라운드로 시작한다.
  --pull      현재 브랜치 최신 커밋을 fast-forward pull 후 재시작한다.
  -h, --help  도움말
EOF
}

ensure_prereqs() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[run_bot.sh] ERROR: ${CONFIG_FILE} 파일이 없습니다." >&2
        echo "  config.yaml.example을 복사하여 설정하세요: cp config/config.yaml.example config/config.yaml" >&2
        exit 1
    fi

    if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
        echo "[run_bot.sh] ERROR: .venv이 없습니다." >&2
        echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock" >&2
        exit 1
    fi
}

load_env() {
    # config.yaml에서 설정을 로드하므로 별도 env 소싱 불필요.
    # 런타임 환경변수 설정이 필요한 경우 여기에 추가.
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
        echo "[run_bot.sh] ERROR: working tree가 비어 있지 않아 --pull을 진행할 수 없습니다." >&2
        echo "  커밋/스태시 후 다시 실행하거나 --pull 없이 --restart만 사용하세요." >&2
        exit 1
    fi

    local branch
    branch="$(git -C "${PROJECT_ROOT}" branch --show-current)"
    if [[ -z "${branch}" ]]; then
        echo "[run_bot.sh] ERROR: 현재 git branch를 확인할 수 없습니다." >&2
        exit 1
    fi

    echo "[run_bot.sh] git pull --ff-only origin ${branch}"
    git -C "${PROJECT_ROOT}" pull --ff-only origin "${branch}"
}

is_systemd_service_active() {
    if ! command -v systemctl >/dev/null 2>&1; then
        return 1
    fi
    # active뿐 아니라 activating(재시작 대기) 상태도 감지한다.
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
    echo "[run_bot.sh] systemd ollama-bot 서비스가 실행 중이므로 먼저 중지합니다."
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
    # systemd가 봇을 관리하고 있으면 먼저 systemd를 통해 중지한다.
    # 직접 kill하면 Restart=on-failure로 systemd가 재시작을 시도하기 때문이다.
    stop_systemd_service_if_running

    collect_existing_bot_pids
    if [[ "${#COLLECTED_PIDS[@]}" -eq 0 ]]; then
        echo "[run_bot.sh] 실행 중인 기존 봇이 없습니다."
        rm -f "${LOCK_FILE}"
        return
    fi

    local -a pids=("${!COLLECTED_PIDS[@]}")
    echo "[run_bot.sh] 기존 봇 종료: ${pids[*]}"
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
        echo "[run_bot.sh] 종료 지연으로 SIGKILL 전송: ${remaining[*]}"
        kill -9 "${remaining[@]}" 2>/dev/null || true
        sleep 1
    fi

    # 종료 후 잔여 프로세스 재확인
    collect_existing_bot_pids
    if [[ "${#COLLECTED_PIDS[@]}" -gt 0 ]]; then
        local -a stragglers=("${!COLLECTED_PIDS[@]}")
        echo "[run_bot.sh] 잔여 프로세스 강제 종료: ${stragglers[*]}"
        kill -9 "${stragglers[@]}" 2>/dev/null || true
        sleep 1
    fi

    # lock 파일 삭제하여 새 봇이 깨끗한 상태에서 시작하도록 보장
    rm -f "${LOCK_FILE}"
}

start_background_bot() {
    echo "[run_bot.sh] 새 봇을 백그라운드로 시작합니다."
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
        echo "[run_bot.sh] ERROR: 새 봇이 시작 직후 종료되었습니다." >&2
        echo "[run_bot.sh] 최근 로그:" >&2
        tail -n 40 "${RESTART_LOG}" >&2 || true
        exit 1
    fi

    echo "[run_bot.sh] 새 봇 PID: ${new_pid}"
    echo "[run_bot.sh] 로그: ${RESTART_LOG}"
}

restart_via_systemd() {
    echo "[run_bot.sh] systemd user service(ollama-bot) 활성 상태입니다. systemctl로 재시작합니다."
    systemctl --user restart ollama-bot
    echo "[run_bot.sh] systemctl --user restart ollama-bot 완료"
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
            echo "[run_bot.sh] 알 수 없는 옵션: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${PULL_LATEST}" == "1" && "${MODE}" != "restart" ]]; then
    echo "[run_bot.sh] ERROR: --pull은 --restart와 함께 사용해야 합니다." >&2
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
        # systemd 서비스가 enabled이면 항상 systemd를 통해 재시작한다.
        # enabled 상태에서 수동 시작하면 systemd가 별도로 봇을 띄워 중복 실행된다.
        # systemd가 active여도 수동 프로세스가 남아 있을 수 있으므로 항상 정리한다.
        stop_existing_bot
        restart_via_systemd
        exit 0
    fi
    stop_existing_bot
    start_background_bot
    exit 0
fi

exec "${VENV_DIR}/bin/python" -m apps.ollama_bot.main
