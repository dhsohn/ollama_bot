#!/usr/bin/env bash
# WSL 호스트에서 sim_host_agent를 실행하는 스크립트.
# Docker 외부에서 직접 실행하여 ORCA/OpenMPI 등 호스트 도구에 접근 가능.
#
# 사용법:
#   ./scripts/run_host_agent.sh
#
# 필수 환경변수 (.env 또는 export):
#   SIM_EXTERNAL_AGENT_TOKEN   - 인증 토큰 (봇 .env와 동일)
#
# 선택 환경변수:
#   SIM_INPUT_DIR_ORCA_AUTO    - orca_runs 호스트 경로 (기본: ~/orca_runs)
#   SIM_OUTPUT_DIR_ORCA_AUTO   - orca_outputs 호스트 경로 (기본: ~/orca_outputs)
#   SIM_TOOL_EXECUTABLE_ORCA_AUTO - orca_auto 실행 파일 (기본: ~/orca_auto/bin/orca_auto)
#   ORCA_AUTO_ORCA_EXECUTABLE  - orca 실행 파일 (기본: ~/opt/orca/orca)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# .env 파일이 있으면 로드 (# 주석, 빈 줄 무시)
ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        key="${key%%[[:space:]]}"
        value="${value##[[:space:]]}"
        # 이미 설정된 환경변수는 덮어쓰지 않음
        if [[ -z "${!key:-}" ]]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
    set +a
fi

# 필수 체크
if [[ -z "${SIM_EXTERNAL_AGENT_TOKEN:-}" ]]; then
    echo "오류: SIM_EXTERNAL_AGENT_TOKEN 환경변수가 필요합니다." >&2
    echo "  export SIM_EXTERNAL_AGENT_TOKEN=<토큰> 또는 .env 파일에 설정하세요." >&2
    exit 1
fi

# 기본값 설정
export SIM_INPUT_DIR_ORCA_AUTO="${SIM_INPUT_DIR_ORCA_AUTO:-$HOME/orca_runs}"
export SIM_OUTPUT_DIR_ORCA_AUTO="${SIM_OUTPUT_DIR_ORCA_AUTO:-$HOME/orca_outputs}"
export SIM_TOOL_EXECUTABLE_ORCA_AUTO="${SIM_TOOL_EXECUTABLE_ORCA_AUTO:-$HOME/orca_auto/bin/orca_auto}"
export ORCA_AUTO_ORCA_EXECUTABLE="${ORCA_AUTO_ORCA_EXECUTABLE:-$HOME/opt/orca/orca}"

# 경로 검증
for var in SIM_INPUT_DIR_ORCA_AUTO SIM_OUTPUT_DIR_ORCA_AUTO; do
    dir="${!var}"
    if [[ ! -d "$dir" ]]; then
        echo "경고: ${var}=${dir} 디렉터리가 존재하지 않습니다." >&2
    fi
done

for var in SIM_TOOL_EXECUTABLE_ORCA_AUTO ORCA_AUTO_ORCA_EXECUTABLE; do
    exe="${!var}"
    if [[ ! -x "$exe" ]]; then
        echo "경고: ${var}=${exe} 실행 파일이 없거나 실행 권한이 없습니다." >&2
    fi
done

echo "=== sim_host_agent 시작 ===" >&2
echo "  config:     ${REPO_ROOT}/config/config.yaml" >&2
echo "  input_dir:  ${SIM_INPUT_DIR_ORCA_AUTO}" >&2
echo "  output_dir: ${SIM_OUTPUT_DIR_ORCA_AUTO}" >&2
echo "  orca_auto:  ${SIM_TOOL_EXECUTABLE_ORCA_AUTO}" >&2
echo "  orca:       ${ORCA_AUTO_ORCA_EXECUTABLE}" >&2

exec python3 "${REPO_ROOT}/scripts/sim_host_agent.py" \
    --config "${REPO_ROOT}/config/config.yaml" \
    --host 0.0.0.0 \
    --port 18081
