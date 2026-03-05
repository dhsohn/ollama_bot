#!/usr/bin/env bash
# WSL에서 ollama_bot을 직접 실행하는 스크립트.
#
# 사용법:
#   ./scripts/run_bot.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[run_bot.sh] ERROR: ${ENV_FILE} 파일이 없습니다." >&2
    echo "  .env.example을 복사하여 설정하세요: cp .env.example .env" >&2
    exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
    echo "[run_bot.sh] ERROR: .venv이 없습니다." >&2
    echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock" >&2
    exit 1
fi

# .env 로드
set -a
source "${ENV_FILE}"
set +a

# 데이터 디렉터리 보장
mkdir -p \
    "${PROJECT_ROOT}/data/conversations" \
    "${PROJECT_ROOT}/data/memory" \
    "${PROJECT_ROOT}/data/logs" \
    "${PROJECT_ROOT}/data/reports" \
    "${PROJECT_ROOT}/data/hf_cache/fastembed" \
    "${PROJECT_ROOT}/kb/orca_runs" \
    "${PROJECT_ROOT}/kb/orca_outputs"

cd "${PROJECT_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${PROJECT_ROOT}/data/hf_cache"
export FASTEMBED_CACHE_PATH="${PROJECT_ROOT}/data/hf_cache/fastembed"

exec "${VENV_DIR}/bin/python" -m apps.ollama_bot.main
