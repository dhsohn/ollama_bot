#!/usr/bin/env bash
# ollama_bot 초기 설정 스크립트 (WSL 네이티브)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"
RUN_AFTER_SETUP=0
INSTALL_BOOT_SERVICE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]

기본 동작:
  - .env 생성/확인
  - .venv 확인
  - data/, kb/ 디렉토리 준비
  - retrieval 모델 상태 점검

Options:
  --run                    setup 후 run_bot.sh 실행
  --install-boot-service   systemd 부팅 서비스 설치/활성화
  -h, --help               도움말
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_AFTER_SETUP=1
            shift
            ;;
        --install-boot-service)
            INSTALL_BOOT_SERVICE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[setup.sh] 알 수 없는 옵션: $1" >&2
            usage
            exit 1
            ;;
    esac
done

extract_yaml_value() {
    local section="$1"
    local key="$2"
    if [ ! -f "${CONFIG_FILE}" ]; then
        return
    fi
    awk -v section="${section}" -v key="${key}" '
        $0 ~ "^[[:space:]]*" section ":[[:space:]]*$" {
            in_section=1
            next
        }
        in_section && $0 ~ "^[^[:space:]]" {
            in_section=0
        }
        in_section {
            pattern = "^[[:space:]]*" key ":[[:space:]]*"
            if ($0 ~ pattern) {
                value = $0
                sub(pattern, "", value)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
                gsub(/^"/, "", value)
                gsub(/"$/, "", value)
                print value
                exit
            }
        }
    ' "${CONFIG_FILE}"
}

check_ollama_model() {
    local model_name="$1"
    if [ -z "${model_name}" ]; then
        return
    fi
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${model_name}"; then
        echo "  - OK: ${model_name}"
    else
        echo "  - 누락: ${model_name} (다운로드: ollama pull ${model_name})"
    fi
}

cd "${PROJECT_ROOT}"

echo "=== ollama_bot setup ==="

# .env 생성
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env 파일이 생성되었습니다. 환경에 맞게 편집하세요."
else
    echo ".env 파일이 이미 존재합니다."
fi

# venv 확인
if [ ! -f .venv/bin/python ]; then
    echo "WARNING: .venv이 없습니다."
    echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock"
fi

# 데이터 디렉토리 생성
mkdir -p data/conversations data/memory data/logs data/reports data/hf_cache/fastembed
mkdir -p kb

lemonade_host="$(extract_yaml_value "lemonade" "host")"
ollama_host="$(extract_yaml_value "ollama" "host")"
embedding_model="$(extract_yaml_value "ollama" "embedding_model")"
reranker_model="$(extract_yaml_value "ollama" "reranker_model")"

echo "아키텍처: Lemonade(응답) + Ollama(임베딩/리랭킹)"
if [ -n "${lemonade_host}" ]; then
    echo "- Lemonade host: ${lemonade_host}"
fi
if [ -n "${ollama_host}" ]; then
    echo "- Ollama host: ${ollama_host}"
fi

# Ollama 모델 확인 (retrieval 전용)
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama CLI가 설치되어 있습니다. retrieval 모델 상태를 확인합니다."
    check_ollama_model "${embedding_model}"
    check_ollama_model "${reranker_model}"
else
    echo "참고: Ollama CLI가 없습니다."
    if [ -n "${ollama_host}" ]; then
        echo "      retrieval 서버(${ollama_host})가 원격이면 정상입니다."
    fi
fi

echo ""
echo "=== setup 완료 ==="
echo "1. .env 파일을 환경에 맞게 편집하세요."
echo "2. config/config.yaml의 lemonade/ollama host 및 모델값을 확인하세요."
echo "3. 실행: bash scripts/run_bot.sh"
echo "4. (선택) 부팅 자동 실행: bash scripts/install_boot_service.sh"

if [[ "${INSTALL_BOOT_SERVICE}" == "1" ]]; then
    echo "[setup.sh] boot service 설치"
    bash "${PROJECT_ROOT}/scripts/install_boot_service.sh"
fi

if [[ "${RUN_AFTER_SETUP}" == "1" ]]; then
    bash "${PROJECT_ROOT}/scripts/run_bot.sh"
fi
