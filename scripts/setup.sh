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
  - config/config.yaml 생성/확인
  - .venv 확인
  - data/, kb/ 디렉토리 준비
  - Ollama 모델 상태 점검

Options:
  --run                    setup 후 run_bot.sh 실행
  --install-boot-service   systemd 부팅 서비스 설치/활성화
  -h, --help               도움말
EOF
}

is_local_ollama_host() {
    local host="$1"
    if [[ -z "${host}" ]]; then
        return 0
    fi
    [[ "${host}" == http://localhost* || "${host}" == https://localhost* || \
       "${host}" == http://127.0.0.1* || "${host}" == https://127.0.0.1* || \
       "${host}" == localhost* || "${host}" == 127.0.0.1* ]]
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

# config.yaml 생성
if [ ! -f config/config.yaml ]; then
    cp config/config.yaml.example config/config.yaml
    echo "config/config.yaml 파일이 생성되었습니다. 환경에 맞게 편집하세요."
else
    echo "config/config.yaml 파일이 이미 존재합니다."
fi

# venv 확인
if [ ! -f .venv/bin/python ]; then
    echo "WARNING: .venv이 없습니다."
    echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock"
fi

# 데이터 디렉토리 생성
mkdir -p data/conversations data/memory data/logs data/reports data/hf_cache/fastembed
mkdir -p kb

ollama_host="$(extract_yaml_value "ollama" "host")"
chat_model="$(extract_yaml_value "ollama" "chat_model")"
embedding_model="$(extract_yaml_value "ollama" "embedding_model")"
reranker_model="$(extract_yaml_value "ollama" "reranker_model")"

echo "아키텍처: Ollama 단일 스택 (chat + embedding + reranking)"
if [ -n "${ollama_host}" ]; then
    echo "- Ollama host: ${ollama_host}"
fi
if [ -n "${chat_model}" ]; then
    echo "- Chat model: ${chat_model}"
fi
if [ -n "${embedding_model}" ]; then
    echo "- Embedding model: ${embedding_model}"
fi
if [ -n "${reranker_model}" ]; then
    echo "- Reranker model: ${reranker_model}"
fi

# Ollama 모델 확인
if command -v ollama >/dev/null 2>&1 && is_local_ollama_host "${ollama_host}"; then
    echo "Ollama CLI가 설치되어 있습니다. 로컬 모델 상태를 확인합니다."
    check_ollama_model "${chat_model}"
    check_ollama_model "${embedding_model}"
    check_ollama_model "${reranker_model}"
elif command -v ollama >/dev/null 2>&1; then
    echo "참고: Ollama host가 로컬이 아니므로 CLI 모델 검사를 건너뜁니다."
    echo "      원격 Ollama 서버(${ollama_host})에서 모델 상태를 확인하세요."
else
    echo "참고: Ollama CLI가 없습니다."
    if [ -n "${ollama_host}" ]; then
        echo "      Ollama 서버(${ollama_host})가 원격이면 정상입니다."
    fi
fi

echo ""
echo "=== setup 완료 ==="
echo "1. config/config.yaml 파일을 환경에 맞게 편집하세요."
echo "2. config/config.yaml의 ollama host 및 모델값을 확인하세요."
echo "3. 실행: bash scripts/run_bot.sh"
echo "4. (선택) 부팅 자동 실행: bash scripts/install_boot_service.sh"

if [[ "${INSTALL_BOOT_SERVICE}" == "1" ]]; then
    echo "[setup.sh] boot service 설치"
    bash "${PROJECT_ROOT}/scripts/install_boot_service.sh"
fi

if [[ "${RUN_AFTER_SETUP}" == "1" ]]; then
    bash "${PROJECT_ROOT}/scripts/run_bot.sh"
fi
