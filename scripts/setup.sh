#!/bin/bash
# ollama_bot 초기 설정 스크립트
set -euo pipefail

CONFIG_FILE="config/config.yaml"

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

echo "=== ollama_bot Setup ==="

# Docker 확인
command -v docker >/dev/null 2>&1 || { echo "오류: Docker가 필요합니다."; exit 1; }

# .env 생성
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env 파일이 생성되었습니다."
else
    echo ".env 파일이 이미 존재합니다."
fi

# 데이터 디렉토리 생성
mkdir -p data/conversations data/memory data/logs data/reports

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
echo "=== 설정 완료 ==="
echo "빠른 실행(추천): bash scripts/bootstrap.sh"
echo "1. .env 파일을 환경에 맞게 편집하세요."
echo "2. config/config.yaml의 lemonade/ollama host 및 모델값을 확인하세요."
echo "3. (로컬 Ollama 사용 시) ollama serve"
echo "4. 봇 실행: docker compose -f docker-compose.yml up -d"
echo "5. (선택) 부팅 자동 실행: sudo bash scripts/install_boot_service.sh"
