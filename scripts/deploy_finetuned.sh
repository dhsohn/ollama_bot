#!/usr/bin/env bash
# 파인튜닝된 모델을 Ollama에 배포하는 스크립트.
#
# 사용법:
#   ./scripts/deploy_finetuned.sh <모델_디렉터리> [모델_이름]
#
# 예시:
#   ./scripts/deploy_finetuned.sh models/finetuned ollama_bot_ft

set -euo pipefail

MODEL_DIR="${1:?사용법: $0 <모델_디렉터리> [모델_이름]}"
MODEL_NAME="${2:-ollama_bot_ft}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/Modelfile.template"

if [ ! -d "$MODEL_DIR" ]; then
    echo "오류: 모델 디렉터리를 찾을 수 없습니다: $MODEL_DIR" >&2
    exit 1
fi

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "오류: Modelfile 템플릿을 찾을 수 없습니다: $TEMPLATE_FILE" >&2
    exit 1
fi

# Modelfile 생성
MODELFILE="${MODEL_DIR}/Modelfile"
sed "s|{{MODEL_PATH}}|${MODEL_DIR}|g" "$TEMPLATE_FILE" > "$MODELFILE"
echo "Modelfile 생성됨: $MODELFILE"

# Ollama에 모델 생성
echo "Ollama 모델 생성 중: $MODEL_NAME"
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo "배포 완료: $MODEL_NAME"
echo "테스트: ollama run $MODEL_NAME"
