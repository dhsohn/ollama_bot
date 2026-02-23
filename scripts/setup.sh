#!/bin/bash
# ollama_bot 초기 설정 스크립트
set -euo pipefail

echo "=== ollama_bot Setup ==="

# Docker 확인
command -v docker >/dev/null 2>&1 || { echo "오류: Docker가 필요합니다."; exit 1; }

# .env 생성
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env 파일이 생성되었습니다. 텔레그램 봇 토큰 등을 설정하세요."
else
    echo ".env 파일이 이미 존재합니다."
fi

# 데이터 디렉토리 생성
mkdir -p data/conversations data/memory data/logs data/reports

# Ollama 확인
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama가 설치되어 있습니다."
    if ! ollama list 2>/dev/null | grep -q "gpt-oss:20b"; then
        echo "기본 모델이 없습니다. 다운로드: ollama pull gpt-oss:20b"
    else
        echo "기본 모델(gpt-oss:20b)이 준비되어 있습니다."
    fi
else
    echo "경고: Ollama가 설치되지 않았습니다. https://ollama.com 에서 설치하세요."
fi

echo ""
echo "=== 설정 완료 ==="
echo "1. .env 파일을 편집하여 TELEGRAM_BOT_TOKEN을 설정하세요."
echo "2. Ollama를 실행하세요: ollama serve"
echo "3. 봇을 실행하세요: docker compose up -d"
echo "4. (선택) 부팅 자동 실행: sudo bash scripts/install_boot_service.sh"
