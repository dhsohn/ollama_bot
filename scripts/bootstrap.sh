#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_FLAG=""
INSTALL_BOOT_SERVICE=0
SKIP_WINDOWS_FIX=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [options]

원클릭 설치/실행:
1) setup.sh 실행
2) (lemonade provider일 때) Windows 방화벽/portproxy 자동 설정
3) up.sh 실행

Options:
  --build                 docker image를 강제 재빌드하여 실행
  --install-boot-service  ollama_bot.service 설치/활성화 (sudo 필요)
  --skip-windows-fix      Windows 네트워크 자동 설정 단계 건너뜀
  -h, --help              도움말
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD_FLAG="--build"
      shift
      ;;
    --install-boot-service)
      INSTALL_BOOT_SERVICE=1
      shift
      ;;
    --skip-windows-fix)
      SKIP_WINDOWS_FIX=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bootstrap.sh] 알 수 없는 옵션: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

echo "[bootstrap.sh] step 1/3: 기본 설정"
bash "${PROJECT_ROOT}/scripts/setup.sh"

if [[ "${SKIP_WINDOWS_FIX}" == "0" ]]; then
  echo "[bootstrap.sh] step 2/3: Windows 네트워크 자동 설정(lemonade일 때만)"
  bash "${PROJECT_ROOT}/scripts/configure_windows_lemonade.sh"
else
  echo "[bootstrap.sh] step 2/3: Windows 네트워크 자동 설정 건너뜀"
fi

if [[ "${INSTALL_BOOT_SERVICE}" == "1" ]]; then
  echo "[bootstrap.sh] boot service 설치"
  sudo bash "${PROJECT_ROOT}/scripts/install_boot_service.sh"
fi

echo "[bootstrap.sh] step 3/3: 컨테이너 실행"
if [[ -n "${BUILD_FLAG}" ]]; then
  bash "${PROJECT_ROOT}/scripts/up.sh" --build
else
  bash "${PROJECT_ROOT}/scripts/up.sh"
fi

echo "[bootstrap.sh] 완료"
