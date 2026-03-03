#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TMP_LOCK="$(mktemp "${TMPDIR:-/tmp}/requirements.lock.check.XXXXXX")"

cleanup() {
  rm -f "${TMP_LOCK}"
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"

PYTHON_BIN=""
for candidate in \
  "${PROJECT_ROOT}/.venv/bin/python" \
  "$(command -v python 2>/dev/null || true)" \
  "$(command -v python3 2>/dev/null || true)"; do
  if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
    continue
  fi
  if ! "${candidate}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
    continue
  fi
  if "${candidate}" -c "import piptools" >/dev/null 2>&1; then
    PYTHON_BIN="${candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[check_requirements_lock] ERROR: Python 3.11+ with pip-tools is required."
  echo "Install dev dependencies first:"
  echo "  pip install -r requirements-dev.txt"
  exit 1
fi

"${PYTHON_BIN}" -m piptools compile \
  --quiet \
  --no-strip-extras \
  --constraint=requirements.lock \
  --output-file="${TMP_LOCK}" \
  --pip-args='--use-feature=fast-deps' \
  requirements.txt

normalize_lock() {
  sed -nE 's/^([A-Za-z0-9_.-]+==[^[:space:]]+).*/\1/p' "$1" | sort
}

if ! diff -u <(normalize_lock requirements.lock) <(normalize_lock "${TMP_LOCK}") >/dev/null; then
  echo "[check_requirements_lock] ERROR: requirements.lock is out of date."
  echo "Run:"
  echo "  pip-compile --output-file=requirements.lock --pip-args='--use-feature=fast-deps' requirements.txt"
  echo
  diff -u <(normalize_lock requirements.lock) <(normalize_lock "${TMP_LOCK}") || true
  exit 1
fi

echo "[check_requirements_lock] OK: requirements.lock is up to date."
