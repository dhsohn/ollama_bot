#!/usr/bin/env bash
# Initial setup script for ollama_bot (WSL native)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config/config.yaml"
RUN_AFTER_SETUP=0
INSTALL_BOOT_SERVICE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Default behavior:
  - create or verify config/config.yaml
  - verify .venv
  - prepare the data/ and kb/ directories
  - check Ollama model status

Options:
  --run                    run run_bot.sh after setup
  --install-boot-service   install and enable the systemd boot service
  -h, --help               show help
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
            echo "[setup.sh] unknown option: $1" >&2
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
        echo "  - Missing: ${model_name} (download with: ollama pull ${model_name})"
    fi
}

cd "${PROJECT_ROOT}"

echo "=== ollama_bot setup ==="

# Create config.yaml
if [ ! -f config/config.yaml ]; then
    cp config/config.yaml.example config/config.yaml
    echo "Created config/config.yaml. Edit it for your environment."
else
    echo "config/config.yaml already exists."
fi

# Verify the virtual environment
if [ ! -f .venv/bin/python ]; then
    echo "WARNING: .venv is missing."
    echo "  python -m venv .venv && .venv/bin/pip install -r requirements.lock"
fi

# Create data directories
mkdir -p data/conversations data/memory data/logs data/reports data/hf_cache/fastembed
mkdir -p kb

ollama_host="$(extract_yaml_value "ollama" "host")"
chat_model="$(extract_yaml_value "ollama" "chat_model")"
embedding_model="$(extract_yaml_value "ollama" "embedding_model")"
reranker_model="$(extract_yaml_value "ollama" "reranker_model")"

echo "Architecture: Ollama single-stack (chat + embedding + reranking)"
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

# Check Ollama models
if command -v ollama >/dev/null 2>&1 && is_local_ollama_host "${ollama_host}"; then
    echo "Ollama CLI is installed. Checking local model status."
    check_ollama_model "${chat_model}"
    check_ollama_model "${embedding_model}"
    check_ollama_model "${reranker_model}"
elif command -v ollama >/dev/null 2>&1; then
    echo "Note: the Ollama host is not local, so CLI model checks are skipped."
    echo "      Check model status on the remote Ollama server (${ollama_host})."
else
    echo "Note: Ollama CLI is not installed."
    if [ -n "${ollama_host}" ]; then
        echo "      This is fine if the Ollama server (${ollama_host}) is remote."
    fi
fi

echo ""
echo "=== Setup complete ==="
echo "1. Edit config/config.yaml for your environment."
echo "2. Verify the ollama host and model values in config/config.yaml."
echo "3. Run: bash scripts/run_bot.sh"
echo "4. Optional boot auto-start: bash scripts/install_boot_service.sh"

if [[ "${INSTALL_BOOT_SERVICE}" == "1" ]]; then
    echo "[setup.sh] installing boot service"
    bash "${PROJECT_ROOT}/scripts/install_boot_service.sh"
fi

if [[ "${RUN_AFTER_SETUP}" == "1" ]]; then
    bash "${PROJECT_ROOT}/scripts/run_bot.sh"
fi
