# ollama_bot

[![CI](https://github.com/dhsohn/ollama_bot/actions/workflows/ci.yml/badge.svg)](https://github.com/dhsohn/ollama_bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A private Telegram bot powered by a local [Ollama](https://ollama.com) server. Designed for WSL2/Linux environments.

- **Chat** with a local LLM via Telegram — streaming responses, long conversation memory
- **Skills** system — trigger specialized actions with keywords (YAML-based, hot-reload)
- **Automations** — scheduled tasks that run in the background
- **Feedback** — thumbs up/down with optional reason collection
- **RAG** — knowledge base retrieval using local embedding + reranking models

---

## Language Support

The bot supports **English** and **Korean** (`en` / `ko`).

When you send `/start` for the first time, the bot displays a language selection prompt:

```
Welcome to ollama_bot!

Please select your language:

[ 한국어 ]  [ English ]
```

Tap your preferred language — the choice is saved and all subsequent bot messages
(commands, help text, status, feedback, etc.) will be in that language.

You can change the language at any time from the **Settings** menu (`/start` → ⚙️ Settings).

The default fallback language is configured in `config/config.yaml` under `bot.language`.

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> && cd ollama_bot
python -m venv .venv
.venv/bin/pip install -r requirements.lock
```

### 2. Configure

```bash
cp config/config.yaml.example config/config.yaml
```

Edit `config/config.yaml` and set at minimum:

```yaml
telegram:
  bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
  allowed_users: "YOUR_TELEGRAM_CHAT_ID"   # comma-separated for multiple users
```

> `config/config.yaml` contains secrets — it is gitignored by default.

### 3. Pull Ollama models

```bash
ollama pull gpt-oss:20b
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
ollama pull dengcao/bge-reranker-v2-m3:latest
```

| Service | Port  | Role                              |
|---------|-------|-----------------------------------|
| Ollama  | 11434 | Chat · Embedding · Reranking      |

### 4. Run

```bash
bash scripts/run_bot.sh
```

Runs in the foreground. Logs are printed directly to the terminal.

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Start the bot — select language on first use |
| `/help` | List all available commands |
| `/skills` | Show loaded skills (`/skills reload` to hot-reload) |
| `/auto` | Manage automations (`list` · `run <name>` · `disable <name>` · `reload`) |
| `/memory` | View memory stats (`/memory clear` · `/memory export`) |
| `/status` | System status (uptime, models, skills, automations) |
| `/feedback` | Feedback statistics |
| `/continue` | Continue a long response that was cut off |

---

## Update & Restart

```bash
bash scripts/run_bot.sh --restart           # kill current bot and restart in background
bash scripts/run_bot.sh --restart --pull    # git pull then restart
```

`--pull` only runs if the working tree is clean. Restart logs are written to `data/logs/manual_restart.log`.

---

## Auto-start with systemd

```bash
bash scripts/install_boot_service.sh
systemctl --user start ollama-bot
```

```bash
journalctl --user -u ollama-bot -f      # follow logs
systemctl --user status ollama-bot      # service status
systemctl --user restart ollama-bot     # restart
```

---

## Configuration Reference

All settings live in `config/config.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `bot` | Bot name, default language (`ko`/`en`), conversation length |
| `telegram` | Bot token, allowed user IDs |
| `ollama` | Host, chat model, embedding model, reranker model |
| `rag` | Knowledge base directories, retrieval settings |
| `memory` | SQLite backend, retention period |
| `feedback` | Enable/disable, reason collection, DICL examples |
| `semantic_cache` | Similarity threshold, TTL, exclusion patterns |
| `intent_router` | Route config path, confidence threshold |
| `response_planner` | Trigger conditions, token budget |
| `response_reviewer` | Review-then-rewrite pipeline settings |
| `scheduler` | Timezone for automation schedules |

---

## Architecture

Request processing follows this layered pipeline:

```
Telegram Update
  → TelegramHandler
  → Engine
     → Skill trigger          (keyword match → YAML skill)
     → Instant responder      (rule-based fast reply)
     → Intent router          (semantic routing)
     → Semantic cache         (cached similar queries)
     → Full LLM
        → Response planner    (JSON plan for complex queries)
        → Draft answer
        → Response reviewer   (quality check + optional rewrite)
        → RAG                 (knowledge base retrieval)
        → Final answer
  → Memory / Feedback / Automation
  → Telegram response
```

### Module Map

| Module | Role |
|--------|------|
| `core/telegram_handler.py` | Telegram entrypoint — handler registration and dependency wiring |
| `core/telegram_commands*.py` | `/start`, `/help`, `/skills`, `/auto`, `/memory`, `/status` |
| `core/telegram_menus.py` | Inline menus, onboarding flow, language selection |
| `core/telegram_streaming.py` | Streaming message rendering |
| `core/telegram_feedback.py` | 👍/👎 callbacks, reason collection |
| `core/engine.py` | Orchestration public API |
| `core/engine_routing.py` | Skill / instant / cache / full routing logic |
| `core/engine_context.py` | Prompt assembly, history compression, DICL injection |
| `core/engine_planner.py` | JSON response plan for complex queries |
| `core/engine_reviewer.py` | Draft review and conditional rewrite |
| `core/engine_rag.py` | RAG retrieval and context injection |
| `core/memory.py` | SQLite-backed conversation and long-term memory |
| `core/semantic_cache.py` | Embedding-based query cache |
| `core/intent_router.py` | Semantic intent routing |
| `core/i18n.py` | `en`/`ko` string catalog — `t(key, lang)` |
| `core/auto_scheduler.py` | Automation scheduler |

---

## Project Structure

```
ollama_bot/
├── apps/ollama_bot/main.py     # entry point
├── core/                       # bot logic
├── config/
│   ├── config.yaml             # runtime config (gitignored)
│   └── config.yaml.example     # config template
├── skills/
│   ├── _builtin/               # built-in skill YAMLs
│   └── custom/                 # your custom skills
├── auto/
│   ├── _builtin/               # built-in automation YAMLs
│   └── custom/                 # your custom automations
├── kb/                         # knowledge base files (RAG)
├── scripts/                    # operational scripts
├── tests/                      # test suite
└── data/                       # runtime data (DB, logs, reports)
```

---

## Operations Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_bot.sh` | Run / restart the bot (`--restart`, `--pull`) |
| `scripts/setup.sh` | Initial setup — config, directories, model check |
| `scripts/install_boot_service.sh` | Install systemd user service |
| `scripts/update_wsl_hosts.sh` | Update `/etc/hosts` with WSL gateway IP |
| `scripts/healthcheck.sh` | Health check |
| `scripts/soak_monitor.sh` | Long-running stability monitor |
| `scripts/check_requirements_lock.sh` | Validate dependency lock file |

---

## Dependencies

```bash
# Regenerate lock file
pip-compile --output-file=requirements.lock requirements.txt

# Validate
bash scripts/check_requirements_lock.sh
```

---

## Quality Checks

```bash
.venv/bin/ruff check .
.venv/bin/mypy core apps main.py
.venv/bin/pytest -q
.venv/bin/pytest -q --cov=core --cov=apps --cov-report=term-missing:skip-covered
```

| Check | Command | CI artifact |
|-------|---------|-------------|
| Lock file | `bash scripts/check_requirements_lock.sh` | drift check |
| Lint | `.venv/bin/ruff check .` | ruff report |
| Type check | `.venv/bin/mypy core apps main.py` | mypy report |
| Tests | `.venv/bin/pytest -q` | test results |
| Coverage | `pytest --cov ...` | `coverage.xml` + console summary |

---

## License

MIT
