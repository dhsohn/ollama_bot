# ollama_bot

[![CI](https://github.com/dhsohn/ollama_bot/actions/workflows/ci.yml/badge.svg)](https://github.com/dhsohn/ollama_bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

WSL2/리눅스 환경용 텔레그램 private-chat 봇. Dual-Provider 아키텍처 기반.

- **Lemonade Server**: LLM 응답 (`gpt-oss-20b-NPU`)
- **Ollama Server**: 임베딩/리랭킹 (`Qwen3-Embedding-0.6B`, `bge-reranker-v2-m3`)
- **시뮬레이션**: ORCA/orca_auto 작업을 직접 subprocess로 실행

## 핵심 기능

- 텔레그램 1:1 대화 + 스트리밍 응답
- 스킬/자동화 시스템 (YAML 기반, hot-reload)
- 피드백 수집 (👍/👎 + 사유)
- 계층형 응답: 스킬 → 즉시응답 → 인텐트라우팅 → 시맨틱캐시 → Full LLM(RAG)
- 시뮬레이션 큐: `/sim submit|list|status|cancel` — 리소스 관리, 재시도, 알림
- 장기 메모리/대화 보관 (SQLite)
- DFT 출력 자동 인덱싱 + 모니터링

## 빠른 시작

```bash
git clone <repo-url> && cd ollama_bot
python -m venv .venv
.venv/bin/pip install -r requirements.lock
cp .env.example .env   # 토큰/ID 설정
```

### `.env` 설정

```env
TELEGRAM_BOT_TOKEN=...
ALLOWED_TELEGRAM_USERS=123456789
SIM_INPUT_DIR_ORCA_AUTO=kb/orca_runs
```

- `ALLOWED_TELEGRAM_USERS`: 숫자 Chat ID (쉼표 구분). placeholder 상태면 시작 시 종료됨.
- `SIM_INPUT_DIR_ORCA_AUTO`: `/sim submit orca_auto mj1` → `kb/orca_runs/mj1` 으로 해석
- `strict_startup`: optional 컴포넌트 초기화 실패 시 즉시 종료 여부 (`false` 기본)
- 런타임 설정은 `config/config.yaml` 에서 관리

### LLM 백엔드

로컬/원격 환경에서 두 서버 실행:

| 서버 | 포트 | 역할 |
|---|---|---|
| Lemonade | 8000 | LLM 응답 (`gpt-oss-20b-NPU`) |
| Ollama | 11434 | 임베딩 + 리랭킹 |

```bash
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
ollama pull dengcao/bge-reranker-v2-m3:latest
```

### 실행

```bash
bash scripts/run_bot.sh
```

- 포그라운드 실행. 터미널을 붙여 둔 상태에서 로그를 직접 볼 때 사용.

### 업데이트/재시작

```bash
bash scripts/run_bot.sh --restart
bash scripts/run_bot.sh --restart --pull
```

- `--restart`: 현재 프로젝트에서 실행 중인 기존 봇을 종료하고 새 봇을 백그라운드로 다시 시작
- `--restart --pull`: 현재 브랜치에 대해 `git pull --ff-only` 후 재시작
- `--pull`은 working tree가 깨끗할 때만 동작
- 재시작 로그: `data/logs/manual_restart.log`
- `ollama-bot` systemd user service가 활성 상태면 수동 kill 대신 `systemctl --user restart ollama-bot`를 사용

### systemd 자동 시작

```bash
bash scripts/install_boot_service.sh
systemctl --user start ollama-bot
```

```bash
journalctl --user -u ollama-bot -f      # 로그
systemctl --user status ollama-bot       # 상태
```

## 텔레그램 명령어

| 명령어 | 설명 |
|---|---|
| `/start`, `/help` | 시작/도움말 |
| `/skills` | 스킬 목록 (`/skills reload` 리로드) |
| `/auto list` | 자동화 목록 (`/auto run <name>` 수동 실행) |
| `/memory` | 메모리 통계 (`/memory clear` 삭제, `/memory export` 내보내기) |
| `/status` | 시스템 상태 |
| `/feedback` | 피드백 통계 |
| `/sim submit <tool> <path>` | 시뮬레이션 제출 |
| `/sim list`, `/sim status` | 작업 목록/상태 |
| `/sim cancel <id>` | 작업 취소 |

## 아키텍처 개요

요청 처리 흐름은 아래 순서를 따른다.

```text
Telegram Update
  -> TelegramHandler
  -> Engine
     -> Skill trigger
     -> Instant responder
     -> Intent router
     -> Semantic cache
     -> Full LLM + RAG
  -> Memory / Feedback / Automation / Sim queue
  -> Telegram response
```

### 레이어별 책임

| 레이어 | 주요 파일 | 책임 |
|---|---|---|
| Runtime | `core/app_runtime.py`, `core/runtime_*` | 설정 로딩, 의존성 생성, startup/shutdown, background task |
| Interface | `core/telegram_handler.py`, `core/telegram_*` | Telegram command/callback/message 처리, 스트리밍 UX, feedback 버튼 |
| Orchestration | `core/engine.py`, `core/engine_*` | 계층형 라우팅, 컨텍스트 조립, 모델 준비, 요청 추적, RAG/요약 orchestration |
| Domain services | `core/memory.py`, `core/security.py`, `core/feedback_manager.py`, `core/semantic_cache.py`, `core/intent_router.py` | 보안, 메모리, 피드백, 캐시, 라우팅 정책 |
| Retrieval | `core/rag/*` | 인덱싱, 검색, rerank, RAG context 구성 |
| Operations | `core/auto_scheduler.py`, `core/sim_scheduler.py`, `core/automation_callables*.py` | 자동화 실행, 시뮬레이션 큐, 운영형 작업 orchestration |

### 주요 모듈 맵

| 모듈 | 역할 |
|---|---|
| `core/telegram_handler.py` | Telegram entrypoint. handler registration과 dependency wiring만 담당 |
| `core/telegram_commands*.py` | `/start`, `/help`, `/skills`, `/auto`, `/memory`, `/status` 등 명령 처리 |
| `core/telegram_streaming.py` | 일반 메시지, streaming render, analyze-all fallback |
| `core/telegram_continuation.py` | 긴 응답 이어보기 state와 follow-up message 생성 |
| `core/telegram_feedback.py` | 👍/👎 callback, reason 수집, preview cache 관리 |
| `core/telegram_sim.py` | `/sim` 큐 명령 처리 |
| `core/engine.py` | public API와 최상위 orchestration 유지 |
| `core/engine_routing.py` | skill/instant/cache/full 라우팅 판단 |
| `core/engine_context.py` | prompt/history 조립, preference/guideline/DICL 주입 |
| `core/engine_summary.py` | summarize skill과 chunked summary pipeline |
| `core/engine_rag.py` | full request 준비, RAG 주입, full-scan analyze/reindex |
| `core/engine_tracking.py` | request lifecycle, stream meta, persistence, request logging |
| `core/engine_background.py`, `core/engine_models.py` | background summary task, model prepare/role resolution |

## 프로젝트 구조

```text
ollama_bot/
├── apps/ollama_bot/main.py    # 엔트리포인트
├── core/                      # 코어 로직
├── config/config.yaml         # 런타임 설정
├── skills/{_builtin,custom}/  # 스킬 YAML
├── auto/{_builtin,custom}/    # 자동화 YAML
├── scripts/                   # 운영 스크립트
├── tests/                     # 테스트
├── kb/                        # 지식베이스 (orca_runs, orca_outputs)
├── data/                      # 런타임 데이터 (DB, 로그, 리포트)
└── .env                       # 시크릿
```

## 모듈 분리 원칙

- 줄 수가 아니라 책임 경계로 분리한다. 예: `telegram`은 `commands/messages/feedback/sim`, `engine`은 `routing/context/summary/rag/tracking`으로 나눈다.
- public entrypoint는 유지한다. 예: `core.engine.Engine`, `core.telegram_handler.TelegramHandler`는 계속 최상위 import 경로를 보존한다.
- orchestrator는 얇게 유지한다. 상위 파일은 흐름 제어와 wiring 중심으로 두고, 세부 구현은 helper module로 이동한다.
- 구조 변경과 동작 변경을 섞지 않는다. 리팩터링 단계에서는 동작을 바꾸지 않고, helper 추출 후 테스트로 회귀를 막는다.
- 테스트가 모듈 경계의 안전망 역할을 한다. `pytest`, `mypy`, `ruff`를 리팩터링마다 반복 실행해 patch 포인트와 내부 API 호환성을 확인한다.

## 설정

- `config/config.yaml`: 전역 런타임 설정
- `.env`: 텔레그램 토큰 + 시뮬레이션 경로

주요 YAML 섹션: `bot`, `lemonade`, `ollama`, `rag`, `sim_queue`, `dft`, `security`, `memory`, `feedback`, `semantic_cache`, `intent_router`

## 운영 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/run_bot.sh` | 봇 실행/재시작 (`--restart`, `--pull`) |
| `scripts/setup.sh` | 초기 설정 (`.env`, 디렉터리, 모델 점검) |
| `scripts/install_boot_service.sh` | systemd 서비스 설치 (봇 + hosts 갱신) |
| `scripts/update_wsl_hosts.sh` | WSL 게이트웨이 IP → `/etc/hosts` 별칭 갱신 (`HOSTNAME_ALIAS` 지원) |
| `scripts/healthcheck.sh` | 헬스체크 |
| `scripts/soak_monitor.sh` | 장시간 안정성 모니터링 (systemd) |
| `scripts/check_requirements_lock.sh` | 의존성 잠금 파일 검사 |
| `scripts/configure_windows_lemonade.sh` | Windows 방화벽/portproxy 설정 |

## 의존성

```bash
# 갱신
pip-compile --output-file=requirements.lock requirements.txt
# 검증
bash scripts/check_requirements_lock.sh
```

## 품질 검증

```bash
.venv/bin/ruff check .
.venv/bin/mypy core apps main.py
.venv/bin/pytest -q
.venv/bin/pytest -q --cov=core --cov=apps --cov-report=term-missing:skip-covered
```

## 품질 게이트

| 항목 | 로컬 명령 | CI 산출물 |
|---|---|---|
| 의존성 잠금 검증 | `bash scripts/check_requirements_lock.sh` | lock drift 차단 |
| 린트 | `.venv/bin/ruff check .` | `ruff` 결과 |
| 타입 검사 | `.venv/bin/mypy core apps main.py` | `mypy` 결과 |
| 테스트 | `.venv/bin/pytest -q` | 테스트 결과 |
| 커버리지 | `.venv/bin/pytest -q --cov=core --cov=apps --cov-report=term-missing:skip-covered` | 콘솔 coverage summary + `coverage.xml` artifact |

## 라이선스

MIT
