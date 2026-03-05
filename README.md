# ollama_bot

WSL2 homelab용 텔레그램 private-chat 봇. Dual-Provider 아키텍처 기반.

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
- 런타임 설정은 `config/config.yaml` 에서 관리

### LLM 백엔드

Windows 호스트에서 두 서버 실행:

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

## 설정

- `config/config.yaml`: 전역 런타임 설정
- `.env`: 텔레그램 토큰 + 시뮬레이션 경로

주요 YAML 섹션: `bot`, `lemonade`, `ollama`, `rag`, `sim_queue`, `dft`, `security`, `memory`, `feedback`, `semantic_cache`, `intent_router`

## 운영 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/run_bot.sh` | 봇 실행 (포그라운드) |
| `scripts/setup.sh` | 초기 설정 (`.env`, 디렉터리, 모델 점검) |
| `scripts/install_boot_service.sh` | systemd 서비스 설치 (봇 + hosts 갱신) |
| `scripts/update_wsl_hosts.sh` | WSL 게이트웨이 IP → `/etc/hosts` 갱신 |
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
.venv/bin/pytest -q
.venv/bin/ruff check .
```

## 라이선스

MIT
