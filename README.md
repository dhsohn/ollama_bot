# ollama_bot

**Dual-Provider 아키텍처** 기반 텔레그램 private-chat 봇입니다.

- **Lemonade Server (8000)**: LLM 응답 전담 — `gpt-oss-20b-NPU` 단일 모델 상주
- **Ollama Server (11434)**: 쿼리 최적화 전담 — 임베딩(`dengcao/Qwen3-Embedding-0.6B:Q8_0`) + 리랭킹(`dengcao/bge-reranker-v2-m3:latest`)

현재 코드베이스는 **단일 앱 구조**입니다.
- 엔트리포인트: `apps/ollama_bot/main.py` (`main.py`는 단순 래퍼)
- 코어 로직: `core/`
- 설정: `config/config.yaml`
- 실행: `bash scripts/run_bot.sh`

## 핵심 기능

- 텔레그램 1:1(private chat) 기반 대화
- 스킬 시스템(YAML): `skills/_builtin`, `skills/custom`
- 자동화 시스템(YAML + APScheduler): `auto/_builtin`, `auto/custom`
- 피드백 버튼(👍/👎) + 부정 피드백 사유 수집(옵션)
- 계층형 응답 최적화
  - Tier 0: 스킬 트리거
  - Tier 1: 규칙 기반 즉시 응답(InstantResponder)
  - Tier 2: 인텐트 라우팅(IntentRouter)
  - Tier 3: 시맨틱 캐시(SemanticCache)
  - Tier 4: Full LLM — Ollama(임베딩/리랭킹) → Lemonade(gpt-oss-20b-NPU 응답)
- 장기 메모리/대화 보관(SQLite)
- 보안 기본값
  - 화이트리스트 사용자 인증(`ALLOWED_TELEGRAM_USERS`)
  - 레이트리밋/전역 동시성 제한
  - 경로 검증, 입력 정제

## 동작 플로우차트

### 대화 플로우

```mermaid
flowchart TD
    U[Telegram 사용자 입력] --> H[Telegram Handler]
    H --> S[Security 검사/Rate Limit]
    S --> T0{Tier 0 스킬 트리거?}
    T0 -->|Yes| SK[Skill 실행]
    T0 -->|No| T1{Tier 1 Instant Rule?}
    T1 -->|Yes| IR[즉시 응답 반환]
    T1 -->|No| T2{Tier 2 Intent Router}
    T2 --> T3{Tier 3 Semantic Cache Hit?}
    T3 -->|Yes| CACHED[캐시 응답 반환]
    T3 -->|No| T4[Tier 4 Full LLM]
    T4 --> OPT["쿼리 최적화 — Ollama:11434"]
    OPT --> EMB[임베딩 Qwen3-Embedding]
    EMB --> RAG[RAG 벡터 검색]
    RAG --> RR[리랭킹 bge-reranker]
    RR --> CTX[컨텍스트 빌드]
    CTX --> LLM["LLM 응답 — Lemonade:8000\ngpt-oss-20b-NPU"]
    SK --> OUT[응답 전송 + 메모리 저장]
    IR --> OUT
    CACHED --> OUT
    LLM --> OUT
```

### 자동화 플로우

```mermaid
flowchart TD
    A[APScheduler 자동화 트리거] --> AR[auto/*.yaml 로드]
    AR --> ACT{Action Type}
    ACT -->|prompt| P[Engine.process_prompt]
    ACT -->|skill| K[Engine.execute_skill]
    ACT -->|callable| C[등록 callable 실행]
    P --> LLM["Lemonade gpt-oss-20b-NPU"]
    K --> LLM
    C --> LLM
    LLM --> AO[텔레그램 전송/파일 저장]
```

## 사전 요구사항

- Python 3.11+
- 임베딩 런타임은 `fastembed`(ONNX Runtime CPU) 기반
- 텔레그램 봇 토큰 (`@BotFather`)
- Dual-Provider 백엔드 (Windows 호스트에서 실행)
  - **Lemonade Server** (포트 8000) — LLM 응답 (`gpt-oss-20b-NPU`)
  - **Ollama Server** (포트 11434) — 임베딩/리랭킹 (`dengcao/Qwen3-Embedding-0.6B:Q8_0`, `dengcao/bge-reranker-v2-m3:latest`)

## 빠른 시작

### 1) 설치

```bash
git clone <repo-url>
cd ollama_bot
python -m venv .venv
.venv/bin/pip install -r requirements.lock
cp .env.example .env
```

### 2) `.env` 설정

`.env` 기본 예시:

```env
TELEGRAM_BOT_TOKEN=...
ALLOWED_TELEGRAM_USERS=123456789
SIM_INPUT_DIR_ORCA_AUTO=kb/orca_runs
SIM_EXTERNAL_AGENT_TOKEN=<긴_랜덤_토큰>
```

- `ALLOWED_TELEGRAM_USERS`는 **숫자 Chat ID CSV**만 허용됩니다.
- placeholder(`your_telegram_chat_id_here`) 상태면 시작 시 fail-fast로 종료됩니다.
- `/sim submit <tool> <이름>` shorthand를 쓰려면 `SIM_INPUT_DIR_<TOOL>` 또는 `SIM_INPUT_DIR`를 설정하세요.
  - 예: `SIM_INPUT_DIR_ORCA_AUTO=kb/orca_runs`이면 `/sim submit orca_auto mj1` → `kb/orca_runs/mj1`
- `SIM_EXTERNAL_AGENT_TOKEN`은 `sim_host_agent` 인증 토큰입니다. 충분히 긴 랜덤 문자열을 사용하세요.
- 런타임 일반 설정(model/host/log/data_dir 등)은 `config/config.yaml`에서 관리합니다.

### 3) LLM 백엔드 준비 (Dual-Provider)

두 서버 모두 Windows 호스트에서 실행되어야 합니다.

#### Lemonade Server (LLM 응답)

- 포트 8000에서 OpenAI-compatible API 제공
- `gpt-oss-20b-NPU` 모델을 상주 로드
- 시작 시 `prepare_model`로 모델 선로드를 보장

#### Ollama Server (쿼리 최적화)

```bash
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
ollama pull dengcao/bge-reranker-v2-m3:latest
ollama serve
```

- 포트 11434에서 임베딩/리랭킹 전담
- 두 모델을 `keep_alive` 설정으로 상주 유지 권장
- RAG 파이프라인에서 벡터 검색 + 리랭킹에 사용

### 4) 실행

```bash
bash scripts/run_bot.sh
```

### 5) systemd 서비스 등록 (부팅 시 자동 시작)

```bash
bash scripts/install_boot_service.sh
systemctl --user start ollama-bot sim-host-agent
```

로그 확인:

```bash
journalctl --user -u ollama-bot -f
```

상태 확인:

```bash
systemctl --user status ollama-bot sim-host-agent
```

### 시뮬레이션 실행 경로

`/sim submit|list|status|info|cancel`은 모두 `sim_host_agent`를 통해 실행/조회/취소됩니다.

- `ollama_bot`: 큐/텔레그램 인터페이스 담당
- `sim_host_agent`: host에서 실제 시뮬레이션 실행과 제어 담당
- 인증: `SIM_EXTERNAL_AGENT_TOKEN` Bearer 토큰 필수
- 취소 가드: 감지된 시뮬레이션 작업만 종료 허용 (임의 PID 차단)

## 실행 정책/제약

- 시뮬레이션 작업은 **host agent 단일 실행 경로**를 사용합니다. (`ollama_bot` 내부에서 직접 실행하지 않음)
- 실행은 `bash scripts/run_bot.sh`를 사용하세요.
- 텔레그램은 private chat(1:1)만 처리합니다. 그룹/채널 메시지는 거절됩니다.

## CLI 점검

```bash
python -m apps.cli chat
python -m apps.cli dry-run "테스트 질문"
python -m apps.cli test
```

- CLI는 Dual-Provider 설정을 그대로 사용합니다 (Lemonade 응답 + Ollama 임베딩/리랭킹).

## 텔레그램 명령어

| 명령어 | 설명 |
|---|---|
| `/start` | 시작 안내 |
| `/help` | 도움말 |
| `/skills` | 스킬 목록 |
| `/skills reload` | 스킬 strict 리로드 |
| `/auto` 또는 `/auto list` | 자동화 목록 |
| `/auto disable <name>` | 자동화 비활성화 |
| `/auto run <name>` | 자동화 1회 수동 실행 |
| `/auto reload` | 자동화 strict 리로드 |
| `/memory` | 메모리 통계 |
| `/memory clear` | 현재 채팅 대화 기록 삭제 |
| `/memory export` | 현재 채팅 기록 markdown 내보내기 |
| `/status` | 시스템 상태 |
| `/feedback` | 피드백 통계 (`feedback.enabled: true`일 때) |
| `/skip` | 부정 피드백 사유 입력 건너뛰기 (`collect_reason: true`일 때) |

- 일반 대화에 `분석` 단어가 포함되면 `/analyze_all` 없이도 전체 문서 분석(full-scan)이 자동 실행됩니다.

## 내장 스킬 (`skills/_builtin`)

| 이름 | 트리거 |
|---|---|
| `summarize` | `/summarize`, `요약해줘` |
| `translate` | `/translate`, `번역해줘` |
| `code_review` | `/review`, `코드 리뷰` |

## 내장 자동화 (`auto/_builtin`)

기본 타임존은 `scheduler.timezone`(기본 `Asia/Seoul`) 기준입니다.

| 이름 | 스케줄(cron) | 설명 |
|---|---|---|
| `preference_extraction` | `0 0 * * *` | 선호도/고정 정보 추출 |
| `feedback_analysis` | `0 2 * * *` | 피드백 분석 후 가이드라인 갱신 |
| `memory_hygiene` | `30 3 * * *` | 메모리 정리 |
| `memory_consolidation` | `0 4 * * sun` | 메모리 통합 압축 |
| `export_training_data` | `0 3 * * 0` | KTO 학습 데이터 내보내기 |
| `daily_summary` | `0 9 * * *` | 전일 대화 요약 |
| `rag_reindex` | `30 3 * * 0` | 주 1회 RAG 증분 재인덱싱 |
| `error_log_triage` | `0 */6 * * *` | 오류 로그 triage |
| `health_check` | `*/30 * * * *` | 주기 헬스체크 |

## 커스텀 스킬 추가

`skills/custom/my_skill.yaml` 예시:

```yaml
name: "my_skill"
description: "내 스킬"
version: "1.0"
triggers:
  - "/myskill"
  - "내 스킬"
system_prompt: |
  당신은 특정 업무를 수행하는 전문가입니다.
allowed_tools: []
parameters:
  - name: "input_text"
    type: "string"
    required: true
    description: "입력"
timeout: 30
model_role: "default"  # optional: default
temperature: 0.7     # optional
max_tokens: 1024     # optional
streaming: true      # optional
security_level: "safe"
```

주의사항:
- `name` 중복 불가
- 트리거 중복 불가
- `/skills reload`는 strict 모드라 오류가 하나라도 있으면 실패합니다.

## 커스텀 자동화 추가

`auto/custom/my_auto.yaml` 예시:

```yaml
name: "my_auto"
description: "매일 리포트"
enabled: true
schedule: "0 8 * * *"
action:
  type: "prompt"   # skill | prompt | callable | command
  target: "오늘의 핵심 이슈를 요약해줘"
  model_role: "default"
  parameters: {}
output:
  send_to_telegram: true
  save_to_file: "reports/my_auto_{date}.md"
retry:
  max_attempts: 2
  delay_seconds: 30
timeout: 120
```

주의사항:
- `name` 중복 불가
- `schedule`은 유효한 cron이어야 함
- `/auto reload`는 strict 모드
- 자동화 LLM 호출은 기본 모델(`gpt-oss-20b-NPU`)을 사용합니다.
- `command` 액션 타입은 현재 버전(v0.1)에서 보안상 비활성화되어 실제 시스템 명령을 실행하지 않습니다.
- `save_to_file` 경로는 `DATA_DIR` 기준으로 검증됩니다(`{date}` 플레이스홀더 지원).

## 설정 파일

- `config/config.yaml`: 전역 런타임 설정
- `.env`: 텔레그램 시크릿 + 시뮬레이션 환경변수

주요 섹션:
- `bot`, `lemonade`, `telegram`, `security`, `memory`, `scheduler`
- `feedback`, `auto_evaluation`
- `instant_responder`, `semantic_cache`, `intent_router`, `context_compressor`
- `ollama`(retrieval), `rag`

`rag` 다중 코퍼스 디렉토리 예시:

```yaml
rag:
  enabled: true
  startup_index_enabled: false   # 부팅 시 백그라운드 인덱싱 비활성화
  kb_dirs:
    - "kb/orca_runs"
    - "kb/orca_outputs"
  max_file_size_mb: 2
  supported_extensions:
    - ".md"
    - ".json"
    - ".py"
    - ".js"
    - ".ts"
```

Dual-Provider 설정 예시:

```yaml
# LLM 응답 — Lemonade Server
lemonade:
  host: "http://homelab:8000"
  default_model: "gpt-oss-20b-NPU"

# 쿼리 최적화 — Ollama Server (임베딩 + 리랭킹)
ollama:
  host: "http://homelab:11434"
  embedding_model: "dengcao/Qwen3-Embedding-0.6B:Q8_0"
  reranker_model: "dengcao/bge-reranker-v2-m3:latest"
```

`.env` 우선순위 관련:
- `APP_ENV_FILE` 또는 `APP_ENV_FILES`(CSV)로 로드 파일 지정 가능
- 앱 런타임 설정 반영은 기본적으로 텔레그램 시크릿 2개만 지원하며,
  시뮬레이션 shorthand 경로(`SIM_INPUT_DIR`, `SIM_INPUT_DIR_<TOOL>`)는 실행 시 `os.environ`에서 직접 참조합니다.

## 운영 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/configure_windows_lemonade.sh` | Lemonade용 Windows 방화벽/portproxy 자동 설정 |
| `scripts/run_bot.sh` | 봇 실행 |
| `scripts/run_host_agent.sh` | 시뮬레이션 호스트 에이전트 실행 |
| `scripts/install_boot_service.sh` | systemd user service 설치 (부팅 시 자동 시작) |
| `scripts/healthcheck.sh` | 헬스체크 |
| `scripts/soak_monitor.sh` | 장시간 안정성 모니터링 |
| `scripts/check_requirements_lock.sh` | `requirements.lock` 최신성 검사 |

## 의존성 잠금 정책

- `requirements.txt`: 직접 의존성 선언 파일
- `requirements.lock`: 배포/런타임 기준 파일
- `requirements-dev.txt`: 개발 도구 + `requirements.lock` 포함

의존성 변경 시 아래 순서로 갱신합니다:

```bash
pip-compile --output-file=requirements.lock --pip-args='--use-feature=fast-deps' requirements.txt
bash scripts/check_requirements_lock.sh
```

## 품질 검증

```bash
bash scripts/check_requirements_lock.sh
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest -q
```

장시간 모니터링 예시:

```bash
bash scripts/soak_monitor.sh --minutes 180 --max-restarts 0 --max-error-lines 0
```

## 데이터

기본 데이터 경로(`data_dir`)는 프로젝트 루트의 `data/`입니다.
기본 지식베이스 경로는 `kb/orca_runs`, `kb/orca_outputs`입니다.

주요 산출물:
- `data/memory/ollama_bot.db` (대화/장기 메모리)
- `data/memory/feedback.db` (피드백)
- `data/memory/cache.db` (시맨틱 캐시)
- `data/conversations/` (내보낸 대화 markdown)
- `data/reports/` (자동화 리포트)
- `data/logs/` (애플리케이션 로그)

## 프로젝트 구조

```text
ollama_bot/
├── apps/ollama_bot/main.py
├── core/
├── config/
├── skills/
│   ├── _builtin/
│   └── custom/
├── auto/
│   ├── _builtin/
│   └── custom/
├── packages/hw_amd_npu/
├── scripts/
├── tests/
└── .env.example
```

## 라이선스

MIT
