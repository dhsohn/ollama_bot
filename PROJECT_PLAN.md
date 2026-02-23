# ollama_bot - 프로젝트 계획서

> 로컬 LLM(Ollama) 기반 24시간 자동화 개인 AI 에이전트
> OpenClaw 경량화 벤치마킹 버전 | 보안 강화 | Docker 운영 | 텔레그램 연동

---

## 1. 프로젝트 개요

### 1.1 목적
OpenClaw(구 Clawdbot/Moltbot)의 핵심 아이디어를 벤치마킹하되, **경량화**와 **보안 강화**에 집중한 로컬 LLM 기반 24시간 자동화 봇을 구축한다.

### 1.2 핵심 목표
| 항목 | 설명 |
|------|------|
| 경량화 | OpenClaw의 거대한 스킬 생태계(10,700+) 대신, 핵심 기능만 탑재한 미니멀 아키텍처 |
| 보안 강화 | OpenClaw에서 보고된 CVE들(RCE, SSRF, Path Traversal 등)을 설계 단계에서 차단 |
| 로컬 우선 | Ollama를 통한 완전 로컬 LLM 운영, 외부 API 의존도 제로 |
| Docker 운영 | 컨테이너 격리를 통한 안전한 실행 환경 |
| 텔레그램 연동 | 텔레그램 봇을 주요 사용자 인터페이스로 활용 |
| WSL 호환 | Windows(WSL2) 환경에서 원활한 실행 보장 |
| 사용자 커스텀 | SKILL/AUTO 디렉토리를 통한 확장 가능한 구조 |

### 1.3 OpenClaw과의 비교
| 항목 | OpenClaw | ollama_bot |
|------|----------|------------|
| LLM | 클라우드 API + 로컬(Ollama) | Ollama 전용 (로컬 Only) |
| 메시징 | WhatsApp, Telegram, Discord, Slack, Signal, iMessage | 텔레그램 전용 (경량화) |
| 스킬 수 | 10,700+ (ClawHub) | 핵심 스킬만 내장 + 사용자 커스텀 |
| 설치 | npm/curl 원라이너 | Docker Compose 원커맨드 |
| 보안 | 다수 CVE 보고됨 | 설계 단계부터 보안 내재화 |
| 언어 | TypeScript/Node.js | Python (경량, Ollama 친화) |
| 크기 | 대규모 | 최소 의존성 경량 구조 |

### 1.4 MVP 범위 (v0.1)
| 구분 | 포함 (In Scope) | 제외 (Out of Scope) |
|------|------------------|----------------------|
| 인터페이스 | 텔레그램 1:1 대화, 기본 명령어(`/start`, `/help`, `/status`) | WebUI, 멀티 채널(Discord/Slack) |
| 모델 | Ollama 로컬 모델 단일 기본값(`gpt-oss:20b`) | 다중 모델 자동 라우팅, 클라우드 API |
| 기능 | 대화, 메모리, 스킬, 자동화(AUTO) | 에이전트 간 협업, 장기 워크플로 오케스트레이션 |
| 보안 | 화이트리스트 인증, 입력 검증, 컨테이너 하드닝 | 멀티테넌트 RBAC, SSO |

### 1.5 성공 지표 (초기 운영 기준)
| 항목 | 목표 |
|------|------|
| 응답 지연 시간 | 일반 대화 기준 p95 12초 이하 |
| 안정성 | 24시간 연속 운영 중 비정상 종료 0회 |
| 보안 | 미인증 사용자 요청 100% 차단 |
| 운영성 | 장애 발생 시 30분 이내 원인 식별 가능 (로그 기준) |

---

## 2. 아키텍처 설계

### 2.1 전체 구조
```
┌─────────────────────────────────────────────────┐
│                 Docker Container                 │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ Telegram  │──▶│   Core   │──▶│   Ollama    │  │
│  │ Handler   │◀──│  Engine   │◀──│   Client    │  │
│  └──────────┘   └────┬─────┘   └─────────────┘  │
│                      │                            │
│              ┌───────┴───────┐                    │
│              │               │                    │
│         ┌────▼────┐   ┌─────▼─────┐              │
│         │  SKILL  │   │   AUTO    │              │
│         │ Manager │   │ Scheduler │              │
│         └────┬────┘   └─────┬─────┘              │
│              │               │                    │
│         ┌────▼────┐   ┌─────▼─────┐              │
│         │ skills/ │   │   auto/   │              │
│         │ (YAML)  │   │  (YAML)   │              │
│         └─────────┘   └───────────┘              │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │           Data (Volume Mount)             │    │
│  │  conversations/ | memory/ | logs/         │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
         │                          ▲
         ▼                          │
   ┌───────────┐            ┌──────────────┐
   │ Telegram   │            │ Ollama Server │
   │ Bot API    │            │ (Host/Remote) │
   └───────────┘            └──────────────┘
```

### 2.2 핵심 모듈
| 모듈 | 역할 |
|------|------|
| `core/engine.py` | LLM 대화 관리, 컨텍스트 유지, 라우팅 |
| `core/ollama_client.py` | Ollama API 통신, 모델 관리, 스트리밍 응답 |
| `core/telegram_handler.py` | 텔레그램 봇 메시지 수신/발신 처리 |
| `core/skill_manager.py` | SKILL 디렉토리의 스킬 로드/실행/관리 |
| `core/auto_scheduler.py` | AUTO 디렉토리의 자동화 작업 스케줄링 |
| `core/memory.py` | 대화 기록 및 장기 메모리 관리 |
| `core/security.py` | 입력 검증, 샌드박싱, 권한 관리 |

### 2.3 요청 처리 플로우
1. 텔레그램 메시지 수신 후 Chat ID 인증/권한 검증
2. 입력 sanitization 및 명령어/스킬 트리거 라우팅
3. 메모리에서 최근 대화 컨텍스트 로드
4. `core/ollama_client.py`를 통해 `gpt-oss:20b` 추론 요청
5. 응답 후 메모리/로그 저장 및 텔레그램 전송
6. 실패 시 재시도 정책 적용 후 에러 메시지 반환

---

## 3. 디렉토리 구조

```
ollama_bot/
├── PROJECT_PLAN.md          # 본 계획서
├── docker-compose.yml       # Docker Compose 설정
├── Dockerfile               # 컨테이너 빌드 정의
├── .env.example             # 환경변수 템플릿
├── .dockerignore
├── requirements.txt         # 런타임 의존성
├── requirements-dev.txt     # 테스트/개발 의존성
├── config/
│   └── config.yaml          # 봇 전역 설정
├── core/                    # 핵심 엔진
│   ├── __init__.py
│   ├── engine.py            # 메인 엔진 (대화 라우팅, 컨텍스트)
│   ├── ollama_client.py     # Ollama API 클라이언트
│   ├── telegram_handler.py  # 텔레그램 봇 핸들러
│   ├── skill_manager.py     # 스킬 로더/실행기
│   ├── auto_scheduler.py    # 자동화 스케줄러
│   ├── memory.py            # 대화/장기 메모리
│   └── security.py          # 보안 모듈
├── skills/                  # 사용자 커스텀 스킬 (SKILL 디렉토리)
│   ├── README.md            # 스킬 작성 가이드
│   ├── _builtin/            # 내장 스킬
│   │   ├── summarize.yaml   # 텍스트 요약
│   │   ├── translate.yaml   # 번역
│   │   ├── code_review.yaml # 코드 리뷰
│   │   └── web_search.yaml  # 웹 검색 (선택적)
│   └── custom/              # 사용자 정의 스킬
│       └── .gitkeep
├── auto/                    # 사용자 커스텀 자동화 (AUTO 디렉토리)
│   ├── README.md            # 자동화 작성 가이드
│   ├── _builtin/            # 내장 자동화
│   │   ├── daily_summary.yaml    # 일일 요약
│   │   └── health_check.yaml     # 헬스체크
│   └── custom/              # 사용자 정의 자동화
│       └── .gitkeep
├── data/                    # 런타임 데이터 (볼륨 마운트)
│   ├── conversations/       # 대화 기록
│   ├── memory/              # 장기 메모리
│   └── logs/                # 로그 파일
├── tests/                   # 테스트
│   ├── test_engine.py
│   ├── test_ollama_client.py
│   ├── test_telegram.py
│   ├── test_skill_manager.py
│   └── test_security.py
└── scripts/
    ├── setup.sh             # 초기 설정 스크립트
    └── healthcheck.sh       # Docker 헬스체크
```

---

## 4. 보안 설계 (OpenClaw CVE 대응)

OpenClaw에서 보고된 주요 보안 취약점과 ollama_bot의 대응 방안:

### 4.1 알려진 OpenClaw 보안 이슈 및 대응

| CVE | 취약점 | 심각도 | ollama_bot 대응 방안 |
|-----|--------|--------|---------------------|
| CVE-2026-25253 | 1-click RCE (Gateway URL 조작) | Critical (8.8) | WebUI 없음. 텔레그램만 사용하여 공격 표면 자체를 제거 |
| CVE-2026-24763 | Docker 샌드박스 우회 | High | 최소 권한 컨테이너 (no-new-privileges, read-only rootfs, capabilities drop) |
| CVE-2026-25593 | 권한 상승 | High | 단일 사용자 모드, 권한 계층 불필요 |
| CVE-2026-25475 | 인증 우회 | High | 텔레그램 Chat ID 기반 화이트리스트 인증 |
| CVE-2026-26322 | SSRF | High (7.6) | 스킬 실행 시 URL 화이트리스트 + 내부 IP 대역 차단 |
| CVE-2026-26319 | 웹훅 인증 미비 | High (7.5) | 외부 웹훅 미사용, 텔레그램 Polling 방식 채택 |
| CVE-2026-26329 | Path Traversal | Medium | 파일 접근 경로를 data/ 디렉토리로 한정 (chroot-like) |
| ClawHavoc | 악성 스킬 공급망 공격 | Critical | 외부 스킬 마켓플레이스 미연동, 로컬 스킬만 허용 |

### 4.2 보안 원칙
1. **최소 공격 표면**: WebUI 없음, 텔레그램 단일 채널
2. **최소 권한 원칙**: Docker 컨테이너 내 non-root 실행, capabilities 최소화
3. **입력 검증**: 모든 사용자 입력에 대한 sanitization
4. **경로 격리**: 파일 시스템 접근을 지정된 디렉토리로 제한
5. **비밀 관리**: 환경변수 또는 Docker Secrets로 민감 정보 관리
6. **로깅/감사**: 모든 스킬 실행과 자동화 작업에 대한 감사 로그
7. **네트워크 격리**: 불필요한 아웃바운드 트래픽 차단

### 4.3 Docker 보안 설정
```yaml
# docker-compose.yml 보안 관련 발췌
services:
  ollama_bot:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # 네트워크 통신에 필요한 최소 권한만
    user: "botuser"       # non-root 실행
```

### 4.4 추가 하드닝 체크리스트 (권장)
- 이미지 고정 버전 사용 (`python:3.11-slim` digest pinning)
- `pip install --require-hashes` 또는 lock 파일 기반 의존성 고정
- 컨테이너 아웃바운드 도메인 allowlist 적용 (필요 최소 범위)
- 주기적 비밀값 회전 (Telegram Bot Token, 사용자 키)
- 감사 로그에 스킬/자동화 실행자, 입력 요약, 결과 상태 기록

---

## 5. SKILL 시스템 설계

### 5.1 스킬 정의 형식 (YAML)
OpenClaw의 스킬 시스템을 벤치마킹하되, 단순화된 YAML 형식을 사용한다.

```yaml
# skills/custom/my_skill.yaml
name: "my_custom_skill"
description: "스킬에 대한 설명"
version: "1.0"

# 트리거: 사용자가 이 키워드를 사용하면 스킬 활성화
triggers:
  - "/myskill"
  - "커스텀작업"

# LLM에게 전달할 시스템 프롬프트
system_prompt: |
  당신은 특정 작업을 수행하는 전문가입니다.
  다음 규칙을 따르세요:
  1. ...
  2. ...

# 스킬이 사용할 수 있는 도구 (허용 목록)
allowed_tools:
  - file_read
  - file_write

# 매개변수 정의
parameters:
  - name: "input_text"
    type: "string"
    required: true
    description: "처리할 텍스트"

# 타임아웃 (초)
timeout: 30

# 보안 등급: safe | cautious | restricted
security_level: "safe"
```

### 5.2 내장 스킬
| 스킬 | 설명 | 트리거 |
|------|------|--------|
| summarize | 텍스트/URL 요약 | `/summarize`, `요약해줘` |
| translate | 다국어 번역 | `/translate`, `번역해줘` |
| code_review | 코드 리뷰 및 개선 제안 | `/review`, `코드 리뷰` |
| web_search | 웹 검색 (선택적, 외부 API 필요) | `/search`, `검색해줘` |

### 5.3 스킬 보안
- 각 스킬에 `security_level` 지정
- `allowed_tools` 화이트리스트로 스킬별 도구 접근 제한
- 외부 스킬 마켓플레이스 연동 없음 (ClawHavoc 대응)
- 스킬 파일 무결성 해시 검증

---

## 6. AUTO(자동화) 시스템 설계

### 6.1 자동화 정의 형식 (YAML)
OpenClaw의 Heartbeat/Cron 시스템을 벤치마킹한 경량 스케줄러.

```yaml
# auto/custom/my_automation.yaml
name: "daily_report"
description: "매일 아침 9시에 일일 요약 보고서 생성"
version: "1.0"
enabled: true

# 스케줄 (cron 표현식)
schedule: "0 9 * * *"

# 실행할 작업
action:
  type: "skill"           # skill | command | prompt
  target: "summarize"     # 실행할 스킬 이름
  parameters:
    input_text: "어제의 대화 내역을 요약해줘"

# 결과 전달
output:
  send_to_telegram: true
  save_to_file: "data/reports/daily_{date}.md"

# 실패 시 재시도
retry:
  max_attempts: 3
  delay_seconds: 60

# 타임아웃 (초)
timeout: 120
```

### 6.2 내장 자동화
| 자동화 | 설명 | 기본 스케줄 |
|--------|------|------------|
| daily_summary | 일일 대화 요약 | 매일 09:00 |
| health_check | 시스템 상태 점검 | 매 30분 |

### 6.3 자동화 타입
| 타입 | 설명 |
|------|------|
| `skill` | 등록된 스킬을 실행 |
| `command` | 허용된 시스템 명령 실행 (샌드박스 내) |
| `prompt` | LLM에 직접 프롬프트 전송 |

---

## 7. 기술 스택

| 계층 | 기술 | 선택 이유 |
|------|------|----------|
| 언어 | Python 3.11+ | Ollama Python SDK 친화, 경량, 빠른 개발 |
| LLM | Ollama | 완전 로컬, 무료, 다양한 모델 지원 |
| 봇 프레임워크 | python-telegram-bot | 안정적, 비동기 지원, 풍부한 문서 |
| 스케줄러 | APScheduler | 경량 cron 스케줄러, Python 네이티브 |
| 설정 | PyYAML + Pydantic | YAML 설정 + 타입 검증 |
| 컨테이너 | Docker + Docker Compose | 격리 실행, 쉬운 배포 |
| 데이터 | SQLite + Markdown 파일 | 경량 영구 저장소 |
| 로깅 | Python logging (structlog) | 구조화된 로그 |
| 테스트 | pytest + pytest-asyncio | 비동기 테스트 지원 |

### 7.1 모델 운용 정책
- 기본 모델: `gpt-oss:20b`
- 모델 변경은 `/model` 명령으로 허용하되, 허용 목록 기반으로 제한
- 서버 시작 시 기본 모델 사전 로드 여부 점검 (`ollama list` 검증)

---

## 8. Docker 구성

### 8.1 docker-compose.yml 구성
```yaml
services:
  ollama_bot:
    build: .
    container_name: ollama_bot
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data            # 런타임 데이터 영구 보존
      - ./skills/custom:/app/skills/custom    # 사용자 스킬 핫 리로드
      - ./auto/custom:/app/auto/custom        # 사용자 자동화 핫 리로드
      - ./config:/app/config         # 설정 파일
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
    user: "botuser"
    healthcheck:
      test: ["CMD", "bash", "scripts/healthcheck.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - bot_network
    # WSL2 환경에서 Ollama 호스트 접근을 위한 설정
    extra_hosts:
      - "host.docker.internal:host-gateway"

networks:
  bot_network:
    driver: bridge
```

### 8.2 Dockerfile 구성 방향
- Python 3.11-slim 기반 (경량 이미지)
- 멀티 스테이지 빌드 (빌드 의존성 분리)
- non-root 사용자 실행
- 최소 패키지만 설치

### 8.3 WSL2 호환성
- `extra_hosts`로 호스트의 Ollama 서버 접근
- Volume mount 경로 Windows/Linux 호환
- Docker Desktop for Windows 또는 WSL2 내부 Docker 엔진 모두 지원

### 8.4 실행 전 모델 준비 체크
```bash
# 기본 모델 다운로드
ollama pull gpt-oss:20b

# 모델 보유 여부 확인
ollama list | rg gpt-oss:20b
```

---

## 9. 텔레그램 봇 인터페이스

### 9.1 명령어 체계
| 명령어 | 설명 |
|--------|------|
| `/start` | 봇 시작 및 인증 |
| `/help` | 도움말 표시 |
| `/skills` | 스킬 목록 확인 및 재로드 (`/skills reload`) |
| `/auto` | 자동화 작업 목록/관리/재로드 (`/auto reload`) |
| `/model` | 현재 Ollama 모델 확인/변경 |
| `/memory` | 메모리 관리 (조회/초기화) |
| `/status` | 시스템 상태 확인 |

### 9.2 대화 모드
- **일반 대화**: 자연어로 LLM과 자유롭게 대화
- **스킬 모드**: 특정 스킬을 트리거하여 전문 작업 수행
- **관리 모드**: 봇 설정, 자동화 관리 등 시스템 제어

### 9.3 인증
- 텔레그램 Chat ID 기반 화이트리스트
- `.env` 파일에 허용된 사용자 ID 목록 관리
- 미인증 사용자의 접근 완전 차단

---

## 10. 개발 로드맵

### Phase 1: 기반 구축 (핵심)
- [ ] 프로젝트 기본 구조 및 Docker 환경 설정
- [ ] Ollama 클라이언트 구현 (API 연동, 모델 관리)
- [ ] 텔레그램 봇 핸들러 구현 (메시지 수신/발신)
- [ ] Core Engine 구현 (대화 라우팅, 컨텍스트 관리)
- [ ] 기본 보안 모듈 구현 (인증, 입력 검증)

### Phase 2: 스킬 & 메모리
- [ ] SKILL 시스템 구현 (YAML 로더, 실행기)
- [ ] 내장 스킬 구현 (summarize, translate, code_review)
- [ ] 대화 메모리 및 장기 메모리 시스템
- [ ] 스킬 보안 (화이트리스트, 타임아웃, 샌드박싱)

### Phase 3: 자동화
- [ ] AUTO 스케줄러 구현 (APScheduler 기반)
- [ ] 내장 자동화 구현 (daily_summary, health_check)
- [ ] 자동화 결과 텔레그램 전송
- [ ] 자동화 관리 인터페이스 (/auto 명령어)

### Phase 4: 안정화 & 배포
- [ ] 통합 테스트 작성
- [ ] 로깅/모니터링 시스템 완성
- [ ] WSL2 환경 테스트 및 호환성 검증
- [ ] 사용자 문서 작성 (README, 스킬/자동화 가이드)
- [ ] Docker Hub 이미지 배포 (선택)

### 10.5 Phase 완료 기준 (Definition of Done)
- Phase 1 완료 기준: 텔레그램 메시지 수신부터 LLM 응답까지 E2E 동작
- Phase 2 완료 기준: 스킬 3종(summarize/translate/code_review) 정상 실행 + 테스트 통과
- Phase 3 완료 기준: 자동화 2종(daily_summary/health_check) 스케줄 기반 실행 확인
- Phase 4 완료 기준: 보안/회귀 테스트 통과 및 24시간 안정성 테스트 완료

---

## 11. 설정 파일 예시

### 11.1 config.yaml
```yaml
# ollama_bot 전역 설정
bot:
  name: "ollama_bot"
  language: "ko"                    # 기본 언어
  max_conversation_length: 50       # 최대 대화 턴 수
  response_timeout: 60              # LLM 응답 타임아웃 (초)

ollama:
  host: "http://host.docker.internal:11434"  # Ollama 서버 주소
  model: "gpt-oss:20b"              # 기본 모델
  temperature: 0.7
  max_tokens: 2048
  system_prompt: |
    당신은 유용한 AI 어시스턴트입니다.
    한국어로 답변하며, 간결하고 정확한 정보를 제공합니다.

telegram:
  polling_interval: 1               # 폴링 간격 (초)
  max_message_length: 4096          # 최대 메시지 길이

security:
  allowed_users: []                 # .env에서 관리
  rate_limit: 30                    # 분당 최대 요청 수
  max_file_size: 10485760           # 최대 파일 크기 (10MB)
  blocked_paths:                    # 접근 금지 경로 패턴
    - "/etc/*"
    - "/proc/*"
    - "/sys/*"

memory:
  backend: "sqlite"                 # sqlite | markdown
  max_long_term_entries: 1000       # 장기 메모리 최대 항목
  conversation_retention_days: 30   # 대화 보관 기간
```

### 11.2 .env.example
```env
# 텔레그램 봇 토큰 (BotFather에서 발급)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# 허용된 텔레그램 사용자 ID (쉼표 구분)
ALLOWED_TELEGRAM_USERS=your_telegram_chat_id_here

# Ollama 서버 주소 (WSL2에서는 host.docker.internal 사용)
OLLAMA_HOST=http://host.docker.internal:11434

# 기본 모델
OLLAMA_MODEL=gpt-oss:20b

# 로그 레벨
LOG_LEVEL=INFO

# 데이터 디렉토리
DATA_DIR=/app/data
```

---

## 12. 참고 자료

- [OpenClaw 공식 사이트](https://openclaw.ai/)
- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)
- [OpenClaw 보안 이슈 (Cisco)](https://blogs.cisco.com/ai/personal-ai-agents-like-openclaw-are-a-security-nightmare)
- [OpenClaw CVE-2026-25253 (1-Click RCE)](https://thehackernews.com/2026/02/openclaw-bug-enables-one-click-remote.html)
- [OpenClaw 보안 가이드 (Adversa AI)](https://adversa.ai/blog/openclaw-security-101-vulnerabilities-hardening-2026/)
- [Ollama 공식 사이트](https://ollama.com/)
- [python-telegram-bot 문서](https://python-telegram-bot.readthedocs.io/)

---

*작성일: 2026-02-23*
*프로젝트: ollama_bot v0.1*
