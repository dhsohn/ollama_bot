# ollama_bot

로컬 LLM(Ollama) 기반 24시간 자동화 텔레그램 봇

---

## 주요 기능

- **로컬 LLM 대화** — Ollama를 통한 완전 로컬 AI 채팅 (외부 API 의존 없음)
- **Private Chat 전용** — 텔레그램 1:1 대화만 지원 (그룹/채널 미지원)
- **스킬 시스템** — YAML 기반 확장 가능한 전문 기능 (요약, 번역, 코드 리뷰 등)
- **자동화** — cron 스케줄 기반 자동 작업 (일일 요약, 헬스체크)
- **보안 내재화** — Chat ID 화이트리스트 인증, 입력 검증, 경로 격리
- **Docker 지원** — 컨테이너 격리 실행, 보안 하드닝 적용

---

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- [Ollama](https://ollama.com/) 설치 및 실행
- 텔레그램 봇 토큰 ([BotFather](https://t.me/BotFather)에서 발급)

### 1. 설치

```bash
git clone <repository-url>
cd ollama_bot
pip install -r requirements-dev.txt
```

### 2. 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 필수 값을 입력합니다:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
ALLOWED_TELEGRAM_USERS=your_telegram_chat_id_here
OLLAMA_HOST=http://localhost:11434  # Ollama 서버 주소
OLLAMA_MODEL=gpt-oss:20b           # 사용할 모델
SCHEDULER_TIMEZONE=Asia/Seoul      # 자동화 cron 기준 타임존
```

> `ALLOWED_TELEGRAM_USERS`를 placeholder 그대로 두거나 형식이 잘못되면 봇은 시작되지 않습니다(fail-fast).

> 텔레그램 ID 확인: [@userinfobot](https://t.me/userinfobot)에게 메시지를 보내면 확인할 수 있습니다.

### 3. Ollama 모델 준비

```bash
ollama pull gpt-oss:20b
ollama serve
```

### 4. 실행

```bash
docker compose up --build -d
```

> 보안/운영 정책상 컨테이너 외부에서 `python main.py` 실행은 차단됩니다.

### 5. WSL 부팅 시 자동 실행 (선택)

`scripts/up.sh`는 WSL 환경에서 Windows 게이트웨이 IP를 읽어 `.env`의 `OLLAMA_HOST`를 자동 갱신한 뒤
`docker compose up -d`를 실행합니다.

```bash
sudo bash scripts/install_boot_service.sh
```

상태 확인:

```bash
systemctl status ollama_bot.service
```

수동 실행:

```bash
bash scripts/up.sh
```

#### Windows 재부팅 후 WSL 자동 기동

`ollama_bot.service`를 `systemd`로 자동 시작(`enable`)해도, 먼저 WSL 인스턴스가 떠 있어야 서비스가 실행됩니다.
Windows 로그인 시 WSL을 1회 기동하려면 작업 스케줄러를 등록하세요.

```powershell
schtasks /Create /TN "WSL AutoStart" /SC ONLOGON /RU "$env:USERDOMAIN\$env:USERNAME" /TR "wsl.exe -d Ubuntu-20.04 -u root --exec /usr/bin/true" /F
```

배포판 이름은 `wsl -l -q`로 확인해서 `Ubuntu-20.04` 부분을 본인 환경에 맞게 바꾸면 됩니다.

#### `systemd` 서비스 vs `/etc/wsl.conf`의 `[boot] command`

- `systemd` 서비스: 프로세스 생명주기 관리(자동 재시작, 의존성, 로그, 상태 확인)에 적합
- `[boot] command`: WSL 시작 시 단발성 명령 실행 훅

`ollama_bot`처럼 계속 실행되어야 하는 프로세스는 `systemd` 방식이 권장됩니다.

---

## 텔레그램 명령어

| 명령어 | 설명 |
|--------|------|
| `/start` | 봇 시작 |
| `/help` | 도움말 표시 |
| `/skills` | 스킬 목록/리로드 (`/skills`, `/skills reload`) |
| `/auto` | 자동화 관리 (`/auto list`, `/auto enable <이름>`, `/auto disable <이름>`, `/auto reload`) |
| `/model` | 모델 확인/변경 (`/model list`, `/model <모델명>`) |
| `/memory` | 메모리 관리 (`/memory clear`, `/memory export`) |
| `/feedback` | 피드백 통계 확인 (설정 `feedback.enabled: true`일 때 노출) |
| `/status` | 시스템 상태 확인 |

명령어 없이 자유롭게 메시지를 보내면 일반 대화 모드로 작동합니다.
단, private chat(1:1)에서만 동작합니다.

---

## 내장 스킬

| 스킬 | 트리거 | 설명 |
|------|--------|------|
| summarize | `/summarize`, `요약해줘` | 텍스트 요약 |
| translate | `/translate`, `번역해줘` | 다국어 번역 |
| code_review | `/review`, `코드 리뷰` | 코드 리뷰 및 개선 제안 |

사용 예시:
```
/summarize 여기에 긴 텍스트를 붙여넣으세요...
번역해줘 Hello, how are you?
/review def foo(x): return x+1
```

---

## 커스텀 스킬 추가

`skills/custom/` 디렉토리에 YAML 파일을 생성합니다:

```yaml
# skills/custom/my_skill.yaml
name: "my_skill"
description: "나만의 스킬"
version: "1.0"
triggers:
  - "/myskill"
  - "내 스킬"
system_prompt: |
  당신은 특정 작업을 수행하는 전문가입니다.
allowed_tools: []
timeout: 30
security_level: "safe"
```

파일 변경 후 `/skills reload`로 다시 로드하고 `/skills`로 확인할 수 있습니다.

> 스킬 `name` 또는 `triggers`가 기존 스킬과 중복되면 로드가 실패합니다(오류 처리).

---

## 커스텀 자동화 추가

`auto/custom/` 디렉토리에 YAML 파일을 생성합니다:

```yaml
# auto/custom/my_auto.yaml
name: "my_automation"
description: "매일 아침 인사"
enabled: true
schedule: "0 8 * * *"          # cron 표현식: 매일 08:00
action:
  type: "prompt"               # skill | prompt | callable
  target: "오늘의 동기부여 명언을 알려줘"
output:
  send_to_telegram: true
retry:
  max_attempts: 2
  delay_seconds: 30
timeout: 60
```

파일 변경 후 `/auto reload`로 다시 로드할 수 있습니다.

> 자동화 `name`이 기존 자동화와 중복되면 로드가 실패합니다(오류 처리).

---

## 프로젝트 구조

```
ollama_bot/
├── main.py                  # 진입점
├── config/config.yaml       # 전역 설정
├── core/                    # 핵심 엔진
│   ├── engine.py            # 대화 오케스트레이션
│   ├── ollama_client.py     # Ollama API 클라이언트
│   ├── telegram_handler.py  # 텔레그램 봇 핸들러
│   ├── skill_manager.py     # 스킬 로더/실행기
│   ├── auto_scheduler.py    # 자동화 스케줄러
│   ├── memory.py            # 대화/장기 메모리 (SQLite)
│   ├── security.py          # 보안 모듈
│   └── config.py            # 설정 로더
├── skills/                  # 스킬 디렉토리
│   ├── _builtin/            # 내장 스킬 (YAML)
│   └── custom/              # 사용자 커스텀 스킬
├── auto/                    # 자동화 디렉토리
│   ├── _builtin/            # 내장 자동화 (YAML)
│   └── custom/              # 사용자 커스텀 자동화
├── data/                    # 런타임 데이터
│   ├── conversations/       # 대화 내보내기
│   ├── memory/              # SQLite DB
│   └── logs/                # 로그
├── tests/                   # 테스트
├── Dockerfile
└── docker-compose.yml
```

---

## 설정

### config/config.yaml

주요 설정 항목:

```yaml
bot:
  max_conversation_length: 50    # 대화 컨텍스트 최대 메시지 수
  response_timeout: 60           # LLM 응답 타임아웃 (초)

ollama:
  temperature: 0.7
  max_tokens: 2048

security:
  rate_limit: 30                 # 분당 최대 요청 수

scheduler:
  timezone: "Asia/Seoul"         # 자동화 cron 실행 기준 타임존
```

### 환경변수 (.env)

| 변수 | 설명 | 필수 |
|------|------|------|
| `TELEGRAM_BOT_TOKEN` | 텔레그램 봇 토큰 | O |
| `ALLOWED_TELEGRAM_USERS` | 허용 사용자 ID (쉼표 구분) | O |
| `OLLAMA_HOST` | Ollama 서버 주소 | X |
| `OLLAMA_MODEL` | 기본 모델 | X |
| `SCHEDULER_TIMEZONE` | 자동화 스케줄 타임존 (IANA, 예: `Asia/Seoul`) | X |
| `LOG_LEVEL` | 로그 레벨 (DEBUG/INFO/WARNING) | X |

---

## 테스트

```bash
python3 -m pytest tests/ -v
```

---

## WSL2 참고사항

WSL2 환경에서 Docker를 사용하는 경우, Ollama 서버는 호스트에서 실행하고
`OLLAMA_HOST`를 `http://host.docker.internal:11434`로 설정하세요 (기본값).

---

## 라이선스

MIT
