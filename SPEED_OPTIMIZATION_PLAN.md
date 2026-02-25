# ollama_bot - 응답 속도 최적화 계획서

> 모델 변경 없이 `gpt-oss:20b`의 체감 및 실제 응답 속도를 극대화하는 구현 계획
> 계층형 사전 처리 로직 + Ollama 런타임 튜닝 + 지능형 캐싱

---

## 1. 개요

### 1.1 현재 문제

현재 ollama_bot의 모든 메시지는 **동일한 경로**를 따른다:

```
사용자 메시지 → 보안 검증 → 히스토리 50개 로드 → DICL 주입 → gpt-oss:20b 호출 → 응답
```

"안녕"이라는 단순 인사도, 복잡한 코드 리뷰 요청도 동일하게 20B 파라미터 모델을 풀 컨텍스트로 호출한다.
이로 인해 단순한 질문에도 수 초~십수 초의 지연이 발생하고 있다.

### 1.2 최적화 목표

| 지표 | 현재 | 목표 |
|------|------|------|
| 단순 응답 (인사, 감사) | 5-15초 | < 0.1초 (LLM 우회) |
| 반복 질문 (캐시 히트) | 5-15초 | < 0.5초 |
| 일반 대화 (LLM 경유) | 5-15초 | 3-8초 (컨텍스트 축소) |
| 첫 토큰 표시 (체감) | 5-15초 | < 2초 (스트리밍 최적화) |

### 1.3 핵심 원칙

1. **모델은 바꾸지 않는다** — `gpt-oss:20b` 유지
2. **기존 기능 훼손 없음** — 스킬, 자동화, 피드백 시스템 모두 정상 동작
3. **점진적 적용** — 각 Phase를 독립적으로 배포/롤백 가능
4. **측정 가능** — 최적화 전후 latency를 구조화 로그로 비교

### 1.4 최적화 전략 개요 (계층형 아키텍처)

```
사용자 메시지
    │
    ├─ [선행] Skill Trigger 매칭
    │      ├─ 매칭 성공 → 스킬 전용 LLM 경로
    │      └─ 매칭 실패 → 아래 Tier 진행
    │
    ▼
┌─────────────────────────────────────────┐
│ Tier 1: 규칙 기반 즉시 응답              │  ← 0ms, 정규식/키워드 매칭
│ "안녕", "고마워", 시간 질문 등           │
└────────────────┬────────────────────────┘
                 │ (매칭 실패)
                 ▼
┌─────────────────────────────────────────┐
│ Tier 2: 인텐트 라우터                    │  ← <1ms, 의도 분류
│ intent 분류 + 컨텍스트/캐시 전략 결정    │
└────────────────┬────────────────────────┘
                 │ (분류 완료)
                 ▼
┌─────────────────────────────────────────┐
│ Tier 3: 시맨틱 캐시                      │  ← 5-20ms, 임베딩 유사도
│ intent+prompt+model 기준 캐시 조회       │
└────────────────┬────────────────────────┘
                 │ (캐시 미스)
                 ▼
┌─────────────────────────────────────────┐
│ Tier 4: Full LLM (최적화된 컨텍스트)     │  ← 3-8초
│ 압축된 히스토리 + 최적화된 Ollama 설정    │
└─────────────────────────────────────────┘
```

### 1.5 선행 조건 및 현실 제약 (반드시 반영)

- 현재 `docker-compose.yml`에는 Ollama 서비스가 없고, 봇 컨테이너만 실행한다. 따라서 Ollama 성능 튜닝 환경변수는 **호스트 Ollama 프로세스(systemd/shell)**에 설정해야 한다.
- 의존성 설치는 `requirements.txt`가 아니라 `requirements.lock`을 사용한다. 패키지 추가 시 `requirements.lock` 재생성이 반드시 필요하다.
- 현재 메모리 저장 정책은 대화를 최대 50개로 즉시 트리밍한다. Phase 5 요약을 위해서는 요약 원천 데이터(archive) 저장 구조를 먼저 도입해야 한다.
- 토큰/usage 로깅은 현재 `OllamaClient.chat()` 반환 타입(`str`)만으로는 불가능하다. 메타데이터 전달 구조를 먼저 추가해야 한다.

---

## 1.6 Phase 0: 계측 기준선 수집 (모든 최적화의 전제)

> 최적화 효과를 정량적으로 입증하려면, **변경 전 기준선(baseline)**이 반드시 필요하다.
> Phase 0을 건너뛰면 이후 모든 Phase의 "개선 효과"가 추정치에 불과해진다.

### Phase 0 작업 내용

#### 0.1 OllamaClient 응답 메타데이터 전달 구조 추가

현재 `chat()`과 `chat_stream()`은 `str`만 반환하므로 토큰 사용량을 측정할 수 없다.
이를 선행 리팩터링하여 usage 정보를 함께 전달해야 한다.

```python
# 신규 응답 객체
@dataclass
class ChatResponse:
    """LLM 응답 + 메타데이터."""
    content: str
    usage: ChatUsage | None = None  # prompt_eval_count, eval_count, eval_duration 등

@dataclass
class ChatUsage:
    prompt_eval_count: int = 0   # 입력 토큰 수
    eval_count: int = 0          # 출력 토큰 수
    eval_duration: int = 0       # 추론 소요 시간 (ns)
    total_duration: int = 0      # 전체 소요 시간 (ns)
```

**변경 영향**: `chat()` 반환 타입이 `str → ChatResponse`로 변경되므로,
모든 호출부(engine.py, auto_evaluator.py 등)에서 `.content` 접근으로 수정 필요.
하위 호환을 위해 `ChatResponse.__str__`를 구현하되, 명시적 마이그레이션 권장.

#### 0.1.1 스트리밍 메타데이터 계약 추가

`process_message_stream` 경로에서도 최종 메타데이터를 전달하도록 타입 계약을 명시한다.
스트리밍 본문 청크는 기존과 동일하게 유지하고, 종료 시점에 별도 메타를 반환한다.

```python
@dataclass
class RequestMeta:
    tier: str                        # skill|instant|intent|cache|full
    intent: str | None = None
    cache_hit: bool = False
    cache_id: int | None = None
    usage: ChatUsage | None = None

@dataclass
class StreamResult:
    full_response: str
    last_message: Message | None
    tier: str
    intent: str | None = None
    cache_id: int | None = None
    usage: ChatUsage | None = None
```

`result.cache_id`를 피드백 연동에서 사용할 수 있도록, 엔진/렌더러/텔레그램 핸들러 간
메타 계약을 동일하게 유지해야 한다.

#### 0.2 Latency 계측 로그 추가

```python
# core/engine.py — process_message / process_message_stream 시작부
import time
t0 = time.monotonic()
# ... 처리 ...
elapsed_ms = (time.monotonic() - t0) * 1000
self._logger.info(
    "request_baseline",
    chat_id=chat_id,
    latency_ms=round(elapsed_ms, 1),
    history_count=len(history),
    tokens_input=response.usage.prompt_eval_count if response.usage else None,
    tokens_output=response.usage.eval_count if response.usage else None,
)
```

#### 0.3 기준선 데이터 수집 (1-3일 운영)

| 수집 항목 | 목적 |
|----------|------|
| 요청당 latency (ms) | p50/p90/p95 기준값 |
| 입력/출력 토큰 수 | 컨텍스트 축소 효과 측정 기준 |
| 히스토리 메시지 수 | 압축 대상 규모 파악 |
| 요청 유형 분포 | 인사/잡담/코드/복잡 비율 파악 → 인텐트 라우트 설계 근거 |

#### 0.4 Phase 0 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `core/ollama_client.py` | 수정 | `ChatResponse`/`ChatUsage` 도입, `chat()` 반환 타입 변경 |
| `core/engine.py` | 수정 | `ChatResponse.content` 접근, latency 로그 추가 |
| `core/auto_evaluator.py` | 수정 | `ChatResponse.content` 접근 마이그레이션 |
| `core/automation_callables_impl/` | 수정 | LLM 호출부 `.content` 접근 마이그레이션 |

#### 0.5 Phase 0 완료 기준

- [ ] `chat()` / `chat_stream()`이 usage 메타데이터를 전달
- [ ] 기존 모든 호출부가 새 반환 타입에 대응
- [ ] latency/토큰 로그가 구조화 로그에 기록됨
- [ ] 1일 이상 기준선 데이터 수집 완료
- [ ] 기존 테스트 통과

---

## 2. Phase 1: Ollama 런타임 튜닝 (즉시 적용 가능)

코드 변경을 최소화하면서 **가장 높은 영향/노력 비율**을 가진 최적화.

### 2.1 환경변수 기반 Ollama 서버 설정

현재 구조에서는 Ollama 서버가 봇 컨테이너 외부(호스트/원격)에서 실행되므로,
아래 설정은 **Ollama 서버 프로세스 환경**에 적용해야 한다.

**원칙**: `keep_alive`는 서버 환경변수(`OLLAMA_KEEP_ALIVE`)를 단일 소스로 사용한다.
클라이언트 요청 단위 `keep_alive` 파라미터는 사용하지 않는다.

```bash
# Ollama 서버 환경변수
OLLAMA_KEEP_ALIVE=-1              # 모델을 메모리에 상주 (콜드스타트 5-30초 제거)
OLLAMA_FLASH_ATTENTION=1          # Flash Attention 활성화 (Ampere+ GPU, 메모리 ~30% 절감)
OLLAMA_KV_CACHE_TYPE=q8_0         # KV 캐시 8비트 양자화 (메모리 절반 절감)
OLLAMA_NUM_PARALLEL=4             # 봇의 max_concurrent_requests와 정렬
OLLAMA_MAX_LOADED_MODELS=1        # 단일 모델 운영이므로 1로 고정
OLLAMA_MAX_QUEUE=512              # 대기열 크기
```

#### 적용 방법 A (권장): systemd override

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_MAX_QUEUE=512"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

#### 적용 방법 B: 수동 실행 셸

```bash
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=512
ollama serve
```

#### 검증 명령

```bash
systemctl show ollama --property=Environment --no-pager
ollama ps
```

#### 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `scripts/setup.sh` | Ollama systemd/shell 환경변수 설정 가이드 추가 |
| `README.md` | 호스트 Ollama 튜닝 섹션 추가 |
| `.env.example` | 참고 주석만 추가 (`OLLAMA_*` 서버 변수는 `.env` 대상이 아님) |

### 2.2 API 호출 시 `num_ctx` 명시 (`keep_alive`는 서버 정책 단일화)

**변경 파일: `core/ollama_client.py`**

`chat()` 및 `chat_stream()` 메서드에서 `num_ctx`를 명시적으로 전달한다.
`keep_alive`는 서버 환경변수 정책으로만 운영한다.

```python
# chat() 메서드 내 options 딕셔너리 확장
options = {
    "temperature": self._temperature if temperature is None else temperature,
    "num_predict": self._max_tokens if max_tokens is None else max_tokens,
    "num_ctx": self._num_ctx,           # 신규: 컨텍스트 윈도우 크기 제한
}
```

**변경 파일: `core/config.py`** — `OllamaConfig`에 필드 추가

```python
class OllamaConfig(BaseModel):
    # ... 기존 필드 ...
    num_ctx: int = 8192               # 기본 운영값 (Phase 5 완료 전)
    prompt_version: str = "v1"        # 캐시 버전 분리 키
```

**변경 파일: `config/config.yaml`**

```yaml
ollama:
  # ... 기존 설정 ...
  num_ctx: 8192
  prompt_version: "v1"
```

### 2.2.1 `num_ctx` 트레이드오프

기본 운영값을 `num_ctx: 8192`로 시작하면 긴 대화에서의 컨텍스트 절단 위험을 줄일 수 있다.
Phase 5(컨텍스트 압축) 완료 이후에는 `4096`으로 축소하여 VRAM/지연 최적화를 진행한다.

- Phase 5 적용 전: 50개 히스토리 × 평균 100토큰 = ~5000토큰 → 4096은 절단 위험 높음
- Phase 5 적용 후: 10개 원본 + 요약 = ~800-1500토큰 → 4096 운용 가능

따라서 기본값은 `8192`로 유지하고, **Phase 5 완료 시점에 4096으로 단계 전환**한다.

### 2.3 Phase 1 완료 기준

- [ ] Ollama 서버 환경변수 문서화 완료
- [ ] host/systemd에 적용된 환경변수 실측 확인 (`systemctl show`)
- [ ] `ollama_client.py`에서 `num_ctx` 전달
- [ ] `keep_alive`가 서버 환경변수 단일 정책으로 운영됨
- [ ] `config.yaml`에 새 설정값 반영
- [ ] 기존 테스트 통과 확인

---

## 3. Phase 2: 규칙 기반 즉시 응답 레이어 (Tier 1)

LLM을 호출하지 않고 **정규식/키워드 매칭**으로 즉시 응답하는 계층.

### 3.1 아키텍처

```
engine.py: process_message()
    │
    ├─ [선행] skill_manager.match_trigger()  ← 항상 최우선
    │       ├─ 매칭 성공 → 스킬 전용 LLM 경로
    │       └─ 매칭 실패 → Tier 1 진행
    │
    ├─ [Tier 1] instant_responder.match()
    │       ├─ 매칭 성공 → 즉시 응답 반환 (LLM 우회)
    │       └─ 매칭 실패 → Tier 2 진행
    │
    ├─ [Tier 2] intent_router.classify()      (Phase 4 연동)
    ├─ [Tier 3] semantic_cache.get()           (Phase 3 연동)
    └─ [Tier 4] _build_context() → ollama.chat()
```

### 3.2 신규 모듈: `core/instant_responder.py`

YAML 기반으로 규칙을 정의하여 확장성을 확보한다. 하드코딩된 응답이 아닌 설정 파일 기반.

```python
"""규칙 기반 즉시 응답 엔진.

정규식/키워드 매칭으로 LLM을 우회하여 즉시 응답을 반환한다.
규칙은 YAML 파일로 정의하며, 런타임에 리로드 가능하다.
"""

@dataclass(frozen=True)
class InstantMatch:
    """즉시 응답 매칭 결과."""
    response: str
    rule_name: str

class InstantResponder:
    """정규식/키워드 매칭 기반 즉시 응답기."""

    def __init__(self, rules_path: str = "config/instant_rules.yaml"):
        self._rules: list[InstantRule] = []
        self._rules_path = rules_path

    def match(self, text: str) -> InstantMatch | None:
        """입력 텍스트에 매칭되는 즉시 응답을 반환한다.

        Returns:
            InstantMatch 또는 None (매칭 실패).
        """
        ...

    async def reload_rules(self) -> int:
        """규칙 파일을 다시 로드한다."""
        ...
```

### 3.3 규칙 정의 파일: `config/instant_rules.yaml`

```yaml
# 즉시 응답 규칙 정의
# 우선순위: 파일 내 순서대로 평가, 첫 매칭에서 중단

rules:
  # ── 인사 ──
  - name: "greeting"
    patterns:
      - "^(안녕|하이|헬로|hi|hello|hey)\\b"
      - "^좋은\\s*(아침|저녁|오후)"
    responses:
      - "안녕하세요! 무엇을 도와드릴까요?"
      - "반갑습니다! 어떤 도움이 필요하세요?"
    case_insensitive: true

  # ── 감사 ──
  - name: "thanks"
    patterns:
      - "^(고마워|감사|땡큐|thanks|thank you)"
    responses:
      - "천만에요!"
      - "도움이 되었다니 기쁩니다!"
    case_insensitive: true

  # ── 작별 ──
  - name: "farewell"
    patterns:
      - "^(잘가|바이|안녕히|bye|goodbye)"
    responses:
      - "안녕히 가세요! 필요하면 언제든 불러주세요."
    case_insensitive: true

  # ── 동적 응답 (callable) ──
  - name: "current_time"
    patterns:
      - "지금\\s*몇\\s*시"
      - "현재\\s*시간"
    type: "callable"
    callable: "get_current_time"

  - name: "current_date"
    patterns:
      - "오늘\\s*날짜"
      - "오늘\\s*며칠"
    type: "callable"
    callable: "get_current_date"
```

### 3.3.1 동적 응답 callable 등록 메커니즘

`callable: "get_current_time"` 등의 동적 응답은 `InstantResponder` 내부에 등록된
함수 맵에서 해결한다. 기존 `automation_callables` 시스템과는 별도로 관리한다
(목적이 다름: 자동화는 스케줄 기반, 즉시 응답은 사용자 입력 기반).

```python
# core/instant_responder.py
class InstantResponder:
    _BUILTIN_CALLABLES: dict[str, Callable[[], str]] = {
        "get_current_time": lambda: datetime.now().strftime("%H시 %M분입니다."),
        "get_current_date": lambda: datetime.now().strftime("%Y년 %m월 %d일입니다."),
    }

    def _resolve_callable(self, name: str) -> str:
        fn = self._BUILTIN_CALLABLES.get(name)
        if fn is None:
            raise ValueError(f"Unknown callable: {name}")
        return fn()
```

### 3.4 Engine 통합

**변경 파일: `core/engine.py`**

```python
async def process_message(self, chat_id, text, model_override=None):
    # [선행] 스킬 트리거는 모든 Tier보다 우선
    skill = self._skills.match_trigger(text)
    if skill is not None:
        prepared = await self._prepare_skill_request(chat_id, text, skill=skill)
        response = await self._ollama.chat(...)
        ...
        return response

    # [Tier 1] 즉시 응답 확인
    instant = self._instant_responder.match(text)
    if instant is not None:
        await self._persist_turn(chat_id, text, instant.response)
        self._logger.info("instant_response", chat_id=chat_id, rule=instant.rule_name)
        return instant.response

    # [Tier 2~4] 인텐트 → 캐시 → LLM
    prepared = await self._prepare_request(chat_id, text, stream=False)
    ...
```

### 3.5 Phase 2 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `core/instant_responder.py` | 신규 | 규칙 기반 즉시 응답 엔진 |
| `config/instant_rules.yaml` | 신규 | 즉시 응답 규칙 정의 |
| `core/engine.py` | 수정 | `__init__`에 `InstantResponder` 주입, `process_message`/`process_message_stream`에 Tier 1 체크 추가 |
| `core/config.py` | 수정 | `InstantResponderConfig` 추가 (활성화 여부, 규칙 파일 경로) |
| `config/config.yaml` | 수정 | `instant_responder` 섹션 추가 |
| `main.py` | 수정 | `InstantResponder` 초기화 및 Engine에 주입 |
| `tests/test_instant_responder.py` | 신규 | 단위 테스트 |

### 3.6 Phase 2 완료 기준

- [ ] 인사/감사/작별 패턴이 LLM 없이 즉시 응답됨
- [ ] 동적 응답 (시간, 날짜) callable 동작 확인
- [ ] 규칙 미매칭 시 기존 LLM 경로 정상 동작
- [ ] 규칙 런타임 리로드 가능
- [ ] 단위 테스트 통과

---

## 4. Phase 3: 시맨틱 캐싱 (Tier 3)

사용자의 질문을 임베딩 벡터로 변환하고, 과거에 유사한 질문이 있었으면 **LLM 호출 없이 캐시된 응답을 반환**한다.

### 4.1 아키텍처

```
engine.py: process_message()
    │
    ├─ [선행] skill_manager.match_trigger()
    ├─ [Tier 1] instant_responder.match()
    │
    ├─ [Tier 2] intent_router.classify(text)     ← <1ms
    │       └─ route.intent / context_strategy 결정
    │
    ├─ [Tier 3, 신규] semantic_cache.get(text, context)  ← 5-20ms
    │       ├─ 유사 질문 발견 (similarity ≥ threshold)
    │       │   → 캐시된 응답 반환
    │       └─ 캐시 미스
    │           → LLM 호출 후 semantic_cache.put(text, response)
    │
    └─ [Tier 4] _build_context() → ollama.chat()
```

### 4.2 임베딩 전략

경량 임베딩 모델을 사용하여 추가 GPU 부담을 최소화한다.

**방안 A: Ollama 자체 임베딩 (추가 의존성 없음)**

```python
# Ollama의 embed API 활용
response = await client.embed(model="gpt-oss:20b", input=text)
embedding = response["embeddings"][0]
```

장점: 추가 패키지 불필요, 동일 모델 사용
단점: 20B 모델의 임베딩 → 느릴 수 있음

**방안 B: 경량 sentence-transformers (권장)**

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("intfloat/multilingual-e5-small")  # ~120MB, 다국어 균형
embedding = encoder.encode(text)
```

장점: 한국어 포함 다국어 질의에서 안정적, CPU 전용 가능, GPU와 LLM 경합 없음
단점: 추가 패키지 (~120MB)

**선택: `intfloat/multilingual-e5-small`** — 한국어 품질과 속도의 균형을 우선.

### 4.2.1 한국어 서비스 기준 임베딩 선택 기준

기본 운영 모델은 `intfloat/multilingual-e5-small`로 고정한다.
한국어 서비스에서는 다국어 임베딩을 기본값으로 유지하며, 문서/설정/Docker preload를 동일 모델로 통일한다.

- 기본값(고정): `intfloat/multilingual-e5-small` (~120MB, 다국어 균형, 한국어 지원)
- 기준 지표: 캐시 히트 정확도(오탐률), 평균 인코딩 시간(ms), 메모리 사용량
- 선택 원칙: 속도 우선이 아니라 **오탐 최소화 + 실측 latency** 기준으로 채택

### 4.3 신규 모듈: `core/semantic_cache.py`

```python
"""시맨틱 캐싱 엔진.

사용자 질문의 임베딩 유사도를 기반으로 캐시 히트를 판정한다.
캐시는 SQLite에 영구 저장하고, 인메모리 인덱스로 빠른 검색을 제공한다.
"""

@dataclass(frozen=True)
class CacheContext:
    model: str
    prompt_ver: str
    intent: str | None
    scope: str                       # "global" | "user"
    chat_id: int | None = None       # scope=="user"일 때만 사용

class SemanticCache:
    """임베딩 기반 시맨틱 캐시."""

    def __init__(
        self,
        db_path: str,
        model_name: str = "intfloat/multilingual-e5-small",
        similarity_threshold: float = 0.92,
        max_entries: int = 5000,
        ttl_hours: int = 168,           # 7일
    ):
        ...

    async def is_cacheable(self, query: str) -> bool:
        """시간 민감/짧은 질의 등 캐시 제외 여부를 판단한다."""
        ...

    async def get(self, query: str, context: CacheContext) -> CacheResult | None:
        """유사한 캐시 항목을 검색한다.

        Args:
            query: 사용자 입력 텍스트.
            context: cache key 구성 정보(model, prompt_ver, intent, scope, chat_id).

        Returns:
            CacheResult(response, similarity, cached_query) 또는 None.
        """
        ...

    async def put(self, query: str, response: str, context: CacheContext) -> int:
        """질문-응답 쌍을 캐시에 저장하고 cache_id를 반환한다."""
        ...

    async def invalidate(self, chat_id: int | None = None) -> int:
        """캐시를 무효화한다. chat_id 지정 시 해당 사용자만."""
        ...

    async def link_feedback_target(
        self,
        chat_id: int,
        bot_message_id: int,
        cache_id: int,
    ) -> None:
        """텔레그램 메시지와 캐시 항목을 연결한다."""
        ...

    async def get_stats(self) -> dict:
        """캐시 통계 (히트율, 항목 수, 평균 유사도 등)."""
        ...
```

### 4.4 저장소 설계

```
SQLite 테이블: semantic_cache
┌─────┬───────┬────────┬───────┬─────────┬──────────┬───────┬────────────┬───────────┬────────────┬────────────┐
│ id  │ scope │ chat_id│ query │ response│ embedding│ model │ prompt_ver │ intent    │ created_at │ last_hit_at│
└─────┴───────┴────────┴───────┴─────────┴──────────┴───────┴────────────┴───────────┴────────────┴────────────┘
인덱스: (scope, chat_id, created_at), (scope, model, prompt_ver, intent)

SQLite 테이블: semantic_cache_feedback_links
┌────────┬────────────────┬──────────┬────────────┐
│ chat_id│ bot_message_id │ cache_id │ created_at │
└────────┴────────────────┴──────────┴────────────┘
인덱스: (chat_id, bot_message_id)
```

인메모리 검색:
- 시작 시 전체 임베딩을 numpy 배열로 로드 (5000개 × 384차원 ≈ 7.3MB)
- 코사인 유사도 벡터 연산으로 전체 탐색 (~1ms)
- 캐시가 커지면 FAISS 인덱스로 전환 가능 (Phase 확장)

### 4.4.1 캐시 범위: per-user vs global

캐시 키는 `model + prompt_version + intent + scope (+chat_id)`를 사용한다.
`scope=global`은 공유 캐시, `scope=user`는 사용자별 캐시를 의미한다.

**권장 전략: 2-레벨 캐시**

1. **글로벌 캐시** — `scope=global`, `chat_id=None`. 일반 QA/잡담에 적용.
2. **사용자별 캐시** — `scope=user`, `chat_id=...`. 개인 컨텍스트 의존 질의에 적용.

인텐트 라우터(Phase 4) 도입 후, `chitchat`/`simple_qa` 인텐트는 글로벌 캐시,
`code`/`complex` 인텐트는 사용자별 캐시로 분리하면 히트율을 크게 높일 수 있다.
초기 구현은 per-user로 시작하되, 글로벌 캐시는 Phase 4와 함께 도입한다.

### 4.4.2 스트리밍 경로(`process_message_stream`) 캐시 처리

`process_message_stream`에서 캐시 히트 시 스트리밍이 아닌 즉시 응답이 반환되므로,
호출부(telegram_handler.py)에서 이를 처리할 수 있어야 한다.
또한 종료 시 메타(`tier`, `intent`, `cache_id`, `usage`)를 타입으로 전달해야 한다.

```python
@dataclass
class StreamResult:
    full_response: str
    last_message: Message | None
    tier: str
    intent: str | None = None
    cache_id: int | None = None
    usage: ChatUsage | None = None
```

```python
async def process_message_stream(self, chat_id, text, model_override=None):
    # 선행: skill trigger
    skill = self._skills.match_trigger(text)
    if skill is not None:
        ...
        return

    # Tier 1: 즉시 응답
    instant = self._instant_responder.match(text)
    if instant is not None:
        await self._persist_turn(chat_id, text, instant.response)
        yield instant.response  # 단일 청크로 반환
        return

    # Tier 2: 인텐트 분류
    route = self._intent_router.classify(text) if self._intent_router else None
    intent = route.intent if route else None

    # Tier 3: 시맨틱 캐시
    if self._semantic_cache is not None and await self._semantic_cache.is_cacheable(text):
        cached = await self._semantic_cache.get(text, context=cache_ctx)
        if cached is not None:
            await self._persist_turn(chat_id, text, cached.response)
            self._stream_meta = RequestMeta(
                tier="cache",
                intent=intent,
                cache_hit=True,
                cache_id=cached.cache_id,
            )
            yield cached.response  # 단일 청크로 반환
            return

    # Tier 4: LLM 스트리밍 (기존 경로)
    prepared = await self._prepare_request(chat_id, text, stream=True)
    full_response = ""
    async for chunk in self._ollama.chat_stream(...):
        full_response += chunk
        yield chunk
    # 캐시 저장
    ...
```

### 4.5 캐시 무효화 정책

| 조건 | 동작 |
|------|------|
| TTL 만료 (기본 7일) | 자동 삭제 |
| 사용자가 👎 피드백 | 해당 응답 캐시 삭제 |
| `/memory clear` 명령 | 엔진 `clear_conversation()`에서 즉시 `semantic_cache.invalidate(chat_id=...)` 호출 |
| 캐시 최대 크기 초과 | LRU 방식으로 오래된 항목 제거 |
| 모델 변경 (`/model`) | 엔진 `change_model()`에서 즉시 `semantic_cache.invalidate()` 호출 |
| 프롬프트 버전 변경 (`ollama.prompt_version`) | 해당 버전 캐시 무효화 |
| 시간 민감 질의 (`지금`, `오늘`, `현재`) | 캐시 저장/조회 모두 건너뜀 |

### 4.6 Engine 통합

**변경 파일: `core/engine.py`**

```python
async def process_message(self, chat_id, text, model_override=None):
    # [선행] 스킬 우선 처리
    skill = self._skills.match_trigger(text)
    if skill is not None:
        ...
        return ...

    # [Tier 1] 즉시 응답
    instant = self._instant_responder.match(text)
    if instant is not None:
        ...

    # [Tier 2] 인텐트 라우팅
    route = self._intent_router.classify(text) if self._intent_router else None
    intent = route.intent if route else None
    scope = "global" if intent in {"chitchat", "simple_qa"} else "user"

    is_cacheable = (
        await self._semantic_cache.is_cacheable(text)
        if self._semantic_cache
        else False
    )
    cache_ctx = {
        "model": model_override or self._ollama.default_model,
        "prompt_ver": self._config.ollama.prompt_version,
        "intent": intent,
        "scope": scope,
        "chat_id": chat_id if scope == "user" else None,
    }

    # [Tier 3] 시맨틱 캐시
    if self._semantic_cache is not None and is_cacheable:
        cached = await self._semantic_cache.get(text, context=cache_ctx)
        if cached is not None:
            await self._persist_turn(chat_id, text, cached.response)
            self._logger.info(
                "cache_hit",
                chat_id=chat_id,
                similarity=cached.similarity,
            )
            return cached.response

    # [Tier 4] LLM 호출
    prepared = await self._prepare_request(chat_id, text, stream=False, route=route)
    response = await self._ollama.chat(...)

    # 캐시에 저장
    if self._semantic_cache is not None and is_cacheable:
        cache_id = await self._semantic_cache.put(text, response, context=cache_ctx)

    ...
```

### 4.6.1 피드백 연동을 위한 캐시 식별자 연결

👎 피드백으로 정확히 캐시를 삭제하려면 텔레그램 메시지와 캐시 항목을 연결해야 한다.
이를 위해 `stream_and_render()` 반환 타입(`StreamResult`)에 `cache_id` 필드를 포함한다.

```python
# core/telegram_handler.py (스트리밍 완료 직후)
if result.last_message and result.cache_id is not None:
    await self._semantic_cache.link_feedback_target(
        chat_id=chat_id,
        bot_message_id=result.last_message.message_id,
        cache_id=result.cache_id,
    )
```

### 4.7 Phase 3 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `core/semantic_cache.py` | 신규 | 시맨틱 캐싱 엔진 |
| `core/engine.py` | 수정 | Tier 3 캐시 체크/저장 로직 |
| `core/config.py` | 수정 | `SemanticCacheConfig` 추가 |
| `config/config.yaml` | 수정 | `semantic_cache` 섹션 추가 |
| `main.py` | 수정 | `SemanticCache` 초기화/주입 |
| `core/telegram_handler.py` | 수정 | `bot_message_id ↔ cache_id` 링크 저장 |
| `requirements.txt` | 수정 | `sentence-transformers`, `numpy` 추가 |
| `requirements.lock` | 수정 | 잠금 의존성 동기화 |
| `tests/test_semantic_cache.py` | 신규 | 단위 테스트 |

### 4.8 설정 추가

```yaml
# config/config.yaml
semantic_cache:
  enabled: true
  model_name: "intfloat/multilingual-e5-small"  # 임베딩 모델 (한국어 지원)
  embedding_device: "cpu"           # LLM과 GPU 경합 방지
  similarity_threshold: 0.92        # 캐시 히트 판정 기준
  min_query_chars: 4                # 짧은 질의 캐시 제외
  exclude_patterns:                 # 시간/상태성 질문 캐시 제외
    - "(지금|현재)\\s*몇\\s*시"
    - "오늘\\s*(날짜|며칠|요일)"
  max_entries: 5000                 # 최대 캐시 항목 수
  ttl_hours: 168                    # 캐시 TTL (7일)
  invalidate_on_negative_feedback: true  # 👎 시 캐시 삭제
```

### 4.9 Phase 3 완료 기준

- [ ] 동일/유사 질문에 대해 캐시 히트 동작 확인 (LLM 우회)
- [ ] 캐시 미스 시 LLM 응답 후 자동 캐시 저장
- [ ] 시간 민감 질의 캐시 제외 정책 동작 확인
- [ ] 👎 피드백 시 해당 캐시 항목 삭제
- [ ] `/memory clear` 시 사용자별 캐시 삭제
- [ ] `/model` 변경 시 캐시 전체 무효화
- [ ] 캐시 통계 (`/status` 또는 별도 명령)
- [ ] sentence-transformers 모델 Docker 이미지에 포함
- [ ] 단위 테스트 통과

---

## 5. Phase 4: 인텐트 라우팅 (Tier 2)

임베딩 기반으로 사용자 의도를 **1ms 이내에 분류**하고, 간단한 의도는 경량 처리기로, 복잡한 의도만 Full LLM으로 전달한다.

### 5.1 아키텍처

```
engine.py: process_message()
    │
    ├─ [선행] skill_manager.match_trigger()
    ├─ [Tier 1] instant_responder.match()
    ├─ [Tier 2, 신규] intent_router.classify(text)
    │       ├─ "simple_qa"    → 짧은 시스템프롬프트 + 최소 히스토리 (5개)
    │       ├─ "chitchat"     → 캐주얼 톤 + 최소 히스토리 (3개)
    │       ├─ "code"         → 코드 특화 프롬프트 + 중간 히스토리 (15개)
    │       ├─ "complex"      → 풀 시스템프롬프트 + 풀 히스토리 (50개)
    │       └─ None (미분류)  → 기본 경로 (기존과 동일)
    │
    ├─ [Tier 3] semantic_cache.get(text, context)
    └─ [Tier 4] _build_context() → ollama.chat()
```

### 5.2 인텐트 라우터 라이브러리 선택

**[semantic-router](https://github.com/aurelio-labs/semantic-router)** 사용.
sentence-transformers와 동일한 임베딩 모델을 공유하므로 추가 메모리 부담 최소.

### 5.3 신규 모듈: `core/intent_router.py`

```python
"""인텐트 기반 메시지 라우팅 엔진.

사용자 의도를 임베딩 유사도로 분류하고,
의도별 최적화된 처리 경로를 결정한다.
"""

@dataclass
class RouteResult:
    """라우팅 결과."""
    intent: str                     # 분류된 의도
    confidence: float               # 확신도 (0.0-1.0)
    context_strategy: ContextStrategy  # 컨텍스트 빌드 전략

@dataclass
class ContextStrategy:
    """의도별 컨텍스트 빌드 전략."""
    system_prompt_override: str | None = None  # 의도별 시스템 프롬프트 (None이면 기본값)
    max_history: int = 50           # 사용할 대화 히스토리 수
    include_dicl: bool = True       # DICL 예시 포함 여부
    include_preferences: bool = True # 사용자 선호도 포함 여부
    max_tokens: int | None = None   # 응답 최대 토큰 (None이면 기본값)

class IntentRouter:
    """임베딩 기반 인텐트 분류기."""

    def __init__(self, routes_path: str = "config/intent_routes.yaml"):
        ...

    def classify(self, text: str) -> RouteResult | None:
        """사용자 입력의 의도를 분류한다.

        Returns:
            RouteResult 또는 None (확신도 미달).
        """
        ...
```

### 5.4 인텐트 라우트 정의: `config/intent_routes.yaml`

```yaml
# 인텐트 라우팅 규칙 정의
# 각 라우트는 예시 발화(utterances)와 컨텍스트 전략을 정의한다.

min_confidence: 0.75  # 이 이하의 확신도는 기본 경로로 처리

routes:
  # ── 잡담/캐주얼 ──
  - name: "chitchat"
    utterances:
      - "심심해"
      - "뭐하고 있어?"
      - "오늘 기분이 어때?"
      - "재미있는 얘기 해줘"
      - "농담 하나 해봐"
    strategy:
      max_history: 3
      include_dicl: false
      include_preferences: false
      max_tokens: 512

  # ── 단순 질의응답 ──
  - name: "simple_qa"
    utterances:
      - "파이썬이 뭐야?"
      - "HTTP 상태 코드 설명해줘"
      - "리스트와 튜플의 차이"
      - "도커가 뭐야?"
    strategy:
      max_history: 5
      include_dicl: false
      max_tokens: 1024

  # ── 코드 관련 ──
  - name: "code"
    utterances:
      - "코드 짜줘"
      - "이 함수 고쳐줘"
      - "파이썬으로 구현해줘"
      - "디버깅 도와줘"
      - "에러가 나는데"
    strategy:
      max_history: 15
      include_dicl: true
      system_prompt_suffix: |
        코드 응답 시 간결한 코드와 핵심 설명만 제공하세요.

  # ── 복잡한 분석/토론 ──
  - name: "complex"
    utterances:
      - "이 아키텍처를 분석해줘"
      - "장단점을 비교해줘"
      - "전략을 세워줘"
      - "심층 분석해줘"
    strategy:
      max_history: 50
      include_dicl: true
      include_preferences: true
      # max_tokens: 기본값 사용 (2048)
```

### 5.5 Engine 통합 — `_build_context` 수정

인텐트 라우터의 `ContextStrategy`를 `_build_context`에 전달하여, 의도에 따라 **히스토리 수, DICL 포함 여부, 토큰 제한**을 동적으로 조절한다.

```python
async def _prepare_request(self, chat_id, text, *, stream, route=None):
    # route는 process_message/process_message_stream에서 선분류한 결과를 재사용
    strategy = route.context_strategy if route else None
    messages = await self._build_context(chat_id, text, strategy=strategy)
    timeout = self._config.bot.response_timeout
    return _PreparedRequest(skill=None, messages=messages, timeout=timeout)
```

### 5.6 Phase 4 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `core/intent_router.py` | 신규 | 임베딩 기반 인텐트 분류기 |
| `config/intent_routes.yaml` | 신규 | 인텐트 라우트 정의 |
| `core/engine.py` | 수정 | `_prepare_request`에 인텐트 라우팅, `_build_context`에 `ContextStrategy` 적용 |
| `core/config.py` | 수정 | `IntentRouterConfig` 추가 |
| `config/config.yaml` | 수정 | `intent_router` 섹션 추가 |
| `main.py` | 수정 | `IntentRouter` 초기화/주입 |
| `requirements.txt` | 수정 | `semantic-router` 추가 |
| `tests/test_intent_router.py` | 신규 | 단위 테스트 |

### 5.7 설정 추가

```yaml
# config/config.yaml
intent_router:
  enabled: true
  routes_path: "config/intent_routes.yaml"
  min_confidence: 0.75
  encoder_model: "intfloat/multilingual-e5-small"  # semantic_cache와 모델 공유
```

### 5.8 Phase 4 완료 기준

- [ ] 잡담 의도 → 히스토리 3개, DICL 미포함으로 동작
- [ ] 코드 의도 → 히스토리 15개, 코드 특화 프롬프트 적용
- [ ] 미분류 → 기존 풀 컨텍스트 경로로 폴백
- [ ] semantic_cache와 임베딩 모델 인스턴스 공유 확인
- [ ] 단위 테스트 통과

### 5.9 Phase 3 연동: 라우트 단위 캐시 분리

인텐트는 캐시 조회 전에 먼저 확정한다.
`route.intent`와 `scope(global|user)`를 `CacheContext`에 포함한 뒤 캐시 조회/저장을 수행하여,
서로 다른 의도 간 캐시 오염(예: 잡담 응답이 코드 질의에 재사용되는 문제)을 방지한다.

---

## 6. Phase 5: 컨텍스트 윈도우 최적화

### 6.0 선행 변경: 대화 저장 정책 분리 (필수)

현재 구조에서는 `max_conversation_length`를 초과한 메시지가 즉시 삭제되므로,
요약에 사용할 오래된 대화 원본이 남지 않는다. 따라서 Phase 5 착수 전에 저장 정책을 분리한다.

- 온라인 컨텍스트용: 기존 `conversations` (최근 N개 유지)
- 요약/분석 원본용: 신규 `conversations_archive` (append-only, retention 기준 정리)

추가 운영 원칙:

- `conversations_archive`와 `context_summaries`도 `memory.conversation_retention_days` 기준으로 prune 대상에 포함
- `/memory clear` 실행 시 `conversations`, `conversations_archive`, `context_summaries`를 함께 삭제

### 6.1 대화 히스토리 요약 (Context Summarization)

50개의 원본 메시지를 모두 보내는 대신, **오래된 메시지를 요약**하여 토큰 수를 줄인다.

```
[현재] 최근 50개 원본 전송 → ~3000-5000 토큰
[최적화] 최근 10개 원본 + archive 기반 요약(2-3문장) → ~800-1500 토큰
```

### 6.2 요약 전략

```
conversations_archive (요약 원본)
    │
    ├─ 요약 대상 구간 추출
    │
    └─ LLM 요약 (백그라운드, 1회) → context_summaries 캐시

conversations (온라인 경로)
    └─ 최근 N개(기본 10) 원본 유지
```

### 6.3 신규 모듈: `core/context_compressor.py`

```python
"""대화 컨텍스트 압축 엔진.

오래된 대화 히스토리를 요약하여 LLM에 전달하는 토큰 수를 줄인다.
요약은 SQLite에 캐시하여 반복 생성을 방지한다.
"""

class ContextCompressor:
    """대화 히스토리 요약 및 압축."""

    def __init__(
        self,
        ollama: OllamaClient,
        memory: MemoryManager,
        recent_keep: int = 10,          # 원본 유지할 최근 메시지 수
        summary_refresh_interval: int = 10,  # 요약 갱신 주기 (새 메시지 수)
    ):
        ...

    async def build_compressed_history(
        self,
        chat_id: int,
        max_history: int = 50,
    ) -> list[dict[str, str]]:
        """압축된 대화 히스토리를 반환한다.

        Returns:
            [{"role": "system", "content": "이전 대화 요약: ..."},
             {"role": "user", "content": "최근 메시지1"},
             {"role": "assistant", "content": "최근 응답1"},
             ...]
        """
        ...
```

### 6.4 아카이브/요약 캐시 테이블

```sql
CREATE TABLE IF NOT EXISTS conversations_archive (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content       TEXT NOT NULL,
    message_id    INTEGER NOT NULL,   -- conversations.id와 연결
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_archive_chat_id_id
    ON conversations_archive(chat_id, id);

CREATE TABLE IF NOT EXISTS context_summaries (
    chat_id        INTEGER NOT NULL,
    summary        TEXT NOT NULL,
    last_archive_id INTEGER NOT NULL,  -- 요약에 포함된 마지막 archive id
    message_count  INTEGER NOT NULL,   -- 요약에 포함된 메시지 수
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id)
);
```

보관/삭제 정책:

- `conversations_archive.timestamp < cutoff`는 주기적으로 삭제
- `context_summaries`는 보관 기준을 초과한 사용자 데이터 또는 orphan(chat_id 미존재) 항목 삭제
- 유지보수 루프(`main.py`의 정리 작업)에서 본 테이블도 함께 관리

### 6.5 요약 생성 프롬프트

```python
SUMMARY_SYSTEM_PROMPT = (
    "당신은 대화 요약 전문가입니다. "
    "아래 대화 내역을 2-3문장으로 간결하게 요약하세요. "
    "핵심 주제, 사용자의 관심사, 중요한 결정사항만 포함하세요. "
    "구체적인 코드나 긴 설명은 생략하세요."
)
```

**주의**: 요약 자체도 LLM 호출이 필요하므로, 요약 생성은 **백그라운드**에서 수행하고 캐시에 저장한다.
실시간 요청 경로에서는 캐시된 요약만 사용한다. 캐시가 없으면 요약 없이 최근 N개만 전송한다.

**경합 방지 제약 (필수)**:

- 요약 작업 동시성은 1로 제한
- 사용자 요청이 진행 중이면 요약 작업은 큐 대기
- 큐 길이 상한 초과 시 신규 요약은 스킵 (응답 latency 우선)

### 6.6 Engine `_build_context` 수정

```python
async def _build_context(self, chat_id, text, skill=None, strategy=None):
    if skill:
        history = await self._memory.get_conversation(chat_id, limit=5)
    else:
        max_hist = strategy.max_history if strategy else self._max_conversation_length
        if self._context_compressor is not None and max_hist > self._context_compressor.recent_keep:
            history = await self._context_compressor.build_compressed_history(
                chat_id, max_history=max_hist
            )
        else:
            history = await self._memory.get_conversation(chat_id, limit=max_hist)

    # ... 이하 기존 로직 동일 ...
```

### 6.7 Phase 5 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `core/context_compressor.py` | 신규 | 대화 히스토리 요약/압축 엔진 |
| `core/engine.py` | 수정 | `_build_context`에서 `ContextCompressor` 사용 |
| `core/memory.py` | 수정 | `conversations_archive` + `context_summaries` 테이블 추가 |
| `core/config.py` | 수정 | `ContextCompressorConfig` 추가 |
| `config/config.yaml` | 수정 | `context_compressor` 섹션 추가 |
| `main.py` | 수정 | `ContextCompressor` 초기화/주입 |
| `tests/test_context_compressor.py` | 신규 | 단위 테스트 |

### 6.8 설정 추가

```yaml
# config/config.yaml
context_compressor:
  enabled: true
  recent_keep: 10                    # 원본 유지할 최근 메시지 수
  summary_refresh_interval: 10       # 요약 갱신 주기 (새 메시지 수)
  summary_max_tokens: 200            # 요약 최대 토큰
  background_summarize: true         # 백그라운드 요약 생성
  archive_enabled: true              # 요약 원천용 archive 저장
  summarize_concurrency: 1           # 요약 작업 동시성 상한
  run_only_when_idle: true           # 사용자 요청 중에는 요약 지연
```

### 6.9 Phase 5 완료 기준

- [ ] 대화 10개 이하 → 원본 전체 전송 (기존과 동일)
- [ ] 대화 10개 초과 → 최근 10개 원본 + 나머지 요약으로 전송
- [ ] archive 테이블에 요약 원천 데이터 누적 확인
- [ ] 요약 캐시 히트 시 LLM 추가 호출 없음
- [ ] 캐시 미스 시 최근 N개만으로 폴백 (블로킹 없음)
- [ ] 요약 작업이 사용자 응답 latency를 악화시키지 않음 (p95 회귀 없음)
- [ ] 백그라운드 요약 생성 동작 확인
- [ ] archive/summary 테이블 retention prune 동작 확인
- [ ] `/memory clear` 시 archive/summary까지 함께 정리됨
- [ ] 단위 테스트 통과

---

## 7. Phase 6: 스트리밍 UX 강화

현재 스트리밍 인프라는 존재하지만, **체감 속도를 더 개선**할 수 있는 부분이 있다.

### 7.1 현재 스트리밍 흐름

```
[현재]
1. placeholder 메시지 전송 ("...")
2. LLM 스트리밍 시작
3. 1초 간격으로 메시지 편집
4. 스트리밍 완료 후 피드백 버튼 부착
```

### 7.2 개선 사항

#### 7.2.a 타이핑 인디케이터 연속 유지

현재 `ChatAction.TYPING`을 1회만 전송한다. 텔레그램의 typing 인디케이터는 ~5초 후 사라지므로,
LLM 응답이 오래 걸릴 때 사용자는 봇이 멈춘 것으로 오해할 수 있다.

```python
# 스트리밍 대기 중 typing 인디케이터 주기적 갱신
async def _keep_typing(chat_id: int, stop_event: asyncio.Event):
    while not stop_event.is_set():
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue
```

#### 7.2.b 첫 청크 도착 전 상태 메시지

placeholder를 "..." 대신 더 정보성 있는 메시지로 교체.

```python
# Tier 2 인텐트 결과를 활용한 상태 메시지
status_messages = {
    "code": "코드를 분석하고 있습니다...",
    "complex": "심층 분석 중입니다...",
    "simple_qa": "답변을 준비하고 있습니다...",
    None: "생각하고 있습니다...",
}
```

#### 7.2.c 편집 간격 동적 조절

현재 고정 1초 간격. 초반에는 더 자주 편집하여 반응성을 높이고, 후반에는 간격을 넓혀 API 제한을 방지.

```python
# 동적 편집 간격
def _calc_edit_interval(elapsed_seconds: float) -> float:
    if elapsed_seconds < 3.0:
        return 0.5   # 초반: 빠른 피드백
    elif elapsed_seconds < 10.0:
        return 1.0   # 중반: 표준
    else:
        return 2.0   # 후반: 안정적
```

### 7.3 Phase 6 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `core/telegram_handler.py` | 수정 | 타이핑 인디케이터 연속 유지 |
| `core/telegram_message_renderer.py` | 수정 | 동적 편집 간격, 상태 메시지 |

### 7.4 Phase 6 완료 기준

- [ ] LLM 응답 대기 중 typing 인디케이터가 끊기지 않음
- [ ] 인텐트 기반 상태 메시지 표시
- [ ] 스트리밍 초반 빠른 편집, 후반 안정적 편집 동작

---

## 8. 의존성 변경 요약

### 8.1 신규 패키지

```
# requirements.txt 추가
sentence-transformers>=3.0.0,<4.0    # Phase 3, 4: 임베딩 모델
numpy>=1.24.0,<3.0                   # Phase 3: 벡터 연산
semantic-router>=0.1.0               # Phase 4: 인텐트 라우팅
```

> 주의: 실제 Docker 빌드는 `requirements.lock`을 사용하므로, 위 변경 후 lock 재생성이 필수다.

### 8.2 Docker 이미지 영향

| 항목 | 현재 | 변경 후 |
|------|------|---------|
| 이미지 크기 | ~300MB (추정) | ~600MB (sentence-transformers 포함) |
| 메모리 사용 | LLM 전용 | +~200MB (임베딩 모델 + 캐시 인덱스) |
| CPU 사용 | 최소 | +임베딩 인코딩 (요청당 ~5ms) |

### 8.3 Dockerfile 변경

```dockerfile
# read-only 컨테이너에서 임베딩 캐시 저장 경로를 /app/data로 고정
ENV HF_HOME=/app/data/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/.cache/sentence_transformers

# 임베딩 모델을 빌드 시 다운로드하여 이미지에 포함
RUN mkdir -p /app/data/.cache/huggingface /app/data/.cache/sentence_transformers \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
```

### 8.4 잠금 파일 갱신 절차

```bash
pip-compile --output-file=requirements.lock requirements.txt
docker compose build --no-cache
```

검증 포인트:

- `requirements.txt`와 `requirements.lock` 동기화
- 컨테이너 빌드 시 신규 의존성 설치 확인

---

## 9. 설정 파일 최종 구조

### 9.1 config.yaml 전체 (최적화 적용 후)

```yaml
# ollama_bot 전역 설정 (속도 최적화 포함)

bot:
  name: "ollama_bot"
  language: "ko"
  max_conversation_length: 50
  response_timeout: 60

ollama:
  host: "http://host.docker.internal:11434"
  model: "gpt-oss:20b"
  temperature: 0.7
  max_tokens: 2048
  num_ctx: 8192                        # [신규] 기본 운영값 (Phase 5 완료 후 4096으로 축소)
  prompt_version: "v1"                 # [신규] 캐시 버전 분리 키
  system_prompt: |
    당신은 유용한 AI 어시스턴트입니다.
    한국어로 답변하며, 간결하고 정확한 정보를 제공합니다.

telegram:
  polling_interval: 1
  max_message_length: 4096

security:
  allowed_users: []
  rate_limit: 30
  max_concurrent_requests: 4
  max_file_size: 10485760
  blocked_paths:
    - "/etc/*"
    - "/proc/*"
    - "/sys/*"

memory:
  backend: "sqlite"
  max_long_term_entries: 1000
  conversation_retention_days: 30

scheduler:
  timezone: "Asia/Seoul"

feedback:
  enabled: true
  show_buttons: true
  # ... (기존 설정 유지)

auto_evaluation:
  enabled: false
  # ... (기존 설정 유지)

# ── 속도 최적화 신규 섹션 ──

instant_responder:                     # [Phase 2]
  enabled: true
  rules_path: "config/instant_rules.yaml"

semantic_cache:                        # [Phase 3]
  enabled: true
  model_name: "intfloat/multilingual-e5-small"
  embedding_device: "cpu"
  similarity_threshold: 0.92
  min_query_chars: 4
  exclude_patterns:
    - "(지금|현재)\\s*몇\\s*시"
    - "오늘\\s*(날짜|며칠|요일)"
  max_entries: 5000
  ttl_hours: 168
  invalidate_on_negative_feedback: true

intent_router:                         # [Phase 4]
  enabled: true
  routes_path: "config/intent_routes.yaml"
  min_confidence: 0.75
  encoder_model: "intfloat/multilingual-e5-small"

context_compressor:                    # [Phase 5]
  enabled: true
  recent_keep: 10
  summary_refresh_interval: 10
  summary_max_tokens: 200
  background_summarize: true
  archive_enabled: true
  summarize_concurrency: 1
  run_only_when_idle: true
```

---

## 10. 측정 및 모니터링

### 10.1 Latency 구조화 로그

모든 요청 경로에 소요 시간을 기록하여 최적화 효과를 정량 측정한다.
토큰 사용량 로깅을 위해 `OllamaClient.chat()`/`chat_stream()`은 문자열만 반환하지 않고,
usage 메타데이터를 포함한 응답 객체를 함께 전달하도록 선행 리팩터링한다.

스트리밍 경로는 본문 청크와 별도로 종료 시점 메타를 반환한다.
(`tier`, `intent`, `cache_hit`, `cache_id`, `usage`)

```python
@dataclass
class RequestMeta:
    tier: str                      # skill|instant|intent|cache|full
    intent: str | None = None
    cache_hit: bool = False
    cache_id: int | None = None
    usage: ChatUsage | None = None
```

```python
# core/engine.py
self._logger.info(
    "request_completed",
    chat_id=chat_id,
    tier=result.tier,                    # skill|instant|intent|cache|full
    intent=result.intent,
    cache_hit=result.cache_hit,
    cache_id=result.cache_id,
    latency_ms=elapsed_ms,
    tokens_input=result.usage.prompt_eval_count if result.usage else None,
    tokens_output=result.usage.eval_count if result.usage else None,
)
```

### 10.2 `/status` 명령 확장

```
📊 시스템 상태

가동 시간: 12시간 34분
Ollama: 🟢 정상
모델: gpt-oss:20b

⚡ 속도 최적화
  즉시 응답 규칙: 5개 로드
  시맨틱 캐시: 1,234/5,000 (히트율 64.2%)
  인텐트 라우터: 4개 라우트 (정확도 93.1%)
  컨텍스트 압축: 활성 (평균 68% 토큰 절감)

  최근 24시간 응답 시간:
  - Tier 1 (즉시): 0ms (312건)
  - Tier 2 (인텐트 라우팅): <1ms avg (501건)
  - Tier 3 (시맨틱 캐시): 15ms avg (189건)
  - Tier 4 (풀 LLM): 8.1초 avg (43건)
```

### 10.3 자동화 연동: 속도 분석 리포트

기존 `auto/_builtin/` 시스템에 속도 분석 자동화를 추가한다.
YAML 파일 추가만으로는 동작하지 않으며, callable 등록이 함께 필요하다.

```yaml
# auto/_builtin/speed_analysis.yaml
name: "speed_analysis"
description: "일일 응답 속도 분석 리포트"
enabled: true
schedule: "0 10 * * *"   # 매일 10:00
action:
  type: "callable"
  target: "speed_analysis"
output:
  send_to_telegram: true
  save_to_file: "reports/speed_{date}.md"
```

```python
# core/automation_callables.py
scheduler.register_callable(
    "speed_analysis",
    build_speed_analysis_callable(engine=engine, memory=memory, logger=logger),
)
```

### 10.4 공개 API/타입 계약

| 항목 | 기존 | 변경 |
|------|------|------|
| `OllamaClient.chat()` | `str` 반환 | `ChatResponse(content, usage)` 반환 |
| 스트리밍 경로 메타 | 없음 | 스트림 종료 시 `RequestMeta(tier, intent, cache_hit, cache_id, usage)` 제공 |
| `StreamResult` 계약 | `full_response`, `last_message` | `cache_id`, `tier`, `intent`, `usage` 필드 추가 |
| `CacheContext` | `chat_id, model, prompt_ver` 중심 | `intent`, `scope(global/user)` 필수 포함 |
| `OllamaConfig` | `keep_alive` 포함 제안 | `keep_alive` 제거, `num_ctx`, `prompt_version` 유지 |

---

## 11. 디렉토리 구조 변경

```
ollama_bot/
├── core/
│   ├── engine.py                  # [수정] 계층형 라우팅 통합
│   ├── ollama_client.py           # [수정] num_ctx 전달 + ChatResponse/usage 계약
│   ├── config.py                  # [수정] 신규 Config 클래스 추가
│   ├── instant_responder.py       # [신규] Phase 2: 규칙 기반 즉시 응답
│   ├── semantic_cache.py          # [신규] Phase 3: 시맨틱 캐싱
│   ├── intent_router.py           # [신규] Phase 4: 인텐트 라우팅
│   ├── context_compressor.py      # [신규] Phase 5: 컨텍스트 압축
│   ├── telegram_handler.py        # [수정] Phase 6: 스트리밍 UX 강화
│   ├── telegram_message_renderer.py # [수정] Phase 6: 동적 편집 간격 + StreamResult 메타
│   └── automation_callables.py    # [수정] speed_analysis callable 등록
├── config/
│   ├── config.yaml                # [수정] 신규 섹션 추가
│   ├── instant_rules.yaml         # [신규] Phase 2: 즉시 응답 규칙
│   └── intent_routes.yaml         # [신규] Phase 4: 인텐트 라우트 정의
├── tests/
│   ├── test_instant_responder.py  # [신규]
│   ├── test_semantic_cache.py     # [신규]
│   ├── test_intent_router.py      # [신규]
│   └── test_context_compressor.py # [신규]
├── auto/_builtin/
│   └── speed_analysis.yaml        # [신규] 속도 분석 자동화
├── requirements.txt               # [수정] 신규 패키지 추가
├── requirements.lock              # [수정] lock 재생성
├── Dockerfile                     # [수정] 임베딩 모델 사전 다운로드
└── docker-compose.yml             # [수정] Ollama 환경변수 문서화
```

---

## 12. 개발 로드맵 및 우선순위

| Phase | 내용 | 영향도 | 난이도 | 예상 변경 규모 |
|-------|------|--------|--------|---------------|
| **Phase 0** | 계측/벤치마크 기준선 수집 | ★★★★★ | ★☆☆☆☆ | 로그 필드 추가 + 측정 스크립트 |
| **Phase 1** | Ollama 런타임 튜닝 | ★★★★★ | ★☆☆☆☆ | 설정 변경 + 코드 3줄 |
| **Phase 2** | 규칙 기반 즉시 응답 | ★★★☆☆ | ★★☆☆☆ | 신규 모듈 1개 + YAML 1개 |
| **Phase 3** | 시맨틱 캐싱 | ★★★★☆ | ★★★☆☆ | 신규 모듈 1개 + 의존성 추가 |
| **Phase 4** | 인텐트 라우팅 | ★★★☆☆ | ★★★☆☆ | 신규 모듈 1개 + YAML 1개 |
| **Phase 5** | 컨텍스트 압축 | ★★★★☆ | ★★★★☆ | 신규 모듈 1개 + DB 스키마(archive 포함) 변경 |
| **Phase 6** | 스트리밍 UX 강화 | ★★★☆☆ | ★★☆☆☆ | 기존 모듈 수정 |

### 12.1 Phase 완료 기준 (전체)

- Phase 0 완료: Tier별 latency/usage 기준선 로그 수집 완료
- Phase 1 완료: Ollama 설정 적용 후 기존 테스트 통과
- Phase 2 완료: 인사/감사 패턴 즉시 응답 + 미매칭 시 LLM 폴백
- Phase 3 완료: 반복 질문 캐시 히트 + 피드백 연동 캐시 무효화
- Phase 4 완료: 4개 이상 인텐트 분류 + 의도별 컨텍스트 차별화
- Phase 5 완료: archive 기반 요약 + 캐시 동작 + 토큰 수 50% 이상 절감
- Phase 6 완료: 연속 타이핑 + 동적 편집 간격 + 상태 메시지

---

## 12.2 Graceful Degradation 원칙 (전 Phase 공통)

각 최적화 계층이 실패하더라도 **기존 LLM 경로가 반드시 동작**해야 한다.

| 구성 요소 | 실패 시나리오 | 대응 |
|----------|-------------|------|
| InstantResponder | 규칙 파일 파싱 실패 | 경고 로그 + Tier 1 비활성화, Tier 4로 폴백 |
| SemanticCache | 임베딩 모델 로드 실패 | 경고 로그 + Tier 3 비활성화, Tier 4로 폴백 |
| SemanticCache | SQLite DB 손상 | 경고 로그 + 캐시 파일 재생성 (빈 상태로 시작) |
| IntentRouter | encoder 초기화 실패 | 경고 로그 + Tier 2 비활성화, 기본 컨텍스트로 폴백 |
| ContextCompressor | 요약 LLM 호출 실패 | 요약 없이 최근 N개 원본만 전송 |
| sentence-transformers | 패키지 미설치 | `ImportError` 캐치 → 해당 기능 비활성화 |

**구현 패턴** — 각 모듈의 `__init__`에서:

```python
try:
    self._encoder = SentenceTransformer(model_name, device=device)
except Exception as e:
    self._logger.warning("encoder_init_failed", error=str(e))
    self._enabled = False  # 기능 비활성화, LLM 폴백
```

이 원칙은 모든 Phase에 걸쳐 적용되어야 하며,
최적화 모듈 하나의 장애가 봇 전체를 다운시키는 것을 방지한다.

---

## 13. 리스크 및 완화 방안

| 리스크 | 영향 | 완화 방안 |
|--------|------|----------|
| sentence-transformers가 GPU 경합 | LLM 추론 속도 저하 | CPU 전용 모드 강제 (`CUDA_VISIBLE_DEVICES=""`) |
| 한국어 임베딩 품질 부족 | 캐시 오탐/미탐 증가 | 다국어 모델 후보 A/B 벤치마크 후 채택 |
| 시맨틱 캐시 오탐 (유사하지만 다른 질문) | 잘못된 응답 반환 | threshold 보수적 설정 (0.92+), 👎 피드백 시 캐시 삭제 |
| 인텐트 분류 오류 | 최적이 아닌 컨텍스트 사용 | 기본 경로 폴백 (미분류 시 기존과 동일) |
| 요약 품질 저하 | 중요 컨텍스트 유실 | 요약 캐시 미존재 시 최근 N개만 전송 (품질 우선) |
| 요약 백그라운드 작업이 본응답과 경합 | p95 latency 악화 | 요약 동시성 1, 요청 처리 중 요약 지연 |
| Docker 이미지 크기 증가 | 배포 시간 증가 | 멀티스테이지 빌드, 모델 캐시 레이어 분리 |
| `requirements.lock` 미갱신 | 배포 시 의존성 불일치 | CI에서 lock diff 검사 및 빌드 검증 |
| Ollama Flash Attention 미지원 모델 | 설정 무효화 | 시작 시 지원 여부 확인, 미지원 시 경고 로그 |

---

## 13.1 테스트 전략

각 Phase의 단위 테스트 외에, 최적화 효과를 검증하기 위한 **통합/부하 테스트** 전략이 필요하다.

### 단위 테스트 (각 Phase별)

- 이미 계획된 `test_instant_responder.py`, `test_semantic_cache.py` 등
- 각 모듈의 정상 경로 + 에러 경로 + 엣지 케이스 커버

### 통합 테스트 (Phase 3 이후)

- Tier 1 → 2 → 3 → 4 전체 폴백 체인이 올바르게 동작하는지 검증
- 캐시 히트/미스 → 인텐트 라우팅 → LLM 호출 일련의 흐름
- 피드백(👎) → 캐시 무효화 → 재질문 시 LLM 경유 확인

### Latency 회귀 테스트

- Phase 0에서 수집한 기준선 대비 p50/p95 latency 비교
- 각 Phase 적용 후 회귀 없는지 확인하는 스크립트
- CI에서 자동 실행 가능한 형태로 구성

```bash
# 예시: scripts/benchmark_latency.sh
# 미리 정의된 테스트 메시지셋으로 봇에 요청 → latency 측정 → 기준선 대비 비교
```

---

## 14. 참고 자료

### 14.1 Ollama 최적화
- [Ollama FAQ — keep_alive](https://docs.ollama.com/faq)
- [Ollama Performance Tuning — GPU Optimization (Collabnix)](https://collabnix.com/ollama-performance-tuning-gpu-optimization-techniques-for-production/)
- [KV Context Quantisation in Ollama (smcleod.net)](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [Ollama VRAM Fine-Tune with KV Cache (Peddals Blog)](https://blog.peddals.com/en/ollama-vram-fine-tune-with-kv-cache/)

### 14.2 시맨틱 캐싱
- [GPTCache — GitHub](https://github.com/zilliztech/GPTCache)
- [Redis Semantic Caching](https://redis.io/blog/what-is-semantic-caching/)
- [Aashmit/Redis_Cache — Ollama + Redis](https://github.com/Aashmit/Redis_Cache)

### 14.3 인텐트 라우팅
- [semantic-router — GitHub](https://github.com/aurelio-labs/semantic-router)
- [Intent Classification in <1ms (Medium)](https://medium.com/@durgeshrathod.777/intent-classification-in-1ms-how-we-built-a-lightning-fast-classifier-with-embeddings-db76bfb6d964)

### 14.4 컨텍스트 관리
- [LLM Chat History Summarization Guide (mem0.ai)](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025)
- [Context Window Management (apxml.com)](https://apxml.com/courses/langchain-production-llm/chapter-3-advanced-memory-management/context-window-management)
- [JetBrains Research: Efficient Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

### 14.5 참고 프로젝트
- [jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord) — 스트리밍 최적화
- [DocShotgun/LLM-discordchatbot](https://github.com/DocShotgun/LLM-discordchatbot) — 동적 컨텍스트
- [The0mikkel/ollama-discord-bot](https://github.com/The0mikkel/ollama-discord-bot) — Redis 히스토리
- [LLMLingua (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/) — 프롬프트 압축

---

*작성일: 2026-02-25*
*프로젝트: ollama_bot v0.2 — 응답 속도 최적화*
*기반 문서: PROJECT_PLAN.md (v0.1)*
