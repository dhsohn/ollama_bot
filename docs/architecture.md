# 아키텍처 설계 문서

## 1. 계층형 라우팅 (4-Tier Response Pipeline)

모든 사용자 메시지는 비용 효율과 응답 속도를 위해 상위 tier부터 순차 평가된다.
상위 tier에서 매칭되면 하위 tier는 실행하지 않는다.

```
메시지 수신
  │
  ├─ [Skill] 트리거 키워드 매칭 → 전용 프롬프트로 LLM 호출
  │
  ├─ [Instant] 규칙 기반 즉시 응답 (LLM 호출 없음, <1ms)
  │
  ├─ [Cache] 시맨틱 캐시 유사도 검색 (LLM 호출 없음, ~10ms)
  │
  └─ [Full] 컨텍스트 빌드 + RAG + LLM 호출 (최대 지연)
```

### 설계 의도
- **비용 절감**: Instant/Cache tier에서 처리되면 LLM API 호출이 0
- **응답 속도**: 단순 질문은 ms 단위로 응답
- **품질 유지**: 복잡한 질문은 Full tier에서 RAG + 최적화 컨텍스트로 처리

### 트레이드오프
- Cache tier의 유사도 임계값(0.92)이 높으면 hit rate 감소, 낮으면 부정확 응답 위험
- Instant 규칙은 수동 관리 필요 (config/instant_rules.yaml)
- Skill 트리거는 문자열 prefix 매칭이므로 자연어 변형에 취약

### 타입 안전성
`RoutingTier(str, Enum)`으로 tier를 정의하여 문자열 비교 오류를 컴파일 타임에 방지한다.
(`core/enums.py` 참조)

---

## 2. 컨텍스트 압축 (Context Compression)

장기 대화에서 토큰 한도를 초과하지 않도록 히스토리를 관리한다.

```
전체 대화 히스토리
  │
  ├─ 최근 N턴: 원문 유지 (recent_keep=10)
  │
  └─ 이전 턴: 백그라운드 요약 → 압축 텍스트로 대체
      │
      ├─ 요약 갱신: summary_refresh_interval(10턴)마다
      │
      └─ 아카이브: retention_days(30일) 초과 시 삭제
```

### 핵심 모듈
- `core/context_compressor.py`: 요약 생성 및 히스토리 병합
- `core/engine_context.py`: 시스템 프롬프트 조립 (선호도/가이드라인/DICL/언어정책)
- `core/memory.py`: SQLite 기반 대화 저장소

---

## 3. RAG 파이프라인

```
질의
  │
  ├─ 트리거 키워드 검사 → RAG 실행 여부 결정
  │
  ├─ Embedding (Qwen3-Embedding-0.6B) → 벡터 검색 (k0=40)
  │
  ├─ Reranker (bge-reranker-v2-m3) → 상위 topk=8 선별
  │
  └─ Context Builder → 시스템 프롬프트에 주입
```

### Full-Scan 분석 (Map-Reduce)
대규모 코퍼스 전체를 읽어 분석할 때 사용:
1. **Map**: 각 세그먼트(12K chars)에서 근거 추출 (JSON 응답)
2. **Reduce**: 근거 통합/중복 제거 (최대 6 패스)
3. **Final**: 통합 근거로 최종 답변 생성

### 핵심 모듈
- `core/rag/chunker.py`: 문서 청킹 (500-1200 토큰, 15% 오버랩)
- `core/rag/retriever.py`: 임베딩 기반 벡터 검색
- `core/rag/reranker.py`: 교차 인코더 리랭킹
- `core/rag/context_builder.py`: 검색 결과를 프롬프트 형식으로 변환
- `core/engine_rag.py`: RAG 오케스트레이션 + Full-Scan 구현

---

## 4. 메모리 / 영속성

### 스토리지 구조
```
data/
  ├─ memory.db          # 대화 히스토리 + 장기 메모리 (SQLite, WAL 모드)
  ├─ semantic_cache.db  # 시맨틱 캐시 (SQLite)
  ├─ feedback.db        # 피드백 데이터 (SQLite)
  └─ reports/           # 자동화 리포트
```

### 데이터 관리
- **WAL 저널링**: 읽기/쓰기 동시성 보장
- **원자적 트랜잭션**: `_execute` 메서드로 추상화
- **마이그레이션**: `core/db_migrations.py`에서 스키마 버전 관리
- **Retention**: conversation_retention_days(30일) 초과 데이터 자동 정리

---

## 5. 보안 모델

```
요청 수신
  │
  ├─ [1] 인증: allowed_users 화이트리스트 (chat_id 기반)
  │
  ├─ [2] 레이트리밋: 슬라이딩 윈도우 (rate_limit=30/분)
  │
  ├─ [3] 글로벌 동시성: max_concurrent_requests=4
  │
  ├─ [4] 입력 검증:
  │      ├─ 길이 제한 (max_input_length=10,000)
  │      ├─ Unicode NFC 정규화
  │      ├─ Null byte / ANSI escape 제거
  │      └─ 경로 탐색 방어 (../../etc/passwd 차단)
  │
  └─ [5] 출력 위생처리:
         ├─ 프롬프트 인젝션 패턴 제거
         └─ 모델 출력 이상치 탐지
```

### 핵심 모듈
- `core/security.py`: SecurityManager (인증/레이트리밋/입력검증)
- `core/text_utils.py`: sanitize_model_output, detect_output_anomalies
- `core/engine_context.py`: _strip_prompt_injection (DICL 예시 내 인젝션 방어)

---

## 6. 모듈 의존성 방향

```
telegram_handler (UI)
  └─ engine (오케스트레이션)
       ├─ engine_routing  (라우팅 판정)
       ├─ engine_context  (프롬프트 조립)
       ├─ engine_rag      (RAG + Full-Scan)
       ├─ engine_summary  (요약 파이프라인)
       ├─ engine_tracking (메타데이터/로깅)
       ├─ engine_models   (모델 선택)
       └─ engine_background (비동기 태스크)

공유 레이어:
  ├─ config.py      (Pydantic 설정)
  ├─ constants.py   (공유 상수)
  ├─ enums.py       (RoutingTier)
  ├─ memory.py      (대화 저장)
  └─ llm_protocol.py (LLM 클라이언트 Protocol)
```

의존성은 항상 위→아래 방향. 순환 의존을 방지하기 위해 `TYPE_CHECKING` 가드를 사용한다.
