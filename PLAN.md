너는 Lemonade Server(OpenAI 호환 로컬 LLM 서버)를 대상으로 “모델 라우팅 + RAG + Reranker” 시스템을 구현하는 시니어 엔지니어다.

[환경]
- Lemonade Server가 로컬에서 실행 중이며 base_url은 http://localhost:11434/api/v1 (또는 사용자가 설정한 포트) 이다.
- Lemonade Server는 OpenAI 호환 API를 제공한다. /api/v1/chat/completions 를 사용해 모델을 호출하며, 요청 body의 "model" 필드로 실행할 모델을 지정한다.
- (중요) 모델 런타임은 내부적으로 llama.cpp(GGUF) + Ryzen AI SW(NPU) 등이 섞여 있을 수 있지만, 클라이언트(우리 라우터)는 이를 신경쓰지 않고 단일 OpenAI 호환 API로만 호출한다.
- 타임존은 Asia/Seoul.

[모델(라우팅 대상)]
- router embedding: Qwen3-Embedding-0.6B-GGUF (임베딩 전용)
- reranker: bge-reranker-v2-m3-GGUF (cross-encoder reranker; query-passage scoring)
- vision: Qwen3-VL-8B-Instruct-GGUF
- low-cost: GLM-4.7-Flash-GGUF
- reasoning: Qwen3-14B-Hybrid (Ryzen AI LLM / NPU)
- coding: Qwen3-Coder-Next-GGUF

[목표]
입력(텍스트/이미지/메타데이터) + 로컬 지식베이스(문서/노트/코드/논문 등)를 활용하여:
1) 필요 시 RAG로 관련 근거 컨텍스트를 검색/정리하고(citations 포함),
2) 규칙/임베딩 기반 라우팅으로 적절한 생성 모델을 선택하여,
3) lemonade-server의 OpenAI 호환 API로 호출한 뒤,
4) 답변 + 라우팅결정 + RAG trace(검색/리랭크/근거)를 반환한다.

========================
A. RAG(인덱싱/검색/리랭크)
========================

[A-0. 코퍼스/인덱스]
- 코퍼스는 로컬 폴더(예: ./kb 또는 metadata로 지정)에서 로딩한다.
- 최소 지원 확장자: .md .txt .pdf(가능하면) .docx .html .json .csv .py .js 등
- 파이프라인:
  1) 문서 로딩 및 정규화(제목/헤더/코드블록 보존)
  2) 청킹(chunking)
     - 기본 chunk: 500~1200 tokens(또는 1~3k chars)
     - overlap: 10~20%
     - 코드 파일은 함수/클래스 단위 우선, 불가하면 라인 기반 chunk
  3) chunk 임베딩 생성(embedding 모델 사용)
  4) 벡터 인덱스 구축(로컬): FAISS/HNSWlib/SQLite vector 등 택1(구현 난이도/의존성 고려)
  5) 메타데이터 저장:
     - doc_id, source_path, chunk_id, section_title, content_hash, mtime, tokens_estimate
- 인덱스는 디스크에 저장하고 재시작 시 로드한다.
- 증분 인덱싱: content_hash 또는 mtime 기반으로 변경된 문서만 재임베딩한다.

[A-1. Retrieval(1차 검색)]
- retrieve(query, k0):
  1) query 임베딩 생성(embedding 모델)
  2) 벡터 검색으로 top-k0 후보 반환 (권장 k0=30~80)
  3) (선택) BM25 키워드 검색과 병합(하이브리드) 가능
  4) 동일 문서 내 과도한 인접 chunk는 중복 제거(최대 1~2개만 남김)
  5) 후보 리스트(candidates[]) 반환: {chunk_text, score, metadata}

[A-2. Rerank(2차 리랭크; bge-reranker-v2-m3)]
- rerank(query, candidates, k):
  - bge-reranker-v2-m3-GGUF로 (query, chunk_text) 쌍을 점수화하여 내림차순 정렬
  - 최종 top-k context만 선택 (권장 k=5~12)
  - 운영 정책(기본값):
    - rerank_enabled=true
    - rerank_topk0=40
    - rerank_topk=8
    - rerank_budget_ms=예: 800~1500ms (환경에 맞게)
    - retrieval_score_floor: 1차 score가 너무 낮으면 rerank 스킵
- 중요: reranker 호출은 가능한 한 “점수 반환” 형태의 엔드포인트를 사용한다.
  - 우선순위:
    1) (있다면) POST {base_url}/rerank 같은 전용 엔드포인트
    2) (없다면) chat/completions를 이용한 우회(비추, 마지막 수단): “각 후보에 대해 0~1 점수만 출력” 형태로 강제
  - lemonade-server에서 reranker를 호출하는 정확한 방법이 다를 수 있으므로, 구현 시 endpoint capability probe를 수행한다.

[A-3. Context packing + citations]
- build_context(contexts):
  - 각 context에 citation 키를 부여: [#1], [#2]...
  - 답변에서 문서 기반 주장에는 해당 키를 인용하도록 유도한다.
  - 컨텍스트가 부족하면 “근거 부족”을 명시하고, 일반지식/추론은 가정임을 표현한다.

[A-4. RAG 트리거 정책]
- 아래 중 하나면 RAG 수행:
  - metadata.use_rag == true
  - “내 문서/프로젝트/레포/노트/폴더/논문/결과/출처/인용/어디에 적혀” 류 요청
  - 특정 파일명/경로/키워드가 포함되어 로컬 검색이 유효해 보임
- 이미지 단독(텍스트 질문 없음)이면 RAG 생략 가능.
- 이미지+텍스트 질의면 텍스트로 retrieve 후 vision prompt에 contexts를 함께 주입 가능.

========================
B. 라우팅 규칙(필수)
========================

0) 이미지가 포함되면 -> vision 모델
1) 텍스트만이면:
   1-1) 규칙 기반 code detection (``` 코드블록, 에러/스택트레이스/디버그/리팩토링/테스트/빌드/배포 키워드, 파일 확장자/파일명 패턴 등)
        -> coding 모델
   1-2) 그 외는 semantic routing으로 cheap vs reasoning 선택:
        - embedding 모델로 입력 임베딩 생성
        - anchors(CHEAP_TEXT, REASONING_TEXT) 임베딩과 cosine similarity 비교
        - top1_score >= THRESHOLD AND (top1_score-top2_score) >= MARGIN이면 확정
        - 아니면 low-cost 모델(GLM Flash)에게 “CHEAP_TEXT 또는 REASONING_TEXT 중 하나만 출력”하게 1줄 분류 후 확정
   1-3) CHEAP_TEXT -> low-cost 모델
        REASONING_TEXT -> reasoning 모델

- (중요) RAG는 “라우팅 전/후” 둘 다 가능하나, 기본 구현은:
  1) 먼저 라우팅(vision/coding/cheap/reasoning) 결정
  2) RAG 트리거가 켜지면 retrieval + rerank 수행
  3) 선택된 생성 모델에 contexts를 주입하여 답변 생성

========================
C. API 호출 요구사항
========================

[모델 목록/가용성 확인]
- GET {base_url}/models 를 구현(있으면 사용)하고, 시작 시 등록된 모델명을 검증한다.
- 필수 모델: embedding, vision, low-cost, reasoning, coding, reranker
- 일부 누락 시:
  - reranker 누락: rerank 스킵(검색 상위만 사용)
  - embedding 누락: semantic routing과 RAG 둘 다 제한 → 규칙 기반으로만 최소 동작(가능하면 강하게 경고)
  - 특정 생성 모델 누락: fallback 규칙에 따라 대체

[Chat 생성]
- POST {base_url}/chat/completions
  - body: { "model": "<model_name>", "messages": [...], "stream": false, ... }

[Embeddings]
- POST {base_url}/embeddings (가능하면 사용. 아니면 lemonade 문서에 맞는 임베딩 엔드포인트로)
  - body 예: { "model": "Qwen3-Embedding-0.6B-GGUF", "input": ["..."] }

[Rerank]
- 가능한 경우 전용 rerank endpoint 사용(있으면 우선):
  - POST {base_url}/rerank
    - body 예: { "model": "bge-reranker-v2-m3-GGUF", "query": "...", "documents": ["doc1", "doc2", ...] }
- 전용 endpoint가 없으면, “chat 기반 점수화” 우회를 마지막 수단으로 구현하되:
  - 반드시 짧은 출력만 허용(숫자/라벨만), timeout 엄격, 실패하면 rerank 스킵

[Timeout/Retry/Fallback]
- 모든 호출은 timeout, 1회 retry, 실패 시 fallback 정책:
  - cheap -> reasoning 또는 reasoning -> cheap
  - 단, image는 vision 유지, coding은 coding 유지 우선
- RAG 단계 실패 시:
  - retrieval 실패: RAG 미사용으로 답변
  - rerank 실패: retrieval 상위 k만 사용

========================
D. Anchors
========================
- CHEAP_TEXT / REASONING_TEXT 각각 12~30개의 한국어 anchor 문장을 JSON/YAML로 분리 저장
- 시작 시 anchor 임베딩을 미리 계산하여 메모리에 올리거나, 파일 캐시로 저장
- THRESHOLD, MARGIN은 config로 노출 (예: THRESHOLD=0.35~0.55, MARGIN=0.03~0.08 범위에서 튜닝)

========================
E. 캐시/로그
========================
- embedding 결과는 LRU 캐시(예: 2000~10000개)로 저장
- 라우팅/점수/선택 모델/재시도/폴백 여부를 JSON 로그로 남긴다.
- 추가 RAG 로그(rag_trace):
  - rag_used(bool), rerank_used(bool)
  - retrieve_k0, rerank_k
  - retrieved_items: 상위 N개 {doc_id, source_path, chunk_id, retrieval_score, rerank_score}
  - context_tokens_estimate
  - citations_keys_used: 답변에 포함된 [#id] 목록(가능하면)

========================
F. 인터페이스/산출물
========================
- route_request(text, images=None, metadata=None) -> routing_decision
- retrieve(text, metadata=None) -> {candidates, contexts, rag_trace_partial}
- generate(text, images=None, metadata=None) -> {answer, routing_decision, rag_trace}
- CLI:
  - chat: 실제 대화
  - dry-run: 라우팅 + retrieval/rerank 결과/점수만 출력
  - test: 테스트 케이스 실행 및 리포트 출력

[테스트]
- 최소 30개 한국어 테스트 케이스(vision 5, coding 10, cheap 10, reasoning 5)
- + RAG 전용 10개 추가(문서 근거가 있어야 성능이 좋아지는 질문)
- 각 테스트는 기대 라우팅 결과(모델 타입)와, RAG 사용 여부 기대값(rag_used)을 포함한다.

[출력 형식]
- generate()는 반드시:
  - answer(텍스트)
  - routing_decision(선택 모델/근거 점수/규칙 트리거)
  - rag_trace(검색/리랭크/근거 메타데이터)
를 함께 반환한다.