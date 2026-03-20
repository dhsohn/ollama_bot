# Architecture Notes

## 1. Hierarchical Routing (4-Tier Response Pipeline)

Every user message is evaluated from the highest-priority tier downward to balance
latency and cost. Once a tier matches, lower tiers are skipped.

```
Incoming message
  │
  ├─ [Skill] Trigger keyword match -> call the LLM with a specialized prompt
  │
  ├─ [Instant] Rule-based immediate response (no LLM call, <1 ms)
  │
  ├─ [Cache] Semantic cache similarity lookup (no LLM call, ~10 ms)
  │
  └─ [Full] Context build + RAG + LLM call (highest latency)
```

### Design Goals
- **Lower cost**: Instant and Cache tiers avoid LLM API calls entirely.
- **Fast responses**: Simple questions can complete in milliseconds.
- **Preserved quality**: Complex requests fall through to the Full tier with RAG and optimized context.

### Trade-offs
- A high Cache-tier similarity threshold (`0.92`) lowers hit rate, while a low threshold risks inaccurate reuse.
- Instant rules require manual maintenance in `config/instant_rules.yaml`.
- Skill triggers use string-prefix matching and are less robust to natural-language variation.

### Type Safety
`RoutingTier(StrEnum)` defines the routing tiers so string-comparison mistakes are avoided at compile time.
See `core/enums.py`.

---

## 2. Context Compression

Conversation history is managed so long-running chats stay within token limits.

```
Full conversation history
  │
  ├─ Recent N turns: keep verbatim (`recent_keep=10`)
  │
  └─ Older turns: replace with a background summary
      │
      ├─ Refresh summary every `summary_refresh_interval` turns
      │
      └─ Delete archived data older than `retention_days` (30 days)
```

### Key Modules
- `core/context_compressor.py`: builds summaries and merges history
- `core/engine_context.py`: assembles the system prompt (preferences, guidelines, DICL, language policy)
- `core/memory.py`: SQLite-backed conversation storage

---

## 3. RAG Pipeline

```
Query
  │
  ├─ Check trigger keywords -> decide whether to run RAG
  │
  ├─ Embedding (Qwen3-Embedding-0.6B) -> vector search (`k0=40`)
  │
  ├─ Reranker (bge-reranker-v2-m3) -> keep top `topk=8`
  │
  └─ Context Builder -> inject into the system prompt
```

### Full-Scan Analysis (Map-Reduce)
Used when the system must read and analyze the entire corpus:
1. **Map**: extract evidence from each segment (12K chars, JSON output)
2. **Reduce**: merge and deduplicate evidence (up to 6 passes)
3. **Final**: generate the final answer from the merged evidence

### Key Modules
- `core/rag/chunker.py`: document chunking (500-1200 tokens, 15% overlap)
- `core/rag/retriever.py`: embedding-based vector retrieval
- `core/rag/reranker.py`: cross-encoder reranking
- `core/rag/context_builder.py`: converts retrieval output into prompt-ready context
- `core/engine_rag.py`: RAG orchestration and Full-Scan implementation

---

## 4. Memory and Persistence

### Storage Layout
```
data/
  ├─ memory.db          # conversation history + long-term memory (SQLite, WAL mode)
  ├─ semantic_cache.db  # semantic cache (SQLite)
  ├─ feedback.db        # feedback data (SQLite)
  └─ reports/           # automation reports
```

### Data Management
- **WAL journaling**: allows concurrent reads and writes
- **Atomic transactions**: abstracted behind the `_execute` method
- **Migrations**: schema versioning is handled in `core/db_migrations.py`
- **Retention**: data older than `conversation_retention_days` (30 days) is cleaned up automatically

---

## 5. Security Model

```
Incoming request
  │
  ├─ [1] Authentication: `allowed_users` whitelist (based on `chat_id`)
  │
  ├─ [2] Rate limiting: sliding window (`rate_limit=30/min`)
  │
  ├─ [3] Global concurrency: `max_concurrent_requests=4`
  │
  ├─ [4] Input validation:
  │      ├─ length limit (`max_input_length=10,000`)
  │      ├─ Unicode NFC normalization
  │      ├─ null-byte / ANSI escape removal
  │      └─ path traversal defense (blocks `../../etc/passwd`)
  │
  └─ [5] Output hygiene:
         ├─ remove prompt-injection patterns
         └─ detect anomalous model output
```

### Key Modules
- `core/security.py`: `SecurityManager` for auth, rate limiting, and input validation
- `core/text_utils.py`: `sanitize_model_output`, `detect_output_anomalies`
- `core/engine_context.py`: `_strip_prompt_injection` for injection defense inside DICL examples

---

## 6. Dependency Direction

```
telegram_handler (UI)
  └─ engine (orchestration)
       ├─ engine_routing    (routing decisions)
       ├─ engine_context    (prompt assembly)
       ├─ engine_rag        (RAG + Full-Scan)
       ├─ engine_summary    (summary pipeline)
       ├─ engine_tracking   (metadata / logging)
       ├─ engine_models     (model selection)
       └─ engine_background (async tasks)

Shared layer:
  ├─ config.py       (Pydantic settings)
  ├─ constants.py    (shared constants)
  ├─ enums.py        (`RoutingTier`)
  ├─ memory.py       (conversation storage)
  └─ llm_protocol.py (LLM client protocol)
```

Dependencies always flow from top to bottom. `TYPE_CHECKING` guards are used to prevent circular imports.
