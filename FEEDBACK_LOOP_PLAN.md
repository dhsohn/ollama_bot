# 사용자 피드백 루프 기능 추가 (2차 보완 반영)

## 1) 목표와 범위

현재 `ollama_bot`은 응답 품질에 대한 직접 피드백 수집 루프가 없다.
이번 변경의 목표는 다음 2가지다.

1. 각 봇 응답에 대해 `👍/👎` 피드백을 수집한다.
2. 누적된 피드백을 주기적으로 분석해 `feedback_guidelines`를 생성하고, 다음 대화 시스템 프롬프트에 반영한다.

제약사항: Ollama는 API fine-tuning 기반의 온라인 학습을 제공하지 않으므로, 본 루프는 **피드백 기반 프롬프트 개선** 방식으로 구현한다.

---

## 2) 설계 원칙

1. 사용자별 개선 루프
- 피드백 분석은 `allowed_users` 각각에 대해 수행하고, 가이드라인도 사용자별로 저장한다.

2. 재평가 허용 (upsert)
- 동일 응답 메시지(`chat_id + bot_message_id`)에 대한 재클릭은 기존 평가를 덮어쓴다.
- 재평가를 지원하기 위해 버튼은 제거하지 않고 유지한다.

3. 무소음 자동화
- 분석 결과가 없거나 최소 건수 미달이면 callable은 `""`를 반환한다.
- `AutoScheduler`는 빈 결과를 정상 완료로 처리하므로 불필요한 텔레그램 알림이 발생하지 않는다.

4. 실행 가능한 저장 전략
- 가이드라인은 `long_term_memory(category='feedback_guidelines')`에 저장한다.
- 키는 고정 패턴(`feedback_guideline_01`~)을 사용하고, 저장 전 기존 category를 정리한다.
- 엔진 주입 시 key 정렬을 적용해 가이드라인 순서를 안정화한다.

5. 현재 코드 구조와 일치
- `register_builtin_callables(...)`, `TelegramHandler`, `Engine`, `MemoryManager`의 현재 시그니처를 기준으로 확장한다.

6. 리소스 관리
- 프리뷰 캐시는 TTL + 최대 크기 제한으로 무한 증가를 방지한다.
- `message_feedback` 테이블은 보존 정책을 두어 오래된 데이터를 정리한다.

---

## 3) 수정/생성 파일 목록

| 파일 | 작업 |
|------|------|
| `core/feedback_manager.py` | **신규** — 피드백 저장/조회/통계/보존정책 |
| `core/automation_callables_impl/feedback_analysis.py` | **신규** — 피드백 분석 callable |
| `core/automation_callables_impl/common.py` | 수정 — `FEEDBACK_ANALYSIS_SCHEMA` 추가 |
| `auto/_builtin/feedback_analysis.yaml` | **신규** — 스케줄 정의 |
| `core/config.py` | 수정 — `FeedbackConfig` 추가 및 로드 |
| `config/config.yaml` | 수정 — `feedback` 섹션 추가 |
| `core/memory.py` | 수정 — `db` 프로퍼티 + category 단위 삭제 메서드 추가 |
| `core/telegram_handler.py` | 수정 — 인라인 버튼/콜백 핸들러/`/feedback` 명령 |
| `core/telegram_message_renderer.py` | 수정 — `stream_and_render` 반환 타입 확장 (마지막 Message 반환) |
| `core/engine.py` | 수정 — `feedback_guidelines` 주입 |
| `core/automation_callables.py` | 수정 — `feedback_analysis` 등록 경로 추가 |
| `main.py` | 수정 — `FeedbackManager` 초기화, 의존성 주입, `RuntimeState` 확장 |
| `tests/test_feedback_manager.py` | **신규** — 피드백 매니저 테스트 |
| `tests/test_automation_callables.py` | 수정 — `feedback_analysis` 테스트 추가 |
| `tests/test_telegram.py` | 수정 — 버튼/콜백/`/feedback` 테스트 추가 |
| `tests/test_engine.py` | 수정 — 가이드라인 주입 테스트 추가 |
| `tests/test_config.py` | 수정 — `feedback` 섹션 로드 테스트 추가 |
| `README.md` | 수정 — `/feedback` 명령어 문서화 |

---

## 4) 구현 순서

### Step 1. `core/config.py` / `config/config.yaml`

`FeedbackConfig` 모델 추가:

```python
class FeedbackConfig(BaseModel):
    enabled: bool = True
    show_buttons: bool = True
    min_feedback_for_analysis: int = 5
    max_guidelines: int = 5
    preview_max_chars: int = 300
    preview_cache_max_size: int = 500
    preview_cache_ttl_hours: int = 24
    retention_days: int = 90
```

`AppSettings`에 필드 추가:

```python
feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
```

`load_config()` YAML 오버레이에 추가 (기존 `if "scheduler" in yaml_data:` 블록 다음):

```python
if "feedback" in yaml_data:
    settings.feedback = FeedbackConfig(**yaml_data["feedback"])
```

`config/config.yaml`:

```yaml
feedback:
  enabled: true
  show_buttons: true
  min_feedback_for_analysis: 5
  max_guidelines: 5
  preview_max_chars: 300
  preview_cache_max_size: 500
  preview_cache_ttl_hours: 24
  retention_days: 90
```

### Step 2. `core/memory.py`

`FeedbackManager`와 분석 callable에서 재사용할 최소 확장:

1. `db` 프로퍼티 노출 — `self._db`에 대한 읽기 전용 접근

```python
@property
def db(self) -> aiosqlite.Connection:
    """내부 DB 커넥션을 반환한다 (외부 모듈 공유용)."""
    if self._db is None:
        raise RuntimeError("MemoryManager가 아직 초기화되지 않았습니다.")
    return self._db
```

2. `delete_memories_by_category(chat_id, category)` 메서드 추가

```python
async def delete_memories_by_category(self, chat_id: int, category: str) -> int:
    """지정된 chat_id/category에 해당하는 장기 메모리를 모두 삭제한다."""
    if self._db is None:
        return 0
    cursor = await self._db.execute(
        "DELETE FROM long_term_memory WHERE chat_id = ? AND category = ?",
        (chat_id, category),
    )
    await self._db.commit()
    return cursor.rowcount
```

### Step 3. `core/feedback_manager.py` (신규)

스키마:

```sql
CREATE TABLE IF NOT EXISTS message_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    bot_message_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
    user_message_preview TEXT,
    bot_response_preview TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, bot_message_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_chat_created
    ON message_feedback(chat_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_rating
    ON message_feedback(rating);
```

메서드:

- `initialize_schema()`
- `store_feedback(chat_id, bot_message_id, rating, user_preview=None, bot_preview=None)` (upsert, `is_update: bool` 반환)
- `get_user_stats(chat_id)` → `{total, positive, negative, satisfaction_rate}`
- `get_global_stats()` → `{total, positive, negative, satisfaction_rate}`
- `get_recent_feedback(chat_id, rating, limit)` → `list[dict]`
- `count_feedback(chat_id)` → `int`
- `prune_old_feedback(retention_days: int)` → `int` (삭제 건수 반환)

`prune_old_feedback` 구현:

```python
async def prune_old_feedback(self, retention_days: int) -> int:
    """retention_days보다 오래된 피드백을 삭제한다."""
    cursor = await self._db.execute(
        "DELETE FROM message_feedback WHERE created_at < datetime('now', ?)",
        (f"-{retention_days} days",),
    )
    await self._db.commit()
    return cursor.rowcount
```

### Step 4. `core/telegram_message_renderer.py`

`stream_and_render` 반환 타입 확장 — 현재는 `-> str`만 반환하지만, 버튼 부착 대상 메시지를 특정하기 위해 **마지막 Message 객체도 함께 반환**한다.

변경 전:

```python
async def stream_and_render(...) -> str:
    ...
    return full_response
```

변경 후:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class StreamResult:
    full_response: str
    last_message: Any  # telegram.Message

async def stream_and_render(...) -> StreamResult:
    ...
    # 최종 메시지 분할 전송 루프에서 last_msg 추적
    last_msg = sent_message
    if full_response:
        parts = split_message_fn(full_response)
        for idx, part in enumerate(parts):
            if idx == 0:
                try:
                    await sent_message.edit_text(part)
                    last_msg = sent_message
                except Exception:
                    last_msg = await reply_text(part)
            else:
                last_msg = await reply_text(part)
    return StreamResult(full_response=full_response, last_message=last_msg)
```

> **주의**: `reply_text`의 반환값(`telegram.Message`)을 캡처해야 하므로, 기존에 반환값을 무시하던 부분을 수정한다. `stream_and_render`를 호출하는 **모든 기존 코드**는 `result.full_response`로 접근하도록 업데이트한다 (현재 `_handle_message`에서만 사용하며, 반환값을 캡처하지 않으므로 호환성 이슈 없음).

### Step 5. `core/telegram_handler.py`

주요 변경:

**0. Import 추가**

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler
```

**1. 생성자 확장**

```python
class TelegramHandler:
    def __init__(
        self,
        config: AppSettings,
        engine: Engine,
        security: SecurityManager,
        feedback: FeedbackManager | None = None,  # 추가
    ) -> None:
        ...
        self._feedback = feedback
        # 프리뷰 캐시: {(chat_id, bot_message_id): {"user": str, "bot": str, "ts": float}}
        # TTL + 최대 크기 제한
        self._preview_cache: dict[tuple[int, int], dict] = {}
```

**2. 프리뷰 캐시 관리**

```python
import time

def _cache_preview(self, chat_id: int, bot_message_id: int, user_text: str, bot_text: str) -> None:
    """프리뷰를 캐시에 저장한다. TTL 초과/크기 초과 시 정리."""
    max_chars = self._config.feedback.preview_max_chars
    now = time.monotonic()

    # TTL 만료 항목 정리
    ttl_seconds = self._config.feedback.preview_cache_ttl_hours * 3600
    expired = [k for k, v in self._preview_cache.items() if now - v["ts"] > ttl_seconds]
    for k in expired:
        del self._preview_cache[k]

    # 최대 크기 초과 시 가장 오래된 항목 제거
    max_size = self._config.feedback.preview_cache_max_size
    while len(self._preview_cache) >= max_size:
        oldest_key = min(self._preview_cache, key=lambda k: self._preview_cache[k]["ts"])
        del self._preview_cache[oldest_key]

    self._preview_cache[(chat_id, bot_message_id)] = {
        "user": user_text[:max_chars],
        "bot": bot_text[:max_chars],
        "ts": now,
    }
```

**3. 핸들러 등록 (`initialize` 메서드)**

`feedback`이 활성화된 경우에만 관련 핸들러를 조건부 등록한다:

```python
if self._feedback and self._config.feedback.enabled:
    handlers.append(CommandHandler("feedback", self._cmd_feedback))
    # CallbackQueryHandler는 일반 핸들러 목록 밖에서 별도 등록
    self._app.add_handler(
        CallbackQueryHandler(self._handle_feedback_callback, pattern=r"^fb:")
    )
```

`set_my_commands`에 `/feedback` 추가:

```python
commands = [
    BotCommand("start", "봇 시작"),
    BotCommand("help", "도움말"),
    BotCommand("skills", "스킬 목록"),
    BotCommand("auto", "자동화 관리"),
    BotCommand("model", "모델 관리"),
    BotCommand("memory", "메모리 관리"),
    BotCommand("status", "시스템 상태"),
]
if self._feedback and self._config.feedback.enabled:
    commands.append(BotCommand("feedback", "피드백 통계"))
await self._app.bot.set_my_commands(commands)
```

**4. `_handle_message` 수정 — `stream_and_render` 반환값 캡처 + 버튼 부착**

현재 코드 (`telegram_handler.py:403`):

```python
await stream_and_render(
    stream=self._engine.process_message_stream(chat_id, text),
    sent_message=sent_message,
    ...
)
```

변경 후:

```python
result = await stream_and_render(
    stream=self._engine.process_message_stream(chat_id, text),
    sent_message=sent_message,
    reply_text=update.effective_message.reply_text,
    split_message_fn=self._split_message,
    edit_interval=_EDIT_INTERVAL,
    edit_char_threshold=_EDIT_CHAR_THRESHOLD,
)

# 피드백 버튼 부착
if (
    self._feedback
    and self._config.feedback.enabled
    and self._config.feedback.show_buttons
    and result.last_message
):
    target_msg = result.last_message
    self._cache_preview(chat_id, target_msg.message_id, text, result.full_response)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍", callback_data=f"fb:1:{target_msg.message_id}"),
            InlineKeyboardButton("👎", callback_data=f"fb:-1:{target_msg.message_id}"),
        ]
    ])
    try:
        await target_msg.edit_reply_markup(reply_markup=keyboard)
    except Exception:
        pass  # 편집 실패 시 버튼 없이 진행
```

**5. 콜백 처리**

```python
async def _handle_feedback_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not update.effective_chat:
        return
    if update.effective_chat.type != ChatType.PRIVATE:
        return

    try:
        _, rating_str, msg_id_str = query.data.split(":")
        rating = int(rating_str)
        bot_message_id = int(msg_id_str)
    except (ValueError, AttributeError):
        await query.answer("잘못된 피드백 요청입니다.", show_alert=True)
        return

    chat_id = update.effective_chat.id
    try:
        self._security.authenticate(chat_id)
        self._security.check_rate_limit(chat_id)
    except AuthenticationError:
        return
    except RateLimitError:
        await query.answer("요청이 너무 많습니다. 잠시 후 다시 시도해주세요.", show_alert=True)
        return

    if rating not in (-1, 1):
        await query.answer("지원하지 않는 피드백 값입니다.", show_alert=True)
        return

    preview = self._preview_cache.get((chat_id, bot_message_id), {})
    is_update = await self._feedback.store_feedback(
        chat_id=chat_id,
        bot_message_id=bot_message_id,
        rating=rating,
        user_preview=preview.get("user"),
        bot_preview=preview.get("bot"),
    )

    if is_update:
        await query.answer("피드백을 업데이트했어요.", show_alert=False)
    else:
        await query.answer("피드백 감사합니다!", show_alert=False)
```

**6. `/feedback` 명령**

```python
@_auth_required
async def _cmd_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    stats = await self._feedback.get_user_stats(chat_id)
    text = (
        "📊 <b>피드백 통계</b>\n\n"
        f"전체: {stats['total']}건\n"
        f"👍 긍정: {stats['positive']}건\n"
        f"👎 부정: {stats['negative']}건\n"
        f"만족도: {stats['satisfaction_rate']:.0%}"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)
```

**7. `_cmd_help` 업데이트**

도움말 텍스트에 `/feedback` 추가:

```python
"/feedback — 피드백 통계\n"
```

**8. `_cmd_status` 확장 (선택)**

`/status`에 피드백 수집 현황 추가 — `self._feedback`이 존재할 때:

```python
if self._feedback:
    global_stats = await self._feedback.get_global_stats()
    status_text += (
        f"\n📊 피드백: {global_stats['total']}건 "
        f"(만족도 {global_stats['satisfaction_rate']:.0%})"
    )
```

### Step 6. `core/engine.py`

`_build_context()`에서 preferences 주입 뒤 (현재 168행 이후)에 가이드라인 주입.

**가이드라인 개수는 하드코딩하지 않는다** — 엔진에서 `self._config.feedback.max_guidelines`를 상한으로 사용한다:

```python
# 피드백 기반 가이드라인 주입
guidelines = await self._memory.recall_memory(chat_id, category="feedback_guidelines")
if guidelines:
    max_guides = max(1, self._config.feedback.max_guidelines)
    ordered = sorted(guidelines, key=lambda g: g["key"])
    lines = [f"- {g['value']}" for g in ordered[:max_guides]]
    system += (
        "\n\n[응답 품질 가이드라인]\n"
        "사용자 피드백 기반 권장사항:\n"
        + "\n".join(lines)
    )
```

### Step 7. `core/automation_callables_impl/common.py`

기존 스키마 상수들(`PREFERENCES_SCHEMA`, `TRIAGE_SCHEMA` 등)과 동일한 위치에 `FEEDBACK_ANALYSIS_SCHEMA` 추가:

```python
FEEDBACK_ANALYSIS_SCHEMA: dict = {
    "type": "array",
    "maxItems": 10,
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["avoid", "prefer", "style"],
            },
            "guideline": {"type": "string"},
        },
        "required": ["type", "guideline"],
    },
}
```

### Step 8. `core/automation_callables_impl/feedback_analysis.py` (신규)

`build_*_callable` 팩토리 패턴으로 구현:

```python
from core.automation_callables_impl.common import (
    FEEDBACK_ANALYSIS_SCHEMA,
    parse_json_array,
)

def build_feedback_analysis_callable(engine, memory, feedback, allowed_users, logger):
    async def feedback_analysis(
        min_feedback_count: int = 5,
        max_negative_samples: int = 15,
        max_positive_samples: int = 10,
        max_guidelines: int = 5,
    ) -> str:
        """사용자 피드백을 분석해 응답 품질 가이드라인을 갱신한다."""
        results = []

        for chat_id in allowed_users:
            count = await feedback.count_feedback(chat_id)
            if count < min_feedback_count:
                continue

            negatives = await feedback.get_recent_feedback(chat_id, rating=-1, limit=max_negative_samples)
            positives = await feedback.get_recent_feedback(chat_id, rating=1, limit=max_positive_samples)

            if not negatives and not positives:
                continue

            # LLM에 구조화 출력 요청
            prompt = _build_analysis_prompt(negatives, positives)
            raw = await engine.process_prompt(
                prompt=prompt,
                chat_id=chat_id,
                format=FEEDBACK_ANALYSIS_SCHEMA,
            )

            parsed = parse_json_array(raw)
            if not parsed:
                logger.warning("feedback_analysis_parse_failed", chat_id=chat_id)
                continue

            valid: list[dict] = []
            seen: set[str] = set()
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                t = str(item.get("type", "")).strip().lower()
                g = str(item.get("guideline", "")).strip()
                if t not in {"avoid", "prefer", "style"} or not g:
                    continue
                dedupe_key = f"{t}:{g}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                valid.append({"type": t, "guideline": g})

            if not valid:
                continue

            # 유효 결과가 있을 때만 기존 가이드라인 삭제 후 저장
            await memory.delete_memories_by_category(chat_id, "feedback_guidelines")
            for i, item in enumerate(valid[:max_guidelines]):
                key = f"feedback_guideline_{i+1:02d}"
                value = f"[{item['type']}] {item['guideline']}"
                await memory.store_memory(chat_id, key, value, category="feedback_guidelines")

            results.append(f"chat_id={chat_id}: {len(valid[:max_guidelines])}건 갱신")

        if not results:
            return ""

        return "## 피드백 분석 결과\n\n" + "\n".join(f"- {r}" for r in results)

    return feedback_analysis
```

### Step 9. `auto/_builtin/feedback_analysis.yaml`

```yaml
name: "feedback_analysis"
description: "사용자 피드백을 분석해 응답 품질 가이드라인을 갱신합니다"
version: "1.0"
enabled: true
schedule: "0 2 * * *"
action:
  type: "callable"
  target: "feedback_analysis"
  parameters:
    min_feedback_count: 5
    max_negative_samples: 15
    max_positive_samples: 10
    max_guidelines: 5
output:
  send_to_telegram: true
  save_to_file: "reports/feedback_analysis_{date}.md"
retry:
  max_attempts: 2
  delay_seconds: 60
timeout: 180
```

### Step 10. `core/automation_callables.py`

`register_builtin_callables(...)` 확장:

```python
from core.feedback_manager import FeedbackManager

def register_builtin_callables(
    scheduler,
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    data_dir: str = "data",
    feedback: FeedbackManager | None = None,  # 추가
) -> None:
    ...
    # 기존 등록 코드 유지

    # 피드백 분석 callable은 항상 등록한다.
    # 이유: feedback.enabled=false여도 YAML(auto/_builtin/feedback_analysis.yaml)은 로드될 수 있음.
    # 미등록 상태면 스케줄 실행 시 "Callable not registered" 에러가 발생하므로 no-op fallback 제공.
    if feedback is not None:
        from core.automation_callables_impl.feedback_analysis import build_feedback_analysis_callable
        scheduler.register_callable(
            "feedback_analysis",
            build_feedback_analysis_callable(engine, memory, feedback, allowed_users, logger),
        )
    else:
        async def _feedback_analysis_noop(**kwargs) -> str:
            return ""
        scheduler.register_callable("feedback_analysis", _feedback_analysis_noop)
```

### Step 11. `main.py`

**`_build_runtime()` 수정:**

`memory.initialize()` 직후, `OllamaClient` 초기화 전에 `FeedbackManager`를 생성한다:

```python
from core.feedback_manager import FeedbackManager

# memory.initialize() 이후
feedback: FeedbackManager | None = None
if config.feedback.enabled:
    feedback = FeedbackManager(memory.db)
    await feedback.initialize_schema()
```

의존성 주입:

```python
telegram = TelegramHandler(config, engine, security, feedback=feedback)
register_builtin_callables(scheduler, engine, memory, allowed_users, data_dir, feedback=feedback)
```

**`RuntimeState` 확장:**

```python
@dataclass
class RuntimeState:
    config: AppSettings
    logger: Any
    memory: MemoryManager
    ollama: OllamaClient
    app: Any
    scheduler: AutoScheduler
    skill_count: int
    auto_count: int
    cleanup_stack: AsyncExitStack
    feedback: FeedbackManager | None = None  # 추가
```

**피드백 보존 정책 실행 — 스케줄러 초기화 시 한 번 실행:**

```python
if feedback:
    pruned = await feedback.prune_old_feedback(config.feedback.retention_days)
    if pruned:
        logger.info("feedback_pruned", count=pruned)
```

### Step 12. 테스트

`tests/test_feedback_manager.py` (신규):

- 스키마 초기화 및 테이블 생성 확인
- 신규 저장 / 재평가 upsert (동일 `chat_id + bot_message_id`에 rating 변경)
- 사용자 통계 / 글로벌 통계 계산 정확도
- 최근 긍정/부정 샘플 조회 (limit 동작)
- `prune_old_feedback()` — retention 기간 이전 데이터 삭제 확인

`tests/test_telegram.py`:

- 응답 후 버튼 표시 여부 (`enabled/show_buttons` 조합 4가지)
- `feedback=None`일 때 콜백 핸들러/명령어 미등록 확인
- 콜백 파싱 정상/실패 케이스
- 콜백의 인증/레이트리밋/유효 rating 검증 케이스
- 재평가(동일 메시지 재클릭) 동작
- `/feedback` 응답 포맷
- 분할 메시지 시 마지막 메시지에 버튼 부착 확인
- 프리뷰 캐시 TTL 만료 및 최대 크기 제한 동작

`tests/test_engine.py`:

- `feedback_guidelines` 존재 시 시스템 프롬프트에 주입되는지 확인
- `feedback_guidelines` 없을 시 시스템 프롬프트 변화 없음 확인

`tests/test_automation_callables.py`:

- 최소 피드백 미달 시 skip (`""`)
- JSON 파싱 실패 graceful 처리 (빈 문자열 반환)
- 가이드라인 저장/교체 (기존 category 삭제 후 재저장)
- `max_guidelines` 초과 시 잘림 확인
- `parse_json_array` 재사용 확인

`tests/test_config.py`:

- YAML의 `feedback` 섹션 로드 및 기본값 확인
- 각 필드 오버라이드 동작 확인

`tests/test_telegram_message_renderer.py`:

- `stream_and_render`가 `StreamResult`를 반환하는지 확인
- 분할 메시지 시 `last_message`가 마지막 파트의 Message 객체인지 확인
- 단일 메시지 시 `last_message`가 `sent_message`와 동일한지 확인

---

## 5) 데이터 흐름

```text
사용자 메시지
  -> 봇 스트리밍 응답 (stream_and_render → StreamResult)
  -> 마지막 메시지에 [👍][👎] 인라인 버튼 부착
  -> 프리뷰(user_text, bot_text)를 TTL 캐시에 저장
  -> 사용자 클릭 시 message_feedback upsert (캐시에서 프리뷰 조회)
  -> 정기 자동화(feedback_analysis, 매일 02:00) 실행
  -> 사용자별 부정/긍정 샘플 조회 → LLM 분석 → JSON 파싱
  -> 기존 feedback_guidelines 삭제 후 새 가이드라인 저장
  -> 다음 대화 _build_context()에 가이드라인 주입
```

```text
[보존 정책]
봇 시작 시 → prune_old_feedback(retention_days) 실행
프리뷰 캐시 → TTL(24h) + 최대 크기(500) 자동 정리
```

---

## 6) 수동 검증 시나리오

1. `python3 -m pytest tests/ -v`
2. 텔레그램에서 일반 대화 후 **마지막 메시지**에 버튼 표시 확인
3. 긴 응답(분할 메시지)에서도 마지막 파트에만 버튼이 붙는지 확인
4. `👍` 클릭 후 `/feedback`에서 통계 증가 확인
5. 동일 메시지에서 `👎` 재클릭 후 통계 반영(재평가) 확인
6. `/status`에 피드백 현황이 표시되는지 확인
7. `config.yaml`에서 `feedback.enabled: false` 설정 후 버튼/명령어 미표시 확인
8. `/auto list`에서 `feedback_analysis` 로드 확인
9. 피드백 누적 후 수동 실행 시 `feedback_guidelines`가 장기 메모리에 갱신되는지 확인
10. 봇 재시작 후 `prune_old_feedback` 로그 확인

---

## 7) 알려진 제한사항

1. 프리뷰 캐시는 프로세스 메모리 기반이므로 재시작 시 유실된다. 재시작 이전에 수집되었으나 아직 클릭되지 않은 버튼의 프리뷰는 `None`으로 저장된다 (피드백 자체는 정상 기록).
2. 버튼은 텔레그램 메시지 단위로 수집되므로, 사용자의 추가 코멘트형 자유서술 피드백은 본 범위에 포함하지 않는다.
3. `stream_and_render` 반환 타입 변경(`str` → `StreamResult`)은 해당 함수를 호출하는 모든 코드에 영향을 준다. 호출부를 함께 수정하지 않으면 런타임 오류가 발생할 수 있다.
4. 텔레그램 `callback_data`는 최대 64바이트 제한이 있다. `fb:{rating}:{message_id}` 형식은 일반적으로 20바이트 이내이므로 문제없으나, 극단적으로 큰 `message_id`가 발생할 경우를 대비해 길이 검증을 추가할 수 있다.
5. `feedback.enabled=false`에서도 `feedback_analysis` YAML이 로드될 수 있으므로, callable 미등록 오류를 막기 위해 no-op callable 등록이 필요하다.
