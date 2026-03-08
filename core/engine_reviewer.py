"""Full-tier response review and rewrite helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.text_utils import detect_output_anomalies, sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine

_REVIEWER_QUERY_MAX_CHARS = 4_000
_REVIEWER_DRAFT_MAX_CHARS = 8_000
_REVIEWER_SYSTEM_PROMPT = """
당신은 로컬 LLM 최종 답변의 검수기입니다.
사용자 질문과 draft answer를 보고 품질을 판정한 뒤, 반드시 JSON object만 반환하세요.

허용 키:
- pass: boolean
- issues: string[]
- rewrite_needed: boolean
- revised_answer: string

규칙:
- draft가 충분히 좋으면 pass=true, rewrite_needed=false, revised_answer="" 로 반환하세요.
- 문제가 있으면 revised_answer에 고친 최종 답변만 넣으세요.
- revised_answer에는 내부 사고, 정책 메모, JSON 설명을 넣지 마세요.
- RAG citation([#1] 등)이 있으면 가능한 유지하세요.
- 사실을 새로 만들지 마세요.
""".strip()
_BLOCKING_ANOMALY_REASONS = frozenset({
    "empty_after_sanitize",
    "repeated_assignment_pattern",
    "repeated_word_run",
    "repeated_char_run",
    "dominant_repeated_token",
    "low_token_diversity",
})
_LOW_QUALITY_FALLBACK = (
    "방금 답변 생성이 비정상적으로 깨져 제대로 답하지 못했습니다. "
    "같은 메시지를 한 번 더 보내주시면 다시 답하겠습니다."
)


@dataclass(frozen=True)
class ResponseReview:
    """Reviewer가 반환한 품질 판정."""

    passed: bool
    rewrite_needed: bool
    issues: tuple[str, ...]
    revised_answer: str


def should_review_response(
    engine: Engine,
    *,
    text: str,
    images: list[bytes] | None,
    planner_applied: bool,
    anomaly_reasons: list[str] | None = None,
) -> bool:
    """현재 full-tier 응답에 reviewer를 붙일지 결정한다."""
    cfg = engine._config.response_reviewer
    if not cfg.enabled or images:
        return False
    if not text.strip():
        return False
    if has_blocking_anomaly(anomaly_reasons):
        return True
    return not cfg.only_when_planner_used or planner_applied


async def maybe_review_response(
    engine: Engine,
    *,
    chat_id: int,
    text: str,
    response: str,
    raw_response: str,
    intent: str | None,
    target_model: str | None,
    timeout: int,
    planner_applied: bool,
    rag_used: bool,
    images: list[bytes] | None,
    anomaly_reasons: list[str] | None = None,
) -> str:
    """필요 시 draft answer를 검수하고 rewrite를 적용한다."""
    blocking_anomaly = has_blocking_anomaly(anomaly_reasons)
    if not should_review_response(
        engine,
        text=text,
        images=images,
        planner_applied=planner_applied,
        anomaly_reasons=anomaly_reasons,
    ):
        if blocking_anomaly:
            return _fallback_for_blocking_anomaly(
                engine,
                chat_id=chat_id,
                reasons=anomaly_reasons or [],
                stage="review_skipped",
            )
        return response

    cfg = engine._config.response_reviewer
    review_timeout = max(10, min(int(timeout), cfg.timeout_seconds))
    review_messages = _build_review_messages(
        text=text,
        response=response,
        intent=intent,
        planner_applied=planner_applied,
        rag_used=rag_used,
        anomaly_reasons=anomaly_reasons or [],
    )

    try:
        chat_response = await engine._llm_client.chat(
            messages=review_messages,
            model=target_model,
            timeout=review_timeout,
            max_tokens=cfg.max_review_tokens,
            response_format="json",
        )
    except Exception as exc:
        engine._logger.debug(
            "response_reviewer_failed",
            chat_id=chat_id,
            error=str(exc),
        )
        if blocking_anomaly:
            return _fallback_for_blocking_anomaly(
                engine,
                chat_id=chat_id,
                reasons=anomaly_reasons or [],
                stage="review_failed",
            )
        return response

    review = _parse_review_payload(engine, chat_response.content)
    if review is None:
        engine._logger.debug("response_reviewer_invalid_payload", chat_id=chat_id)
        if blocking_anomaly:
            return _fallback_for_blocking_anomaly(
                engine,
                chat_id=chat_id,
                reasons=anomaly_reasons or [],
                stage="review_invalid_payload",
            )
        return response

    await _record_review_observation(
        engine,
        chat_id=chat_id,
        intent=intent,
        review=review,
        planner_applied=planner_applied,
        rag_used=rag_used,
    )

    final_response = response
    if not review.rewrite_needed or not review.revised_answer:
        engine._logger.debug(
            "response_reviewer_passed",
            chat_id=chat_id,
            issues=list(review.issues) or None,
        )
    else:
        revised = sanitize_model_output(review.revised_answer).strip()
        if not revised:
            engine._logger.debug("response_reviewer_empty_rewrite", chat_id=chat_id)
        else:
            revised_anomalies = detect_output_anomalies(review.revised_answer, revised)
            if revised_anomalies:
                engine._logger.warning(
                    "response_reviewer_rewrite_rejected",
                    chat_id=chat_id,
                    reasons=revised_anomalies,
                )
            else:
                engine._logger.info(
                    "response_reviewer_rewritten",
                    chat_id=chat_id,
                    issues=list(review.issues) or None,
                )
                final_response = revised

    if blocking_anomaly:
        final_anomalies = detect_output_anomalies(final_response, final_response)
        if has_blocking_anomaly(final_anomalies):
            return _fallback_for_blocking_anomaly(
                engine,
                chat_id=chat_id,
                reasons=final_anomalies,
                stage="review_persistent_anomaly",
            )
    return final_response


def has_blocking_anomaly(anomaly_reasons: list[str] | tuple[str, ...] | None) -> bool:
    """사용자에게 그대로 노출하면 안 되는 붕괴성 응답인지 판단한다."""
    if not anomaly_reasons:
        return False
    return any(reason in _BLOCKING_ANOMALY_REASONS for reason in anomaly_reasons)


def _fallback_for_blocking_anomaly(
    engine: Engine,
    *,
    chat_id: int,
    reasons: list[str] | tuple[str, ...],
    stage: str,
) -> str:
    engine._logger.warning(
        "response_quality_fallback_applied",
        chat_id=chat_id,
        reasons=list(reasons) or None,
        stage=stage,
    )
    return _LOW_QUALITY_FALLBACK


async def _record_review_observation(
    engine: Engine,
    *,
    chat_id: int,
    intent: str | None,
    review: ResponseReview,
    planner_applied: bool,
    rag_used: bool,
) -> None:
    engine._logger.info(
        "response_review_recorded",
        chat_id=chat_id,
        intent=intent,
        rewritten=review.rewrite_needed,
        issues=list(review.issues) or None,
        planner_applied=planner_applied,
        rag_used=rag_used,
    )
    feedback_manager = getattr(engine, "_feedback_manager", None)
    if feedback_manager is None:
        return

    store_review_result = getattr(feedback_manager, "store_review_result", None)
    if not callable(store_review_result):
        return
    try:
        await store_review_result(
            chat_id,
            intent=intent,
            rewritten=review.rewrite_needed,
            issues=review.issues,
            planner_applied=planner_applied,
            rag_used=rag_used,
        )
    except Exception as exc:
        engine._logger.warning(
            "response_review_persist_failed",
            chat_id=chat_id,
            error=str(exc),
        )


def _build_review_messages(
    *,
    text: str,
    response: str,
    intent: str | None,
    planner_applied: bool,
    rag_used: bool,
    anomaly_reasons: list[str],
) -> list[dict[str, str]]:
    question = sanitize_model_output(text).strip()[:_REVIEWER_QUERY_MAX_CHARS]
    draft = sanitize_model_output(response).strip()[:_REVIEWER_DRAFT_MAX_CHARS]
    anomaly_text = ", ".join(anomaly_reasons) if anomaly_reasons else "none"
    user_content = (
        "[user_question]\n"
        f"{question}\n\n"
        "[draft_answer]\n"
        f"{draft}\n\n"
        "[review_signals]\n"
        f"- intent: {(intent or 'unknown').strip() or 'unknown'}\n"
        f"- planner_applied: {'yes' if planner_applied else 'no'}\n"
        f"- rag_used: {'yes' if rag_used else 'no'}\n"
        f"- anomaly_reasons: {anomaly_text}"
    )
    return [
        {"role": "system", "content": _REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_review_payload(engine: Engine, raw_text: str) -> ResponseReview | None:
    payload = engine._extract_json_payload(raw_text)
    if payload is None:
        return None

    issues = _normalize_string_list(payload.get("issues"), limit=5)
    revised_answer = sanitize_model_output(str(payload.get("revised_answer") or "")).strip()
    rewrite_needed = _coerce_bool(payload.get("rewrite_needed"))
    passed = _coerce_bool(payload.get("pass"))
    if revised_answer and not rewrite_needed:
        rewrite_needed = True

    return ResponseReview(
        passed=passed,
        rewrite_needed=rewrite_needed,
        issues=issues,
        revised_answer=revised_answer,
    )


def _normalize_string_list(value: Any, *, limit: int) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        return ()

    items: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = sanitize_model_output(str(candidate or "")).strip()
        if not normalized:
            continue
        compact = " ".join(normalized.split())
        if compact in seen:
            continue
        seen.add(compact)
        items.append(compact[:120])
        if len(items) >= limit:
            break
    return tuple(items)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
