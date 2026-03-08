"""Full-tier response planner helpers.

로컬 LLM이 바로 최종 장문 답변을 생성할 때 품질이 흔들리는 문제를 줄이기 위해,
최종 생성 전에 짧은 JSON 설계안을 만들고 시스템 프롬프트에 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.text_utils import sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine
    from core.intent_router import ContextStrategy

_PLANNER_QUERY_MAX_CHARS = 4_000
_PLANNER_RAG_HINT_MAX_CHARS = 800
_ALLOWED_RESPONSE_MODES = frozenset({
    "direct",
    "structured",
    "step_by_step",
    "comparison",
    "evidence_first",
})
_ALLOWED_BREVITY = frozenset({"short", "medium", "long"})
_RESPONSE_MODE_LABELS = {
    "direct": "핵심 우선",
    "structured": "구조화",
    "step_by_step": "단계형",
    "comparison": "비교형",
    "evidence_first": "근거 우선",
}
_BREVITY_LABELS = {
    "short": "짧게",
    "medium": "중간 길이",
    "long": "상세하게",
}
_PLANNER_SYSTEM_PROMPT = """
당신은 로컬 LLM의 최종 답변 구조를 설계하는 planner입니다.
최종 답변 본문을 작성하지 말고, 반드시 JSON object만 반환하세요.

허용 키:
- response_mode: "direct" | "structured" | "step_by_step" | "comparison" | "evidence_first"
- brevity: "short" | "medium" | "long"
- use_bullets: boolean
- sections: string[]
- must_cover: string[]
- suggest_next_step: boolean

규칙:
- 사실을 새로 만들지 마세요.
- 질문을 다시 쓰거나 장문 설명을 출력하지 마세요.
- sections/must_cover는 짧은 한국어 구문으로 작성하세요.
- 섹션은 최대 4개, must_cover는 최대 4개만 추천하세요.
""".strip()


@dataclass(frozen=True)
class ResponsePlan:
    """Planner가 반환한 응답 설계."""

    response_mode: str
    brevity: str
    use_bullets: bool
    sections: tuple[str, ...]
    must_cover: tuple[str, ...]
    suggest_next_step: bool


def should_plan_response(
    engine: Engine,
    *,
    text: str,
    intent: str | None,
    rag_used: bool,
    images: list[bytes] | None,
) -> bool:
    """현재 요청에 planner를 적용할지 결정한다."""
    cfg = engine._config.response_planner
    if not cfg.enabled or images:
        return False

    normalized = text.strip()
    if not normalized:
        return False

    if rag_used and cfg.force_for_rag:
        return True

    intent_key = (intent or "").strip().lower()
    trigger_intents = {
        item.strip().lower()
        for item in cfg.trigger_intents
        if isinstance(item, str) and item.strip()
    }
    if intent_key and intent_key in trigger_intents:
        return True

    return len(normalized) >= cfg.min_input_chars


async def maybe_apply_response_plan(
    engine: Engine,
    *,
    chat_id: int,
    text: str,
    intent: str | None,
    strategy: ContextStrategy | None,
    messages: list[dict[str, str]],
    rag_result: Any,
    target_model: str | None,
    timeout: int,
    images: list[bytes] | None,
) -> tuple[list[dict[str, str]], bool]:
    """필요 시 planner를 호출해 최종 응답 프롬프트를 보강한다."""
    rag_used = bool(getattr(rag_result, "contexts", None))
    if not should_plan_response(
        engine,
        text=text,
        intent=intent,
        rag_used=rag_used,
        images=images,
    ):
        return messages, False

    planner_messages = _build_planner_messages(
        text=text,
        intent=intent,
        strategy=strategy,
        rag_result=rag_result,
    )
    planner_cfg = engine._config.response_planner
    planner_timeout = max(10, min(int(timeout), planner_cfg.timeout_seconds))

    try:
        chat_response = await engine._llm_client.chat(
            messages=planner_messages,
            model=target_model,
            timeout=planner_timeout,
            max_tokens=planner_cfg.max_plan_tokens,
            response_format="json",
        )
    except Exception as exc:
        engine._logger.debug(
            "response_planner_failed",
            chat_id=chat_id,
            error=str(exc),
        )
        return messages, False

    plan = _parse_plan_payload(engine, chat_response.content, rag_used=rag_used)
    if plan is None:
        engine._logger.debug("response_planner_invalid_payload", chat_id=chat_id)
        return messages, False

    engine._logger.debug(
        "response_planner_applied",
        chat_id=chat_id,
        mode=plan.response_mode,
        brevity=plan.brevity,
        sections=len(plan.sections),
        must_cover=len(plan.must_cover),
    )
    return inject_response_plan(messages, plan), True


def inject_response_plan(
    messages: list[dict[str, str]],
    plan: ResponsePlan,
) -> list[dict[str, str]]:
    """Planner 설계안을 시스템 프롬프트에 주입한다."""
    scaffold = render_response_plan(plan)
    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {
            "role": "system",
            "content": result[0]["content"].rstrip() + "\n\n" + scaffold,
        }
        return result
    return [{"role": "system", "content": scaffold}, *result]


def render_response_plan(plan: ResponsePlan) -> str:
    """최종 프롬프트에 넣을 응답 설계 지시문을 렌더링한다."""
    lines = [
        "[응답 설계안]",
        "아래 설계는 내부 힌트입니다. 설계안이나 JSON을 그대로 노출하지 말고 자연스러운 최종 답변만 작성하세요.",
        f"- 답변 모드: {_RESPONSE_MODE_LABELS.get(plan.response_mode, plan.response_mode)}",
        f"- 권장 길이: {_BREVITY_LABELS.get(plan.brevity, plan.brevity)}",
        f"- 불릿 사용: {'권장' if plan.use_bullets else '필요 시만'}",
    ]
    if plan.sections:
        lines.append("- 권장 섹션 순서: " + " -> ".join(plan.sections))
    if plan.must_cover:
        lines.append("- 반드시 다룰 항목: " + "; ".join(plan.must_cover))
    if plan.suggest_next_step:
        lines.append("- 마지막에 실행 가능한 다음 단계 1개를 제안하세요.")
    return "\n".join(lines)


def _build_planner_messages(
    *,
    text: str,
    intent: str | None,
    strategy: ContextStrategy | None,
    rag_result: Any,
) -> list[dict[str, str]]:
    truncated_query = sanitize_model_output(text).strip()[:_PLANNER_QUERY_MAX_CHARS]
    rag_hint = _build_rag_hint(rag_result)
    meta_lines = [
        f"- intent: {(intent or 'unknown').strip() or 'unknown'}",
        f"- max_tokens: {strategy.max_tokens if strategy and strategy.max_tokens else 'default'}",
        f"- include_preferences: {strategy.include_preferences if strategy else True}",
        f"- include_dicl: {strategy.include_dicl if strategy else True}",
        f"- rag_available: {'yes' if rag_hint else 'no'}",
    ]
    user_content = (
        "[사용자 질문]\n"
        f"{truncated_query}\n\n"
        "[메타]\n"
        f"{chr(10).join(meta_lines)}"
    )
    if rag_hint:
        user_content += f"\n\n[문서 근거 힌트]\n{rag_hint}"

    return [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_rag_hint(rag_result: Any) -> str:
    contexts = getattr(rag_result, "contexts", None)
    if not contexts:
        return ""
    first_context = sanitize_model_output(str(contexts[0])).strip()
    if len(first_context) <= _PLANNER_RAG_HINT_MAX_CHARS:
        return first_context
    return first_context[:_PLANNER_RAG_HINT_MAX_CHARS].rstrip() + "..."


def _parse_plan_payload(
    engine: Engine,
    raw_text: str,
    *,
    rag_used: bool,
) -> ResponsePlan | None:
    payload = engine._extract_json_payload(raw_text)
    if payload is None:
        return None

    cfg = engine._config.response_planner
    response_mode = _normalize_choice(
        payload.get("response_mode"),
        allowed=_ALLOWED_RESPONSE_MODES,
        default="evidence_first" if rag_used else "structured",
    )
    brevity = _normalize_choice(
        payload.get("brevity"),
        allowed=_ALLOWED_BREVITY,
        default="medium",
    )
    use_bullets = _coerce_bool(payload.get("use_bullets"))
    sections = _normalize_string_list(payload.get("sections"), limit=cfg.max_sections)
    must_cover = _normalize_string_list(payload.get("must_cover"), limit=cfg.max_must_cover)
    suggest_next_step = _coerce_bool(payload.get("suggest_next_step"))

    if not sections:
        sections = _default_sections(
            response_mode=response_mode,
            rag_used=rag_used,
            suggest_next_step=suggest_next_step,
        )

    return ResponsePlan(
        response_mode=response_mode,
        brevity=brevity,
        use_bullets=use_bullets,
        sections=sections,
        must_cover=must_cover,
        suggest_next_step=suggest_next_step,
    )


def _normalize_choice(value: Any, *, allowed: set[str] | frozenset[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return default


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
        text = sanitize_model_output(str(candidate or "")).strip()
        if not text:
            continue
        compact = " ".join(text.split())
        if compact in seen:
            continue
        seen.add(compact)
        items.append(compact[:80])
        if len(items) >= limit:
            break
    return tuple(items)


def _default_sections(
    *,
    response_mode: str,
    rag_used: bool,
    suggest_next_step: bool,
) -> tuple[str, ...]:
    sections: list[str]
    if response_mode == "step_by_step":
        sections = ["핵심 답변", "단계별 설명"]
    elif response_mode == "comparison":
        sections = ["핵심 차이", "장단점"]
    elif response_mode == "evidence_first" or rag_used:
        sections = ["핵심 답변", "근거", "해석"]
    elif response_mode == "direct":
        sections = ["핵심 답변"]
    else:
        sections = ["핵심 답변", "설명"]

    if suggest_next_step:
        sections.append("다음 단계")
    return tuple(sections[:4])


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
