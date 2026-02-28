"""피드백 분석 자동화 callable."""

from __future__ import annotations

from typing import Any

from core.automation_callables_impl.common import (
    FEEDBACK_ANALYSIS_SCHEMA,
    parse_json_array,
)


def _build_analysis_prompt(
    negatives: list[dict],
    positives: list[dict],
    auto_low: list[dict] | None = None,
) -> str:
    """LLM 분석용 프롬프트를 생성한다."""
    parts: list[str] = [
        "아래는 사용자가 봇 응답에 남긴 피드백 데이터이다.\n"
        "부정(👎) 피드백과 긍정(👍) 피드백을 분석하여 "
        "봇이 앞으로 개선해야 할 가이드라인을 JSON 배열로 작성하라.\n"
        "각 항목은 type(avoid/prefer/style)과 guideline(한국어 한 문장) 필드를 포함한다.\n\n"
    ]

    if negatives:
        parts.append("## 부정 피드백 (👎)\n")
        for fb in negatives:
            user = fb.get("user_preview") or "(없음)"
            bot = fb.get("bot_preview") or "(없음)"
            reason = fb.get("reason")
            line = f"- 질문: {user}\n  응답: {bot}\n"
            if reason:
                reason_clean = reason.strip().replace("\n", " ")[:200]
                line += f"  사유: {reason_clean}\n"
            parts.append(line)

    if positives:
        parts.append("\n## 긍정 피드백 (👍)\n")
        for fb in positives:
            user = fb.get("user_preview") or "(없음)"
            bot = fb.get("bot_preview") or "(없음)"
            parts.append(f"- 질문: {user}\n  응답: {bot}\n")

    if auto_low:
        parts.append("\n## 자동 평가 저점 응답 (LLM-as-Judge)\n")
        for ev in auto_low:
            user = ev.get("user_input") or "(없음)"
            bot = ev.get("bot_response") or "(없음)"
            score = ev.get("score", "?")
            explanation = ev.get("explanation") or ""
            line = f"- [점수 {score}/5] 질문: {user}\n  응답: {bot}\n"
            if explanation:
                line += f"  평가사유: {explanation[:200]}\n"
            parts.append(line)

    return "".join(parts)


def build_feedback_analysis_callable(
    engine: Any,
    memory: Any,
    feedback: Any,
    allowed_users: list[int],
    logger: Any,
):
    """피드백 분석 callable 팩토리."""

    async def feedback_analysis(
        min_feedback_count: int = 5,
        max_negative_samples: int = 15,
        max_positive_samples: int = 10,
        max_guidelines: int = 5,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """사용자 피드백을 분석해 응답 품질 가이드라인을 갱신한다."""
        results: list[str] = []

        for chat_id in allowed_users:
            count = await feedback.count_feedback(chat_id)
            if count < min_feedback_count:
                continue

            negatives = await feedback.get_recent_feedback(chat_id, rating=-1, limit=max_negative_samples)
            positives = await feedback.get_recent_feedback(chat_id, rating=1, limit=max_positive_samples)

            # 자동 평가 저점 데이터 수집
            auto_low: list[dict] | None = None
            if hasattr(feedback, "get_low_score_evaluations"):
                try:
                    auto_low = await feedback.get_low_score_evaluations(chat_id=chat_id, max_score=2, limit=10)
                except Exception:
                    pass

            if not negatives and not positives and not auto_low:
                continue

            # LLM에 구조화 출력 요청
            prompt = _build_analysis_prompt(negatives, positives, auto_low=auto_low)
            raw = await engine.process_prompt(
                prompt=prompt,
                chat_id=chat_id,
                response_format=FEEDBACK_ANALYSIS_SCHEMA,
                max_tokens=max_tokens,
                temperature=temperature,
                model_override=model,
                model_role=model_role,
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
                key = f"feedback_guideline_{i + 1:02d}"
                value = f"[{item['type']}] {item['guideline']}"
                await memory.store_memory(chat_id, key, value, category="feedback_guidelines")

            results.append(f"chat_id={chat_id}: {len(valid[:max_guidelines])}건 갱신")

        if not results:
            return ""

        return "## 피드백 분석 결과\n\n" + "\n".join(f"- {r}" for r in results)

    return feedback_analysis
