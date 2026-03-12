"""Stateless helpers for the Lemonade client."""

from __future__ import annotations

import json
import re
from typing import Any

from core.llm_types import ChatUsage

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CJK_RANGE = re.compile(
    r"[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]",
)


def estimate_token_count(text: str) -> int:
    """Estimate token count for mixed Korean/English text.

    Korean characters average ~1.5 chars/token; ASCII averages ~4 chars/token.
    This intentionally over-estimates to leave a safety margin.
    """
    if not text:
        return 0
    cjk_chars = len(_CJK_RANGE.findall(text))
    ascii_chars = len(text) - cjk_chars
    return int(cjk_chars / 1.5) + int(ascii_chars / 3.5) + 4  # +4 for overhead


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total token count of a chat messages list."""
    total = 0
    for msg in messages:
        # role token overhead (~4 tokens per message for role/separators)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_token_count(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_token_count(part["text"])
    return total


def format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def compact_text(value: str, *, max_chars: int = 280) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def parse_loaded_models(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    entries = payload.get("all_models_loaded", [])
    names: set[str] = set()
    if not isinstance(entries, list):
        return names
    for item in entries:
        if isinstance(item, dict):
            name = (
                item.get("model_name")
                or item.get("id")
                or item.get("model")
                or item.get("name")
            )
            if isinstance(name, str) and name:
                names.add(name)
        elif isinstance(item, str) and item:
            names.add(item)
    return names


def extract_content(choice: dict[str, Any]) -> str:
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            return "".join(chunks)
    return ""


def extract_api_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return compact_text(error)
    if not isinstance(error, dict):
        return None
    message = error.get("message") or error.get("detail") or error.get("error")
    code = error.get("code") or error.get("type")
    parts: list[str] = []
    if isinstance(code, str) and code.strip():
        parts.append(code.strip())
    if isinstance(message, str) and message.strip():
        parts.append(message.strip())
    if not parts:
        return compact_text(str(error))
    return compact_text(": ".join(parts))


def usage_from_payload(payload: dict[str, Any]) -> ChatUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    return ChatUsage(
        prompt_eval_count=int(usage.get("prompt_tokens", 0) or 0),
        eval_count=int(usage.get("completion_tokens", 0) or 0),
        eval_duration=0,
        total_duration=int(usage.get("total_duration", 0) or 0),
    )


def build_chat_payload(
    *,
    model: str | None,
    messages: list[dict[str, str]],
    default_temperature: float,
    temperature: float | None,
    default_max_tokens: int,
    max_tokens: int | None,
    response_format: str | dict | None,
    stream: bool,
    logger: Any,
    context_window: int = 0,
    min_output_tokens: int = 128,
) -> dict[str, Any]:
    effective_max_tokens = default_max_tokens if max_tokens is None else max_tokens

    if context_window > 0:
        estimated_input = estimate_messages_tokens(messages)
        budget = context_window - estimated_input
        if budget < min_output_tokens:
            logger.warning(
                "token_budget_tight",
                context_window=context_window,
                estimated_input_tokens=estimated_input,
                remaining=budget,
                forced_min=min_output_tokens,
            )
            budget = min_output_tokens
        if effective_max_tokens > budget:
            logger.info(
                "max_tokens_clamped",
                original=effective_max_tokens,
                clamped=budget,
                context_window=context_window,
                estimated_input_tokens=estimated_input,
            )
            effective_max_tokens = budget

    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": default_temperature if temperature is None else temperature,
        "max_tokens": effective_max_tokens,
        "stream": stream,
    }
    if model:
        payload["model"] = model
    if response_format is not None:
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(response_format, dict):
            format_type = str(response_format.get("type", "")).strip().lower()
            if format_type in {"json_object", "text"}:
                payload["response_format"] = response_format
            else:
                payload["response_format"] = {"type": "json_object"}
                logger.debug(
                    "lemonade_response_format_downgraded",
                    requested_type=format_type or None,
                )
        else:
            payload["response_format"] = response_format
    return payload


def parse_rerank_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    results = payload.get("results", payload.get("data", []))
    if not isinstance(results, list):
        return []
    scored: list[dict[str, Any]] = []
    for item in results:
        scored.append({
            "index": int(item.get("index", 0)),
            "score": float(item.get("relevance_score", item.get("score", 0.0))),
        })
    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return scored


def parse_rerank_chat_response(content: str) -> list[dict[str, Any]]:
    parsed = json.loads(content)
    if isinstance(parsed, list):
        scored = [
            {"index": int(item.get("index", 0)), "score": float(item.get("score", 0.0))}
            for item in parsed
        ]
    elif isinstance(parsed, dict) and "results" in parsed:
        scored = [
            {"index": int(item.get("index", 0)), "score": float(item.get("score", 0.0))}
            for item in parsed["results"]
        ]
    else:
        raise ValueError("Unexpected rerank chat response format")
    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return scored
