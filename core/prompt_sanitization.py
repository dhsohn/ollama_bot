"""Shared prompt-sanitization helpers."""

from __future__ import annotations

import re as _re

_INJECTION_RE = _re.compile(
    r"\[/?(?:system|user|assistant|INST)\]"
    r"|<\|(?:im_start|im_end|system|user|assistant)\|>"
    r"|(?:^|\n)\s*(?:system|user|assistant|human)\s*:",
    _re.IGNORECASE,
)
_CODE_BLOCK_RE = _re.compile(r"```.*?```", _re.DOTALL)


def strip_prompt_injection(text: str) -> str:
    """Remove prompt-injection markers while preserving fenced code blocks."""
    if not text:
        return ""

    parts: list[str] = []
    last = 0
    for match in _CODE_BLOCK_RE.finditer(text):
        outside = text[last:match.start()]
        outside = _INJECTION_RE.sub("", outside)
        outside = _re.sub(r"\n{3,}", "\n\n", outside)
        parts.append(outside)
        parts.append(match.group(0))
        last = match.end()

    tail = text[last:]
    tail = _INJECTION_RE.sub("", tail)
    tail = _re.sub(r"\n{3,}", "\n\n", tail)
    parts.append(tail)

    sanitized = "".join(parts)
    sanitized = _re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()
