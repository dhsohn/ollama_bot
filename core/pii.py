"""PII 마스킹 유틸리티."""

from __future__ import annotations

import re

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{4}-\d{4}\b"),  # 한국 전화번호
    re.compile(r"\b\d{6}-\d{7}\b"),  # 주민등록번호
    re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"),  # 전화번호 일반형
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # 이메일
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # 카드번호(폭넓은 패턴)
]


def redact_pii(text: str) -> str:
    """문자열 내 알려진 PII 패턴을 마스킹한다."""
    redacted = text
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted

