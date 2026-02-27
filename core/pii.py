"""PII 마스킹 유틸리티."""

from __future__ import annotations

import re

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{4}-\d{4}\b"),  # 한국 전화번호
    re.compile(r"\b\d{6}-\d{7}\b"),  # 주민등록번호
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # 이메일
]
_GENERIC_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?"
    r"\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}(?![-.\s]?\d)"
)
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _passes_luhn(number: str) -> bool:
    """Luhn 체크섬 검증."""
    if not number.isdigit():
        return False
    if not 13 <= len(number) <= 19:
        return False

    checksum = 0
    parity = len(number) % 2
    for index, char in enumerate(number):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _redact_card_numbers(text: str) -> str:
    """카드번호 후보 중 Luhn 검증 통과 건만 마스킹한다."""

    def _replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = "".join(ch for ch in candidate if ch.isdigit())
        if _passes_luhn(digits):
            return "[REDACTED]"
        return candidate

    return _CARD_CANDIDATE_RE.sub(_replace, text)


def _redact_generic_phone_numbers(text: str) -> str:
    """일반 전화번호 후보를 자리수 기준으로 검증해 마스킹한다."""

    def _replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = "".join(ch for ch in candidate if ch.isdigit())
        has_plus_prefix = candidate.lstrip().startswith("+")
        if 9 <= len(digits) <= 11:
            return "[REDACTED]"
        if has_plus_prefix and 10 <= len(digits) <= 13:
            return "[REDACTED]"
        return candidate

    return _GENERIC_PHONE_RE.sub(_replace, text)


def redact_pii(text: str) -> str:
    """문자열 내 알려진 PII 패턴을 마스킹한다."""
    redacted = text
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    redacted = _redact_generic_phone_numbers(redacted)
    redacted = _redact_card_numbers(redacted)
    return redacted
