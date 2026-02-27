"""PII 마스킹 테스트."""

from __future__ import annotations

from core.pii import redact_pii


def test_redact_pii_masks_valid_luhn_card_number() -> None:
    text = "결제 카드: 4111 1111 1111 1111"
    redacted = redact_pii(text)
    assert redacted == "결제 카드: [REDACTED]"


def test_redact_pii_keeps_non_luhn_long_number() -> None:
    text = "참고 번호: 1234 5678 9012 3456"
    redacted = redact_pii(text)
    assert redacted == text


def test_redact_pii_still_masks_phone_and_email() -> None:
    text = "연락처 010-1234-5678, 메일 test@example.com"
    redacted = redact_pii(text)
    assert redacted == "연락처 [REDACTED], 메일 [REDACTED]"
