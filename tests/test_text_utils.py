"""text_utils 테스트."""

from __future__ import annotations

from core.text_utils import sanitize_model_output


def test_sanitize_model_output_prefers_final_channel_message() -> None:
    raw = (
        "<|start|>assistant<|channel|>analysis<|message|>draft<|end|>"
        "<|start|>assistant<|channel|>final<|message|>최종 답변<|end|>"
    )
    assert sanitize_model_output(raw) == "최종 답변"


def test_sanitize_model_output_removes_think_block() -> None:
    raw = "<think>chain of thought</think>\n실제 답변"
    assert sanitize_model_output(raw) == "실제 답변"
