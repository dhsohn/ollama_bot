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


def test_sanitize_model_output_handles_channel_tokens_without_start_marker() -> None:
    raw = (
        "assistant<|channel|>analysis<|message|>draft<|end|>"
        "assistant<|channel|>final<|message|>최종 답변<|end|>"
    )
    assert sanitize_model_output(raw) == "최종 답변"


def test_sanitize_model_output_extracts_final_from_assistant_marker_fallback() -> None:
    raw = (
        "저는 수업이수요?We need to respond in Korean. "
        "The user says: assistantfinal}안녕하세요!"
    )
    assert sanitize_model_output(raw) == "안녕하세요!"


def test_sanitize_model_output_hides_analysis_only_output() -> None:
    raw = "assistantanalysis}We need to respond in Korean."
    assert sanitize_model_output(raw) == ""
