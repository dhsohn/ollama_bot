"""lemonade_utils 모듈 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock

from core.lemonade_utils import (
    build_chat_payload,
    compact_text,
    extract_api_error,
    extract_content,
    format_exception,
    parse_loaded_models,
    parse_rerank_chat_response,
    parse_rerank_items,
    usage_from_payload,
)


class TestFormatException:
    def test_with_message(self) -> None:
        assert format_exception(ValueError("bad")) == "ValueError: bad"

    def test_without_message(self) -> None:
        assert format_exception(ValueError()) == "ValueError"

    def test_whitespace_message(self) -> None:
        assert format_exception(ValueError("  ")) == "ValueError"


class TestCompactText:
    def test_short_text(self) -> None:
        assert compact_text("hello  world") == "hello world"

    def test_long_text_truncated(self) -> None:
        result = compact_text("x" * 300, max_chars=10)
        assert len(result) == 13  # 10 + "..."
        assert result.endswith("...")


class TestParseLoadedModels:
    def test_non_dict_returns_empty(self) -> None:
        assert parse_loaded_models("invalid") == set()
        assert parse_loaded_models(None) == set()

    def test_non_list_entries(self) -> None:
        assert parse_loaded_models({"all_models_loaded": "not_a_list"}) == set()

    def test_dict_entries_with_various_keys(self) -> None:
        payload = {
            "all_models_loaded": [
                {"model_name": "model-a"},
                {"id": "model-b"},
                {"model": "model-c"},
                {"name": "model-d"},
            ]
        }
        assert parse_loaded_models(payload) == {"model-a", "model-b", "model-c", "model-d"}

    def test_string_entries(self) -> None:
        payload = {"all_models_loaded": ["model-a", "model-b"]}
        assert parse_loaded_models(payload) == {"model-a", "model-b"}

    def test_empty_names_ignored(self) -> None:
        payload = {"all_models_loaded": [{"model_name": ""}, "", {"name": None}]}
        assert parse_loaded_models(payload) == set()

    def test_mixed_entries(self) -> None:
        payload = {"all_models_loaded": [{"id": "alpha"}, "beta", 42]}
        assert parse_loaded_models(payload) == {"alpha", "beta"}


class TestExtractContent:
    def test_normal_string_content(self) -> None:
        assert extract_content({"message": {"content": "hello"}}) == "hello"

    def test_list_content(self) -> None:
        choice = {"message": {"content": [{"text": "a"}, {"text": "b"}]}}
        assert extract_content(choice) == "ab"

    def test_missing_content(self) -> None:
        assert extract_content({"message": {}}) == ""
        assert extract_content({}) == ""
        assert extract_content("not_a_dict") == ""

    def test_non_text_parts_ignored(self) -> None:
        choice = {"message": {"content": [{"image": "data"}, {"text": "ok"}]}}
        assert extract_content(choice) == "ok"


class TestExtractApiError:
    def test_non_dict(self) -> None:
        assert extract_api_error("string") is None
        assert extract_api_error(None) is None

    def test_string_error(self) -> None:
        assert extract_api_error({"error": "something bad"}) == "something bad"

    def test_dict_error_with_message_and_code(self) -> None:
        result = extract_api_error({"error": {"code": "rate_limit", "message": "too many"}})
        assert "rate_limit" in result
        assert "too many" in result

    def test_dict_error_with_detail(self) -> None:
        result = extract_api_error({"error": {"detail": "not found"}})
        assert "not found" in result

    def test_dict_error_no_known_fields(self) -> None:
        result = extract_api_error({"error": {"unknown_key": "val"}})
        assert result is not None

    def test_no_error_key(self) -> None:
        assert extract_api_error({"data": "ok"}) is None

    def test_empty_string_error(self) -> None:
        assert extract_api_error({"error": "  "}) is None

    def test_non_dict_error_value(self) -> None:
        assert extract_api_error({"error": 42}) is None


class TestUsageFromPayload:
    def test_valid_usage(self) -> None:
        usage = usage_from_payload({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
        assert usage is not None
        assert usage.prompt_eval_count == 10
        assert usage.eval_count == 20

    def test_missing_usage(self) -> None:
        assert usage_from_payload({}) is None
        assert usage_from_payload({"usage": "not_dict"}) is None


class TestBuildChatPayload:
    def test_basic_payload(self) -> None:
        result = build_chat_payload(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            default_temperature=0.7,
            temperature=None,
            default_max_tokens=1024,
            max_tokens=None,
            response_format=None,
            stream=False,
            logger=MagicMock(),
        )
        assert result["model"] == "test"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1024
        assert result["stream"] is False
        assert "response_format" not in result

    def test_json_response_format(self) -> None:
        result = build_chat_payload(
            model=None,
            messages=[],
            default_temperature=0.7,
            temperature=0.3,
            default_max_tokens=1024,
            max_tokens=512,
            response_format="json",
            stream=True,
            logger=MagicMock(),
        )
        assert result["response_format"] == {"type": "json_object"}
        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 512
        assert "model" not in result

    def test_dict_response_format_json_object(self) -> None:
        result = build_chat_payload(
            model="m",
            messages=[],
            default_temperature=0.7,
            temperature=None,
            default_max_tokens=1024,
            max_tokens=None,
            response_format={"type": "json_object"},
            stream=False,
            logger=MagicMock(),
        )
        assert result["response_format"] == {"type": "json_object"}

    def test_dict_response_format_text(self) -> None:
        result = build_chat_payload(
            model="m",
            messages=[],
            default_temperature=0.7,
            temperature=None,
            default_max_tokens=1024,
            max_tokens=None,
            response_format={"type": "text"},
            stream=False,
            logger=MagicMock(),
        )
        assert result["response_format"] == {"type": "text"}

    def test_dict_response_format_downgraded(self) -> None:
        logger = MagicMock()
        result = build_chat_payload(
            model="m",
            messages=[],
            default_temperature=0.7,
            temperature=None,
            default_max_tokens=1024,
            max_tokens=None,
            response_format={"type": "json_schema", "schema": {}},
            stream=False,
            logger=logger,
        )
        assert result["response_format"] == {"type": "json_object"}
        logger.debug.assert_called_once()

    def test_other_response_format_passthrough(self) -> None:
        result = build_chat_payload(
            model="m",
            messages=[],
            default_temperature=0.7,
            temperature=None,
            default_max_tokens=1024,
            max_tokens=None,
            response_format=42,
            stream=False,
            logger=MagicMock(),
        )
        assert result["response_format"] == 42


class TestParseRerankItems:
    def test_non_dict_returns_empty(self) -> None:
        assert parse_rerank_items("string") == []
        assert parse_rerank_items(None) == []

    def test_results_key(self) -> None:
        payload = {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]}
        result = parse_rerank_items(payload)
        assert result[0]["score"] == 0.9
        assert result[0]["index"] == 0

    def test_data_key_fallback(self) -> None:
        payload = {"data": [{"index": 0, "score": 0.5}]}
        result = parse_rerank_items(payload)
        assert result[0]["score"] == 0.5

    def test_non_list_results(self) -> None:
        assert parse_rerank_items({"results": "bad"}) == []

    def test_sorted_by_score_descending(self) -> None:
        payload = {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]}
        result = parse_rerank_items(payload)
        assert result[0]["index"] == 1


class TestParseRerankChatResponse:
    def test_list_format(self) -> None:
        content = '[{"index": 0, "score": 0.8}, {"index": 1, "score": 0.2}]'
        result = parse_rerank_chat_response(content)
        assert result[0]["index"] == 0
        assert result[0]["score"] == 0.8

    def test_dict_with_results_key(self) -> None:
        content = '{"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}]}'
        result = parse_rerank_chat_response(content)
        assert result[0]["index"] == 1

    def test_unexpected_format_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="Unexpected"):
            parse_rerank_chat_response('{"key": "value"}')
