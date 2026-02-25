"""InstantResponder 테스트."""

from __future__ import annotations

import pytest

from core.instant_responder import InstantResponder, InstantMatch


@pytest.fixture
def responder(tmp_path) -> InstantResponder:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - name: "greeting"
    patterns:
      - "^(안녕|하이|hi|hello)"
    responses:
      - "안녕하세요!"
      - "반갑습니다!"
    case_insensitive: true

  - name: "farewell"
    patterns:
      - "^(바이|bye)"
    responses:
      - "잘가요!"
    case_insensitive: true

  - name: "current_time"
    patterns:
      - "지금\\\\s*몇\\\\s*시"
    type: "callable"
    callable: "get_current_time"
""",
        encoding="utf-8",
    )
    return InstantResponder(rules_path=str(rules_path))


class TestMatch:
    def test_greeting_match(self, responder: InstantResponder) -> None:
        result = responder.match("안녕하세요")
        assert result is not None
        assert result.rule_name == "greeting"
        assert result.response in ("안녕하세요!", "반갑습니다!")

    def test_greeting_case_insensitive(self, responder: InstantResponder) -> None:
        result = responder.match("Hello there")
        assert result is not None
        assert result.rule_name == "greeting"

    def test_farewell_match(self, responder: InstantResponder) -> None:
        result = responder.match("바이바이")
        assert result is not None
        assert result.rule_name == "farewell"

    def test_no_match(self, responder: InstantResponder) -> None:
        result = responder.match("파이썬 코드 짜줘")
        assert result is None

    def test_empty_text(self, responder: InstantResponder) -> None:
        result = responder.match("")
        assert result is None

    def test_whitespace_text(self, responder: InstantResponder) -> None:
        result = responder.match("   ")
        assert result is None


class TestCallable:
    def test_time_callable(self, responder: InstantResponder) -> None:
        result = responder.match("지금 몇 시야?")
        assert result is not None
        assert result.rule_name == "current_time"
        assert "시" in result.response


class TestRulesCount:
    def test_rules_count(self, responder: InstantResponder) -> None:
        assert responder.rules_count == 3


class TestMissingFile:
    def test_missing_rules_file(self, tmp_path) -> None:
        responder = InstantResponder(rules_path=str(tmp_path / "nonexistent.yaml"))
        assert responder.rules_count == 0
        assert responder.match("안녕") is None


class TestReloadRules:
    def test_reload_rules(self, responder: InstantResponder) -> None:
        count = responder.reload_rules()
        assert count == 3
