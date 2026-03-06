"""규칙 기반 즉시 응답 엔진.

정규식/키워드 매칭으로 LLM을 우회하여 즉시 응답을 반환한다.
규칙은 YAML 파일로 정의하며, 런타임에 리로드 가능하다.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import yaml

from core.logging_setup import get_logger


@dataclass(frozen=True)
class InstantMatch:
    """즉시 응답 매칭 결과."""

    response: str
    rule_name: str


@dataclass
class _InstantRule:
    """내부 규칙 표현."""

    name: str
    patterns: list[re.Pattern]
    responses: list[str] = field(default_factory=list)
    rule_type: str = "static"  # "static" | "callable"
    callable_name: str | None = None


class InstantResponder:
    """정규식/키워드 매칭 기반 즉시 응답기."""

    _BUILTIN_CALLABLES: ClassVar[dict[str, Callable[[], str]]] = {
        "get_current_time": lambda: datetime.now().strftime("%H시 %M분입니다."),
        "get_current_date": lambda: datetime.now().strftime("%Y년 %m월 %d일입니다."),
    }

    def __init__(self, rules_path: str = "config/instant_rules.yaml") -> None:
        self._rules: list[_InstantRule] = []
        self._rules_path = rules_path
        self._logger = get_logger("instant_responder")
        self._load_rules()

    @property
    def rules_count(self) -> int:
        return len(self._rules)

    def match(self, text: str) -> InstantMatch | None:
        """입력 텍스트에 매칭되는 즉시 응답을 반환한다."""
        text_stripped = text.strip()
        if not text_stripped:
            return None

        for rule in self._rules:
            for pattern in rule.patterns:
                if pattern.search(text_stripped):
                    if rule.rule_type == "callable" and rule.callable_name:
                        response = self._resolve_callable(rule.callable_name)
                        if response is not None:
                            return InstantMatch(response=response, rule_name=rule.name)
                    elif rule.responses:
                        response = random.choice(rule.responses)
                        return InstantMatch(response=response, rule_name=rule.name)
        return None

    def reload_rules(self) -> int:
        """규칙 파일을 다시 로드한다."""
        self._load_rules()
        return len(self._rules)

    def _load_rules(self) -> None:
        """YAML 파일에서 규칙을 로드한다."""
        path = Path(self._rules_path)
        if not path.exists():
            self._logger.warning("instant_rules_not_found", path=self._rules_path)
            self._rules = []
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            rules_data = data.get("rules", [])
            loaded: list[_InstantRule] = []

            for rule_data in rules_data:
                name = rule_data.get("name", "unknown")
                raw_patterns = rule_data.get("patterns", [])
                case_insensitive = rule_data.get("case_insensitive", False)
                flags = re.IGNORECASE if case_insensitive else 0

                compiled_patterns: list[re.Pattern] = []
                for raw in raw_patterns:
                    try:
                        compiled_patterns.append(re.compile(raw, flags))
                    except re.error as exc:
                        self._logger.warning(
                            "instant_rule_pattern_invalid",
                            rule=name,
                            pattern=raw,
                            error=str(exc),
                        )

                if not compiled_patterns:
                    continue

                rule_type = rule_data.get("type", "static")
                loaded.append(
                    _InstantRule(
                        name=name,
                        patterns=compiled_patterns,
                        responses=rule_data.get("responses", []),
                        rule_type=rule_type,
                        callable_name=rule_data.get("callable"),
                    )
                )

            self._rules = loaded
            self._logger.info("instant_rules_loaded", count=len(loaded))

        except Exception as exc:
            self._logger.warning("instant_rules_load_failed", error=str(exc))
            self._rules = []

    def _resolve_callable(self, name: str) -> str | None:
        """내장 callable을 실행한다."""
        fn = self._BUILTIN_CALLABLES.get(name)
        if fn is None:
            self._logger.warning("instant_callable_unknown", name=name)
            return None
        try:
            return fn()
        except Exception as exc:
            self._logger.warning("instant_callable_failed", name=name, error=str(exc))
            return None
