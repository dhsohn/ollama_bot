"""스킬 시스템 — YAML 기반 스킬 로더 및 실행기.

skills/_builtin/ 및 skills/custom/ 디렉토리의 YAML 파일을 로드하고,
사용자 입력에 대한 트리거 매칭 및 LLM 컨텍스트 생성을 담당한다.
스킬은 순수 선언적이며, 코드 실행 없이 시스템 프롬프트만 변경한다.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from core.logging_setup import get_logger
from core.security import SecurityManager, SecurityViolationError


class SecurityLevel(str, Enum):
    SAFE = "safe"
    CAUTIOUS = "cautious"
    RESTRICTED = "restricted"


class SkillParameter(BaseModel):
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""


class SkillDefinition(BaseModel):
    """스킬 YAML 정의를 검증하는 모델."""

    name: str
    description: str
    version: str = "1.0"
    triggers: list[str]
    system_prompt: str
    allowed_tools: list[str] = Field(default_factory=list)
    parameters: list[SkillParameter] = Field(default_factory=list)
    timeout: int = 30
    streaming: bool = True
    model_role: str = "skill"
    temperature: float | None = None
    max_tokens: int | None = None
    security_level: SecurityLevel = SecurityLevel.SAFE

    @field_validator("triggers")
    @classmethod
    def validate_triggers(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one trigger is required")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("timeout must be >= 1")
        return value

    @field_validator("model_role")
    @classmethod
    def validate_model_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("model_role must not be empty")
        return normalized

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1:
            raise ValueError("max_tokens must be >= 1")
        return value


class DuplicateSkillNameError(ValueError):
    """스킬 이름 충돌."""


class DuplicateSkillTriggerError(ValueError):
    """스킬 트리거 충돌."""


class SkillManager:
    """스킬을 로드, 매칭, 관리한다."""

    def __init__(
        self,
        security: SecurityManager,
        skills_dir: str = "skills",
    ) -> None:
        self._security = security
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillDefinition] = {}
        self._trigger_map: dict[str, str] = {}  # trigger → skill name
        self._command_trigger_map: dict[str, str] = {}
        self._keyword_trigger_order: list[tuple[str, str]] = []
        self._ac_goto: list[dict[str, int]] = [{}]
        self._ac_fail: list[int] = [0]
        self._ac_output: list[list[int]] = [[]]
        self._last_load_errors: list[str] = []
        self._logger = get_logger("skill_manager")

    async def load_skills(self, *, strict: bool = False) -> int:
        """_builtin/ 및 custom/ 디렉토리에서 스킬 YAML을 로드한다."""
        new_skills: dict[str, SkillDefinition] = {}
        new_trigger_map: dict[str, str] = {}
        name_sources: dict[str, Path] = {}
        trigger_sources: dict[str, Path] = {}
        self._last_load_errors = []
        loaded = 0

        for sub_dir in ["_builtin", "custom"]:
            skill_path = self._skills_dir / sub_dir
            if not skill_path.exists():
                continue

            for yaml_file in sorted(skill_path.glob("*.yaml")):
                try:
                    skill = self._load_skill_file(yaml_file)
                    if skill:
                        existing_source = name_sources.get(skill.name)
                        if existing_source is not None:
                            raise DuplicateSkillNameError(
                                f"Duplicate skill name '{skill.name}' "
                                f"({existing_source} vs {yaml_file})"
                            )

                        for trigger in skill.triggers:
                            trigger_key = trigger.lower()
                            existing_skill = new_trigger_map.get(trigger_key)
                            if existing_skill and existing_skill != skill.name:
                                existing_trigger_source = trigger_sources.get(trigger_key)
                                raise DuplicateSkillTriggerError(
                                    f"Duplicate trigger '{trigger}' for skills "
                                    f"'{existing_skill}' and '{skill.name}' "
                                    f"({existing_trigger_source} vs {yaml_file})"
                                )

                        new_skills[skill.name] = skill
                        name_sources[skill.name] = yaml_file
                        for trigger in skill.triggers:
                            trigger_key = trigger.lower()
                            if trigger_key in new_trigger_map:
                                # 동일 스킬 내 중복 트리거는 첫 항목을 유지한다.
                                continue
                            new_trigger_map[trigger_key] = skill.name
                            trigger_sources[trigger_key] = yaml_file
                        loaded += 1
                        self._logger.info(
                            "skill_loaded",
                            name=skill.name,
                            triggers=skill.triggers,
                            source=str(yaml_file),
                        )
                except (DuplicateSkillNameError, DuplicateSkillTriggerError):
                    raise
                except Exception as exc:
                    self._last_load_errors.append(f"{yaml_file.name}: {exc}")
                    self._logger.error(
                        "skill_load_failed",
                        file=str(yaml_file),
                        error=str(exc),
                    )

        if strict and self._last_load_errors:
            raise ValueError(self._format_load_error_summary(self._last_load_errors))

        # 전체 로드가 성공했을 때만 활성 스킬 상태를 교체한다.
        self._skills = new_skills
        self._trigger_map = new_trigger_map
        self._rebuild_matcher_index()
        if self._last_load_errors:
            self._logger.warning(
                "skills_loaded_with_errors",
                loaded=loaded,
                error_count=len(self._last_load_errors),
            )
        self._logger.info("skills_loaded_total", count=loaded)
        return loaded

    def _load_skill_file(self, path: Path) -> SkillDefinition | None:
        """단일 YAML 파일에서 스킬을 로드하고 보안을 검증한다."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        skill = SkillDefinition(**data)

        # 보안 등급과 도구 호환성 검증
        try:
            self._security.check_skill_security(
                skill.security_level.value, skill.allowed_tools
            )
        except SecurityViolationError as exc:
            self._last_load_errors.append(f"{path.name}: {exc}")
            self._logger.warning(
                "skill_security_rejected",
                name=skill.name,
                reason=str(exc),
            )
            return None

        return skill

    def get_skill(self, name: str) -> SkillDefinition | None:
        """이름으로 스킬을 조회한다."""
        return self._skills.get(name)

    def match_trigger(self, text: str) -> SkillDefinition | None:
        """사용자 입력에서 스킬 트리거를 매칭한다.

        우선순위: 명령어(/command) > 키워드
        """
        text_lower = text.lower().strip()

        # 1. /command 트리거 (정확한 접두사 매칭)
        first_word = text_lower.split()[0] if text_lower else ""
        if first_word.startswith("/"):
            skill_name = self._command_trigger_map.get(first_word)
            if skill_name:
                return self._skills.get(skill_name)

        # 2. 키워드 트리거 (Aho-Corasick 매칭)
        skill_name = self._match_keyword_trigger(text_lower)
        if skill_name:
            return self._skills.get(skill_name)

        return None

    def _rebuild_matcher_index(self) -> None:
        """트리거 인덱스를 재생성한다."""
        self._command_trigger_map = {}
        self._keyword_trigger_order = []

        for trigger, skill_name in self._trigger_map.items():
            if trigger.startswith("/"):
                self._command_trigger_map[trigger] = skill_name
            else:
                self._keyword_trigger_order.append((trigger, skill_name))

        self._build_keyword_automaton()

    def _build_keyword_automaton(self) -> None:
        """키워드 트리거용 Aho-Corasick 오토마톤을 생성한다."""
        self._ac_goto = [{}]
        self._ac_fail = [0]
        self._ac_output = [[]]

        for order_idx, (trigger, _) in enumerate(self._keyword_trigger_order):
            state = 0
            for ch in trigger:
                next_state = self._ac_goto[state].get(ch)
                if next_state is None:
                    next_state = len(self._ac_goto)
                    self._ac_goto[state][ch] = next_state
                    self._ac_goto.append({})
                    self._ac_fail.append(0)
                    self._ac_output.append([])
                state = next_state
            self._ac_output[state].append(order_idx)

        queue: deque[int] = deque()
        for next_state in self._ac_goto[0].values():
            queue.append(next_state)
            self._ac_fail[next_state] = 0

        while queue:
            state = queue.popleft()
            for ch, next_state in self._ac_goto[state].items():
                queue.append(next_state)
                fail_state = self._ac_fail[state]
                while fail_state and ch not in self._ac_goto[fail_state]:
                    fail_state = self._ac_fail[fail_state]
                self._ac_fail[next_state] = self._ac_goto[fail_state].get(ch, 0)
                self._ac_output[next_state].extend(
                    self._ac_output[self._ac_fail[next_state]]
                )

    def _match_keyword_trigger(self, text_lower: str) -> str | None:
        """본문 키워드 트리거를 검색한다.

        기존 동작(로드 순서상 먼저 등록된 트리거 우선)을 유지하기 위해
        매칭 위치와 무관하게 최소 order 인덱스를 선택한다.
        """
        if not self._keyword_trigger_order:
            return None

        state = 0
        best_order: int | None = None
        for ch in text_lower:
            while state and ch not in self._ac_goto[state]:
                state = self._ac_fail[state]
            state = self._ac_goto[state].get(ch, 0)

            outputs = self._ac_output[state]
            if not outputs:
                continue

            current_best = min(outputs)
            if best_order is None or current_best < best_order:
                best_order = current_best
                if best_order == 0:
                    break

        if best_order is None:
            return None
        return self._keyword_trigger_order[best_order][1]

    def list_skills(self) -> list[dict]:
        """로드된 스킬 목록을 반환한다."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "triggers": s.triggers,
                "security_level": s.security_level.value,
            }
            for s in self._skills.values()
        ]

    def get_last_load_errors(self) -> list[str]:
        """가장 최근 load_skills에서 수집된 오류를 반환한다."""
        return list(self._last_load_errors)

    @staticmethod
    def _format_load_error_summary(errors: list[str], max_items: int = 3) -> str:
        preview = errors[:max_items]
        message = (
            f"Skill loading failed in strict mode "
            f"({len(errors)} error(s)): {'; '.join(preview)}"
        )
        if len(errors) > max_items:
            message += f"; ... and {len(errors) - max_items} more"
        return message

    async def reload_skills(self, *, strict: bool = False) -> int:
        """스킬을 다시 로드한다."""
        return await self.load_skills(strict=strict)

    def get_skill_context(
        self,
        skill: SkillDefinition,
        user_input: str,
    ) -> list[dict[str, str]]:
        """스킬의 시스템 프롬프트와 사용자 입력으로 LLM 메시지를 생성한다."""
        # 트리거 명령어를 입력에서 제거
        clean_input = user_input
        for trigger in skill.triggers:
            if user_input.lower().startswith(trigger.lower()):
                clean_input = user_input[len(trigger):].strip()
                break

        return [
            {"role": "system", "content": skill.system_prompt},
            {"role": "user", "content": clean_input or user_input},
        ]

    @property
    def skill_count(self) -> int:
        return len(self._skills)
