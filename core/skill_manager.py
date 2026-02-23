"""스킬 시스템 — YAML 기반 스킬 로더 및 실행기.

skills/_builtin/ 및 skills/custom/ 디렉토리의 YAML 파일을 로드하고,
사용자 입력에 대한 트리거 매칭 및 LLM 컨텍스트 생성을 담당한다.
스킬은 순수 선언적이며, 코드 실행 없이 시스템 프롬프트만 변경한다.
"""

from __future__ import annotations

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
    security_level: SecurityLevel = SecurityLevel.SAFE

    @field_validator("triggers")
    @classmethod
    def validate_triggers(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one trigger is required")
        return v


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
        self._logger = get_logger("skill_manager")

    async def load_skills(self) -> int:
        """_builtin/ 및 custom/ 디렉토리에서 스킬 YAML을 로드한다."""
        self._skills.clear()
        self._trigger_map.clear()
        loaded = 0

        for sub_dir in ["_builtin", "custom"]:
            skill_path = self._skills_dir / sub_dir
            if not skill_path.exists():
                continue

            for yaml_file in sorted(skill_path.glob("*.yaml")):
                try:
                    skill = self._load_skill_file(yaml_file)
                    if skill:
                        self._skills[skill.name] = skill
                        for trigger in skill.triggers:
                            self._trigger_map[trigger.lower()] = skill.name
                        loaded += 1
                        self._logger.info(
                            "skill_loaded",
                            name=skill.name,
                            triggers=skill.triggers,
                            source=str(yaml_file),
                        )
                except Exception as exc:
                    self._logger.error(
                        "skill_load_failed",
                        file=str(yaml_file),
                        error=str(exc),
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
            skill_name = self._trigger_map.get(first_word)
            if skill_name:
                return self._skills.get(skill_name)

        # 2. 키워드 트리거 (포함 여부 매칭)
        for trigger, skill_name in self._trigger_map.items():
            if not trigger.startswith("/") and trigger in text_lower:
                return self._skills.get(skill_name)

        return None

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

    async def reload_skills(self) -> int:
        """스킬을 다시 로드한다."""
        return await self.load_skills()

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
