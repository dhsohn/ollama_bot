"""스킬 매니저 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.security import SecurityManager
from core.config import SecurityConfig
from core.skill_manager import SkillManager, SkillDefinition


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """테스트용 스킬 디렉토리를 생성한다."""
    builtin = tmp_path / "_builtin"
    custom = tmp_path / "custom"
    builtin.mkdir()
    custom.mkdir()

    # 유효한 스킬
    (builtin / "summarize.yaml").write_text(
        """
name: "summarize"
description: "텍스트 요약"
version: "1.0"
triggers:
  - "/summarize"
  - "요약해줘"
system_prompt: "You are a summarizer."
allowed_tools: []
timeout: 30
security_level: "safe"
""",
        encoding="utf-8",
    )

    # 보안 위반 스킬 (safe인데 shell 요청)
    (builtin / "bad_skill.yaml").write_text(
        """
name: "bad_skill"
description: "위험한 스킬"
version: "1.0"
triggers:
  - "/bad"
system_prompt: "Bad."
allowed_tools:
  - shell
timeout: 30
security_level: "safe"
""",
        encoding="utf-8",
    )

    # 커스텀 스킬
    (custom / "custom_skill.yaml").write_text(
        """
name: "custom_skill"
description: "커스텀 스킬"
version: "1.0"
triggers:
  - "/custom"
  - "커스텀"
system_prompt: "Custom skill prompt."
allowed_tools: []
timeout: 30
security_level: "safe"
""",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def security() -> SecurityManager:
    config = SecurityConfig(allowed_users=[111], rate_limit=30)
    return SecurityManager(config)


@pytest.fixture
def skill_manager(security: SecurityManager, skills_dir: Path) -> SkillManager:
    return SkillManager(security=security, skills_dir=str(skills_dir))


class TestSkillLoading:
    @pytest.mark.asyncio
    async def test_load_valid_skills(self, skill_manager: SkillManager) -> None:
        count = await skill_manager.load_skills()
        # summarize + custom_skill = 2 (bad_skill은 보안 위반으로 제외)
        assert count == 2

    @pytest.mark.asyncio
    async def test_security_violation_skipped(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        assert skill_manager.get_skill("bad_skill") is None

    @pytest.mark.asyncio
    async def test_builtin_and_custom_both_loaded(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        assert skill_manager.get_skill("summarize") is not None
        assert skill_manager.get_skill("custom_skill") is not None


class TestTriggerMatching:
    @pytest.mark.asyncio
    async def test_command_trigger(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        skill = skill_manager.match_trigger("/summarize 이 텍스트를 요약해줘")
        assert skill is not None
        assert skill.name == "summarize"

    @pytest.mark.asyncio
    async def test_keyword_trigger(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        skill = skill_manager.match_trigger("이 텍스트를 요약해줘")
        assert skill is not None
        assert skill.name == "summarize"

    @pytest.mark.asyncio
    async def test_no_match(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        skill = skill_manager.match_trigger("오늘 날씨 어때?")
        assert skill is None

    @pytest.mark.asyncio
    async def test_command_priority(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        # /summarize 명령어가 매칭
        skill = skill_manager.match_trigger("/summarize")
        assert skill is not None
        assert skill.name == "summarize"

    @pytest.mark.asyncio
    async def test_keyword_trigger_preserves_load_order_priority(
        self, skill_manager: SkillManager
    ) -> None:
        """본문 내 위치보다 로드 순서상 앞선 키워드 트리거를 우선한다."""
        await skill_manager.load_skills()
        skill = skill_manager.match_trigger("커스텀 기능으로 이 텍스트를 요약해줘")
        assert skill is not None
        # summarize(요약해줘)가 custom_skill(커스텀)보다 먼저 로드된다.
        assert skill.name == "summarize"


class TestSkillContext:
    @pytest.mark.asyncio
    async def test_get_skill_context(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        skill = skill_manager.get_skill("summarize")
        assert skill is not None

        context = skill_manager.get_skill_context(skill, "/summarize 테스트 텍스트")
        assert len(context) == 2
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "You are a summarizer."
        assert context[1]["role"] == "user"
        assert context[1]["content"] == "테스트 텍스트"


class TestListSkills:
    @pytest.mark.asyncio
    async def test_list_skills(self, skill_manager: SkillManager) -> None:
        await skill_manager.load_skills()
        skills_list = skill_manager.list_skills()
        assert len(skills_list) == 2
        names = [s["name"] for s in skills_list]
        assert "summarize" in names
        assert "custom_skill" in names


class TestReload:
    @pytest.mark.asyncio
    async def test_reload_skills(self, skill_manager: SkillManager) -> None:
        count1 = await skill_manager.load_skills()
        count2 = await skill_manager.reload_skills()
        assert count1 == count2


class TestSkillConflicts:
    @pytest.mark.asyncio
    async def test_duplicate_skill_name_raises(
        self,
        skill_manager: SkillManager,
        skills_dir: Path,
    ) -> None:
        (skills_dir / "custom" / "dup_name.yaml").write_text(
            """
name: "summarize"
description: "중복 이름"
version: "1.0"
triggers:
  - "/dup_name"
system_prompt: "Duplicate name."
allowed_tools: []
timeout: 30
security_level: "safe"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Duplicate skill name"):
            await skill_manager.load_skills()

    @pytest.mark.asyncio
    async def test_duplicate_trigger_raises(
        self,
        skill_manager: SkillManager,
        skills_dir: Path,
    ) -> None:
        (skills_dir / "custom" / "dup_trigger.yaml").write_text(
            """
name: "another_skill"
description: "중복 트리거"
version: "1.0"
triggers:
  - "/summarize"
system_prompt: "Duplicate trigger."
allowed_tools: []
timeout: 30
security_level: "safe"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Duplicate trigger"):
            await skill_manager.load_skills()

    @pytest.mark.asyncio
    async def test_duplicate_trigger_within_same_skill_is_ignored(
        self,
        skill_manager: SkillManager,
        skills_dir: Path,
    ) -> None:
        (skills_dir / "custom" / "same_skill_dup_trigger.yaml").write_text(
            """
name: "self_dup_trigger"
description: "같은 스킬 내부 중복 트리거"
version: "1.0"
triggers:
  - "/selfdup"
  - "/selfdup"
  - "중복키워드"
  - "중복키워드"
system_prompt: "Duplicate trigger in same skill."
allowed_tools: []
timeout: 30
security_level: "safe"
""",
            encoding="utf-8",
        )

        count = await skill_manager.load_skills()
        assert count == 3  # summarize + custom_skill + self_dup_trigger

        skill = skill_manager.match_trigger("/selfdup test")
        assert skill is not None
        assert skill.name == "self_dup_trigger"

        keyword_skill = skill_manager.match_trigger("중복키워드 테스트")
        assert keyword_skill is not None
        assert keyword_skill.name == "self_dup_trigger"
