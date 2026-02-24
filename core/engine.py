"""메인 엔진 — 대화 오케스트레이션, 컨텍스트 관리, 라우팅.

모든 사용자 메시지 처리의 중앙 허브.
텔레그램 핸들러로부터 입력을 받아 적절한 처리 후 응답을 반환한다.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from core.config import AppSettings
from core.logging_setup import get_logger
from core.memory import MemoryManager
from core.ollama_client import OllamaClient
from core.skill_manager import SkillDefinition, SkillManager


@dataclass
class _PreparedRequest:
    """LLM 호출에 필요한 사전 계산 결과."""

    skill: SkillDefinition | None
    messages: list[dict[str, str]]
    timeout: int


class Engine:
    """대화 처리 엔진. 스킬 트리거, 컨텍스트 관리, LLM 호출을 오케스트레이션한다."""

    def __init__(
        self,
        config: AppSettings,
        ollama: OllamaClient,
        memory: MemoryManager,
        skills: SkillManager,
    ) -> None:
        self._config = config
        self._ollama = ollama
        self._memory = memory
        self._skills = skills
        self._system_prompt = config.ollama.system_prompt
        self._max_conversation_length = config.bot.max_conversation_length
        self._start_time = time.monotonic()
        self._logger = get_logger("engine")

    async def process_message(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
    ) -> str:
        """사용자 메시지를 처리하고 응답을 반환한다.

        1. 스킬 트리거 매칭
        2. 컨텍스트 빌드 (시스템 프롬프트 + 대화 기록 + 사용자 입력)
        3. LLM 호출
        4. 메모리 저장
        """
        prepared = await self._prepare_request(chat_id, text, stream=False)

        response = await self._ollama.chat(
            messages=prepared.messages,
            model=model_override,
            timeout=prepared.timeout,
        )
        await self._persist_turn(
            chat_id=chat_id,
            user_text=text,
            assistant_text=response,
            skill=prepared.skill,
        )

        return response

    async def process_message_stream(
        self,
        chat_id: int,
        text: str,
        model_override: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 방식으로 메시지를 처리한다. 청크를 순차 반환한다."""
        prepared = await self._prepare_request(chat_id, text, stream=True)

        full_response = ""
        async for chunk in self._ollama.chat_stream(
            messages=prepared.messages,
            model=model_override,
            timeout=prepared.timeout,
        ):
            full_response += chunk
            yield chunk

        await self._persist_turn(
            chat_id=chat_id,
            user_text=text,
            assistant_text=full_response,
            skill=prepared.skill,
        )

    async def _prepare_request(
        self,
        chat_id: int,
        text: str,
        *,
        stream: bool,
    ) -> _PreparedRequest:
        """스킬 매칭과 컨텍스트 빌드를 공통 처리한다."""
        skill = self._skills.match_trigger(text)
        if skill:
            self._logger.info(
                "skill_triggered_stream" if stream else "skill_triggered",
                chat_id=chat_id,
                skill=skill.name,
            )
            messages = await self._build_context(chat_id, text, skill=skill)
            timeout = skill.timeout
        else:
            messages = await self._build_context(chat_id, text)
            timeout = self._config.bot.response_timeout

        return _PreparedRequest(
            skill=skill,
            messages=messages,
            timeout=timeout,
        )

    async def _persist_turn(
        self,
        chat_id: int,
        user_text: str,
        assistant_text: str,
        skill: SkillDefinition | None = None,
    ) -> None:
        """사용자/어시스턴트 턴을 메모리에 일관되게 저장한다."""
        metadata = {"skill": skill.name} if skill else None
        await self._memory.add_message(chat_id, "user", user_text, metadata=metadata)
        await self._memory.add_message(chat_id, "assistant", assistant_text)

    async def _build_context(
        self,
        chat_id: int,
        text: str,
        skill: SkillDefinition | None = None,
    ) -> list[dict[str, str]]:
        """LLM에 전달할 메시지 목록을 조립한다."""
        if skill:
            # 스킬 모드: 스킬 시스템 프롬프트 사용
            system = skill.system_prompt
            # 스킬에서도 최근 대화 일부를 컨텍스트로 포함
            history = await self._memory.get_conversation(chat_id, limit=5)
        else:
            system = self._system_prompt
            history = await self._memory.get_conversation(
                chat_id, limit=self._max_conversation_length
            )

        # 사용자 선호도를 시스템 프롬프트에 추가
        preferences = await self._memory.recall_memory(chat_id, category="preferences")
        if preferences:
            pref_lines = [f"- {p['key']}: {p['value']}" for p in preferences]
            system += (
                "\n\n[사용자 고정 정보 및 선호도]\n"
                "아래 정보를 참고하여 일관된 응답을 제공하세요:\n"
                + "\n".join(pref_lines)
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        messages.extend(history)

        # 스킬인 경우 트리거 명령어를 제거한 입력 사용
        if skill:
            clean_input = text
            for trigger in skill.triggers:
                if text.lower().startswith(trigger.lower()):
                    clean_input = text[len(trigger):].strip()
                    break
            messages.append({"role": "user", "content": clean_input or text})
        else:
            messages.append({"role": "user", "content": text})

        return messages

    async def execute_skill(
        self,
        skill_name: str,
        parameters: dict,
        chat_id: int | None = None,
    ) -> str:
        """프로그래밍 방식으로 스킬을 실행한다 (auto_scheduler용)."""
        skill = self._skills.get_skill(skill_name)
        if not skill:
            return f"스킬 '{skill_name}'을(를) 찾을 수 없습니다."

        input_text = parameters.get("input_text", parameters.get("query", ""))
        messages = [
            {"role": "system", "content": skill.system_prompt},
            {"role": "user", "content": input_text},
        ]

        response = await self._ollama.chat(
            messages=messages,
            timeout=skill.timeout,
        )

        # 채팅 ID가 있으면 메모리에 저장
        if chat_id:
            await self._memory.add_message(
                chat_id, "assistant", response,
                metadata={"skill": skill_name, "auto": True},
            )

        return response

    async def process_prompt(
        self,
        prompt: str,
        chat_id: int | None = None,
        format: str | dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """단순 프롬프트를 LLM에 전달한다 (auto_scheduler의 prompt 타입용)."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._ollama.chat(
            messages=messages,
            format=format,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def change_model(self, model: str) -> dict:
        """런타임 기본 모델을 변경한다."""
        models = await self._ollama.list_models()
        available_names = [m["name"] for m in models]

        if model not in available_names:
            return {
                "success": False,
                "error": f"모델 '{model}'을(를) 찾을 수 없습니다.",
                "available": available_names,
            }

        old_model = self._ollama.default_model
        self._ollama.default_model = model
        self._logger.info(
            "model_changed", old_model=old_model, new_model=model
        )
        return {"success": True, "old_model": old_model, "new_model": model}

    async def list_models(self) -> list[dict]:
        """설치된 모델 목록을 반환한다."""
        return await self._ollama.list_models()

    def get_current_model(self) -> str:
        """현재 기본 모델 이름을 반환한다."""
        return self._ollama.default_model

    async def reload_skills(self) -> int:
        """스킬 정의를 다시 로드한다."""
        return await self._skills.reload_skills()

    def list_skills(self) -> list[dict]:
        """로드된 스킬 목록을 반환한다."""
        return self._skills.list_skills()

    async def get_memory_stats(self, chat_id: int) -> dict:
        """채팅 메모리 통계를 조회한다."""
        return await self._memory.get_memory_stats(chat_id)

    async def clear_conversation(self, chat_id: int) -> int:
        """채팅 대화 기록을 삭제한다."""
        return await self._memory.clear_conversation(chat_id)

    async def export_conversation_markdown(
        self,
        chat_id: int,
        output_dir: Path,
    ) -> Path:
        """채팅 대화 기록을 마크다운으로 내보낸다."""
        return await self._memory.export_conversation_markdown(chat_id, output_dir)

    async def get_status(self) -> dict:
        """시스템 전체 상태를 반환한다."""
        ollama_health = await self._ollama.health_check()
        uptime_seconds = time.monotonic() - self._start_time

        return {
            "uptime_seconds": int(uptime_seconds),
            "uptime_human": self._format_uptime(uptime_seconds),
            "ollama": ollama_health,
            "skills_loaded": self._skills.skill_count,
            "current_model": self._ollama.default_model,
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}시간 {minutes}분 {secs}초"
        if minutes > 0:
            return f"{minutes}분 {secs}초"
        return f"{secs}초"
