"""대화 컨텍스트 압축 엔진.

오래된 대화 히스토리를 요약하여 LLM에 전달하는 토큰 수를 줄인다.
요약은 SQLite에 캐시하여 반복 생성을 방지한다.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.llm_protocol import LLMClientProtocol
    from core.memory import MemoryManager

_SUMMARY_SYSTEM_PROMPT = (
    "당신은 대화 요약 전문가입니다. "
    "아래 대화 내역을 2-3문장으로 간결하게 요약하세요. "
    "핵심 주제, 사용자의 관심사, 중요한 결정사항만 포함하세요. "
    "구체적인 코드나 긴 설명은 생략하세요."
)


class ContextCompressor:
    """대화 히스토리 요약 및 압축."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        memory: MemoryManager,
        recent_keep: int = 10,
        summary_refresh_interval: int = 10,
        summary_max_tokens: int = 200,
        summarize_concurrency: int = 1,
    ) -> None:
        self._llm_client = llm_client
        self._memory = memory
        self.recent_keep = recent_keep
        self._refresh_interval = summary_refresh_interval
        self._summary_max_tokens = summary_max_tokens
        self._summarize_sem = asyncio.Semaphore(summarize_concurrency)
        self._logger = get_logger("context_compressor")

    async def build_compressed_history(
        self,
        chat_id: int,
        max_history: int = 50,
    ) -> list[dict[str, str]]:
        """압축된 대화 히스토리를 반환한다.

        최근 N개는 원본, 나머지는 요약으로 대체한다.
        요약이 없으면 최근 N개만 반환한다 (graceful fallback).
        """
        recent = await self._memory.get_conversation(
            chat_id, limit=self.recent_keep
        )

        if len(recent) < self.recent_keep:
            # 히스토리가 충분하지 않으면 그대로 반환
            return recent

        # 캐시된 요약 조회
        summary_data = await self._memory.get_summary(chat_id)
        if summary_data is not None:
            summary_msg = {
                "role": "system",
                "content": f"[이전 대화 요약]\n{summary_data['summary']}",
            }
            return [summary_msg] + recent

        # 요약이 없으면 최근 N개만 반환 (블로킹 없음)
        return recent

    async def maybe_refresh_summary(self, chat_id: int) -> bool:
        """필요 시 백그라운드에서 요약을 갱신한다.

        Returns:
            요약이 갱신되었으면 True.
        """
        summary_data = await self._memory.get_summary(chat_id)
        last_id = summary_data["last_archive_id"] if summary_data else 0

        new_archived = await self._memory.get_archived_messages(
            chat_id, after_id=last_id
        )
        if len(new_archived) < self._refresh_interval:
            return False

        # 동시성 제한
        if self._summarize_sem.locked():
            self._logger.debug("summarize_skipped_busy", chat_id=chat_id)
            return False

        async with self._summarize_sem:
            summary_data = await self._memory.get_summary(chat_id)
            base_summary = summary_data["summary"] if summary_data else None
            last_id = summary_data["last_archive_id"] if summary_data else 0

            # 신규 아카이브만 요약 대상에 반영한다.
            new_archived = await self._memory.get_archived_messages(
                chat_id, after_id=last_id
            )
            if not new_archived:
                return False

            try:
                summary_text = await self._generate_summary(
                    new_archived, previous_summary=base_summary
                )
                last_archive_id = max(m["id"] for m in new_archived)
                prev_count = summary_data["message_count"] if summary_data else 0
                await self._memory.store_summary(
                    chat_id,
                    summary_text,
                    last_archive_id,
                    prev_count + len(new_archived),
                )
                self._logger.info(
                    "summary_refreshed",
                    chat_id=chat_id,
                    archive_count=prev_count + len(new_archived),
                )
                return True
            except Exception as exc:
                self._logger.warning(
                    "summary_generation_failed",
                    chat_id=chat_id,
                    error=str(exc),
                )
                return False

    async def _generate_summary(
        self,
        messages: list[dict],
        *,
        previous_summary: str | None = None,
    ) -> str:
        """LLM 호출로 대화를 요약한다."""
        # 최대 50개 메시지만 사용
        truncated = messages[-50:]
        conversation_text = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in truncated
        )
        if previous_summary:
            summary_input = (
                "[기존 요약]\n"
                f"{previous_summary}\n\n"
                "[신규 대화]\n"
                f"{conversation_text}\n\n"
                "기존 요약을 유지하되 신규 대화를 반영해 갱신 요약을 작성하세요."
            )
        else:
            summary_input = conversation_text

        chat_response = await self._llm_client.chat(
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": summary_input},
            ],
            max_tokens=self._summary_max_tokens,
            temperature=0.3,
            timeout=30,
        )
        return chat_response.content
