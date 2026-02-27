"""LLM-as-Judge 자동 응답 평가 모듈.

봇 응답을 비동기로 자동 평가하고 결과를 저장한다.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.config import AutoEvaluationConfig
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.feedback_manager import FeedbackManager
    from core.llm_protocol import LLMClientProtocol

_EVAL_PROMPT_TEMPLATE = """\
당신은 AI 응답 품질 평가자입니다.
아래 사용자 질문과 AI 응답을 1~5점으로 평가하세요.

## 평가 기준
- 5점: 정확하고 완전하며 도움이 됨
- 4점: 대체로 좋지만 사소한 개선 여지 있음
- 3점: 보통. 핵심은 전달하나 부족한 부분 있음
- 2점: 부정확하거나 불완전함
- 1점: 부적절하거나 완전히 잘못된 답변

## 사용자 질문
{user_input}

## AI 응답
{bot_response}

## 출력 형식 (JSON)
{{"score": <1-5>, "explanation": "<한 문장 사유>"}}
"""


class AutoEvaluator:
    """비동기 LLM-as-Judge 자동 평가기."""

    def __init__(
        self,
        config: AutoEvaluationConfig,
        llm_client: "LLMClientProtocol",
        feedback_manager: "FeedbackManager",
        timezone_name: str = "UTC",
    ) -> None:
        self._config = config
        self._llm_client = llm_client
        self._feedback = feedback_manager
        self._timezone_name = timezone_name
        self._logger = get_logger("auto_evaluator")
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._in_flight: set[tuple[int, int]] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._consecutive_failures = 0
        self._cooldown_until: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def schedule_evaluation(
        self,
        chat_id: int,
        bot_message_id: int,
        user_input: str,
        bot_response: str,
    ) -> None:
        """평가를 비동기로 예약한다."""
        if not self._config.enabled:
            return

        if len(bot_response.strip()) < self._config.min_response_length:
            return

        key = (chat_id, bot_message_id)
        if key in self._in_flight:
            return

        self._in_flight.add(key)
        task = asyncio.create_task(
            self._evaluate_and_cleanup(chat_id, bot_message_id, user_input, bot_response),
            name=f"auto_eval_{chat_id}_{bot_message_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)

    async def _evaluate_and_cleanup(
        self,
        chat_id: int,
        bot_message_id: int,
        user_input: str,
        bot_response: str,
    ) -> None:
        """평가 실행 후 in_flight에서 제거한다."""
        key = (chat_id, bot_message_id)
        try:
            await self.evaluate(chat_id, bot_message_id, user_input, bot_response)
        except Exception as exc:
            self._logger.error(
                "auto_eval_failed",
                chat_id=chat_id,
                bot_message_id=bot_message_id,
                error=str(exc),
            )
        finally:
            self._in_flight.discard(key)

    async def evaluate(
        self,
        chat_id: int,
        bot_message_id: int,
        user_input: str,
        bot_response: str,
    ) -> dict | None:
        """응답을 평가하고 결과를 저장한다."""
        if not self._config.enabled:
            return None

        # 쿨다운 확인
        if time.monotonic() < self._cooldown_until:
            self._logger.debug("auto_eval_in_cooldown")
            return None

        # 일일 한도 확인
        start_utc, end_utc = self._utc_bounds_for_today()
        today_count = await self._feedback.count_today_evaluations(
            chat_id=chat_id,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        if today_count >= self._config.daily_limit:
            self._logger.debug("auto_eval_daily_limit_reached", count=today_count)
            return None

        async with self._semaphore:
            # 동시성 제어 후 재확인
            today_count = await self._feedback.count_today_evaluations(
                chat_id=chat_id,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            if today_count >= self._config.daily_limit:
                return None

            prompt = _EVAL_PROMPT_TEMPLATE.format(
                user_input=user_input[:500],
                bot_response=bot_response[:2000],
            )

            try:
                chat_response = await self._llm_client.chat(
                    messages=[
                        {"role": "system", "content": "응답 품질 평가 전문가입니다."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "object", "properties": {"score": {"type": "integer"}, "explanation": {"type": "string"}}, "required": ["score", "explanation"]},
                    temperature=0.1,
                    timeout=30,
                )
                raw = chat_response.content
            except Exception as exc:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._cooldown_until = time.monotonic() + self._config.cooldown_seconds
                    self._logger.warning(
                        "auto_eval_cooldown_activated",
                        seconds=self._config.cooldown_seconds,
                    )
                raise

            self._consecutive_failures = 0

            # JSON 파싱
            result = self._parse_result(raw)
            if result is None:
                self._logger.warning("auto_eval_parse_failed", raw=raw[:200])
                return None

            await self._feedback.store_auto_evaluation(
                chat_id=chat_id,
                bot_message_id=bot_message_id,
                user_input=user_input[:500],
                bot_response=bot_response[:500],
                score=result["score"],
                explanation=result.get("explanation"),
            )

            self._logger.info(
                "auto_eval_stored",
                chat_id=chat_id,
                score=result["score"],
            )
            return result

    def _utc_bounds_for_today(self) -> tuple[str, str]:
        """설정 timezone 기준 오늘의 UTC 경계를 반환한다."""
        tz: tzinfo = timezone.utc
        try:
            tz = ZoneInfo(self._timezone_name)
        except ZoneInfoNotFoundError:
            self._logger.warning(
                "auto_eval_timezone_fallback_utc",
                timezone=self._timezone_name,
            )

        now_local = datetime.now(tz)
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        end_utc = end_local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return start_utc, end_utc

    @staticmethod
    def _parse_result(raw: str) -> dict | None:
        """LLM 출력에서 score/explanation을 추출한다."""
        import json

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # JSON 블록 추출 시도
            import re
            match = re.search(r"\{[^}]+\}", raw)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                return None

        score = data.get("score")
        if not isinstance(score, int) or score < 1 or score > 5:
            return None

        return {
            "score": score,
            "explanation": str(data.get("explanation", ""))[:500],
        }

    async def shutdown(self) -> None:
        """진행 중인 평가가 완료될 때까지 대기한다."""
        if self._tasks:
            self._logger.info(
                "auto_eval_shutdown_waiting",
                pending=len(self._tasks),
            )
            tasks = tuple(self._tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
