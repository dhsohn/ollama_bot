"""AutoEvaluator 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.auto_evaluator import AutoEvaluator
from core.config import AutoEvaluationConfig
from core.llm_types import ChatResponse


@pytest.fixture
def eval_config() -> AutoEvaluationConfig:
    return AutoEvaluationConfig(
        enabled=True,
        daily_limit=50,
        min_response_length=10,
        max_concurrency=2,
        cooldown_seconds=5,
    )


@pytest.fixture
def mock_ollama() -> AsyncMock:
    client = AsyncMock()
    client.chat = AsyncMock(
        return_value=ChatResponse(content='{"score": 4, "explanation": "Good response"}')
    )
    return client


@pytest.fixture
def mock_feedback() -> AsyncMock:
    fm = AsyncMock()
    fm.count_today_evaluations = AsyncMock(return_value=0)
    fm.store_auto_evaluation = AsyncMock()
    return fm


@pytest.fixture
def evaluator(eval_config, mock_ollama, mock_feedback) -> AutoEvaluator:
    return AutoEvaluator(
        config=eval_config,
        llm_client=mock_ollama,
        feedback_manager=mock_feedback,
    )


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_stores_result(self, evaluator, mock_feedback) -> None:
        result = await evaluator.evaluate(111, 1, "질문", "이것은 충분히 긴 응답입니다")
        assert result is not None
        assert result["score"] == 4
        mock_feedback.store_auto_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_disabled(self, eval_config, mock_ollama, mock_feedback) -> None:
        eval_config.enabled = False
        ev = AutoEvaluator(config=eval_config, llm_client=mock_ollama, feedback_manager=mock_feedback)
        result = await ev.evaluate(111, 1, "질문", "응답")
        assert result is None
        mock_feedback.store_auto_evaluation.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_daily_limit(self, evaluator, mock_feedback) -> None:
        mock_feedback.count_today_evaluations = AsyncMock(return_value=100)
        result = await evaluator.evaluate(111, 1, "질문", "충분히 긴 응답입니다")
        assert result is None
        mock_feedback.store_auto_evaluation.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_parse_failure(self, evaluator, mock_ollama, mock_feedback) -> None:
        mock_ollama.chat = AsyncMock(return_value=ChatResponse(content="invalid json"))
        result = await evaluator.evaluate(111, 1, "질문", "충분히 긴 응답입니다")
        assert result is None
        mock_feedback.store_auto_evaluation.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_uses_chat_scoped_daily_budget(
        self,
        evaluator,
        mock_feedback,
    ) -> None:
        await evaluator.evaluate(111, 1, "질문", "충분히 긴 응답입니다")

        assert mock_feedback.count_today_evaluations.await_count >= 2
        first_call = mock_feedback.count_today_evaluations.await_args_list[0]
        assert first_call.kwargs["chat_id"] == 111
        assert first_call.kwargs["start_utc"] < first_call.kwargs["end_utc"]


class TestScheduleEvaluation:
    @pytest.mark.asyncio
    async def test_schedule_deduplicates(self, evaluator, mock_ollama) -> None:
        # schedule_evaluation은 비동기 태스크를 생성
        evaluator.schedule_evaluation(111, 1, "질문", "충분히 긴 응답입니다")
        evaluator.schedule_evaluation(111, 1, "질문", "충분히 긴 응답입니다")  # 중복
        # in_flight에는 하나만 있어야 함
        assert (111, 1) in evaluator._in_flight
        # 태스크가 완료될 때까지 잠시 대기
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_schedule_skips_short_response(self, evaluator, mock_ollama) -> None:
        evaluator.schedule_evaluation(111, 1, "질문", "짧은")
        assert (111, 1) not in evaluator._in_flight

    @pytest.mark.asyncio
    async def test_schedule_skips_when_disabled(self, eval_config, mock_ollama, mock_feedback) -> None:
        eval_config.enabled = False
        ev = AutoEvaluator(config=eval_config, llm_client=mock_ollama, feedback_manager=mock_feedback)
        ev.schedule_evaluation(111, 1, "질문", "충분히 긴 응답입니다")
        assert (111, 1) not in ev._in_flight


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_clears_in_flight(self, evaluator) -> None:
        # shutdown은 in_flight가 비어야 즉시 반환
        await evaluator.shutdown()
        assert len(evaluator._in_flight) == 0

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_running_tasks(
        self,
        evaluator,
        mock_ollama,
    ) -> None:
        gate = asyncio.Event()
        mock_ollama.chat = AsyncMock(side_effect=_slow_chat_response(gate))

        evaluator.schedule_evaluation(111, 1, "질문", "충분히 긴 응답입니다")
        await asyncio.sleep(0)

        shutdown_task = asyncio.create_task(evaluator.shutdown())
        await asyncio.sleep(0.05)
        assert not shutdown_task.done()

        gate.set()
        await shutdown_task
        assert len(evaluator._tasks) == 0
        assert len(evaluator._in_flight) == 0


def _slow_chat_response(gate: asyncio.Event):
    async def _impl(*args, **kwargs):
        await gate.wait()
        return ChatResponse(content='{"score": 4, "explanation": "Good response"}')

    return _impl


class TestParseResult:
    def test_valid_json(self) -> None:
        result = AutoEvaluator._parse_result('{"score": 3, "explanation": "OK"}')
        assert result == {"score": 3, "explanation": "OK"}

    def test_invalid_score(self) -> None:
        result = AutoEvaluator._parse_result('{"score": 0, "explanation": "Bad"}')
        assert result is None

    def test_score_out_of_range(self) -> None:
        result = AutoEvaluator._parse_result('{"score": 6, "explanation": "Too high"}')
        assert result is None

    def test_json_in_text(self) -> None:
        result = AutoEvaluator._parse_result('Here is the result: {"score": 5, "explanation": "Great"}')
        assert result is not None
        assert result["score"] == 5

    def test_invalid_json(self) -> None:
        result = AutoEvaluator._parse_result("not json at all")
        assert result is None
