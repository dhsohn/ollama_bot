"""export_training_data callable 테스트."""

from __future__ import annotations

import json

import pytest

from core.automation_callables_impl.export_training_data import (
    build_export_training_data_callable,
)


class _DummyFeedback:
    async def export_kto_dataset(self, min_preview_length: int = 20) -> list[dict]:
        return [
            {
                "prompt": "충분히 긴 질문 텍스트입니다.",
                "completion": "충분히 긴 답변 텍스트입니다.",
                "label": True,
            }
        ]


class _DummyLogger:
    def __init__(self) -> None:
        self.events: list[tuple[tuple, dict]] = []

    def info(self, *args, **kwargs) -> None:
        self.events.append((args, kwargs))


@pytest.mark.asyncio
async def test_export_training_data_writes_jsonl_and_meta(tmp_path) -> None:
    feedback = _DummyFeedback()
    logger = _DummyLogger()
    callable_fn = build_export_training_data_callable(feedback, str(tmp_path), logger)

    result = await callable_fn(min_preview_length=5)

    assert "KTO 데이터 내보내기 완료" in result

    training_dir = tmp_path / "training"
    jsonl_files = list(training_dir.glob("kto_dataset_*.jsonl"))
    assert len(jsonl_files) == 1

    jsonl_path = jsonl_files[0]
    meta_path = jsonl_path.with_suffix(".meta.json")
    assert meta_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["label"] is True

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["format"] == "kto"
    assert meta["total_samples"] == 1
