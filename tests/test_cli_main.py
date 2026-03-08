"""CLI entrypoint tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.cli import main as cli_main
from core.llm_types import ChatResponse
from core.rag.types import RAGResult, RAGTrace, RetrievedItem


def _fake_chunk(text: str = "chunk") -> RetrievedItem:
    metadata = SimpleNamespace(
        doc_id="doc-1",
        source_path="kb/readme.md",
        chunk_id=0,
        section_title="intro",
        tokens_estimate=42,
    )
    chunk = SimpleNamespace(text=text, metadata=metadata)
    return RetrievedItem(chunk=chunk, retrieval_score=0.7, rerank_score=0.9)


@pytest.mark.asyncio
async def test_cmd_dry_run_reports_rag_trace_and_closes_components(monkeypatch, capsys) -> None:
    llm = SimpleNamespace(default_model="test-model", close=AsyncMock())
    retrieval = SimpleNamespace(close=AsyncMock())
    rag_result = RAGResult(
        contexts=["context"],
        candidates=[_fake_chunk()],
        trace=RAGTrace(rag_used=True, rerank_used=True, total_latency_ms=12.34),
    )
    rag_pipeline = SimpleNamespace(
        should_trigger_rag=lambda query: True,
        execute=AsyncMock(return_value=rag_result),
    )

    monkeypatch.setattr(
        cli_main,
        "_init_components",
        AsyncMock(return_value=(llm, retrieval, rag_pipeline, SimpleNamespace())),
    )

    await cli_main.cmd_dry_run(SimpleNamespace(query="kb에서 찾아줘"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["rag_triggered"] is True
    assert payload["routing"]["selected_model"] == "test-model"
    assert payload["rag_trace"]["rag_used"] is True
    retrieval.close.assert_awaited_once()
    llm.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cmd_test_reports_passes_and_closes_components(monkeypatch, capsys) -> None:
    llm = SimpleNamespace(default_model="test-model", close=AsyncMock())
    retrieval = SimpleNamespace(close=AsyncMock())
    rag_pipeline = SimpleNamespace(
        should_trigger_rag=lambda query: any(keyword in query for keyword in ("README", "kb", "지식베이스", "논문", "내 파일")),
    )

    monkeypatch.setattr(
        cli_main,
        "_init_components",
        AsyncMock(return_value=(llm, retrieval, rag_pipeline, SimpleNamespace())),
    )

    await cli_main.cmd_test(SimpleNamespace())

    output = capsys.readouterr().out
    assert "provider: lemonade" in output
    assert "RAG Trigger:" in output
    retrieval.close.assert_awaited_once()
    llm.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cmd_chat_runs_single_round_trip_with_rag(monkeypatch, capsys) -> None:
    llm = SimpleNamespace(
        default_model="test-model",
        chat=AsyncMock(return_value=ChatResponse(content="답변 완료")),
        close=AsyncMock(),
    )
    retrieval = SimpleNamespace(close=AsyncMock())
    rag_result = RAGResult(
        contexts=["retrieved context"],
        candidates=[_fake_chunk("retrieved context")],
        trace=RAGTrace(rag_used=True, context_tokens_estimate=128),
    )
    rag_pipeline = SimpleNamespace(
        should_trigger_rag=lambda query: True,
        execute=AsyncMock(return_value=rag_result),
    )
    config = SimpleNamespace(
        lemonade=SimpleNamespace(system_prompt="system prompt"),
    )

    monkeypatch.setattr(
        cli_main,
        "_init_components",
        AsyncMock(return_value=(llm, retrieval, rag_pipeline, config)),
    )
    inputs = iter(["프로젝트 문서 찾아줘", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    await cli_main.cmd_chat(SimpleNamespace())

    output = capsys.readouterr().out
    assert "[rag]" in output
    assert "Bot> 답변 완료" in output
    llm.chat.assert_awaited_once()
    retrieval.close.assert_awaited_once()
    llm.close.assert_awaited_once()


def test_main_prints_help_when_no_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_main.sys, "argv", ["python"])

    cli_main.main()

    assert "ollama_bot CLI" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["python", "chat"], "chat"),
        (["python", "dry-run", "hello"], "dry-run"),
        (["python", "test"], "test"),
    ],
)
def test_main_dispatches_selected_command(
    monkeypatch,
    argv: list[str],
    expected: str,
) -> None:
    called: list[tuple[str, object]] = []

    async def fake_chat(args) -> None:
        called.append(("chat", args))

    async def fake_dry_run(args) -> None:
        called.append(("dry-run", args))

    async def fake_test(args) -> None:
        called.append(("test", args))

    monkeypatch.setattr(cli_main, "cmd_chat", fake_chat)
    monkeypatch.setattr(cli_main, "cmd_dry_run", fake_dry_run)
    monkeypatch.setattr(cli_main, "cmd_test", fake_test)
    monkeypatch.setattr(cli_main.sys, "argv", argv)

    cli_main.main()

    assert called and called[0][0] == expected
