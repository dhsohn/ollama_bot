"""CLI interface for chat, dry-run, and test commands.

This lets you exercise the single-model response path and the RAG pipeline
without going through Telegram.

Usage:
  python -m apps.cli chat              interactive chat
  python -m apps.cli dry-run "query"   print model + RAG metadata only
  python -m apps.cli test              run built-in test cases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> None:
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_path()


async def _init_components():
    """Initialize the configured LLM client and optional RAG pipeline."""
    from core.config import OllamaConfig, load_config
    from core.ollama_client import OllamaClient

    config = load_config()
    if not config.ollama.chat_model.strip():
        raise ValueError(
            "ollama.chat_model must be configured."
        )
    llm: Any = OllamaClient(
        OllamaConfig(
            host=config.ollama.host,
            model=config.ollama.chat_model,
            temperature=config.ollama.chat_temperature,
            max_tokens=config.ollama.chat_max_tokens,
            num_ctx=config.ollama.chat_num_ctx,
            system_prompt=config.ollama.chat_system_prompt,
        )
    )
    llm.default_model = config.ollama.chat_model
    await llm.initialize()

    retrieval_client = None
    rag_pipeline = None

    if config.rag.enabled:
        from core.rag.context_builder import RAGContextBuilder
        from core.rag.indexer import RAGIndexer
        from core.rag.pipeline import RAGPipeline
        from core.rag.reranker import RAGReranker
        from core.rag.retriever import RAGRetriever

        retrieval_client = OllamaClient(
            OllamaConfig(
                host=config.ollama.host,
                model=config.ollama.embedding_model,
            )
        )
        await retrieval_client.initialize()

        index_dir = config.rag.index_dir or str(Path(config.data_dir) / "rag_index")
        indexer = RAGIndexer(
            config.rag,
            retrieval_client,
            config.ollama.embedding_model,
        )
        await indexer.initialize(str(Path(index_dir) / "rag.db"))

        kb_dirs = [path for path in config.rag.kb_dirs if Path(path).exists()]
        if kb_dirs:
            await indexer.index_corpus(kb_dirs)

        retriever = RAGRetriever(
            indexer,
            retrieval_client,
            config.ollama.embedding_model,
        )
        reranker = RAGReranker(
            retrieval_client,
            config.ollama.reranker_model,
            config.rag,
        )
        rag_pipeline = RAGPipeline(retriever, reranker, RAGContextBuilder(), config.rag)

    return llm, retrieval_client, rag_pipeline, config


async def _close_components(llm: Any, retrieval_client: Any) -> None:
    if retrieval_client is not None:
        await retrieval_client.close()
    await llm.close()


async def cmd_chat(args: argparse.Namespace) -> None:
    """Run an interactive chat session."""
    llm, retrieval_client, rag_pipeline, config = await _init_components()
    from core.config import get_system_prompt
    from core.text_utils import sanitize_model_output

    system_prompt = get_system_prompt(config)

    print("=== ollama_bot CLI Chat ===")
    print("Exit with Ctrl+C or 'exit'\n")

    try:
        while True:
            try:
                query = input("You> ").strip()
            except EOFError:
                break
            if not query or query.lower() in ("exit", "quit"):
                break

            model_name = llm.default_model

            # RAG
            rag_context = ""
            if rag_pipeline and rag_pipeline.should_trigger_rag(query):
                result = await rag_pipeline.execute(query)
                if result.contexts:
                    rag_context = result.contexts[0]
                    print(f"  [rag] {len(result.candidates)} sources, "
                          f"{result.trace.context_tokens_estimate} tokens")

            # Generation
            messages = [{"role": "system", "content": system_prompt}]
            if rag_context:
                from core.rag.context_builder import RAGContextBuilder
                messages[0]["content"] += RAGContextBuilder.build_rag_system_suffix(rag_context)
            messages.append({"role": "user", "content": query})

            response = await llm.chat(messages=messages, model=model_name)
            print(f"\nBot> {sanitize_model_output(response.content)}\n")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        await _close_components(llm, retrieval_client)


async def cmd_dry_run(args: argparse.Namespace) -> None:
    """Print model and RAG metadata as JSON."""
    llm, retrieval_client, rag_pipeline, _config = await _init_components()

    query = args.query
    result: dict = {"query": query}
    result["routing"] = {
        "selected_model": llm.default_model,
        "trigger": "single_model",
    }

    if rag_pipeline:
        triggered = rag_pipeline.should_trigger_rag(query)
        result["rag_triggered"] = triggered
        if triggered:
            rag_result = await rag_pipeline.execute(query)
            result["rag_trace"] = rag_result.trace.to_dict()
        else:
            result["rag_trace"] = None
    else:
        result["rag_triggered"] = False
        result["rag_trace"] = None

    print(json.dumps(result, ensure_ascii=False, indent=2))
    await _close_components(llm, retrieval_client)


_RAG_TEST_CASES: list[dict[str, Any]] = [
    {"input": "내 프로젝트의 README에 뭐라고 적혀있어?", "expected_rag": True},
    {"input": "kb 폴더에 있는 문서 검색해줘", "expected_rag": True},
    {"input": "내 노트에서 관련 내용을 찾아줘", "expected_rag": True},
    {"input": "프로젝트 문서를 인용해서 설명해줘", "expected_rag": True},
    {"input": "내 파일에 어디에 적혀있어?", "expected_rag": True},
    {"input": "논문의 결과를 출처와 함께 정리해줘", "expected_rag": True},
    {"input": "지식베이스에서 검색해", "expected_rag": True},
    {"input": "안녕하세요", "expected_rag": False},
    {"input": "오늘 날씨 어때?", "expected_rag": False},
    {"input": "파이썬으로 퀵소트 짜줘", "expected_rag": False},
]


async def cmd_test(args: argparse.Namespace) -> None:
    """Run built-in CLI test cases."""
    llm, retrieval_client, rag_pipeline, _config = await _init_components()

    print("=== Single Model Check ===\n")
    print(f"  [PASS] selected_model={llm.default_model}\n")

    print("=== RAG Trigger Test Cases ===\n")
    rag_correct = 0
    rag_total = len(_RAG_TEST_CASES)

    for i, tc in enumerate(_RAG_TEST_CASES, 1):
        if rag_pipeline:
            triggered = rag_pipeline.should_trigger_rag(tc["input"])
            match = triggered == tc["expected_rag"]
            status = "PASS" if match else "FAIL"
            if match:
                rag_correct += 1
            print(f"  [{status}] #{i}: expected_rag={tc['expected_rag']}, "
                  f"triggered={triggered}")
        else:
            print(f"  [SKIP] #{i}: rag_pipeline not available")

    print(f"\nRAG Trigger: {rag_correct}/{rag_total} ({rag_correct/rag_total*100:.0f}%)\n")

    await _close_components(llm, retrieval_client)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ollama_bot CLI - single-model and RAG checks",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="interactive chat")

    dry_p = sub.add_parser("dry-run", help="print model and RAG metadata only")
    dry_p.add_argument("query", type=str, help="test query")

    sub.add_parser("test", help="run built-in test cases")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    func_map = {
        "chat": cmd_chat,
        "dry-run": cmd_dry_run,
        "test": cmd_test,
    }
    asyncio.run(func_map[args.command](args))


if __name__ == "__main__":
    main()
