"""CLI 인터페이스 — chat, dry-run, test.

단일 모델 응답 경로와 RAG 파이프라인을 텔레그램 없이 직접 테스트할 수 있다.

사용법:
  python -m apps.cli chat              대화형 채팅
  python -m apps.cli dry-run "쿼리"    모델+RAG 결과만 출력
  python -m apps.cli test              테스트 케이스 실행
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


async def _init_components():
    """설정 기반 LLM 클라이언트와 선택적 RAGPipeline을 초기화한다."""
    os.environ.setdefault("ALLOW_LOCAL_RUN", "1")
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "cli-mode")
    os.environ.setdefault("ALLOWED_TELEGRAM_USERS", "0")

    from core.config import load_config
    from core.lemonade_multi_client import build_lemonade_client
    from core.ollama_client import OllamaClient

    config = load_config()
    provider = config.llm_provider.strip().lower()
    llm: Any

    if provider == "lemonade":
        llm = build_lemonade_client(
            config.lemonade,
            fallback_ollama=config.ollama,
        )
    elif provider == "ollama":
        llm = OllamaClient(config.ollama)
    else:
        raise ValueError(f"Unsupported llm_provider: {provider}")
    await llm.initialize()

    rag_pipeline = None

    if provider == "lemonade" and config.rag.enabled:
        from core.rag.context_builder import RAGContextBuilder
        from core.rag.indexer import RAGIndexer
        from core.rag.pipeline import RAGPipeline
        from core.rag.reranker import RAGReranker
        from core.rag.retriever import RAGRetriever

        index_dir = config.rag.index_dir or str(
            Path(config.data_dir) / "rag_index"
        )
        indexer = RAGIndexer(config.rag, llm, config.model_registry.embedding_model)
        await indexer.initialize(str(Path(index_dir) / "rag.db"))

        kb_dirs = [path for path in config.rag.kb_dirs if Path(path).exists()]
        if kb_dirs:
            await indexer.index_corpus(kb_dirs)

        retriever = RAGRetriever(indexer, llm, config.model_registry.embedding_model)
        reranker = RAGReranker(llm, config.model_registry.reranker_model, config.rag)
        rag_pipeline = RAGPipeline(retriever, reranker, RAGContextBuilder(), config.rag)

    return llm, rag_pipeline, config, provider


async def cmd_chat(args: argparse.Namespace) -> None:
    """대화형 채팅."""
    llm, rag_pipeline, config, provider = await _init_components()
    from core.text_utils import sanitize_model_output

    print("=== ollama_bot CLI Chat ===")
    print(f"provider: {provider}")
    if provider != "lemonade" and config.rag.enabled:
        print("  [notice] rag는 lemonade provider에서만 활성화됩니다.")
    print("종료: Ctrl+C 또는 'exit'\n")

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

            # 생성
            messages = [{"role": "system", "content": config.ollama.system_prompt}]
            if rag_context:
                from core.rag.context_builder import RAGContextBuilder
                messages[0]["content"] += RAGContextBuilder.build_rag_system_suffix(rag_context)
            messages.append({"role": "user", "content": query})

            response = await llm.chat(messages=messages, model=model_name)
            print(f"\nBot> {sanitize_model_output(response.content)}\n")
    except KeyboardInterrupt:
        print("\n종료.")
    finally:
        await llm.close()


async def cmd_dry_run(args: argparse.Namespace) -> None:
    """모델 + RAG 결과만 JSON 출력."""
    llm, rag_pipeline, _config, provider = await _init_components()

    query = args.query
    result: dict = {"query": query, "provider": provider}
    result["routing"] = {
        "selected_model": llm.default_model,
        "selected_role": "default",
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
    await llm.close()


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
    """테스트 케이스 실행."""
    llm, rag_pipeline, _config, provider = await _init_components()
    print(f"provider: {provider}")
    if provider != "lemonade":
        print("  [notice] rag 테스트는 lemonade provider에서만 실행됩니다.\n")

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

    await llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ollama_bot CLI — 단일 모델/RAG 테스트",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="대화형 채팅")

    dry_p = sub.add_parser("dry-run", help="모델+RAG 결과만 출력")
    dry_p.add_argument("query", type=str, help="테스트 쿼리")

    sub.add_parser("test", help="테스트 케이스 실행")

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
