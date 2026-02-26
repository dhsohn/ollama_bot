"""CLI 인터페이스 — chat, dry-run, test.

모델 라우팅과 RAG 파이프라인을 텔레그램 없이 직접 테스트할 수 있다.

사용법:
  python -m apps.cli chat              대화형 채팅
  python -m apps.cli dry-run "쿼리"    라우팅+RAG 결과만 출력
  python -m apps.cli test              테스트 케이스 실행
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


async def _init_components():
    """설정 기반 LLM 클라이언트와 선택적 ModelRouter/RAGPipeline을 초기화한다."""
    os.environ.setdefault("ALLOW_LOCAL_RUN", "1")
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "cli-mode")
    os.environ.setdefault("ALLOWED_TELEGRAM_USERS", "0")

    from core.config import load_config
    from core.lemonade_client import LemonadeClient
    from core.ollama_client import OllamaClient

    config = load_config()
    provider = config.llm_provider.strip().lower()

    if provider == "lemonade":
        llm = LemonadeClient(config.lemonade, fallback_ollama=config.ollama)
    elif provider == "ollama":
        llm = OllamaClient(config.ollama)
    else:
        raise ValueError(f"Unsupported llm_provider: {provider}")
    await llm.initialize()

    model_router = None
    rag_pipeline = None

    if provider == "lemonade" and config.model_routing.enabled:
        from core.model_registry import ModelRegistry
        from core.model_router import ModelRouter

        registry = ModelRegistry(config.model_registry, llm)
        await registry.initialize()
        model_router = ModelRouter(
            config=config.model_routing,
            registry=registry,
            client=llm,
            embedding_model=config.model_registry.embedding_model,
        )
        await model_router.initialize()

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

        if Path(config.rag.kb_dir).exists():
            await indexer.index_corpus(config.rag.kb_dir)

        retriever = RAGRetriever(indexer, llm, config.model_registry.embedding_model)
        reranker = RAGReranker(llm, config.model_registry.reranker_model, config.rag)
        rag_pipeline = RAGPipeline(retriever, reranker, RAGContextBuilder(), config.rag)

    return llm, model_router, rag_pipeline, config, provider


async def cmd_chat(args: argparse.Namespace) -> None:
    """대화형 채팅."""
    llm, model_router, rag_pipeline, config, provider = await _init_components()
    print("=== ollama_bot CLI Chat ===")
    print(f"provider: {provider}")
    if provider != "lemonade" and (config.model_routing.enabled or config.rag.enabled):
        print("  [notice] model_routing/rag는 lemonade provider에서만 활성화됩니다.")
    print("종료: Ctrl+C 또는 'exit'\n")

    try:
        while True:
            try:
                query = input("You> ").strip()
            except EOFError:
                break
            if not query or query.lower() in ("exit", "quit"):
                break

            # 라우팅
            model_name = llm.default_model
            if model_router:
                decision = await model_router.route(query)
                model_name = decision.selected_model
                print(f"  [routing] {decision.selected_role} ({decision.trigger})")

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
            print(f"\nBot> {response.content}\n")
    except KeyboardInterrupt:
        print("\n종료.")
    finally:
        await llm.close()


async def cmd_dry_run(args: argparse.Namespace) -> None:
    """라우팅 + RAG 결과만 JSON 출력."""
    llm, model_router, rag_pipeline, _config, provider = await _init_components()

    query = args.query
    result: dict = {"query": query, "provider": provider}

    if model_router:
        decision = await model_router.route(query)
        result["routing"] = decision.to_dict()
    else:
        result["routing"] = None

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


# 테스트 케이스
_ROUTING_TEST_CASES = [
    # Vision (5)
    {"input": "이 사진에서 뭐가 보여?", "images": [b"fake"], "expected_role": "vision"},
    {"input": "이미지를 분석해줘", "images": [b"fake"], "expected_role": "vision"},
    {"input": "이 그래프 설명해줘", "images": [b"fake"], "expected_role": "vision"},
    {"input": "사진 속 텍스트를 읽어줘", "images": [b"fake"], "expected_role": "vision"},
    {"input": "이 스크린샷에서 에러 찾아줘", "images": [b"fake"], "expected_role": "vision"},
    # Coding (10)
    {"input": "```python\ndef hello():\n    print('hi')\n```\n이 코드 설명해줘", "expected_role": "coding"},
    {"input": "TypeError: unsupported operand type(s) 이 에러 해결해줘", "expected_role": "coding"},
    {"input": "파이썬으로 퀵소트 구현해줘", "expected_role": "coding"},
    {"input": "이 함수를 리팩토링해줘", "expected_role": "coding"},
    {"input": "Traceback (most recent call last):\n  File \"main.py\", line 10", "expected_role": "coding"},
    {"input": "자바스크립트로 REST API 만들어줘", "expected_role": "coding"},
    {"input": "async def process()에서 디버깅 도와줘", "expected_role": "coding"},
    {"input": "이 코드의 테스트 케이스를 작성해줘", "expected_role": "coding"},
    {"input": "Docker 빌드가 실패해요", "expected_role": "coding"},
    {"input": "main.py에서 import 에러가 나요", "expected_role": "coding"},
    # Cheap/Low-cost (10)
    {"input": "안녕!", "expected_role": "low_cost"},
    {"input": "오늘 점심 뭐 먹을까?", "expected_role": "low_cost"},
    {"input": "고마워", "expected_role": "low_cost"},
    {"input": "좋은 아침이야", "expected_role": "low_cost"},
    {"input": "오늘 날씨 어때?", "expected_role": "low_cost"},
    {"input": "번역해줘: good morning", "expected_role": "low_cost"},
    {"input": "맞춤법 검사해줘: 안녕하새요", "expected_role": "low_cost"},
    {"input": "'resilience'의 뜻이 뭐야?", "expected_role": "low_cost"},
    {"input": "짧게 요약해줘", "expected_role": "low_cost"},
    {"input": "네", "expected_role": "low_cost"},
    # Reasoning (5)
    {"input": "이 시스템의 병목 지점을 찾아서 개선 전략을 세워줘", "expected_role": "reasoning"},
    {"input": "GPT-4와 Claude의 아키텍처 차이를 심층 분석해줘", "expected_role": "reasoning"},
    {"input": "이 논문의 방법론에 대해 비판적으로 평가하고 반론을 제시해줘", "expected_role": "reasoning"},
    {"input": "마이크로서비스 vs 모놀리스 아키텍처의 트레이드오프를 다각도로 분석해줘", "expected_role": "reasoning"},
    {"input": "인과관계를 규명하고 단계별로 논리를 전개해줘", "expected_role": "reasoning"},
]

_RAG_TEST_CASES = [
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
    llm, model_router, rag_pipeline, _config, provider = await _init_components()
    print(f"provider: {provider}")
    if provider != "lemonade":
        print("  [notice] model_routing/rag 테스트는 lemonade provider에서만 실행됩니다.\n")

    print("=== Routing Test Cases ===\n")
    correct = 0
    total = len(_ROUTING_TEST_CASES)

    for i, tc in enumerate(_ROUTING_TEST_CASES, 1):
        images = tc.get("images")
        if model_router:
            decision = await model_router.route(tc["input"], images=images)
            actual = decision.selected_role
            match = actual == tc["expected_role"]
            status = "PASS" if match else "FAIL"
            if match:
                correct += 1
            print(f"  [{status}] #{i}: expected={tc['expected_role']}, "
                  f"actual={actual} ({decision.trigger})")
        else:
            print(f"  [SKIP] #{i}: model_router not available")

    print(f"\nRouting: {correct}/{total} ({correct/total*100:.0f}%)\n")

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
        description="ollama_bot CLI — 모델 라우팅/RAG 테스트",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="대화형 채팅")

    dry_p = sub.add_parser("dry-run", help="라우팅+RAG 결과만 출력")
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
