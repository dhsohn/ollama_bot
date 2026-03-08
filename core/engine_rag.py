from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from core import engine_planner, engine_reviewer
from core.constants import (
    FULL_SCAN_FINAL_MAX_TOKENS,
    FULL_SCAN_MAP_MAX_TOKENS,
    FULL_SCAN_PROGRESS_EVERY_SEGMENTS,
    FULL_SCAN_REDUCE_GROUP_MAX_CHARS,
    FULL_SCAN_REDUCE_MAX_PASSES,
    FULL_SCAN_REDUCE_MAX_TOKENS,
    FULL_SCAN_SEGMENT_MAX_CHARS,
    REASONING_TIMEOUT_SECONDS,
    SUMMARY_MAP_TIMEOUT_SECONDS,
    SUMMARY_REDUCE_TIMEOUT_SECONDS,
)
from core.text_utils import sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine
    from core.intent_router import ContextStrategy


async def emit_full_scan_progress(
    callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        maybe_result = callback(payload)
        if inspect.isawaitable(maybe_result):
            await maybe_result
    except Exception:
        return


def build_full_scan_segments(
    chunks: list[Any],
    *,
    max_chars: int,
    segment_factory: Callable[..., Any],
) -> list[Any]:
    """source_path/chunk_id 순으로 전체 청크를 세그먼트로 패킹한다."""
    if not chunks:
        return []

    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (
            str(getattr(getattr(chunk, "metadata", None), "source_path", "")),
            int(getattr(getattr(chunk, "metadata", None), "chunk_id", 0)),
        ),
    )
    segments: list[Any] = []

    current_source: str | None = None
    current_start = 0
    current_end = 0
    current_parts: list[str] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current_source, current_start, current_end, current_parts, current_chars
        if not current_source or not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        if text:
            segments.append(
                segment_factory(
                    source_path=current_source,
                    start_chunk_id=current_start,
                    end_chunk_id=current_end,
                    text=text,
                )
            )
        current_source = None
        current_start = 0
        current_end = 0
        current_parts = []
        current_chars = 0

    for chunk in sorted_chunks:
        metadata = getattr(chunk, "metadata", None)
        source_path = str(getattr(metadata, "source_path", "") or "")
        chunk_id = int(getattr(metadata, "chunk_id", 0))
        chunk_text = sanitize_model_output(str(getattr(chunk, "text", ""))).strip()
        if not source_path or not chunk_text:
            continue

        block = f"[chunk {chunk_id}]\n{chunk_text}"
        block_len = len(block)
        if (
            current_source is not None
            and (source_path != current_source or current_chars + block_len > max_chars)
        ):
            flush()

        if current_source is None:
            current_source = source_path
            current_start = chunk_id
            current_end = chunk_id
        else:
            current_end = chunk_id

        current_parts.append(block)
        current_chars += block_len

    flush()
    return segments


def pack_blocks_for_reduction(
    blocks: list[str],
    *,
    max_chars: int,
) -> list[str]:
    """여러 텍스트 블록을 reduce 단계 입력 크기에 맞춰 그룹화한다."""
    if not blocks:
        return []

    groups: list[str] = []
    current_lines: list[str] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current_lines, current_chars
        if not current_lines:
            return
        groups.append("\n".join(current_lines).strip())
        current_lines = []
        current_chars = 0

    for block in blocks:
        normalized = sanitize_model_output(block).strip()
        if not normalized:
            continue
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            continue
        for line in lines:
            line_len = len(line)
            if current_lines and current_chars + line_len > max_chars:
                flush()
            if line_len > max_chars:
                cursor = 0
                while cursor < len(line):
                    part = line[cursor:cursor + max_chars].strip()
                    if part:
                        if current_lines:
                            flush()
                        groups.append(part)
                    cursor += max_chars
                continue
            current_lines.append(line)
            current_chars += line_len
    flush()
    return groups


async def prepare_full_request(
    engine: Engine,
    *,
    chat_id: int,
    text: str,
    model_override: str | None,
    images: list[bytes] | None,
    metadata: dict | None,
    intent: str | None,
    strategy: ContextStrategy | None,
    stream: bool,
) -> dict[str, Any]:
    """Tier 4(full LLM) 요청 준비를 공통 처리한다.

    컨텍스트 빌드, RAG 실행, ContextProvider 주입, 모델 선택, 타임아웃 계산을
    통합 처리하여 LLM 호출에 필요한 messages/timeout/max_tokens/target_model을 반환한다.
    """
    target_model = model_override or engine._config.lemonade.default_model
    rag_result = None

    if engine._rag_pipeline and engine._rag_pipeline.should_trigger_rag(text, metadata):
        rag_result = await engine._rag_pipeline.execute(text, metadata)

    prepared = await engine._prepare_request(
        chat_id,
        text,
        stream=stream,
        strategy=strategy,
    )
    effective_timeout = engine._resolve_inference_timeout(
        base_timeout=prepared.timeout,
        intent=intent,
        model_role="default",
        has_images=bool(images),
    )
    prepared_model, _ = await engine._prepare_target_model(
        model=target_model,
        role="default",
        timeout=effective_timeout,
    )
    target_model = prepared_model or target_model

    messages = prepared.messages
    if rag_result and rag_result.contexts:
        messages = inject_rag_context(messages, rag_result)

    for provider in engine._context_providers:
        try:
            extra = await provider.get_context(text)
            if extra:
                messages = inject_extra_context(messages, extra)
        except Exception as exc:
            engine._logger.warning(
                "context_provider_failed",
                provider=type(provider).__name__,
                error=str(exc),
            )

    messages, planner_applied = await engine_planner.maybe_apply_response_plan(
        engine,
        chat_id=chat_id,
        text=text,
        intent=intent,
        strategy=strategy,
        messages=messages,
        rag_result=rag_result,
        target_model=target_model,
        timeout=effective_timeout,
        images=images,
    )
    review_enabled = engine_reviewer.should_review_response(
        engine,
        text=text,
        images=images,
        planner_applied=planner_applied,
    )

    return {
        "messages": messages,
        "timeout": effective_timeout,
        "max_tokens": prepared.max_tokens,
        "target_model": target_model,
        "rag_result": rag_result,
        "planner_applied": planner_applied,
        "review_enabled": review_enabled,
        "stream_buffering": review_enabled and engine._config.response_reviewer.stream_buffering,
    }


def inject_rag_context(
    messages: list[dict[str, str]],
    rag_result: Any,
) -> list[dict[str, str]]:
    """RAG 컨텍스트를 시스템 프롬프트에 주입한다."""
    from core.rag.context_builder import RAGContextBuilder

    if not rag_result or not rag_result.contexts:
        return messages

    context_text = rag_result.contexts[0]
    suffix = RAGContextBuilder.build_rag_system_suffix(context_text)

    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {
            "role": "system",
            "content": result[0]["content"] + suffix,
        }
    else:
        result.insert(0, {"role": "system", "content": suffix})
    return result


def inject_extra_context(
    messages: list[dict[str, str]],
    context: str,
) -> list[dict[str, str]]:
    """추가 컨텍스트를 시스템 프롬프트에 주입한다."""
    if not context:
        return messages

    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {
            "role": "system",
            "content": result[0]["content"] + f"\n\n[추가 컨텍스트]\n{context.strip()}",
        }
    else:
        result.insert(0, {"role": "system", "content": f"[추가 컨텍스트]\n{context.strip()}"})
    return result


def extract_json_payload(text: str) -> dict[str, Any] | None:
    payload_text = text.strip()
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = payload_text.find("{")
    end = payload_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(payload_text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


async def analyze_all_corpus(
    engine: Engine,
    query: str,
    *,
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    """RAG 인덱스 전체를 읽어 map-reduce 방식으로 분석한다."""
    t0 = time.monotonic()
    question = query.strip()
    if not question:
        raise ValueError("query_is_empty")
    if engine._rag_pipeline is None:
        raise RuntimeError("rag_pipeline_disabled")

    await emit_full_scan_progress(
        progress_callback,
        {"phase": "collect", "message": "RAG 인덱스 전체 청크를 수집 중입니다..."},
    )
    chunks = await engine._rag_pipeline.get_all_chunks()
    total_chunks = len(chunks)
    if total_chunks == 0:
        return {
            "answer": "RAG 인덱스가 비어 있어 전체 분석을 수행할 수 없습니다.",
            "stats": {
                "total_chunks": 0,
                "total_segments": 0,
                "mapped_segments": 0,
                "evidence_lines": 0,
                "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            },
        }

    segments = build_full_scan_segments(
        chunks,
        max_chars=FULL_SCAN_SEGMENT_MAX_CHARS,
        segment_factory=lambda **kwargs: kwargs,
    )
    total_segments = len(segments)
    await emit_full_scan_progress(
        progress_callback,
        {
            "phase": "map_start",
            "message": "전체 문서 맵 분석을 시작합니다.",
            "total_chunks": total_chunks,
            "total_segments": total_segments,
        },
    )

    map_timeout = max(int(engine._config.bot.response_timeout), SUMMARY_MAP_TIMEOUT_SECONDS)
    reduce_timeout = max(int(engine._config.bot.response_timeout), SUMMARY_REDUCE_TIMEOUT_SECONDS)
    final_timeout = max(int(engine._config.bot.response_timeout), REASONING_TIMEOUT_SECONDS)

    map_model_candidate = engine._resolve_model_for_role("low_cost")
    if map_model_candidate is None:
        map_model_candidate = engine._resolve_model_for_role("reasoning")
    map_model, _ = await engine._prepare_target_model(
        model=map_model_candidate,
        role="low_cost",
        timeout=map_timeout,
    )

    reduce_model_candidate = engine._resolve_model_for_role("reasoning")
    if reduce_model_candidate is None:
        reduce_model_candidate = map_model
    reduce_model, _ = await engine._prepare_target_model(
        model=reduce_model_candidate,
        role="reasoning",
        timeout=reduce_timeout,
    )
    final_model, _ = await engine._prepare_target_model(
        model=reduce_model_candidate,
        role="reasoning",
        timeout=final_timeout,
    )

    map_system = engine._inject_language_policy(
        "당신은 문서 증거 추출기입니다. 질문과 직접 관련된 사실만 JSON으로 추출하세요. "
        "불확실하면 relevant=false로 답하세요."
    )
    evidence_lines: list[str] = []
    mapped_segments = 0
    for index, segment in enumerate(segments, start=1):
        map_prompt = (
            "[질문]\n"
            f"{question}\n\n"
            "[문서 세그먼트 메타]\n"
            f"source_path: {segment['source_path']}\n"
            f"chunk_range: {segment['start_chunk_id']}-{segment['end_chunk_id']}\n\n"
            "[문서 세그먼트 본문]\n"
            f"{segment['text']}\n\n"
            "다음 JSON만 출력하세요:\n"
            "{\"relevant\": true|false, \"findings\": [\"근거 기반 문장\"], \"confidence\": 0.0~1.0}\n"
            "규칙:\n"
            "- findings는 최대 4개\n"
            "- 질문과 직접 관련된 정보만 포함\n"
            "- 추측 금지"
        )
        try:
            map_resp = await engine._llm_client.chat(
                messages=[
                    {"role": "system", "content": map_system},
                    {"role": "user", "content": map_prompt},
                ],
                model=map_model,
                timeout=map_timeout,
                max_tokens=FULL_SCAN_MAP_MAX_TOKENS,
                temperature=0.0,
                response_format="json",
            )
        except Exception as exc:
            engine._logger.warning(
                "full_scan_map_failed",
                segment_index=index,
                total_segments=total_segments,
                source_path=segment["source_path"],
                error=str(exc),
            )
            continue

        payload = extract_json_payload(map_resp.content)
        if payload is None:
            continue
        relevant = bool(payload.get("relevant", False))
        findings_raw = payload.get("findings", [])
        findings: list[str] = []
        if isinstance(findings_raw, list):
            for item in findings_raw:
                text_item = sanitize_model_output(str(item)).strip()
                if text_item:
                    findings.append(text_item)
        if not relevant and not findings:
            continue
        if not findings:
            continue

        mapped_segments += 1
        citation = f"{segment['source_path']}#{segment['start_chunk_id']}-{segment['end_chunk_id']}"
        for finding in findings[:4]:
            evidence_lines.append(f"- [{citation}] {finding}")

        if (
            index == 1
            or index == total_segments
            or index % FULL_SCAN_PROGRESS_EVERY_SEGMENTS == 0
        ):
            await emit_full_scan_progress(
                progress_callback,
                {
                    "phase": "map",
                    "processed_segments": index,
                    "total_segments": total_segments,
                    "mapped_segments": mapped_segments,
                    "evidence_lines": len(evidence_lines),
                },
            )

    if not evidence_lines:
        duration_ms = round((time.monotonic() - t0) * 1000, 1)
        return {
            "answer": (
                "전체 문서를 읽었지만 질문과 직접 연결되는 근거를 찾지 못했습니다. "
                "질문 범위를 더 구체적으로 지정해 주세요."
            ),
            "stats": {
                "total_chunks": total_chunks,
                "total_segments": total_segments,
                "mapped_segments": mapped_segments,
                "evidence_lines": 0,
                "duration_ms": duration_ms,
            },
        }

    reduced_blocks = list(evidence_lines)
    reduce_pass = 0
    while reduce_pass < FULL_SCAN_REDUCE_MAX_PASSES:
        groups = pack_blocks_for_reduction(
            reduced_blocks,
            max_chars=FULL_SCAN_REDUCE_GROUP_MAX_CHARS,
        )
        if len(groups) <= 1:
            reduced_blocks = groups
            break

        reduce_pass += 1
        await emit_full_scan_progress(
            progress_callback,
            {
                "phase": "reduce",
                "reduce_pass": reduce_pass,
                "groups": len(groups),
            },
        )

        next_blocks: list[str] = []
        reduce_system = engine._inject_language_policy(
            "당신은 근거 통합 요약기입니다. 입력된 근거를 손실 없이 압축하세요. "
            "인용 표식([경로#chunk])은 보존하세요."
        )
        for group_index, group_text in enumerate(groups, start=1):
            reduce_prompt = (
                "[질문]\n"
                f"{question}\n\n"
                "[근거 목록]\n"
                f"{group_text}\n\n"
                "중복을 제거해 핵심 근거만 불릿으로 재작성하세요.\n"
                "출력 규칙:\n"
                "- 최대 12개 불릿\n"
                "- 각 불릿에 최소 1개 인용 표식 유지\n"
                "- 한국어만 사용"
            )
            try:
                reduce_resp = await engine._llm_client.chat(
                    messages=[
                        {"role": "system", "content": reduce_system},
                        {"role": "user", "content": reduce_prompt},
                    ],
                    model=reduce_model,
                    timeout=reduce_timeout,
                    max_tokens=FULL_SCAN_REDUCE_MAX_TOKENS,
                    temperature=0.0,
                )
            except Exception as exc:
                engine._logger.warning(
                    "full_scan_reduce_failed",
                    reduce_pass=reduce_pass,
                    group_index=group_index,
                    groups=len(groups),
                    error=str(exc),
                )
                continue
            reduced = sanitize_model_output(reduce_resp.content).strip()
            if reduced:
                next_blocks.append(reduced)

        if not next_blocks:
            break
        reduced_blocks = next_blocks

    evidence_text = "\n\n".join(reduced_blocks).strip()
    await emit_full_scan_progress(
        progress_callback,
        {"phase": "final", "message": "최종 답변을 생성 중입니다..."},
    )
    final_system = engine._inject_language_policy(
        "당신은 전체 문서를 검토한 수석 분석가입니다. "
        "아래 근거만 사용해 질문에 답하고, 핵심 주장마다 인용 표식([경로#chunk])을 붙이세요. "
        "근거가 부족한 부분은 '근거 부족'이라고 명시하세요."
    )
    final_prompt = (
        "[질문]\n"
        f"{question}\n\n"
        "[통합 근거]\n"
        f"{evidence_text}\n\n"
        "최종 출력 형식:\n"
        "1) 결론(2~4문장)\n"
        "2) 핵심 근거 불릿 3~8개 (각 불릿에 인용 표식)\n"
        "3) 근거 부족/추가 확인 필요 항목 (있으면)"
    )
    final_resp = await engine._llm_client.chat(
        messages=[
            {"role": "system", "content": final_system},
            {"role": "user", "content": final_prompt},
        ],
        model=final_model,
        timeout=final_timeout,
        max_tokens=FULL_SCAN_FINAL_MAX_TOKENS,
        temperature=0.0,
    )
    answer = sanitize_model_output(final_resp.content).strip()
    duration_ms = round((time.monotonic() - t0) * 1000, 1)
    return {
        "answer": answer,
        "stats": {
            "total_chunks": total_chunks,
            "total_segments": total_segments,
            "mapped_segments": mapped_segments,
            "evidence_lines": len(evidence_lines),
            "duration_ms": duration_ms,
            "map_model": map_model,
            "reduce_model": reduce_model,
            "final_model": final_model,
        },
    }


async def reindex_rag_corpus(engine: Engine, kb_dirs: list[str] | None = None) -> dict[str, Any]:
    """RAG 코퍼스를 증분 재인덱싱하고 통계를 반환한다."""
    if not engine._config.rag.enabled or engine._rag_pipeline is None:
        raise RuntimeError("rag_pipeline_disabled")

    roots = [str(path).strip() for path in (kb_dirs or engine._config.rag.kb_dirs)]
    roots = [path for path in roots if path]
    if not roots:
        raise ValueError("rag_kb_dirs_empty")

    async with engine._rag_reindex_lock:
        result = await engine._rag_pipeline.reindex_corpus(roots)

    if isinstance(result, dict):
        result.setdefault("roots", roots)
        return result
    return {"roots": roots}
