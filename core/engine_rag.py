from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core import engine_planner, engine_reviewer
from core.config import get_default_chat_model
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


@dataclass(frozen=True)
class _FullRequestContext:
    rag_result: Any
    target_model: str
    effective_timeout: int
    messages: list[dict[str, str]]
    max_tokens: int | None


@dataclass(frozen=True)
class _FullScanTimeouts:
    map_timeout: int
    reduce_timeout: int
    final_timeout: int


@dataclass(frozen=True)
class _FullScanModels:
    map_model: str
    reduce_model: str
    final_model: str


@dataclass
class _MappedEvidence:
    evidence_lines: list[str]
    mapped_segments: int = 0


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
    """Pack all chunks into segments ordered by `source_path` and `chunk_id`."""
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
    """Group text blocks to fit the reduce-stage input budget."""
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


async def _maybe_execute_rag(
    engine: Engine,
    *,
    text: str,
    metadata: dict | None,
) -> Any:
    if engine._rag_pipeline and engine._rag_pipeline.should_trigger_rag(text, metadata):
        return await engine._rag_pipeline.execute(text, metadata)
    return None


async def _resolve_full_request_target_model(
    engine: Engine,
    *,
    model_override: str | None,
    effective_timeout: int,
) -> str:
    target_model = model_override or get_default_chat_model(engine._config)
    prepared_model, _ = await engine._prepare_target_model(
        model=target_model,
        role=None,
        timeout=effective_timeout,
    )
    return prepared_model or target_model


async def _inject_context_provider_messages(
    engine: Engine,
    *,
    messages: list[dict[str, str]],
    text: str,
) -> list[dict[str, str]]:
    updated_messages = messages
    for provider in engine._context_providers:
        try:
            extra = await provider.get_context(text)
            if extra:
                updated_messages = inject_extra_context(updated_messages, extra)
        except Exception as exc:
            engine._logger.warning(
                "context_provider_failed",
                provider=type(provider).__name__,
                error=str(exc),
            )
    return updated_messages


async def _prepare_full_request_context(
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
) -> _FullRequestContext:
    rag_result = await _maybe_execute_rag(
        engine,
        text=text,
        metadata=metadata,
    )
    prepared = await engine._prepare_request(
        chat_id,
        text,
        stream=stream,
        strategy=strategy,
    )
    effective_timeout = engine._resolve_inference_timeout(
        base_timeout=prepared.timeout,
        intent=intent,
        model_role=None,
        has_images=bool(images),
    )
    target_model = await _resolve_full_request_target_model(
        engine,
        model_override=model_override,
        effective_timeout=effective_timeout,
    )
    messages = prepared.messages
    if rag_result and rag_result.contexts:
        messages = inject_rag_context(messages, rag_result)
    messages = await _inject_context_provider_messages(
        engine,
        messages=messages,
        text=text,
    )
    return _FullRequestContext(
        rag_result=rag_result,
        target_model=target_model,
        effective_timeout=effective_timeout,
        messages=messages,
        max_tokens=prepared.max_tokens,
    )


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
    """Prepare a Tier-4 (full LLM) request.

    This combines context building, optional RAG execution, context-provider
    injection, model selection, and timeout calculation, then returns the
    `messages`, `timeout`, `max_tokens`, and `target_model` needed for the LLM call.
    """
    request_context = await _prepare_full_request_context(
        engine,
        chat_id=chat_id,
        text=text,
        model_override=model_override,
        images=images,
        metadata=metadata,
        intent=intent,
        strategy=strategy,
        stream=stream,
    )

    messages, planner_applied = await engine_planner.maybe_apply_response_plan(
        engine,
        chat_id=chat_id,
        text=text,
        intent=intent,
        strategy=strategy,
        messages=request_context.messages,
        rag_result=request_context.rag_result,
        target_model=request_context.target_model,
        timeout=request_context.effective_timeout,
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
        "timeout": request_context.effective_timeout,
        "max_tokens": request_context.max_tokens,
        "target_model": request_context.target_model,
        "rag_result": request_context.rag_result,
        "planner_applied": planner_applied,
        "review_enabled": review_enabled,
        "stream_buffering": review_enabled and engine._config.response_reviewer.stream_buffering,
    }


def inject_rag_context(
    messages: list[dict[str, str]],
    rag_result: Any,
) -> list[dict[str, str]]:
    """Inject RAG context into the system prompt."""
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
    """Inject additional context into the system prompt."""
    if not context:
        return messages

    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {
            "role": "system",
            "content": result[0]["content"] + f"\n\n[Additional Context]\n{context.strip()}",
        }
    else:
        result.insert(0, {"role": "system", "content": f"[Additional Context]\n{context.strip()}"})
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


def _build_empty_full_scan_result(*, started_at: float) -> dict[str, Any]:
    return {
        "answer": "The RAG index is empty. Cannot perform full analysis.",
        "stats": {
            "total_chunks": 0,
            "total_segments": 0,
            "mapped_segments": 0,
            "evidence_lines": 0,
            "duration_ms": round((time.monotonic() - started_at) * 1000, 1),
        },
    }


def _build_no_evidence_full_scan_result(
    *,
    started_at: float,
    total_chunks: int,
    total_segments: int,
    mapped_segments: int,
) -> dict[str, Any]:
    return {
        "answer": (
            "I read the entire document but could not find evidence directly related to the question. "
            "Please try narrowing down or being more specific with your question."
        ),
        "stats": {
            "total_chunks": total_chunks,
            "total_segments": total_segments,
            "mapped_segments": mapped_segments,
            "evidence_lines": 0,
            "duration_ms": round((time.monotonic() - started_at) * 1000, 1),
        },
    }


async def _collect_full_scan_segments(
    engine: Engine,
    *,
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
) -> tuple[list[dict[str, Any]], int]:
    rag_pipeline = engine._rag_pipeline
    if rag_pipeline is None:
        raise RuntimeError("rag_pipeline_disabled")
    await emit_full_scan_progress(
        progress_callback,
        {"phase": "collect", "message": "Collecting all chunks from RAG index..."},
    )
    chunks = await rag_pipeline.get_all_chunks()
    total_chunks = len(chunks)
    segments = build_full_scan_segments(
        chunks,
        max_chars=FULL_SCAN_SEGMENT_MAX_CHARS,
        segment_factory=lambda **kwargs: kwargs,
    )
    return segments, total_chunks


async def _emit_full_scan_map_start(
    *,
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    total_chunks: int,
    total_segments: int,
) -> None:
    await emit_full_scan_progress(
        progress_callback,
        {
            "phase": "map_start",
            "message": "Starting full document map analysis.",
            "total_chunks": total_chunks,
            "total_segments": total_segments,
        },
    )


def _resolve_full_scan_timeouts(engine: Engine) -> _FullScanTimeouts:
    base_timeout = int(engine._config.bot.response_timeout)
    return _FullScanTimeouts(
        map_timeout=max(base_timeout, SUMMARY_MAP_TIMEOUT_SECONDS),
        reduce_timeout=max(base_timeout, SUMMARY_REDUCE_TIMEOUT_SECONDS),
        final_timeout=max(base_timeout, REASONING_TIMEOUT_SECONDS),
    )


async def _resolve_full_scan_models(
    engine: Engine,
    *,
    timeouts: _FullScanTimeouts,
) -> _FullScanModels:
    default_chat_model = get_default_chat_model(engine._config)
    map_model_candidate = engine._resolve_model_for_role("low_cost")
    map_role = "low_cost" if map_model_candidate is not None else None
    if map_model_candidate is None:
        map_model_candidate = engine._resolve_model_for_role("reasoning")
        map_role = "reasoning" if map_model_candidate is not None else None
    if map_model_candidate is None:
        map_model_candidate = default_chat_model
    map_model, _ = await engine._prepare_target_model(
        model=map_model_candidate,
        role=map_role,
        timeout=timeouts.map_timeout,
    )

    reduce_model_candidate = engine._resolve_model_for_role("reasoning")
    reduce_role = "reasoning" if reduce_model_candidate is not None else None
    if reduce_model_candidate is None:
        reduce_model_candidate = map_model
    reduce_model, _ = await engine._prepare_target_model(
        model=reduce_model_candidate,
        role=reduce_role,
        timeout=timeouts.reduce_timeout,
    )
    final_model, _ = await engine._prepare_target_model(
        model=reduce_model_candidate,
        role=reduce_role,
        timeout=timeouts.final_timeout,
    )
    return _FullScanModels(
        map_model=map_model or map_model_candidate,
        reduce_model=reduce_model or reduce_model_candidate,
        final_model=final_model or reduce_model_candidate,
    )


def _build_full_scan_map_system(engine: Engine) -> str:
    return engine._inject_language_policy(
        "You are a document evidence extractor. Extract only facts directly related to the question as JSON. "
        "If uncertain, respond with relevant=false."
    )


def _build_full_scan_map_prompt(question: str, segment: dict[str, Any]) -> str:
    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Document segment metadata]\n"
        f"source_path: {segment['source_path']}\n"
        f"chunk_range: {segment['start_chunk_id']}-{segment['end_chunk_id']}\n\n"
        "[Document segment body]\n"
        f"{segment['text']}\n\n"
        "Output only the following JSON:\n"
        "{\"relevant\": true|false, \"findings\": [\"evidence-based sentence\"], \"confidence\": 0.0~1.0}\n"
        "Rules:\n"
        "- Maximum 4 findings\n"
        "- Include only information directly related to the question\n"
        "- No speculation"
    )


def _extract_full_scan_findings(payload: dict[str, Any]) -> list[str]:
    findings_raw = payload.get("findings", [])
    findings: list[str] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            text_item = sanitize_model_output(str(item)).strip()
            if text_item:
                findings.append(text_item)
    return findings


def _append_full_scan_evidence(
    evidence_lines: list[str],
    *,
    segment: dict[str, Any],
    findings: list[str],
) -> None:
    citation = f"{segment['source_path']}#{segment['start_chunk_id']}-{segment['end_chunk_id']}"
    for finding in findings[:4]:
        evidence_lines.append(f"- [{citation}] {finding}")


async def _map_full_scan_segments(
    engine: Engine,
    *,
    question: str,
    segments: list[dict[str, Any]],
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    map_model: str,
    map_timeout: int,
) -> _MappedEvidence:
    total_segments = len(segments)
    map_system = _build_full_scan_map_system(engine)
    mapped = _MappedEvidence(evidence_lines=[])

    for index, segment in enumerate(segments, start=1):
        map_prompt = _build_full_scan_map_prompt(question, segment)
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
        findings = _extract_full_scan_findings(payload)
        if not relevant and not findings:
            continue
        if not findings:
            continue

        mapped.mapped_segments += 1
        _append_full_scan_evidence(
            mapped.evidence_lines,
            segment=segment,
            findings=findings,
        )

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
                    "mapped_segments": mapped.mapped_segments,
                    "evidence_lines": len(mapped.evidence_lines),
                },
            )

    return mapped


def _build_full_scan_reduce_system(engine: Engine) -> str:
    return engine._inject_language_policy(
        "You are an evidence consolidation summarizer. Compress the input evidence without loss. "
        "Preserve citation markers ([path#chunk])."
    )


def _build_full_scan_reduce_prompt(question: str, group_text: str) -> str:
    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Evidence list]\n"
        f"{group_text}\n\n"
        "Remove duplicates and rewrite only key evidence as bullet points.\n"
        "Output rules:\n"
        "- Maximum 12 bullet points\n"
        "- Keep at least 1 citation marker per bullet"
    )


async def _reduce_full_scan_evidence(
    engine: Engine,
    *,
    question: str,
    evidence_lines: list[str],
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    reduce_model: str,
    reduce_timeout: int,
) -> list[str]:
    reduced_blocks = list(evidence_lines)
    reduce_pass = 0

    while reduce_pass < FULL_SCAN_REDUCE_MAX_PASSES:
        groups = pack_blocks_for_reduction(
            reduced_blocks,
            max_chars=FULL_SCAN_REDUCE_GROUP_MAX_CHARS,
        )
        if len(groups) <= 1:
            return groups

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
        reduce_system = _build_full_scan_reduce_system(engine)
        for group_index, group_text in enumerate(groups, start=1):
            reduce_prompt = _build_full_scan_reduce_prompt(question, group_text)
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

    return reduced_blocks


def _build_full_scan_final_system(engine: Engine) -> str:
    return engine._inject_language_policy(
        "You are a senior analyst who has reviewed the entire document. "
        "Answer the question using only the evidence below, and attach citation markers ([path#chunk]) to key claims. "
        "Explicitly state 'insufficient evidence' for parts lacking evidence."
    )


def _build_full_scan_final_prompt(question: str, evidence_text: str) -> str:
    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Consolidated evidence]\n"
        f"{evidence_text}\n\n"
        "Final output format:\n"
        "1) Conclusion (2-4 sentences)\n"
        "2) 3-8 key evidence bullets (with citation markers per bullet)\n"
        "3) Items with insufficient evidence / needing further verification (if any)"
    )


async def _generate_full_scan_answer(
    engine: Engine,
    *,
    question: str,
    reduced_blocks: list[str],
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    final_model: str,
    final_timeout: int,
) -> str:
    evidence_text = "\n\n".join(reduced_blocks).strip()
    await emit_full_scan_progress(
        progress_callback,
        {"phase": "final", "message": "Generating final answer..."},
    )
    final_resp = await engine._llm_client.chat(
        messages=[
            {"role": "system", "content": _build_full_scan_final_system(engine)},
            {"role": "user", "content": _build_full_scan_final_prompt(question, evidence_text)},
        ],
        model=final_model,
        timeout=final_timeout,
        max_tokens=FULL_SCAN_FINAL_MAX_TOKENS,
        temperature=0.0,
    )
    return sanitize_model_output(final_resp.content).strip()


async def analyze_all_corpus(
    engine: Engine,
    query: str,
    *,
    progress_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> dict[str, Any]:
    """Read the entire RAG index and analyze it with a map-reduce flow."""
    t0 = time.monotonic()
    question = query.strip()
    if not question:
        raise ValueError("query_is_empty")
    if engine._rag_pipeline is None:
        raise RuntimeError("rag_pipeline_disabled")

    segments, total_chunks = await _collect_full_scan_segments(
        engine,
        progress_callback=progress_callback,
    )
    if total_chunks == 0:
        return _build_empty_full_scan_result(started_at=t0)

    total_segments = len(segments)
    await _emit_full_scan_map_start(
        progress_callback=progress_callback,
        total_chunks=total_chunks,
        total_segments=total_segments,
    )
    timeouts = _resolve_full_scan_timeouts(engine)
    models = await _resolve_full_scan_models(
        engine,
        timeouts=timeouts,
    )
    mapped = await _map_full_scan_segments(
        engine,
        question=question,
        segments=segments,
        progress_callback=progress_callback,
        map_model=models.map_model,
        map_timeout=timeouts.map_timeout,
    )
    if not mapped.evidence_lines:
        return _build_no_evidence_full_scan_result(
            started_at=t0,
            total_chunks=total_chunks,
            total_segments=total_segments,
            mapped_segments=mapped.mapped_segments,
        )

    reduced_blocks = await _reduce_full_scan_evidence(
        engine,
        question=question,
        evidence_lines=mapped.evidence_lines,
        progress_callback=progress_callback,
        reduce_model=models.reduce_model,
        reduce_timeout=timeouts.reduce_timeout,
    )
    answer = await _generate_full_scan_answer(
        engine,
        question=question,
        reduced_blocks=reduced_blocks,
        progress_callback=progress_callback,
        final_model=models.final_model,
        final_timeout=timeouts.final_timeout,
    )
    return {
        "answer": answer,
        "stats": {
            "total_chunks": total_chunks,
            "total_segments": total_segments,
            "mapped_segments": mapped.mapped_segments,
            "evidence_lines": len(mapped.evidence_lines),
            "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            "map_model": models.map_model,
            "reduce_model": models.reduce_model,
            "final_model": models.final_model,
        },
    }


async def reindex_rag_corpus(engine: Engine, kb_dirs: list[str] | None = None) -> dict[str, Any]:
    """Incrementally reindex the RAG corpus and return summary stats."""
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
