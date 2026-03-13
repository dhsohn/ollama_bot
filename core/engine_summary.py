from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.config import get_default_chat_model
from core.constants import (
    SUMMARY_CHUNK_MAX_CHARS,
    SUMMARY_CHUNK_OVERLAP_CHARS,
    SUMMARY_CHUNK_TRIGGER_CHARS,
    SUMMARY_MAP_MAX_TOKENS,
    SUMMARY_MAP_TIMEOUT_SECONDS,
    SUMMARY_REDUCE_MAX_TOKENS,
    SUMMARY_REDUCE_TIMEOUT_SECONDS,
)
from core.skill_manager import SkillDefinition
from core.text_utils import sanitize_model_output

if TYPE_CHECKING:
    from core.engine import Engine


def is_summarize_skill(skill: SkillDefinition) -> bool:
    return skill.name.strip().lower() == "summarize"


def extract_skill_user_input(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            return str(content).strip()
    return ""


def should_use_chunked_summary(*, skill: SkillDefinition, input_text: str) -> bool:
    if not is_summarize_skill(skill):
        return False
    return len(input_text.strip()) >= SUMMARY_CHUNK_TRIGGER_CHARS


def split_text_for_summary(text: str) -> list[str]:
    source = text.strip()
    if not source:
        return []
    if len(source) <= SUMMARY_CHUNK_MAX_CHARS:
        return [source]

    chunks: list[str] = []
    cursor = 0
    total_len = len(source)
    min_split_offset = int(SUMMARY_CHUNK_MAX_CHARS * 0.55)

    while cursor < total_len:
        max_end = min(total_len, cursor + SUMMARY_CHUNK_MAX_CHARS)
        end = max_end

        if max_end < total_len:
            window = source[cursor:max_end]
            split_offset = max(
                window.rfind("\n\n", min_split_offset),
                window.rfind("\n", min_split_offset),
                window.rfind(". ", min_split_offset),
                window.rfind("! ", min_split_offset),
                window.rfind("? ", min_split_offset),
                window.rfind("。", min_split_offset),
                window.rfind("!", min_split_offset),
                window.rfind("?", min_split_offset),
            )
            if split_offset >= 0:
                end = cursor + split_offset + 1

        chunk = source[cursor:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total_len:
            break

        next_cursor = max(0, end - SUMMARY_CHUNK_OVERLAP_CHARS)
        if next_cursor <= cursor:
            next_cursor = end
        cursor = next_cursor

    return chunks


async def run_skill_chat(
    engine: Engine,
    *,
    skill: SkillDefinition,
    messages: list[dict[str, str]],
    model_override: str | None,
    model_role_override: str | None = None,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
    timeout_override: int | None = None,
    chat_id: int | None = None,
) -> tuple[str, Any, str | None]:
    resolved_timeout = int(timeout_override or skill.timeout)
    resolved_role = (model_role_override or skill.model_role).strip().lower()
    resolved_max_tokens = (
        max_tokens_override if max_tokens_override is not None else skill.max_tokens
    )
    resolved_temperature = (
        temperature_override if temperature_override is not None else skill.temperature
    )
    user_input = engine._extract_skill_user_input(messages)
    if engine._should_use_chunked_summary(skill=skill, input_text=user_input):
        try:
            return await run_chunked_summary_pipeline(
                engine,
                skill=skill,
                messages=messages,
                model_override=model_override,
                timeout_override=resolved_timeout,
                chat_id=chat_id,
            )
        except Exception as exc:
            engine._logger.warning(
                "summarize_chunk_pipeline_failed",
                chat_id=chat_id,
                error=str(exc),
            )

    target_model, _ = await engine._prepare_target_model(
        model=model_override,
        role=resolved_role,
        timeout=resolved_timeout,
    )
    chat_response = await engine._llm_client.chat(
        messages=messages,
        model=target_model,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        timeout=resolved_timeout,
    )
    content = sanitize_model_output(chat_response.content)
    return content, chat_response.usage, target_model


async def run_chunked_summary_pipeline(
    engine: Engine,
    *,
    skill: SkillDefinition,
    messages: list[dict[str, str]],
    model_override: str | None,
    timeout_override: int | None = None,
    chat_id: int | None,
) -> tuple[str, Any, str | None]:
    user_input = engine._extract_skill_user_input(messages)
    chunks = engine._split_text_for_summary(user_input)
    if len(chunks) <= 1:
        raise RuntimeError("chunked_summary_not_applicable")

    base_timeout = int(timeout_override or skill.timeout)
    map_timeout = max(base_timeout, SUMMARY_MAP_TIMEOUT_SECONDS)
    reduce_timeout = max(base_timeout, SUMMARY_REDUCE_TIMEOUT_SECONDS)

    if model_override:
        map_model_candidate = model_override
        map_role = "skill"
        reduce_model_candidate = model_override
        reduce_role = "skill"
    else:
        map_model_candidate = get_default_chat_model(engine._config)
        map_role = "default"
        reduce_model_candidate = get_default_chat_model(engine._config)
        reduce_role = "default"

    map_model, _ = await engine._prepare_target_model(
        model=map_model_candidate or None,
        role=map_role,
        timeout=map_timeout,
    )
    reduce_model, _ = await engine._prepare_target_model(
        model=reduce_model_candidate or None,
        role=reduce_role,
        timeout=reduce_timeout,
    )

    engine._logger.info(
        "summarize_chunk_pipeline_started",
        chat_id=chat_id,
        chunk_count=len(chunks),
        map_model=map_model,
        reduce_model=reduce_model,
    )

    map_system = engine._inject_language_policy(
        "You are an expert at accurately summarizing parts of long documents.\n"
        "Extract only the key facts from the input chunk.\n"
        "Avoid speculation, redundancy, and verbose sentences."
    )

    chunk_summaries: list[str] = []
    seen_summaries: set[str] = set()
    for index, chunk_text in enumerate(chunks, start=1):
        map_prompt = (
            f"[Document chunk {index}/{len(chunks)}]\n"
            f"{chunk_text}\n\n"
            "Output rules:\n"
            "- Write only 3-5 bullet points of key points from the original\n"
            "- Do not add content not in the original\n"
            "- No duplicate expressions"
        )
        map_response = await engine._llm_client.chat(
            messages=[
                {"role": "system", "content": map_system},
                {"role": "user", "content": map_prompt},
            ],
            model=map_model,
            timeout=map_timeout,
            max_tokens=SUMMARY_MAP_MAX_TOKENS,
        )
        map_summary = sanitize_model_output(map_response.content).strip()
        if not map_summary:
            continue
        normalized = map_summary.lower()
        if normalized in seen_summaries:
            continue
        seen_summaries.add(normalized)
        chunk_summaries.append(f"[Chunk {index}]\n{map_summary}")

    if not chunk_summaries:
        raise RuntimeError("chunked_summary_empty_intermediate")

    base_system = (
        messages[0]["content"]
        if messages and messages[0].get("role") == "system"
        else engine._inject_language_policy(skill.system_prompt)
    )
    reduce_system = (
        f"{base_system}\n\n"
        "[Long-text summary consolidation rules]\n"
        "- Consolidate the intermediate summaries below into one final summary.\n"
        "- Remove duplicates and keep only key information."
    )
    reduce_prompt = (
        "Below are intermediate summaries from processing a long original text in chunks.\n"
        "Remove duplicates and write the final summary.\n"
        "Final output format:\n"
        "1) 3-7 key point bullets\n"
        "2) Optionally, a one-line conclusion at the end\n\n"
        "[Intermediate summaries]\n"
        + "\n\n".join(chunk_summaries)
    )
    reduce_response = await engine._llm_client.chat(
        messages=[
            {"role": "system", "content": reduce_system},
            {"role": "user", "content": reduce_prompt},
        ],
        model=reduce_model,
        timeout=reduce_timeout,
        max_tokens=SUMMARY_REDUCE_MAX_TOKENS,
    )
    final_summary = sanitize_model_output(reduce_response.content).strip()
    if not final_summary:
        raise RuntimeError("chunked_summary_empty_final")

    engine._logger.info(
        "summarize_chunk_pipeline_completed",
        chat_id=chat_id,
        chunk_count=len(chunks),
        intermediate_count=len(chunk_summaries),
        map_model=map_model,
        reduce_model=reduce_model,
    )
    return final_summary, reduce_response.usage, reduce_model
