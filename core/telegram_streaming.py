from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from telegram.constants import ChatAction
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from telegram import Update

    from core.telegram_handler import TelegramHandler

_EDIT_INTERVAL = 1.0
_EDIT_CHAR_THRESHOLD = 100
_TYPING_INTERVAL = 4.0
_STREAM_DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 120.0
_STREAM_DEFAULT_CHUNK_TIMEOUT_SECONDS = 45.0
_STREAM_DEFAULT_MAX_SECONDS_CAP = 300.0
_STREAM_REASONING_FIRST_CHUNK_TIMEOUT_SECONDS = 600.0
_STREAM_REASONING_CHUNK_TIMEOUT_SECONDS = 60.0
_STREAM_REASONING_MAX_SECONDS_CAP = 3600.0
_STREAM_LONG_TIMEOUT_INTENTS = {"complex", "code"}
_STREAM_MAX_TOTAL_CHARS = 8_192
_STREAM_MAX_REPEATED_CHUNKS = 30
_STREAM_RENDER_WAIT_GRACE_SECONDS = 5.0
_STREAM_RECOVERY_TIMEOUT_SECONDS = 120.0
_AUTO_CONTINUATION_MAX_TURNS = 3
_FULL_SCAN_AUTO_TRIGGER_RE = re.compile(r"분석", re.IGNORECASE)
_THINKING_PLACEHOLDER_TEMPLATE = "{bot_name}이 답변을 위해 생각 중입니다..."


async def handle_message(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """자유 텍스트 메시지를 처리한다. 스트리밍 UX를 제공한다."""
    await self._handle_message_impl(update, context)


async def handle_message_impl(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text_override: str | None = None,
    force_continuation: bool = False,
    auto_continuation_turn: int = 0,
    stream_and_render_fn: Callable[..., Any],
    detect_output_anomalies_fn: Callable[[str, str], list[str]],
) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    chat_id = chat.id
    raw_text = (
        text_override
        if text_override is not None
        else (message.text or message.caption or "")
    )

    images: list[bytes] | None = None
    image_download_failed = False
    if message.photo:
        try:
            photo = message.photo[-1]
            file = await photo.get_file()
            image_bytes = await file.download_as_bytearray()
            images = [bytes(image_bytes)]
        except Exception as exc:
            image_download_failed = True
            self._logger.warning(
                "image_download_failed",
                chat_id=chat_id,
                error=str(exc),
            )

    if not raw_text and not images and not force_continuation:
        if image_download_failed:
            await message.reply_text("이미지 다운로드에 실패했어요. 잠시 후 다시 시도해주세요.")
        return

    text = self._security.sanitize_input(raw_text) if raw_text else ""
    self._cleanup_pending_continuations()
    continuation_state: dict[str, Any] | None = None
    continuation_root_query: str | None = None
    if force_continuation or (not images and self._is_continue_request(text)):
        continuation_state = self._take_pending_continuation(chat_id)
        if continuation_state is None:
            await message.reply_text("이어볼 답변이 없습니다. 먼저 질문을 해주세요.")
            return
        continuation_root_query = str(continuation_state.get("root_query", "")).strip()
        text = self._build_continuation_prompt(continuation_state)
    else:
        self._pending_continuation.pop(chat_id, None)

    if not text.strip() and not images:
        return

    if continuation_state is None and not images and self._should_auto_trigger_analyze_all(text):
        self._logger.info("analyze_all_auto_triggered", chat_id=chat_id)
        await self._run_analyze_all_flow(
            chat=chat,
            message=message,
            query=text,
            auto_triggered=True,
        )
        return

    await chat.send_action(ChatAction.TYPING)

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        self._keep_typing(chat, typing_stop),
        name=f"typing_{chat_id}",
    )
    render_timeout: float | None = None

    try:
        sent_message = await message.reply_text(
            _THINKING_PLACEHOLDER_TEMPLATE.format(bot_name=self._config.bot.name)
        )

        raw_intent = self._engine.classify_intent(text)
        if inspect.isawaitable(raw_intent):
            intent = await raw_intent
        else:
            intent = raw_intent

        intent_key = str(intent).strip().lower() if intent is not None else None
        if images or intent_key in _STREAM_LONG_TIMEOUT_INTENTS:
            first_chunk_timeout_seconds = _STREAM_REASONING_FIRST_CHUNK_TIMEOUT_SECONDS
            chunk_timeout_seconds = _STREAM_REASONING_CHUNK_TIMEOUT_SECONDS
            effective_stream_seconds = max(
                float(self._config.bot.response_timeout),
                _STREAM_REASONING_MAX_SECONDS_CAP,
            )
        else:
            first_chunk_timeout_seconds = _STREAM_DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS
            chunk_timeout_seconds = _STREAM_DEFAULT_CHUNK_TIMEOUT_SECONDS
            effective_stream_seconds = min(
                float(self._config.bot.response_timeout),
                _STREAM_DEFAULT_MAX_SECONDS_CAP,
            )
        render_timeout = effective_stream_seconds + _STREAM_RENDER_WAIT_GRACE_SECONDS
        result = await asyncio.wait_for(
            stream_and_render_fn(
                stream=self._engine.process_message_stream(chat_id, text, images=images),
                sent_message=sent_message,
                reply_text=message.reply_text,
                split_message_fn=self._split_message,
                edit_interval=_EDIT_INTERVAL,
                edit_char_threshold=_EDIT_CHAR_THRESHOLD,
                max_edit_length=self._max_message_length,
                first_chunk_timeout_seconds=first_chunk_timeout_seconds,
                chunk_timeout_seconds=chunk_timeout_seconds,
                max_stream_seconds=effective_stream_seconds,
                max_total_chars=_STREAM_MAX_TOTAL_CHARS,
                max_repeated_chunks=_STREAM_MAX_REPEATED_CHUNKS,
            ),
            timeout=render_timeout,
        )
        consume_meta = getattr(self._engine, "consume_last_stream_meta", None)
        if callable(consume_meta):
            stream_meta = consume_meta(chat_id)
            if inspect.isawaitable(stream_meta):
                stream_meta = await stream_meta
            if isinstance(stream_meta, dict):
                result.tier = stream_meta.get("tier", result.tier)
                result.intent = stream_meta.get("intent")
                result.cache_id = stream_meta.get("cache_id")
                result.usage = stream_meta.get("usage")
                if result.stop_reason is None:
                    result.stop_reason = stream_meta.get("stop_reason")
        stop_reason = getattr(result, "stop_reason", None)
        recovery_reason: str | None = None
        anomaly_reasons: list[str] = []
        if stop_reason in {"chunk_timeout", "repeated_chunks"}:
            recovery_reason = stop_reason
        elif stop_reason is None:
            anomaly_reasons = detect_output_anomalies_fn(
                result.full_response,
                result.full_response,
            )
            actionable_reasons = [
                reason
                for reason in anomaly_reasons
                if reason != "empty_after_sanitize"
            ]
            if actionable_reasons:
                anomaly_reasons = actionable_reasons
                recovery_reason = "response_anomaly"

        if recovery_reason is not None:
            self._logger.warning(
                "stream_recovery_triggered",
                chat_id=chat_id,
                reason=recovery_reason,
                anomaly_reasons=anomaly_reasons or None,
            )
            try:
                rollback_fn = getattr(self._engine, "rollback_last_turn", None)
                if callable(rollback_fn):
                    deleted = await rollback_fn(chat_id)
                    self._logger.info(
                        "stream_recovery_turn_rolled_back",
                        chat_id=chat_id,
                        deleted=deleted,
                    )
            except Exception as rb_exc:
                self._logger.warning(
                    "stream_recovery_rollback_failed",
                    chat_id=chat_id,
                    error=str(rb_exc),
                )
            try:
                recovered_response = await asyncio.wait_for(
                    self._engine.process_message(chat_id, text, images=images),
                    timeout=_STREAM_RECOVERY_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                self._logger.warning(
                    "stream_recovery_failed",
                    chat_id=chat_id,
                    reason=recovery_reason,
                    error=str(exc),
                )
            else:
                recovered_text = str(recovered_response).strip()
                if recovered_text:
                    recovered_parts = self._split_message(recovered_text)
                    if recovered_parts:
                        last_recovered = None
                        try:
                            await sent_message.edit_text(recovered_parts[0])
                            last_recovered = sent_message
                        except Exception:
                            last_recovered = await message.reply_text(recovered_parts[0])

                        for part in recovered_parts[1:]:
                            last_recovered = await message.reply_text(part)

                        if last_recovered is not None:
                            result.last_message = last_recovered
                            result.full_response = recovered_text

        await self._link_feedback_target(chat_id, result)
        await self._attach_feedback_controls(chat_id, text, result)

        stop_reason = getattr(result, "stop_reason", None)
        if stop_reason == "max_total_chars":
            next_turn = 1
            if continuation_state is not None:
                next_turn = max(1, int(continuation_state.get("turn", 0)) + 1)
            root_query = (continuation_root_query or "").strip() or text
            self._set_pending_continuation(
                chat_id,
                root_query=root_query,
                turn=next_turn,
            )
            if auto_continuation_turn < _AUTO_CONTINUATION_MAX_TURNS:
                await message.reply_text("↪️ 답변이 길어 자동으로 이어서 보여드릴게요.")
                await self._handle_message_impl(
                    update,
                    context,
                    text_override="",
                    force_continuation=True,
                    auto_continuation_turn=auto_continuation_turn + 1,
                )
                return
            await message.reply_text(
                self._build_long_response_followup_message(result.full_response)
            )
        else:
            self._pending_continuation.pop(chat_id, None)

    except TimeoutError:
        self._logger.error(
            "stream_render_timeout",
            chat_id=chat_id,
            timeout_seconds=render_timeout,
        )
        await message.reply_text(
            "⚠️ 응답 시간이 길어져 중단했습니다. 질문을 더 짧게 나눠 다시 시도해주세요."
        )
    except Exception as exc:
        self._logger.error(
            "message_processing_error",
            chat_id=chat_id,
            error=str(exc),
        )
        await message.reply_text("죄송합니다. 메시지 처리 중 오류가 발생했습니다.")
    finally:
        typing_stop.set()
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task


def should_auto_trigger_analyze_all(text: str) -> bool:
    """일반 채팅에서 full-scan 분석으로 우회할 문구인지 검사한다."""
    text_norm = text.strip()
    if not text_norm:
        return False
    if text_norm.startswith("/"):
        return False
    return bool(_FULL_SCAN_AUTO_TRIGGER_RE.search(text_norm))


async def run_analyze_all_flow(
    self: TelegramHandler,
    *,
    chat: Any,
    message: Any,
    query: str,
    auto_triggered: bool,
) -> None:
    """전체 문서 분석 실행 + 진행률 렌더링 공통 처리."""
    query_text = query.strip()
    if not query_text:
        await message.reply_text("분석할 질문을 입력해주세요.")
        return

    await chat.send_action(ChatAction.TYPING)
    progress_message = await message.reply_text(
        "전체 문서 분석을 시작합니다.\n- 단계: 준비\n- 진행: 0%"
    )

    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(
        keep_typing(chat, typing_stop),
        name=f"analyze_all_typing_{chat.id}",
    )
    last_progress_update = 0.0

    async def _on_progress(payload: dict[str, Any]) -> None:
        nonlocal last_progress_update
        now = time.monotonic()
        phase = str(payload.get("phase", "")).strip().lower()
        force_update = phase in {"final", "map_start", "collect"}
        if not force_update and (now - last_progress_update) < 1.5:
            return
        last_progress_update = now

        if phase == "collect":
            text = "전체 문서 분석을 시작합니다.\n- 단계: 인덱스 수집\n- 진행: 준비 중"
        elif phase == "map_start":
            total_chunks = int(payload.get("total_chunks", 0))
            total_segments = int(payload.get("total_segments", 0))
            text = (
                "전체 문서 분석을 시작합니다.\n"
                "- 단계: 맵 분석 시작\n"
                f"- 청크: {total_chunks}개\n"
                f"- 세그먼트: {total_segments}개"
            )
        elif phase == "map":
            processed = int(payload.get("processed_segments", 0))
            total = max(1, int(payload.get("total_segments", 1)))
            mapped = int(payload.get("mapped_segments", 0))
            evidence = int(payload.get("evidence_lines", 0))
            percent = int((processed / total) * 100)
            text = (
                "전체 문서 분석 진행 중\n"
                "- 단계: 맵 분석\n"
                f"- 진행: {processed}/{total} ({percent}%)\n"
                f"- 근거 세그먼트: {mapped}개\n"
                f"- 근거 라인: {evidence}개"
            )
        elif phase == "reduce":
            reduce_pass = int(payload.get("reduce_pass", 0))
            groups = int(payload.get("groups", 0))
            text = (
                "전체 문서 분석 진행 중\n"
                "- 단계: 리듀스(통합)\n"
                f"- 패스: {reduce_pass}\n"
                f"- 그룹: {groups}개"
            )
        elif phase == "final":
            text = "전체 문서 분석 진행 중\n- 단계: 최종 답변 생성"
        else:
            return

        with suppress(Exception):
            await progress_message.edit_text(text)

    try:
        result = await self._engine.analyze_all_corpus(
            query_text,
            progress_callback=_on_progress,
        )
        answer = str(result.get("answer", "")).strip()
        stats = result.get("stats", {}) if isinstance(result, dict) else {}
        if not answer:
            answer = "분석 결과를 생성하지 못했습니다."

        stats_lines = []
        if isinstance(stats, dict):
            total_chunks = stats.get("total_chunks")
            total_segments = stats.get("total_segments")
            mapped_segments = stats.get("mapped_segments")
            evidence_lines = stats.get("evidence_lines")
            duration_ms = stats.get("duration_ms")
            if total_chunks is not None:
                stats_lines.append(f"- 총 청크: {total_chunks}")
            if total_segments is not None:
                stats_lines.append(f"- 총 세그먼트: {total_segments}")
            if mapped_segments is not None:
                stats_lines.append(f"- 근거 세그먼트: {mapped_segments}")
            if evidence_lines is not None:
                stats_lines.append(f"- 근거 라인: {evidence_lines}")
            if duration_ms is not None:
                stats_lines.append(f"- 소요 시간: {duration_ms}ms")

        header = "📚 전체 문서 분석 결과"
        if auto_triggered:
            header += " (자동 전환)"
        final_text = f"{header}\n\n{answer}"
        if stats_lines:
            final_text += "\n\n[분석 통계]\n" + "\n".join(stats_lines)

        parts = self._split_message(final_text)
        if parts:
            try:
                await progress_message.edit_text(parts[0])
            except Exception:
                await message.reply_text(parts[0])
            for part in parts[1:]:
                await message.reply_text(part)
    except Exception as exc:
        self._logger.error(
            "analyze_all_failed",
            chat_id=chat.id,
            error=str(exc),
        )
        await message.reply_text(
            "전체 문서 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )
    finally:
        typing_stop.set()
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task


async def keep_typing(chat: Any, stop_event: asyncio.Event) -> None:
    """typing 인디케이터를 주기적으로 전송한다."""
    while not stop_event.is_set():
        with suppress(Exception):
            await chat.send_action(ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_TYPING_INTERVAL)
            return
        except TimeoutError:
            pass
