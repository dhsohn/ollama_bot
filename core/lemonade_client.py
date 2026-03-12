"""Lemonade(OpenAI-compatible) API 클라이언트.

Lemonade Server의 OpenAI 호환 엔드포인트를 사용해
채팅, 스트리밍, 헬스체크, 모델 조회를 제공한다.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from core import lemonade_delegates
from core.config import LemonadeConfig
from core.lemonade_errors import LemonadeClientError, LemonadeModelNotFoundError
from core.lemonade_utils import (
    build_chat_payload,
    extract_api_error,
    extract_content,
    format_exception,
    usage_from_payload,
)
from core.llm_types import ChatResponse, ChatStreamState, ChatUsage
from core.logging_setup import get_logger


_ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _ENV_TRUE_VALUES


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _preview_text(value: str, *, max_chars: int) -> str:
    text = value.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class LemonadeClient:
    """Lemonade OpenAI-compatible 클라이언트."""

    def __init__(
        self,
        config: LemonadeConfig,
    ) -> None:
        self._host = config.host.rstrip("/")
        base_path = config.base_path.strip()
        if base_path:
            self._base_path = "/" + base_path.strip("/")
        else:
            self._base_path = ""
        self._api_key = config.api_key
        self._default_model = config.default_model.strip()
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._context_window = config.context_window
        self._min_output_tokens = config.min_output_tokens
        self._system_prompt = config.system_prompt
        self._timeout_default = config.timeout_seconds
        self._model_load_timeout_default = max(
            config.model_load_timeout_seconds,
            config.timeout_seconds,
        )
        self._heavy_model_load_timeout_default = max(
            config.heavy_model_load_timeout_seconds,
            self._model_load_timeout_default,
        )
        self._client: httpx.AsyncClient | None = None
        self._auto_reconnect_enabled = False
        self._is_healthy = True
        self._last_connection_error: str | None = None
        self._next_reconnect_at = 0.0
        self._reconnect_cooldown_seconds = config.reconnect_cooldown_seconds
        self._reconnect_lock = asyncio.Lock()
        self._prepare_model_lock = asyncio.Lock()
        self._loaded_models: set[str] = set()
        self._loaded_models_checked_at = 0.0
        self._loaded_models_ttl_seconds = 10.0
        self._heavy_roles = {"reasoning", "coding", "vision"}
        self._stream_debug_raw_sse = _env_flag("LEMONADE_DEBUG_STREAM_SSE")
        self._stream_debug_compare = _env_flag("LEMONADE_DEBUG_STREAM_COMPARE")
        self._stream_debug_max_chars = _env_int(
            "LEMONADE_DEBUG_STREAM_MAX_CHARS",
            240,
            minimum=32,
        )
        self._stream_debug_max_lines = _env_int(
            "LEMONADE_DEBUG_STREAM_MAX_LINES",
            120,
            minimum=1,
        )
        self._logger = get_logger("lemonade_client")

    recover_connection = lemonade_delegates.recover_connection
    _refresh_loaded_models = lemonade_delegates._refresh_loaded_models
    prepare_model = lemonade_delegates.prepare_model
    list_models = lemonade_delegates.list_models
    get_model_info = lemonade_delegates.get_model_info
    health_check = lemonade_delegates.health_check
    embed = lemonade_delegates.embed
    rerank = lemonade_delegates.rerank
    _rerank_via_chat = lemonade_delegates._rerank_via_chat
    check_model_availability = lemonade_delegates.check_model_availability
    _retry_with_backoff = lemonade_delegates._retry_with_backoff

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        self._default_model = model.strip()

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def host(self) -> str:
        return self._host

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _endpoint(self, path: str) -> str:
        return f"{self._base_path}{path}"

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LemonadeClient가 아직 초기화되지 않았습니다.")
        return self._client

    def _mark_healthy(self) -> None:
        self._is_healthy = True
        self._last_connection_error = None
        self._next_reconnect_at = 0.0

    def _mark_unhealthy(self, error: Exception | str) -> None:
        self._is_healthy = False
        if isinstance(error, Exception):
            self._last_connection_error = format_exception(error)
        else:
            self._last_connection_error = str(error)
        self._next_reconnect_at = time.monotonic() + self._reconnect_cooldown_seconds

    async def initialize(self) -> None:
        """클라이언트를 생성하고 연결/모델 상태를 확인한다."""
        client = httpx.AsyncClient(
            base_url=self._host,
            headers=self._headers(),
        )
        self._client = client
        self._auto_reconnect_enabled = True
        try:
            models = await self._list_model_names(client)
            self._logger.info(
                "lemonade_connected",
                host=self._host,
                models_count=len(models),
            )
            self._mark_healthy()
        except Exception as exc:
            self._mark_unhealthy(exc)
            await client.aclose()
            self._client = None
            if isinstance(exc, LemonadeClientError):
                raise
            error_text = format_exception(exc)
            raise LemonadeClientError(
                f"Failed to connect to Lemonade at {self._host}: {error_text}"
            ) from exc

    async def close(self) -> None:
        """클라이언트 리소스를 정리한다."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._auto_reconnect_enabled = False
        self._is_healthy = False
        self._loaded_models.clear()
        self._loaded_models_checked_at = 0.0

    async def _list_model_names(self, client: httpx.AsyncClient) -> list[str]:
        response = await client.get(
            self._endpoint("/models"),
            timeout=self._timeout_default,
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            entries = payload.get("data", [])
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []

        names: list[str] = []
        for item in entries:
            if isinstance(item, dict):
                model_name = item.get("id") or item.get("model") or item.get("name")
                if isinstance(model_name, str) and model_name:
                    names.append(model_name)
            elif isinstance(item, str) and item:
                names.append(item)
        return names

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        response_format: str | dict | None = None,
    ) -> ChatResponse:
        """비스트리밍 채팅 요청."""
        requested_model = (model or "").strip()
        target_model = requested_model or self._default_model or None
        payload = build_chat_payload(
            model=target_model,
            messages=messages,
            default_temperature=self._temperature,
            temperature=temperature,
            default_max_tokens=self._max_tokens,
            max_tokens=max_tokens,
            response_format=response_format,
            stream=False,
            logger=self._logger,
            context_window=self._context_window,
            min_output_tokens=self._min_output_tokens,
        )

        async def _do_chat() -> ChatResponse:
            client = self._require_client()
            response = await client.post(
                self._endpoint("/chat/completions"),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            error_text = extract_api_error(body)
            if error_text:
                raise LemonadeClientError(
                    f"Lemonade chat request returned API error for model '{target_model}': {error_text}"
                )
            choices = body.get("choices", [])
            if not choices:
                raise LemonadeClientError("lemonade_chat_failed: missing choices")
            content = extract_content(choices[0])
            usage = usage_from_payload(body)
            return ChatResponse(content=content, usage=usage)

        return await self._retry_with_backoff(_do_chat)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        stream_state: ChatStreamState | None = None,
    ) -> AsyncGenerator[str, None]:
        """스트리밍 채팅 요청."""
        client = self._require_client()
        requested_model = (model or "").strip()
        target_model = requested_model or self._default_model or None
        payload = build_chat_payload(
            model=target_model,
            messages=messages,
            default_temperature=self._temperature,
            temperature=temperature,
            default_max_tokens=self._max_tokens,
            max_tokens=max_tokens,
            response_format=None,
            stream=True,
            logger=self._logger,
            context_window=self._context_window,
            min_output_tokens=self._min_output_tokens,
        )
        state = stream_state or ChatStreamState()
        state.usage = None
        stream_started = time.monotonic()
        last_content_chunk: str | None = None
        last_fallback_snapshot: str | None = None
        repeated_content_count = 0
        max_repeated_content = 200
        sse_line_no = 0
        raw_log_count = 0
        compare_log_count = 0
        raw_limit_logged = False
        compare_limit_logged = False

        def _can_log(kind: str) -> bool:
            nonlocal raw_log_count, compare_log_count, raw_limit_logged, compare_limit_logged

            if kind == "raw_sse":
                if not self._stream_debug_raw_sse:
                    return False
                if raw_log_count >= self._stream_debug_max_lines:
                    if not raw_limit_logged:
                        raw_limit_logged = True
                        self._logger.info(
                            "lemonade_stream_debug_limit_reached",
                            kind=kind,
                            max_lines=self._stream_debug_max_lines,
                            model=target_model,
                        )
                    return False
                raw_log_count += 1
                return True

            if not self._stream_debug_compare:
                return False
            if compare_log_count >= self._stream_debug_max_lines:
                if not compare_limit_logged:
                    compare_limit_logged = True
                    self._logger.info(
                        "lemonade_stream_debug_limit_reached",
                        kind=kind,
                        max_lines=self._stream_debug_max_lines,
                        model=target_model,
                    )
                return False
            compare_log_count += 1
            return True

        def _log_raw_sse(data_raw: str, *, elapsed: float) -> None:
            if not _can_log("raw_sse"):
                return
            self._logger.info(
                "lemonade_stream_sse_line",
                model=target_model,
                line_no=sse_line_no,
                elapsed_ms=round(elapsed * 1000, 1),
                chars=len(data_raw),
                raw_preview=_preview_text(
                    data_raw,
                    max_chars=self._stream_debug_max_chars,
                ),
                truncated=len(data_raw) > self._stream_debug_max_chars,
            )

        def _log_payload_compare(
            *,
            elapsed: float,
            finish_reason: str | None,
            delta_content: str | None,
            message_content: str,
            selected_path: str,
            emitted_content: str | None,
            snapshot_prefix_match: bool = False,
            snapshot_exact_match: bool = False,
            skipped: bool = False,
        ) -> None:
            if not _can_log("compare"):
                return
            self._logger.info(
                "lemonade_stream_payload_compare",
                model=target_model,
                line_no=sse_line_no,
                elapsed_ms=round(elapsed * 1000, 1),
                finish_reason=finish_reason,
                has_delta_content=bool(delta_content),
                delta_len=len(delta_content or ""),
                delta_preview=(
                    _preview_text(delta_content, max_chars=self._stream_debug_max_chars)
                    if delta_content
                    else None
                ),
                has_message_content=bool(message_content),
                message_len=len(message_content),
                message_preview=(
                    _preview_text(message_content, max_chars=self._stream_debug_max_chars)
                    if message_content
                    else None
                ),
                selected_path=selected_path,
                emitted_len=len(emitted_content or ""),
                emitted_preview=(
                    _preview_text(emitted_content, max_chars=self._stream_debug_max_chars)
                    if emitted_content
                    else None
                ),
                snapshot_prefix_match=snapshot_prefix_match,
                snapshot_exact_match=snapshot_exact_match,
                skipped=skipped,
            )

        try:
            async with client.stream(
                "POST",
                self._endpoint("/chat/completions"),
                json=payload,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    sse_line_no += 1
                    # httpx timeout= 은 개별 청크 읽기 지연을 감시하고,
                    # 이 수동 체크는 전체 스트림 지속 시간을 제한한다.
                    elapsed = time.monotonic() - stream_started
                    if elapsed > float(timeout):
                        raise TimeoutError(
                            f"stream_duration_exceeded({elapsed:.1f}s>{timeout}s)"
                        )

                    data_line = line.strip()
                    if not data_line or not data_line.startswith("data:"):
                        continue
                    data_raw = data_line[5:].strip()
                    _log_raw_sse(data_raw, elapsed=elapsed)
                    if data_raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_raw)
                    except json.JSONDecodeError:
                        continue

                    error_text = extract_api_error(chunk)
                    if error_text:
                        raise LemonadeClientError(
                            f"Lemonade streaming API error for model '{target_model}': {error_text}"
                        )

                    usage = usage_from_payload(chunk)
                    if usage is not None:
                        state.usage = usage

                    choices = chunk.get("choices", [])
                    if not choices:
                        _log_payload_compare(
                            elapsed=elapsed,
                            finish_reason=None,
                            delta_content=None,
                            message_content="",
                            selected_path="missing_choices",
                            emitted_content=None,
                            skipped=True,
                        )
                        continue

                    first_choice = choices[0]
                    finish_reason = first_choice.get("finish_reason")
                    delta = first_choice.get("delta", {})
                    content = delta.get("content") if isinstance(delta, dict) else None
                    raw_fallback_content = extract_content(first_choice)
                    if isinstance(content, str) and content:
                        _log_payload_compare(
                            elapsed=elapsed,
                            finish_reason=finish_reason,
                            delta_content=content,
                            message_content=raw_fallback_content,
                            selected_path="delta_content",
                            emitted_content=content,
                        )
                        if content == last_content_chunk:
                            repeated_content_count += 1
                            if repeated_content_count >= max_repeated_content:
                                raise TimeoutError("stream_stalled_repeating_content")
                        else:
                            last_content_chunk = content
                            repeated_content_count = 0
                        yield content
                        if finish_reason:
                            break
                        continue

                    # 일부 구현은 delta 대신 message.content를 전달한다.
                    if raw_fallback_content:
                        fallback_content = raw_fallback_content
                        selected_path = "message_content"
                        snapshot_prefix_match = False
                        snapshot_exact_match = False
                        # message.content가 누적 전체 텍스트인 구현을 델타로 보정한다.
                        if last_fallback_snapshot is not None:
                            if fallback_content == last_fallback_snapshot:
                                snapshot_exact_match = True
                                _log_payload_compare(
                                    elapsed=elapsed,
                                    finish_reason=finish_reason,
                                    delta_content=None,
                                    message_content=raw_fallback_content,
                                    selected_path="message_snapshot_duplicate_skip",
                                    emitted_content=None,
                                    snapshot_exact_match=True,
                                    skipped=True,
                                )
                                repeated_content_count += 1
                                if repeated_content_count >= max_repeated_content:
                                    raise TimeoutError("stream_stalled_repeating_content")
                                if finish_reason:
                                    break
                                continue
                            if fallback_content.startswith(last_fallback_snapshot):
                                snapshot_prefix_match = True
                                selected_path = "message_prefix_delta"
                                fallback_content = fallback_content[len(last_fallback_snapshot):]
                        last_fallback_snapshot = raw_fallback_content
                        if not fallback_content:
                            _log_payload_compare(
                                elapsed=elapsed,
                                finish_reason=finish_reason,
                                delta_content=None,
                                message_content=raw_fallback_content,
                                selected_path=f"{selected_path}_empty_skip",
                                emitted_content=None,
                                snapshot_prefix_match=snapshot_prefix_match,
                                skipped=True,
                            )
                            if finish_reason:
                                break
                            continue
                        _log_payload_compare(
                            elapsed=elapsed,
                            finish_reason=finish_reason,
                            delta_content=None,
                            message_content=raw_fallback_content,
                            selected_path=selected_path,
                            emitted_content=fallback_content,
                            snapshot_prefix_match=snapshot_prefix_match,
                            snapshot_exact_match=snapshot_exact_match,
                        )
                        if fallback_content == last_content_chunk:
                            repeated_content_count += 1
                            if repeated_content_count >= max_repeated_content:
                                raise TimeoutError("stream_stalled_repeating_content")
                        else:
                            last_content_chunk = fallback_content
                            repeated_content_count = 0
                        yield fallback_content
                        if finish_reason:
                            break
                        continue

                    _log_payload_compare(
                        elapsed=elapsed,
                        finish_reason=finish_reason,
                        delta_content=None,
                        message_content="",
                        selected_path="finish_reason_only" if finish_reason else "empty_payload",
                        emitted_content=None,
                        skipped=True,
                    )
                    if finish_reason:
                        break
            self._mark_healthy()
        except TimeoutError as exc:
            # 애플리케이션 레벨 타임아웃(전체 스트림 시간 초과, 반복 콘텐츠 감지)은
            # 연결 장애가 아니므로 unhealthy로 마킹하지 않는다.
            error_text = format_exception(exc)
            raise LemonadeClientError(
                f"Lemonade streaming request failed for model '{target_model}': {error_text}"
            ) from exc
        except (httpx.HTTPError, OSError) as exc:
            self._mark_unhealthy(exc)
            await self.recover_connection()
            error_text = format_exception(exc)
            raise LemonadeClientError(
                f"Lemonade streaming request failed for model '{target_model}': {error_text}"
            ) from exc
