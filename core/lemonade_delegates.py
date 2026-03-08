"""Delegated Lemonade client operations."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from core.lemonade_errors import LemonadeClientError, LemonadeModelNotFoundError
from core.lemonade_utils import (
    compact_text,
    format_exception,
    parse_loaded_models,
    parse_rerank_chat_response,
    parse_rerank_items,
)
from core.llm_types import ChatResponse


async def recover_connection(self: Any, *, force: bool = False) -> bool:
    """연결 장애 시 재연결을 시도한다."""
    if not self._auto_reconnect_enabled:
        return False

    now = time.monotonic()
    if not force and now < self._next_reconnect_at:
        return False

    async with self._reconnect_lock:
        now = time.monotonic()
        if not force and now < self._next_reconnect_at:
            return False

        candidate = httpx.AsyncClient(
            base_url=self._host,
            headers=self._headers(),
        )
        try:
            models = await self._list_model_names(candidate)
        except Exception as exc:
            self._mark_unhealthy(exc)
            self._logger.warning(
                "lemonade_reconnect_failed",
                error=format_exception(exc),
            )
            await candidate.aclose()
            return False

        previous = self._client
        self._client = candidate
        if previous is not None:
            await previous.aclose()
        self._mark_healthy()
        self._logger.info(
            "lemonade_reconnected",
            host=self._host,
            models_count=len(models),
        )
        return True


async def _refresh_loaded_models(self: Any, *, force: bool = False) -> set[str]:
    now = time.monotonic()
    if (
        not force
        and self._loaded_models
        and now - self._loaded_models_checked_at < self._loaded_models_ttl_seconds
    ):
        return set(self._loaded_models)

    client = self._require_client()
    try:
        response = await client.get(
            self._endpoint("/health"),
            timeout=self._timeout_default,
        )
        response.raise_for_status()
        loaded = parse_loaded_models(response.json())
        self._loaded_models = loaded
        self._loaded_models_checked_at = now
        return set(loaded)
    except Exception as exc:
        self._logger.debug(
            "lemonade_loaded_models_refresh_failed",
            error=format_exception(exc),
        )
        return set(self._loaded_models)


async def prepare_model(
    self: Any,
    *,
    model: str | None = None,
    role: str | None = None,
    timeout_seconds: int | None = None,
) -> None:
    """요청 전 모델 로드를 보장한다."""
    target_model = (model or self._default_model or "").strip()
    if not target_model:
        return

    loaded = await self._refresh_loaded_models()
    if target_model in loaded:
        return

    role_key = (role or "").strip().lower()
    role_timeout = (
        self._heavy_model_load_timeout_default
        if role_key in self._heavy_roles
        else self._model_load_timeout_default
    )
    effective_timeout = max(role_timeout, int(timeout_seconds or 0))

    async with self._prepare_model_lock:
        loaded = await self._refresh_loaded_models(force=True)
        if target_model in loaded:
            return

        client = self._require_client()
        started = time.monotonic()
        response = await client.post(
            self._endpoint("/load"),
            json={"model_name": target_model},
            timeout=effective_timeout,
        )
        if response.status_code in (404, 405):
            self._logger.debug(
                "lemonade_model_preload_skipped",
                model=target_model,
                role=role_key or None,
                status_code=response.status_code,
            )
            return
        if response.is_error:
            message = compact_text(response.text)
            raise LemonadeClientError(
                f"Failed to load model '{target_model}' before generation "
                f"(role={role_key or 'unknown'}, status={response.status_code}): {message}"
            )

        self._loaded_models.add(target_model)
        self._loaded_models_checked_at = time.monotonic()
        self._logger.info(
            "lemonade_model_preloaded",
            model=target_model,
            role=role_key or None,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )


async def list_models(self: Any) -> list[dict]:
    """모델 목록을 반환한다."""
    client = self._require_client()
    try:
        names = await self._list_model_names(client)
        self._mark_healthy()
    except Exception as exc:
        self._mark_unhealthy(exc)
        await self.recover_connection()
        raise
    return [{"name": name, "size": None, "modified_at": None} for name in names]


async def get_model_info(self: Any, model: str) -> dict:
    """모델 정보를 반환한다."""
    models = await self.list_models()
    names = {item["name"] for item in models}
    if model not in names:
        raise LemonadeModelNotFoundError(f"Model '{model}' not found")
    return {"model": model, "modelfile": None, "parameters": None}


async def health_check(self: Any, *, attempt_recovery: bool = False) -> dict:
    """서버 상태를 확인한다."""
    client = self._require_client()
    try:
        models = await self._list_model_names(client)
        self._mark_healthy()
        return {
            "status": "ok",
            "host": self._host,
            "models_count": len(models),
            "models": models,
            "default_model": self._default_model,
            "default_model_available": (
                True if not self._default_model else self._default_model in models
            ),
        }
    except Exception as exc:
        self._mark_unhealthy(exc)
        recovered = False
        if attempt_recovery:
            recovered = await self.recover_connection(force=True)
        return {
            "status": "error",
            "host": self._host,
            "error": str(exc),
            "recovery_attempted": attempt_recovery,
            "recovered": recovered,
        }


async def embed(
    self: Any,
    texts: list[str],
    model: str | None = None,
    timeout: int | None = None,
) -> list[list[float]]:
    """임베딩 벡터를 반환한다."""
    client = self._require_client()
    requested_model = (model or "").strip()
    target_model = requested_model or self._default_model or None
    payload: dict[str, Any] = {"input": texts}
    if target_model:
        payload["model"] = target_model
    try:
        response = await client.post(
            self._endpoint("/embeddings"),
            json=payload,
            timeout=timeout or self._timeout_default,
        )
        response.raise_for_status()
    except (TimeoutError, httpx.HTTPError, OSError) as exc:
        self._mark_unhealthy(exc)
        error_text = format_exception(exc)
        raise LemonadeClientError(
            f"Embedding request failed for model '{target_model}': {error_text}"
        ) from exc
    self._mark_healthy()
    body = response.json()
    data = body.get("data", [])
    if not isinstance(data, list):
        raise LemonadeClientError(
            f"Embedding response missing 'data' array for model '{target_model}'"
        )
    results: list[list[float]] = []
    for item in sorted(data, key=lambda entry: entry.get("index", 0)):
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise LemonadeClientError(
                f"Embedding response item missing 'embedding' field "
                f"for model '{target_model}'"
            )
        results.append(embedding)
    return results


async def rerank(
    self: Any,
    query: str,
    documents: list[str],
    model: str | None = None,
    top_n: int | None = None,
    timeout: int | None = None,
) -> list[dict[str, Any]]:
    """리랭크 점수를 반환한다."""
    client = self._require_client()
    requested_model = (model or "").strip()
    target_model = requested_model or self._default_model or None
    effective_timeout = timeout or self._timeout_default

    try:
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
        }
        if target_model:
            payload["model"] = target_model
        if top_n is not None:
            payload["top_n"] = top_n
        response = await client.post(
            self._endpoint("/rerank"),
            json=payload,
            timeout=effective_timeout,
        )
        response.raise_for_status()
        scored = parse_rerank_items(response.json())
        self._mark_healthy()
        return scored
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (404, 405):
            self._logger.debug("rerank_endpoint_not_available", status=exc.response.status_code)
        else:
            error_text = format_exception(exc)
            raise LemonadeClientError(f"Rerank request failed: {error_text}") from exc
    except (TimeoutError, httpx.HTTPError, OSError) as exc:
        self._mark_unhealthy(exc)
        error_text = format_exception(exc)
        raise LemonadeClientError(f"Rerank request failed: {error_text}") from exc

    return await self._rerank_via_chat(
        query,
        documents,
        target_model,
        top_n,
        effective_timeout,
    )


async def _rerank_via_chat(
    self: Any,
    query: str,
    documents: list[str],
    model: str | None,
    top_n: int | None,
    timeout: int,
) -> list[dict[str, Any]]:
    """chat/completions를 사용한 리랭크 폴백."""
    doc_list = "\n".join(
        f"[{i}] {doc[:500]}" for i, doc in enumerate(documents)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Documents:\n{doc_list}\n\n"
        "위 각 문서의 query 관련성을 0.0~1.0 점수로 평가하세요.\n"
        "JSON 배열만 출력: [{\"index\": 0, \"score\": 0.8}, ...]\n"
        "다른 텍스트 없이 JSON만 출력하세요."
    )
    try:
        chat_resp = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=256,
            timeout=timeout,
            response_format="json",
        )
        scored = parse_rerank_chat_response(chat_resp.content)
        if top_n is not None:
            scored = scored[:top_n]
        return scored
    except (LemonadeClientError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise LemonadeClientError(f"Chat-based rerank failed: {exc}") from exc


async def check_model_availability(self: Any, model_names: list[str]) -> dict[str, bool]:
    """주어진 모델 이름들의 가용성을 확인한다."""
    client = self._require_client()
    try:
        available = await self._list_model_names(client)
    except Exception:
        return {name: False for name in model_names}
    available_set = set(available)
    return {name: name in available_set for name in model_names}


async def _retry_with_backoff(
    self: Any,
    coro_factory,
    max_retries: int = 2,
) -> ChatResponse:
    backoff_delays = [1, 3]
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await coro_factory()
            self._mark_healthy()
            return response
        except (TimeoutError, httpx.HTTPError, OSError, LemonadeClientError) as exc:
            last_error = exc
            self._mark_unhealthy(exc)
            if attempt < max_retries:
                await self.recover_connection(force=True)
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                self._logger.warning(
                    "lemonade_retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error=format_exception(exc),
                )
                await asyncio.sleep(delay)

    last_error_text = format_exception(last_error) if last_error is not None else "unknown"
    raise LemonadeClientError(
        f"Lemonade request failed after {max_retries + 1} attempts: {last_error_text}"
    )
