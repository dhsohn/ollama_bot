from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from core.config import get_default_chat_model

if TYPE_CHECKING:
    from core.engine import Engine


async def prepare_target_model(
    engine: Engine,
    *,
    model: str | None,
    role: str | None,
    timeout: int,
) -> tuple[str | None, str | None]:
    """Prepare model loading before sending an LLM request.

    This only runs for clients that implement ``prepare_model``.
    """
    target_role = role.strip().lower() if isinstance(role, str) and role.strip() else None
    target_model = model.strip() if isinstance(model, str) and model.strip() else None
    if target_model is None and target_role is not None:
        target_model = engine._resolve_model_for_role(target_role)
    prepare_model = getattr(engine._llm_client, "prepare_model", None)
    if not callable(prepare_model):
        return target_model, target_role
    try:
        maybe_result = prepare_model(
            model=target_model,
            role=target_role,
            timeout_seconds=timeout,
        )
        if inspect.isawaitable(maybe_result):
            await maybe_result
        return target_model, target_role
    except Exception as exc:
        engine._logger.warning(
            "model_prepare_failed",
            model=target_model,
            role=target_role,
            error=str(exc),
        )
    return target_model, target_role


def _default_chat_model(engine: Engine) -> str:
    """Return the configured Ollama chat model."""
    return get_default_chat_model(engine._config)


def resolve_model_for_role(engine: Engine, role: str | None) -> str | None:
    """Resolve a role name to the configured model identifier."""
    role_key = (role or "").strip().lower()
    if not role_key:
        return None
    role_model_map = {
        "default": _default_chat_model(engine),
        "embedding": engine._config.ollama.embedding_model,
        "reranker": engine._config.ollama.reranker_model,
    }
    mapped = role_model_map.get(role_key)
    if not mapped:
        return None
    normalized = mapped.strip()
    return normalized or None
