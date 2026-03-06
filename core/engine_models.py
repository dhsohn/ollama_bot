from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.engine import Engine


async def prepare_target_model(
    engine: Engine,
    *,
    model: str | None,
    role: str | None,
    timeout: int,
) -> tuple[str | None, str | None]:
    """LLM 요청 전에 모델 로드를 준비한다.

    prepare_model을 구현한 클라이언트에서만 동작한다.
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


def resolve_model_for_role(engine: Engine, role: str | None) -> str | None:
    """role 이름을 설정 모델명으로 해석한다."""
    role_key = (role or "").strip().lower()
    if not role_key:
        return None
    role_model_map = {
        "default": engine._config.lemonade.default_model,
        "embedding": engine._config.ollama.embedding_model,
        "reranker": engine._config.ollama.reranker_model,
    }
    mapped = role_model_map.get(role_key)
    if not mapped:
        return None
    normalized = mapped.strip()
    return normalized or None
