"""모델 가용성 관리.

Dual-Provider 구조에서 기본 모델(chat)과 retrieval 모델(임베딩/리랭커)의
가용성을 관리한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.config import ModelRegistryConfig
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.llm_protocol import RetrievalClientProtocol

_ROLE_TO_CONFIG_ATTR = {
    "default": "default_model",
    "embedding": "embedding_model",
    "reranker": "reranker_model",
}


@dataclass
class ModelInfo:
    """단일 모델의 상태 정보."""

    role: str
    name: str
    available: bool = False
    last_checked: float = 0.0


class ModelRegistry:
    """모델 가용성 관리 (단일 기본 모델 + retrieval 모델)."""

    def __init__(
        self,
        config: ModelRegistryConfig,
        retrieval_client: RetrievalClientProtocol,
    ) -> None:
        self._config = config
        self._retrieval_client = retrieval_client
        self._models: dict[str, ModelInfo] = {}
        self._logger = get_logger("model_registry")

        for role, attr in _ROLE_TO_CONFIG_ATTR.items():
            name = getattr(config, attr)
            self._models[role] = ModelInfo(role=role, name=name)

    async def initialize(self) -> None:
        """시작 시 retrieval 모델 가용성을 확인한다."""
        retrieval_models = [
            info.name for role, info in self._models.items()
            if role in ("embedding", "reranker")
        ]
        availability = await self._retrieval_client.check_model_availability(
            retrieval_models,
        )
        now = time.monotonic()

        for role, info in self._models.items():
            if role == "default":
                # 기본 모델(lemonade)은 별도 체크 — 항상 가용 가정
                info.available = True
            else:
                info.available = availability.get(info.name, False)
            info.last_checked = now

        available_roles = [r for r, m in self._models.items() if m.available]
        missing_roles = [r for r, m in self._models.items() if not m.available]

        self._logger.info(
            "model_registry_initialized",
            available=available_roles,
            missing=missing_roles,
        )

        if not self._models["embedding"].available:
            self._logger.warning(
                "embedding_model_unavailable",
                model=self._models["embedding"].name,
                impact="rag_disabled",
            )

    def get_model(self, role: str) -> str | None:
        """역할에 해당하는 모델명을 반환한다. 미가용이면 None."""
        info = self._models.get(role)
        if info is None or not info.available:
            return None
        return info.name

    def get_model_name(self, role: str) -> str:
        """역할에 해당하는 모델명을 반환한다 (가용성 무관)."""
        info = self._models.get(role)
        if info is None:
            raise ValueError(f"Unknown model role: {role}")
        return info.name

    def is_available(self, role: str) -> bool:
        """해당 역할의 모델이 가용한지 확인한다."""
        info = self._models.get(role)
        return info is not None and info.available

    def resolve_model(self, role: str) -> tuple[str, str, bool]:
        """역할에 맞는 모델을 해결한다. 단일 모델이므로 폴백 없음.

        Returns:
            (model_name, actual_role, fallback_used)
        """
        model = self.get_model(role)
        if model is not None:
            return model, role, False
        raise ValueError(f"No available model for role '{role}'")

    async def refresh_availability(self) -> None:
        """가용성을 재확인한다."""
        retrieval_models = [
            info.name for role, info in self._models.items()
            if role in ("embedding", "reranker")
        ]
        availability = await self._retrieval_client.check_model_availability(
            retrieval_models,
        )
        now = time.monotonic()

        changes: list[str] = []
        for role, info in self._models.items():
            if role == "default":
                continue
            new_status = availability.get(info.name, False)
            if new_status != info.available:
                changes.append(f"{role}: {info.available} -> {new_status}")
            info.available = new_status
            info.last_checked = now

        if changes:
            self._logger.info("model_availability_changed", changes=changes)

    def get_status(self) -> dict[str, dict]:
        """전체 모델 상태를 반환한다."""
        return {
            role: {
                "name": info.name,
                "available": info.available,
                "last_checked": info.last_checked,
            }
            for role, info in self._models.items()
        }
