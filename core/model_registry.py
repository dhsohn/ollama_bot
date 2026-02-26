"""모델 가용성 관리 및 폴백 체인.

Lemonade Server에 등록된 모델 목록을 확인하고,
역할별 모델의 가용성/폴백을 관리한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.config import ModelRegistryConfig
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.lemonade_client import LemonadeClient

_ROLE_TO_CONFIG_ATTR = {
    "embedding": "embedding_model",
    "reranker": "reranker_model",
    "vision": "vision_model",
    "low_cost": "low_cost_model",
    "reasoning": "reasoning_model",
    "coding": "coding_model",
}


@dataclass
class ModelInfo:
    """단일 모델의 상태 정보."""

    role: str
    name: str
    available: bool = False
    last_checked: float = 0.0


class ModelRegistry:
    """모델 가용성 관리 및 폴백 체인."""

    def __init__(
        self,
        config: ModelRegistryConfig,
        client: LemonadeClient,
    ) -> None:
        self._config = config
        self._client = client
        self._models: dict[str, ModelInfo] = {}
        self._logger = get_logger("model_registry")

        for role, attr in _ROLE_TO_CONFIG_ATTR.items():
            name = getattr(config, attr)
            self._models[role] = ModelInfo(role=role, name=name)

    async def initialize(self) -> None:
        """시작 시 모든 모델 가용성을 확인한다."""
        model_names = [m.name for m in self._models.values()]
        availability = await self._client.check_model_availability(model_names)
        now = time.monotonic()

        for role, info in self._models.items():
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
                impact="semantic_routing_and_rag_disabled",
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

    def get_fallback(self, role: str) -> str | None:
        """폴백 모델명을 반환한다. 폴백도 불가하면 None."""
        chain = self._config.fallback_chain.get(role, [])
        for fallback_role in chain:
            model = self.get_model(fallback_role)
            if model is not None:
                return model
        return None

    def get_fallback_role(self, role: str) -> str | None:
        """폴백 역할명을 반환한다."""
        chain = self._config.fallback_chain.get(role, [])
        for fallback_role in chain:
            if self.is_available(fallback_role):
                return fallback_role
        return None

    def resolve_model(self, role: str) -> tuple[str, str, bool]:
        """역할에 맞는 모델을 해결한다. 폴백 포함.

        Returns:
            (model_name, actual_role, fallback_used)
        """
        model = self.get_model(role)
        if model is not None:
            return model, role, False

        fallback_role = self.get_fallback_role(role)
        if fallback_role is not None:
            fallback_model = self.get_model(fallback_role)
            if fallback_model is not None:
                self._logger.warning(
                    "model_fallback_used",
                    original_role=role,
                    fallback_role=fallback_role,
                    fallback_model=fallback_model,
                )
                return fallback_model, fallback_role, True

        raise ValueError(
            f"No available model for role '{role}' and no fallback available"
        )

    async def refresh_availability(self) -> None:
        """가용성을 재확인한다."""
        model_names = [m.name for m in self._models.values()]
        availability = await self._client.check_model_availability(model_names)
        now = time.monotonic()

        changes: list[str] = []
        for role, info in self._models.items():
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
