"""Ollama 모델 가용성 관리."""

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
    """기본 채팅 모델과 retrieval 모델의 가용성을 관리한다."""

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
                # 기본 채팅 모델은 별도 체크 없이 항상 가용하다고 가정한다.
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

    def is_available(self, role: str) -> bool:
        """해당 역할의 모델이 가용한지 확인한다."""
        info = self._models.get(role)
        return info is not None and info.available
