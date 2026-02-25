"""인텐트 기반 메시지 라우팅 엔진.

사용자 의도를 임베딩 유사도로 분류하고,
의도별 최적화된 처리 경로를 결정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from core.logging_setup import get_logger

try:
    import numpy as np
except ImportError:
    np = cast(Any, None)

try:
    import sentence_transformers as sentence_transformers_module

    _HAS_ENCODER = True
except ImportError:
    _HAS_ENCODER = False
    sentence_transformers_module = cast(Any, None)

SentenceTransformer = (
    sentence_transformers_module.SentenceTransformer if _HAS_ENCODER else cast(Any, None)
)


@dataclass
class ContextStrategy:
    """의도별 컨텍스트 빌드 전략."""

    max_history: int = 50
    include_dicl: bool = True
    include_preferences: bool = True
    max_tokens: int | None = None
    system_prompt_suffix: str | None = None


@dataclass
class RouteResult:
    """라우팅 결과."""

    intent: str
    confidence: float
    context_strategy: ContextStrategy


@dataclass
class _RouteDefinition:
    """내부 라우트 정의."""

    name: str
    utterances: list[str]
    strategy: ContextStrategy
    embeddings: Any = None  # numpy array (N, dim)


class IntentRouter:
    """임베딩 기반 인텐트 분류기."""

    def __init__(
        self,
        routes_path: str = "config/intent_routes.yaml",
        encoder_model: str = "intfloat/multilingual-e5-small",
        min_confidence: float = 0.75,
        encoder: Any = None,
    ) -> None:
        self._routes_path = routes_path
        self._encoder_model = encoder_model
        self._min_confidence = min_confidence
        self._logger = get_logger("intent_router")
        self._routes: list[_RouteDefinition] = []
        self._encoder: Any = encoder
        self._enabled = False

        self._initialize()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def routes_count(self) -> int:
        return len(self._routes)

    def _initialize(self) -> None:
        """인코더 로드 + 라우트 임베딩 사전 계산."""
        if not _HAS_ENCODER:
            self._logger.warning("intent_router_no_encoder", reason="sentence-transformers not installed")
            return

        if self._encoder is None:
            try:
                self._encoder = SentenceTransformer(self._encoder_model)
                self._logger.info("intent_router_encoder_loaded", model=self._encoder_model)
            except Exception as exc:
                self._logger.warning("intent_router_encoder_failed", error=str(exc))
                return

        self._load_routes()
        if self._routes:
            self._enabled = True

    def _load_routes(self) -> None:
        """YAML에서 라우트를 로드하고 임베딩을 사전 계산한다."""
        path = Path(self._routes_path)
        if not path.exists():
            self._logger.warning("intent_routes_not_found", path=self._routes_path)
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            global_min_conf = data.get("min_confidence")
            if global_min_conf is not None:
                self._min_confidence = float(global_min_conf)

            routes_data = data.get("routes", [])
            loaded: list[_RouteDefinition] = []

            for route_data in routes_data:
                name = route_data.get("name", "unknown")
                utterances = route_data.get("utterances", [])
                if not utterances:
                    continue

                strategy_data = route_data.get("strategy", {})
                strategy = ContextStrategy(
                    max_history=strategy_data.get("max_history", 50),
                    include_dicl=strategy_data.get("include_dicl", True),
                    include_preferences=strategy_data.get("include_preferences", True),
                    max_tokens=strategy_data.get("max_tokens"),
                    system_prompt_suffix=strategy_data.get("system_prompt_suffix"),
                )

                # 예시 발화 임베딩 사전 계산
                embeddings = self._encoder.encode(
                    utterances, normalize_embeddings=True
                )

                loaded.append(
                    _RouteDefinition(
                        name=name,
                        utterances=utterances,
                        strategy=strategy,
                        embeddings=np.array(embeddings, dtype=np.float32),
                    )
                )

            self._routes = loaded
            self._logger.info("intent_routes_loaded", count=len(loaded))

        except Exception as exc:
            self._logger.warning("intent_routes_load_failed", error=str(exc))
            self._routes = []

    def classify(self, text: str) -> RouteResult | None:
        """사용자 입력의 의도를 분류한다."""
        if not self._enabled or not self._routes:
            return None

        try:
            query_emb = self._encoder.encode(text, normalize_embeddings=True)
            query_vec = np.array(query_emb, dtype=np.float32)

            best_route: _RouteDefinition | None = None
            best_score = -1.0

            for route in self._routes:
                # 각 라우트의 모든 예시와 유사도 계산, 최대값 사용
                similarities = route.embeddings @ query_vec
                max_sim = float(np.max(similarities))
                if max_sim > best_score:
                    best_score = max_sim
                    best_route = route

            if best_route is None or best_score < self._min_confidence:
                return None

            return RouteResult(
                intent=best_route.name,
                confidence=best_score,
                context_strategy=best_route.strategy,
            )

        except Exception as exc:
            self._logger.warning("intent_classify_failed", error=str(exc))
            return None
