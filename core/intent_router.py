"""인텐트 기반 메시지 라우팅 엔진.

사용자 의도를 임베딩 유사도로 분류하고,
의도별 최적화된 처리 경로를 결정한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
import numpy as np
import yaml

from core.embedding_utils import embed_texts
from core.logging_setup import get_logger

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_CODE_FILE_RE = re.compile(
    r"\.(?:py|js|ts|java|cpp|c|h|go|rs|rb|php|swift|kt|sql|sh|yml|yaml|json)\b",
    re.IGNORECASE,
)
_CODE_HINT_RE = re.compile(
    r"(?:\b(?:traceback|stack\s*trace|exception|error|bug|api|sdk|regex)\b"
    r"|코드|코딩|프로그래밍|디버깅|함수|클래스|변수|컴파일|빌드"
    r"|에러|오류|알고리즘|파이썬|자바|자바스크립트|타입스크립트"
    r"|도커|깃|sql|정규식)",
    re.IGNORECASE,
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
        encoder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
        if self._encoder is None:
            try:
                self._encoder = TextEmbedding(
                    model_name=self._encoder_model,
                    providers=["CPUExecutionProvider"],
                )
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
                embeddings = embed_texts(self._encoder, utterances, normalize=True)

                loaded.append(
                    _RouteDefinition(
                        name=name,
                        utterances=utterances,
                        strategy=strategy,
                        embeddings=np.asarray(embeddings, dtype=np.float32),
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
            query_vec = embed_texts(self._encoder, [text], normalize=True)[0]

            scored_routes: list[tuple[_RouteDefinition, float]] = []
            for route in self._routes:
                # 각 라우트의 모든 예시와 유사도 계산, 최대값 사용
                similarities = route.embeddings @ query_vec
                max_sim = float(np.max(similarities))
                scored_routes.append((route, max_sim))

            scored_routes.sort(key=lambda item: item[1], reverse=True)
            for route, score in scored_routes:
                if score < self._min_confidence:
                    break
                if route.name == "code" and not self._looks_like_code_query(text):
                    # 코드 신호 없는 질의를 code intent로 보내는 오탐을 방지한다.
                    self._logger.info(
                        "intent_code_guard_rejected",
                        score=round(score, 4),
                    )
                    continue

                return RouteResult(
                    intent=route.name,
                    confidence=score,
                    context_strategy=route.strategy,
                )
            return None

        except Exception as exc:
            self._logger.warning("intent_classify_failed", error=str(exc))
            return None

    @staticmethod
    def _looks_like_code_query(text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if _CODE_FENCE_RE.search(normalized):
            return True
        if _CODE_FILE_RE.search(normalized):
            return True
        if _CODE_HINT_RE.search(normalized):
            return True
        return False
