"""모델 라우팅 — 입력 기반 최적 생성 모델 자동 선택.

규칙 기반(이미지/코드 감지) + 시맨틱 라우팅(앵커 임베딩)을 결합하여
vision / coding / low_cost / reasoning 중 적절한 모델을 결정한다.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from core.config import ModelRoutingConfig
from core.logging_setup import get_logger

if TYPE_CHECKING:
    from core.llm_protocol import RetrievalClientProtocol
    from core.model_registry import ModelRegistry


@dataclass
class RoutingDecision:
    """모델 라우팅 결정 결과."""

    selected_model: str
    selected_role: str
    trigger: str
    confidence: float = 0.0
    anchor_scores: dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False
    original_role: str | None = None
    classifier_used: bool = False
    latency_ms: float = 0.0
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "selected_role": self.selected_role,
            "trigger": self.trigger,
            "confidence": round(self.confidence, 4),
            "anchor_scores": {k: round(v, 4) for k, v in self.anchor_scores.items()},
            "fallback_used": self.fallback_used,
            "original_role": self.original_role,
            "classifier_used": self.classifier_used,
            "latency_ms": round(self.latency_ms, 1),
            "degraded": self.degraded,
            "degradation_reasons": self.degradation_reasons,
        }


# 코드 블록 / 에러 스택트레이스 정규식
_CODE_FENCE_RE = re.compile(r"```\w*\n[\s\S]*?```")
_STACK_TRACE_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|"
    r"File \"[^\"]+\", line \d+|"
    r"\w+Error:|"
    r"\w+Exception:|"
    r"at [\w.$]+\([\w.:]+\))",
    re.IGNORECASE,
)
_FILE_EXT_RE = re.compile(
    r"(?:^|\s)[\w/\\.-]+\.(?:py|js|ts|java|cpp|c|h|go|rs|rb|php|swift|kt)\b",
    re.IGNORECASE,
)


class _LRUEmbeddingCache:
    """간단한 LRU 임베딩 캐시."""

    def __init__(self, maxsize: int = 5000) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value


class ModelRouter:
    """입력 기반 모델 선택기."""

    def __init__(
        self,
        config: ModelRoutingConfig,
        registry: ModelRegistry,
        client: RetrievalClientProtocol,
        embedding_model: str,
    ) -> None:
        self._config = config
        self._registry = registry
        self._client = client
        self._embedding_model = embedding_model
        self._anchor_embeddings: dict[str, np.ndarray] = {}
        self._embedding_cache = _LRUEmbeddingCache(config.embedding_cache_size)
        self._code_keyword_pattern = self._build_code_pattern(config.code_keywords)
        self._logger = get_logger("model_router")
        self._initialized = False

    def get_status(self) -> dict[str, Any]:
        """라우터 가용성/저하 상태를 반환한다."""
        embedding_available = self._registry.is_available("embedding")
        classifier_available = self._registry.is_available("low_cost")
        semantic_available = self._initialized and embedding_available
        return {
            "initialized": self._initialized,
            "embedding_available": embedding_available,
            "classifier_available": classifier_available,
            "semantic_available": semantic_available,
        }

    def resolve_fallback_model(self, role: str) -> tuple[str, str] | None:
        """주어진 역할의 폴백 모델(있으면)을 반환한다."""
        fallback_role = self._registry.get_fallback_role(role)
        if fallback_role is None:
            return None
        fallback_model = self._registry.get_model(fallback_role)
        if fallback_model is None:
            return None
        return fallback_model, fallback_role

    @staticmethod
    def _build_code_pattern(keywords: list[str]) -> re.Pattern:
        escaped = [re.escape(kw) for kw in keywords if kw]
        if not escaped:
            # 빈 패턴의 무한 매칭을 방지한다.
            return re.compile(r"$^")
        return re.compile("|".join(escaped), re.IGNORECASE)

    async def initialize(self) -> None:
        """앵커 임베딩을 사전 계산한다."""
        if not self._registry.is_available("embedding"):
            self._logger.warning(
                "model_router_init_skipped",
                reason="embedding_model_unavailable",
            )
            return

        anchors = self._load_anchors()
        if not anchors:
            self._logger.warning("model_router_init_skipped", reason="no_anchors")
            return

        for category, texts in anchors.items():
            try:
                embeddings = await self._client.embed(
                    texts, model=self._embedding_model,
                )
                self._anchor_embeddings[category] = np.array(
                    embeddings, dtype=np.float32,
                )
                self._logger.debug(
                    "anchor_embeddings_computed",
                    category=category,
                    count=len(texts),
                )
            except Exception as exc:
                self._logger.error(
                    "anchor_embedding_failed",
                    category=category,
                    error=str(exc),
                )

        self._initialized = bool(self._anchor_embeddings)
        if self._initialized:
            self._logger.info(
                "model_router_initialized",
                categories=list(self._anchor_embeddings.keys()),
            )

    def _load_anchors(self) -> dict[str, list[str]]:
        anchors_path = Path(self._config.anchors_path)
        if not anchors_path.exists():
            self._logger.warning("anchors_file_not_found", path=str(anchors_path))
            return {}
        with open(anchors_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {k: v for k, v in data.items() if isinstance(v, list)}

    async def route(
        self,
        text: str,
        images: list[bytes] | None = None,
        metadata: dict | None = None,
    ) -> RoutingDecision:
        """입력을 분석하여 최적 모델을 결정한다."""
        t0 = time.monotonic()

        # Rule 0: 이미지 → vision
        if images:
            return self._make_decision("vision", "image", t0=t0)

        # Rule 1: 코드 감지 → coding
        if self._detect_code(text):
            return self._make_decision("coding", "code_regex", t0=t0)

        semantic_degradation_reasons: list[str] = []

        # Rule 2: 시맨틱 라우팅 (임베딩 사용 가능할 때)
        if self._initialized and self._registry.is_available("embedding"):
            try:
                query_emb = await self._get_embedding(text)
                scores = self._compute_anchor_scores(query_emb)

                top_category = max(scores, key=scores.get)  # type: ignore[arg-type]
                top_score = scores[top_category]
                ranked_scores = sorted(scores.values(), reverse=True)
                second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0

                if (
                    top_score >= self._config.threshold
                    and (top_score - second_score) >= self._config.margin
                ):
                    role = "low_cost" if top_category == "cheap" else "reasoning"
                    return self._make_decision(
                        role, "semantic",
                        confidence=top_score,
                        anchor_scores=scores,
                        t0=t0,
                    )

                # 불확실 → LLM 분류기
                return await self._route_via_classifier(
                    text,
                    anchor_scores=scores,
                    t0=t0,
                )
            except Exception as exc:
                self._logger.warning("semantic_routing_failed", error=str(exc))
                semantic_degradation_reasons.append("semantic_routing_failed")
        else:
            if not self._registry.is_available("embedding"):
                semantic_degradation_reasons.append("embedding_unavailable_classifier_only")
            elif not self._initialized:
                semantic_degradation_reasons.append("semantic_router_not_initialized")

        # 폴백: LLM 분류기 (임베딩 불가 시)
        return await self._route_via_classifier(
            text,
            t0=t0,
            degradation_reasons=semantic_degradation_reasons,
        )

    def _detect_code(self, text: str) -> bool:
        """정규식 기반 코드 감지."""
        if _CODE_FENCE_RE.search(text):
            return True
        if _STACK_TRACE_RE.search(text):
            return True
        if _FILE_EXT_RE.search(text):
            return True
        keyword_matches = 0
        for _ in self._code_keyword_pattern.finditer(text):
            keyword_matches += 1
            if keyword_matches >= 2:
                return True
        return False

    async def _get_embedding(self, text: str) -> np.ndarray:
        """LRU 캐시 포함 임베딩 생성."""
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        embeddings = await self._client.embed(
            [text], model=self._embedding_model,
        )
        arr = np.array(embeddings[0], dtype=np.float32)
        self._embedding_cache.put(text, arr)
        return arr

    def _compute_anchor_scores(self, query_emb: np.ndarray) -> dict[str, float]:
        """각 앵커 카테고리와의 코사인 유사도 최대값."""
        scores: dict[str, float] = {}
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        for category, anchor_matrix in self._anchor_embeddings.items():
            norms = np.linalg.norm(anchor_matrix, axis=1, keepdims=True) + 1e-10
            normalized = anchor_matrix / norms
            similarities = normalized @ query_norm
            scores[category] = float(np.max(similarities))

        return scores

    async def _classify_with_llm(self, text: str) -> str:
        """low-cost 모델로 1줄 분류."""
        model_name = self._registry.get_model("low_cost")
        if model_name is None:
            raise RuntimeError("low_cost_classifier_unavailable")
        prompt = (
            "아래 사용자 입력이 '간단한 질문/일상 대화'인지 "
            "'깊은 분석/추론이 필요한 질문'인지 판단하세요.\n"
            "반드시 CHEAP 또는 REASONING 중 하나만 출력하세요.\n\n"
            f"입력: {text[:500]}\n\n"
            "분류:"
        )
        chat_resp = await self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=10,
            temperature=0.0,
            timeout=self._config.classifier_timeout_seconds,
        )
        answer = chat_resp.content.strip().upper()
        if "REASONING" in answer:
            return "reasoning"
        return "low_cost"

    async def _route_via_classifier(
        self,
        text: str,
        *,
        anchor_scores: dict[str, float] | None = None,
        t0: float = 0.0,
        degradation_reasons: list[str] | None = None,
    ) -> RoutingDecision:
        reasons = list(degradation_reasons or [])
        try:
            role = await self._classify_with_llm(text)
            return self._make_decision(
                role,
                "classifier",
                anchor_scores=anchor_scores,
                classifier_used=True,
                t0=t0,
                degradation_reasons=reasons,
            )
        except Exception as exc:
            classifier_available = self._registry.is_available("low_cost")
            self._logger.warning(
                "classifier_routing_failed",
                error=str(exc),
                classifier_available=classifier_available,
            )
            if not classifier_available and self._registry.is_available("reasoning"):
                reasons.extend([
                    "low_cost_classifier_unavailable",
                    "forced_reasoning_without_classifier",
                ])
                return self._make_decision(
                    "reasoning",
                    "classifier_unavailable",
                    anchor_scores=anchor_scores,
                    t0=t0,
                    degradation_reasons=reasons,
                )
            reasons.append("classifier_failed_fallback")
            return self._make_decision(
                "low_cost",
                "fallback",
                anchor_scores=anchor_scores,
                t0=t0,
                degradation_reasons=reasons,
            )

    def _make_decision(
        self,
        role: str,
        trigger: str,
        *,
        confidence: float = 0.0,
        anchor_scores: dict[str, float] | None = None,
        classifier_used: bool = False,
        t0: float = 0.0,
        degradation_reasons: list[str] | None = None,
    ) -> RoutingDecision:
        """라우팅 결정을 생성한다. 폴백 처리 포함."""
        try:
            model_name, actual_role, fallback_used = self._registry.resolve_model(role)
        except ValueError:
            # 모든 모델 불가 시 default model 사용
            model_name = self._client.default_model
            actual_role = role
            fallback_used = True

        reasons = list(degradation_reasons or [])
        if fallback_used and "model_fallback_used" not in reasons:
            reasons.append("model_fallback_used")

        return RoutingDecision(
            selected_model=model_name,
            selected_role=actual_role,
            trigger=trigger,
            confidence=confidence,
            anchor_scores=anchor_scores or {},
            fallback_used=fallback_used,
            original_role=role if fallback_used else None,
            classifier_used=classifier_used,
            latency_ms=(time.monotonic() - t0) * 1000 if t0 else 0.0,
            degraded=bool(reasons),
            degradation_reasons=reasons,
        )
