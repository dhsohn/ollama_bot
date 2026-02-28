"""텍스트 유틸리티 — 키워드 추출/응답 정제."""

from __future__ import annotations

import re
from collections import Counter

_KOREAN_STOPWORDS = frozenset({
    "이", "그", "저", "것", "수", "등", "더", "좀", "잘", "안",
    "는", "은", "가", "를", "에", "의", "로", "와", "과", "도",
    "을", "이다", "하다", "있다", "되다", "없다", "같다", "보다",
    "그리고", "하지만", "그래서", "또는", "때문에", "하면",
    "합니다", "입니다", "있습니다", "됩니다", "없습니다",
    "해주세요", "알려주세요", "설명해주세요", "무엇", "어떻게",
})

_ENGLISH_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "and", "or", "but", "not", "no", "if", "so", "as", "it",
    "this", "that", "what", "how", "why", "when", "where", "who",
    "which", "all", "each", "every", "some", "any", "my", "your",
    "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "its", "our", "their",
})

_TOKEN_RE = re.compile(r"[가-힣]{2,}|[a-zA-Z]{2,}")
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)
_ASSISTANT_CHANNEL_BLOCK_RE = re.compile(
    r"(?:<\|start\|>)?(?:assistant)?\s*"
    r"<\|channel\|>(?P<channel>[a-zA-Z0-9_]+)\s*"
    r"<\|message\|>(?P<message>.*?)(?=(?:<\|end\|>"
    r"|(?:<\|start\|>)?(?:assistant)?\s*<\|channel\|>|$))",
    re.DOTALL,
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+?\|>")
_ASSISTANT_MARKER_RE = re.compile(
    r"assistant\s*(?:_|-)?\s*(?P<channel>analysis|commentary|final)\s*[:}\]\)]*",
    re.IGNORECASE,
)


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """텍스트에서 한/영 키워드를 빈도순으로 추출한다."""
    tokens = _TOKEN_RE.findall(text)
    filtered = [
        t.lower()
        for t in tokens
        if t.lower() not in _KOREAN_STOPWORDS and t.lower() not in _ENGLISH_STOPWORDS
    ]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(max_keywords)]


def sanitize_model_output(text: str) -> str:
    """모델 출력에서 내부 사고/채널 토큰을 제거해 사용자 노출 텍스트로 정제한다."""
    raw = text.strip()
    if not raw:
        return ""

    cleaned = _THINK_BLOCK_RE.sub("", raw)
    blocks = list(_ASSISTANT_CHANNEL_BLOCK_RE.finditer(cleaned))
    if blocks:
        final_blocks = [
            m.group("message").strip()
            for m in blocks
            if m.group("channel").lower() == "final"
        ]
        non_analysis_blocks = [
            m.group("message").strip()
            for m in blocks
            if m.group("channel").lower() not in {"analysis", "commentary"}
        ]
        if final_blocks:
            cleaned = final_blocks[-1]
        elif non_analysis_blocks:
            cleaned = non_analysis_blocks[-1]
        else:
            # analysis/commentary-only 출력은 내부 사고 가능성이 높아 노출하지 않는다.
            cleaned = ""
    else:
        markers = list(_ASSISTANT_MARKER_RE.finditer(cleaned))
        if markers:
            final_markers = [m for m in markers if m.group("channel").lower() == "final"]
            if final_markers:
                cleaned = cleaned[final_markers[-1].end():]
            else:
                cleaned = ""

    cleaned = _SPECIAL_TOKEN_RE.sub("", cleaned).strip()
    return cleaned
