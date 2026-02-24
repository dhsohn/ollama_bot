"""텍스트 유틸리티 — 키워드 추출 등."""

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
