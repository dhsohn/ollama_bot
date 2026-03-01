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
_THINK_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning|scratchpad|thoughts?)\b[^>]*>"
    r".*?(?:</(?P=tag)\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)
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
_LOOSE_FINAL_MARKER_RE = re.compile(
    r"(?:\bto\s*=\s*final\b|(?:assistant\s*(?:_|-)?\s*)?final(?:\s+answer)?\b|최종\s*답변\b)",
    re.IGNORECASE,
)
_LEADING_FINAL_ARTIFACT_RE = re.compile(
    r"^(?:\s|[*`#>|:._-]|assistant|analysis|commentary|code|output|text|ko번역|번역)+",
    re.IGNORECASE,
)
_REPEATED_WORD_RUN_RE = re.compile(
    r"\b(?P<word>[a-zA-Z가-힣_]{2,})\b(?:[\s,.:;!?]+\b(?P=word)\b){6,}",
    re.IGNORECASE,
)
_REPEATED_ASSIGNMENT_RE = re.compile(
    r"\b(?P<word>[a-zA-Z_]{2,})(?:=(?P=word)){6,}=?",
    re.IGNORECASE,
)
_INTERNAL_REASONING_PHRASE_RE = re.compile(
    r"(?:\bwe need to respond\b|\bwe need to analyze\b|\bwe have a conversation\b|"
    r"\bthe user says\b|\bthe user asks\b|\blet me think\b|\banalysis:\b|"
    r"\bas per policy\b|\binternal instructions?\b|\bmust not mention policies?\b)",
    re.IGNORECASE,
)
_QUALITY_TOKEN_RE = re.compile(r"[a-zA-Z가-힣_]{2,}")


def _extract_after_loose_final_marker(text: str) -> str | None:
    """엄격한 채널 토큰이 없을 때 final 마커 뒤 텍스트를 보수적으로 추출한다."""
    matches = list(_LOOSE_FINAL_MARKER_RE.finditer(text))
    if not matches:
        return None

    candidate = text[matches[-1].end():].strip()
    if not candidate:
        return None

    # final 마커 직후에 붙는 잔여 토큰(code/analysis/markdown 장식 등)을 제거한다.
    for _ in range(3):
        updated = _LEADING_FINAL_ARTIFACT_RE.sub("", candidate, count=1).lstrip()
        if updated == candidate:
            break
        candidate = updated
        if not candidate:
            return None
    return candidate


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
            # 다만 to=final/final 같은 느슨한 마커가 있으면 마지막 후보를 복구한다.
            cleaned = _extract_after_loose_final_marker(cleaned) or ""
    else:
        markers = list(_ASSISTANT_MARKER_RE.finditer(cleaned))
        if markers:
            final_markers = [m for m in markers if m.group("channel").lower() == "final"]
            if final_markers:
                cleaned = cleaned[final_markers[-1].end():]
            else:
                cleaned = _extract_after_loose_final_marker(cleaned) or ""
        elif _INTERNAL_REASONING_PHRASE_RE.search(cleaned):
            recovered = _extract_after_loose_final_marker(cleaned)
            if recovered is not None:
                cleaned = recovered

    cleaned = _SPECIAL_TOKEN_RE.sub("", cleaned).strip()
    return cleaned


def detect_output_anomalies(text: str, cleaned: str | None = None) -> list[str]:
    """사용자 노출 관점에서 비정상 출력 패턴을 감지한다."""
    raw = text or ""
    visible = sanitize_model_output(raw) if cleaned is None else (cleaned or "")
    reasons: list[str] = []
    seen: set[str] = set()

    def _add(reason: str) -> None:
        if reason not in seen:
            seen.add(reason)
            reasons.append(reason)

    if not visible.strip():
        _add("empty_after_sanitize")

    if (
        _THINK_BLOCK_RE.search(raw)
        or "<|channel|>" in raw
        or _ASSISTANT_MARKER_RE.search(raw) is not None
    ):
        _add("internal_channel_marker")

    if _INTERNAL_REASONING_PHRASE_RE.search(raw):
        _add("internal_reasoning_phrase")

    if _REPEATED_ASSIGNMENT_RE.search(visible):
        _add("repeated_assignment_pattern")
    if _REPEATED_WORD_RUN_RE.search(visible):
        _add("repeated_word_run")

    tokens = [token.lower() for token in _QUALITY_TOKEN_RE.findall(visible)]
    if len(tokens) >= 18:
        token_counts = Counter(tokens)
        top_count = token_counts.most_common(1)[0][1]
        unique_count = len(token_counts)
        dominance = top_count / len(tokens)
        if dominance >= 0.5:
            _add("dominant_repeated_token")
        if unique_count <= 3 and top_count >= 8:
            _add("low_token_diversity")

    return reasons
