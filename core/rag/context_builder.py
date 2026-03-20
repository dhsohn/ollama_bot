"""RAG context builder for citation formatting and prompt injection."""

from __future__ import annotations

from core.rag.types import ChunkMetadata, RetrievedItem


class RAGContextBuilder:
    """Format reranked results into citation-aware context."""

    def build_context(
        self,
        items: list[RetrievedItem],
        max_tokens: int = 4000,
    ) -> tuple[str, dict[str, ChunkMetadata]]:
        """Return the context string and citation map.

        Returns:
            (context_text, citation_map)
            citation_map: {"#1": ChunkMetadata, "#2": ...}
        """
        if not items:
            return "", {}

        citation_map: dict[str, ChunkMetadata] = {}
        sections: list[str] = []
        total_chars = 0
        chars_limit = max_tokens * 3  # Approximate token-to-character conversion.

        for i, item in enumerate(items):
            key = f"#{i + 1}"
            meta = item.chunk.metadata
            source = meta.source_path
            title = meta.section_title or ""

            header = f"[{key}] {source}"
            if title:
                header += f" — {title}"

            text = item.chunk.text.strip()
            section = f"{header}\n{text}"

            if total_chars + len(section) > chars_limit:
                break

            sections.append(section)
            citation_map[key] = meta
            total_chars += len(section)

        context_text = "\n\n".join(sections)
        return context_text, citation_map

    @staticmethod
    def build_rag_system_suffix(context_text: str) -> str:
        """Return the RAG instruction suffix appended to the system prompt."""
        return (
            "\n\n[참고 문서]\n"
            "아래 문서를 근거로 답변하세요. "
            "문서 기반 주장에는 [#번호]로 인용을 표시하세요. "
            "근거가 부족하면 '근거 부족'을 명시하고, "
            "일반 지식/추론은 가정임을 표현하세요.\n"
            "최종 답변 설명은 한국어로 작성하고, "
            "원문 인용/파일 경로/코드 식별자만 원문 표기를 유지하세요.\n\n"
            f"{context_text}"
        )
