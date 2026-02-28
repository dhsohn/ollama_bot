"""RAG 재인덱싱 자동화 callable 구현."""

from __future__ import annotations

import asyncio
from typing import Any

from core.engine import Engine


def build_rag_reindex_callable(
    engine: Engine,
    logger: Any,
):
    lock = asyncio.Lock()

    async def rag_reindex(
        kb_dirs: list[str] | str | None = None,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """RAG 코퍼스를 증분 재인덱싱하고 결과를 포매팅한다."""
        _ = (model, model_role, temperature, max_tokens)

        if lock.locked():
            logger.warning("rag_reindex_already_running")
            return ""

        roots: list[str] | None
        if kb_dirs is None:
            roots = None
        elif isinstance(kb_dirs, str):
            path_text = kb_dirs.strip()
            roots = [path_text] if path_text else []
        elif isinstance(kb_dirs, list):
            roots = [str(item).strip() for item in kb_dirs if str(item).strip()]
        else:
            raise ValueError("kb_dirs must be string, list, or null")

        async with lock:
            result = await engine.reindex_rag_corpus(roots)

        indexed = int(result.get("indexed", 0))
        skipped = int(result.get("skipped", 0))
        removed = int(result.get("removed", 0))
        failed = int(result.get("failed", 0))
        skipped_large = int(result.get("skipped_large", 0))
        total_chunks = int(result.get("total_chunks", 0))
        used_roots = result.get("roots", roots or [])
        if not isinstance(used_roots, list):
            used_roots = [str(used_roots)]

        lines = [
            "🧱 RAG 재인덱싱 결과",
            f"- 대상 경로: {', '.join(str(path) for path in used_roots) if used_roots else '(none)'}",
            f"- indexed: {indexed}",
            f"- skipped: {skipped} (대용량 제외: {skipped_large})",
            f"- removed: {removed}",
            f"- failed: {failed}",
            f"- total_chunks: {total_chunks}",
        ]
        return "\n".join(lines)

    return rag_reindex
