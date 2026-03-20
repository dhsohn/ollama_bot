"""Shared data types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkMetadata:
    """Chunk metadata."""

    doc_id: str
    source_path: str
    chunk_id: int
    section_title: str | None = None
    content_hash: str = ""
    mtime: float = 0.0
    tokens_estimate: int = 0
    file_type: str = ""


@dataclass
class Chunk:
    """An indexed text chunk."""

    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


@dataclass
class RetrievedItem:
    """An item produced by retrieval and optional reranking."""

    chunk: Chunk
    retrieval_score: float
    rerank_score: float | None = None


@dataclass
class RAGTrace:
    """Trace metadata for a RAG execution."""

    rag_used: bool = False
    rerank_used: bool = False
    retrieve_k0: int = 0
    rerank_k: int = 0
    retrieved_items: list[dict[str, Any]] = field(default_factory=list)
    context_tokens_estimate: int = 0
    citations_keys_used: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rag_used": self.rag_used,
            "rerank_used": self.rerank_used,
            "retrieve_k0": self.retrieve_k0,
            "rerank_k": self.rerank_k,
            "retrieved_items": self.retrieved_items,
            "context_tokens_estimate": self.context_tokens_estimate,
            "citations_keys_used": self.citations_keys_used,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "error": self.error,
        }


@dataclass
class RAGResult:
    """Full RAG pipeline result."""

    contexts: list[str]
    candidates: list[RetrievedItem]
    trace: RAGTrace
    citation_map: dict[str, ChunkMetadata] = field(default_factory=dict)
