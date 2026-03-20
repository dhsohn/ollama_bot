"""Document chunking utilities for text and code files."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from core.config import RAGConfig
from core.logging_setup import get_logger
from core.rag.types import Chunk, ChunkMetadata

# Rough token estimate: Korean characters are dense, English averages more bytes.
# Conservatively estimate tokens as chars / 3.
_CHARS_PER_TOKEN = 3

# Function/class boundary patterns for code files
_PY_FUNC_RE = re.compile(r"^(?:async\s+)?def\s+\w+|^class\s+\w+", re.MULTILINE)
_JS_FUNC_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+\w+|"
    r"^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(",
    re.MULTILINE,
)

# Markdown header pattern
_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Section-header patterns for structured output files (.out, .log)
_STRUCTURED_OUTPUT_HEADERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^={3,}\s*.+\s*={3,}$", re.MULTILINE), "SECTION SEPARATOR"),
    (re.compile(r"^-{3,}\s*.+\s*-{3,}$", re.MULTILINE), "SUBSECTION SEPARATOR"),
]


class _HTMLTextExtractor(HTMLParser):
    """Extract only text nodes from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


class DocumentChunker:
    """Split documents into chunks."""

    def __init__(self, config: RAGConfig) -> None:
        self._min_chars = config.chunk_min_tokens * _CHARS_PER_TOKEN
        self._max_chars = config.chunk_max_tokens * _CHARS_PER_TOKEN
        self._overlap_ratio = config.chunk_overlap_ratio
        self._logger = get_logger("rag_chunker")

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Read a file and return its chunk list."""
        path = Path(file_path)
        ext = path.suffix.lower()
        try:
            text = self._load_file(path, ext)
        except Exception as exc:
            self._logger.warning(
                "chunk_file_read_failed",
                path=file_path,
                error=str(exc),
            )
            return []
        if not text.strip():
            return []

        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
        content_hash = self.content_hash(file_path)
        mtime = os.path.getmtime(file_path)

        if ext in (".py",):
            raw_chunks = self._chunk_code(text, _PY_FUNC_RE)
        elif ext in (".js", ".ts"):
            raw_chunks = self._chunk_code(text, _JS_FUNC_RE)
        elif ext == ".md":
            raw_chunks = self._chunk_markdown(text)
        elif ext in (".out", ".log"):
            raw_chunks = self._chunk_structured_output(text)
        else:
            raw_chunks = self._chunk_text(text)

        chunks: list[Chunk] = []
        for i, (chunk_text, section_title) in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            tokens_est = max(1, len(chunk_text) // _CHARS_PER_TOKEN)
            meta = ChunkMetadata(
                doc_id=doc_id,
                source_path=file_path,
                chunk_id=i,
                section_title=section_title,
                content_hash=content_hash,
                mtime=mtime,
                tokens_estimate=tokens_est,
                file_type=ext,
            )
            chunks.append(Chunk(text=chunk_text, metadata=meta))

        return chunks

    @staticmethod
    def _load_file(path: Path, ext: str) -> str:
        """Extract text based on file extension."""
        if ext in (".md", ".txt", ".py", ".js", ".ts"):
            return path.read_text(encoding="utf-8", errors="replace")
        if ext in (".html", ".htm"):
            return DocumentChunker._load_html(path)
        if ext == ".docx":
            return DocumentChunker._load_docx(path)
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if ext == ".csv":
            text = path.read_text(encoding="utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            return "\n".join(" | ".join(row) for row in rows)
        # Other text files
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _load_html(path: Path) -> str:
        parser = _HTMLTextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return parser.get_text()

    @staticmethod
    def _load_docx(path: Path) -> str:
        with zipfile.ZipFile(path) as zf:
            xml_data = zf.read("word/document.xml")
        root = ET.fromstring(xml_data)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for para in root.findall(".//w:p", ns):
            texts = [node.text for node in para.findall(".//w:t", ns) if node.text]
            if texts:
                paragraphs.append("".join(texts))
        return "\n".join(paragraphs)

    def _chunk_text(self, text: str) -> list[tuple[str, str | None]]:
        """Chunk plain text with a token-estimate sliding window."""
        overlap_chars = int(self._max_chars * self._overlap_ratio)
        # Clamp overlap so the cursor always advances.
        overlap_chars = min(overlap_chars, self._max_chars - 1)
        chunks: list[tuple[str, str | None]] = []
        start = 0
        while start < len(text):
            end = start + self._max_chars
            if end < len(text):
                # Prefer sentence-like boundaries.
                boundary = text.rfind("\n", start + self._min_chars, end)
                if boundary == -1:
                    boundary = text.rfind(". ", start + self._min_chars, end)
                if boundary > start:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, None))
            next_start = end - overlap_chars
            if next_start <= start:
                next_start = start + 1  # Guarantee forward progress.
            start = next_start
        return chunks

    def _chunk_markdown(self, text: str) -> list[tuple[str, str | None]]:
        """Chunk markdown by section headers."""
        headers = list(_MD_HEADER_RE.finditer(text))
        if not headers:
            return self._chunk_text(text)

        sections: list[tuple[str, str | None]] = []
        for i, match in enumerate(headers):
            section_title = match.group(2).strip()
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            section_text = text[start:end].strip()

            if len(section_text) <= self._max_chars:
                if section_text:
                    sections.append((section_text, section_title))
            else:
                # Split oversized sections further.
                sub_chunks = self._chunk_text(section_text)
                for chunk_text, _ in sub_chunks:
                    sections.append((chunk_text, section_title))

        # Text before the first header
        if headers and headers[0].start() > 0:
            preamble = text[: headers[0].start()].strip()
            if preamble:
                sections.insert(0, (preamble, None))

        return sections

    def _chunk_code(
        self, text: str, func_pattern: re.Pattern,
    ) -> list[tuple[str, str | None]]:
        """Chunk code files by preferring function and class boundaries."""
        boundaries = list(func_pattern.finditer(text))
        if not boundaries:
            return self._chunk_text(text)

        chunks: list[tuple[str, str | None]] = []

        # Imports/constants before the first function or class
        if boundaries[0].start() > 0:
            preamble = text[: boundaries[0].start()].strip()
            if preamble:
                chunks.append((preamble, "imports"))

        for i, match in enumerate(boundaries):
            func_name = match.group().strip()
            start = match.start()
            end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
            block = text[start:end].strip()

            if len(block) <= self._max_chars:
                if block:
                    chunks.append((block, func_name))
            else:
                # Split oversized functions/classes by line ranges.
                sub_chunks = self._chunk_text(block)
                for j, (chunk_text, _) in enumerate(sub_chunks):
                    label = f"{func_name} (part {j + 1})" if j > 0 else func_name
                    chunks.append((chunk_text, label))

        return chunks

    def _chunk_structured_output(self, text: str) -> list[tuple[str, str | None]]:
        """Chunk structured output files (`.out`, `.log`) by sections.

        Detect separator patterns to preserve semantic boundaries and split
        oversized sections further when needed.
        """
        # Collect all section start positions.
        boundaries: list[tuple[int, str]] = []
        for pattern, label in _STRUCTURED_OUTPUT_HEADERS:
            for m in pattern.finditer(text):
                boundaries.append((m.start(), label))

        if not boundaries:
            return self._chunk_text(text)

        # Sort by position.
        boundaries.sort(key=lambda x: x[0])

        chunks: list[tuple[str, str | None]] = []

        # Content before the first section (for example input metadata)
        if boundaries[0][0] > 0:
            preamble = text[: boundaries[0][0]].strip()
            if preamble:
                if len(preamble) <= self._max_chars:
                    chunks.append((preamble, "HEADER"))
                else:
                    for ct, _ in self._chunk_text(preamble):
                        chunks.append((ct, "HEADER"))

        # Each section
        for i, (start, label) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            section = text[start:end].strip()
            if not section:
                continue

            if len(section) <= self._max_chars:
                chunks.append((section, label))
            else:
                sub_chunks = self._chunk_text(section)
                for j, (ct, _) in enumerate(sub_chunks):
                    sub_label = f"{label} (part {j + 1})" if j > 0 else label
                    chunks.append((ct, sub_label))

        return chunks

    @staticmethod
    def content_hash(file_path: str) -> str:
        """Compute a stable content hash for a file."""
        data = Path(file_path).read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
