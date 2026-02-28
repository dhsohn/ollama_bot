"""문서 청킹 — 텍스트/코드 파일을 적절한 크기의 청크로 분할한다."""

from __future__ import annotations

import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import os
import re
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

from core.config import RAGConfig
from core.logging_setup import get_logger
from core.rag.types import Chunk, ChunkMetadata

# 대략적 토큰 추정: 한국어 1글자 ≈ 1~2 tokens, 영어 1단어 ≈ 1.3 tokens
# 보수적으로 char 수 / 3 을 토큰 수로 추정
_CHARS_PER_TOKEN = 3

# 코드 함수/클래스 경계 패턴
_PY_FUNC_RE = re.compile(r"^(?:async\s+)?def\s+\w+|^class\s+\w+", re.MULTILINE)
_JS_FUNC_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+\w+|"
    r"^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(",
    re.MULTILINE,
)

# 마크다운 헤더 패턴
_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class _HTMLTextExtractor(HTMLParser):
    """HTML에서 텍스트 노드만 추출한다."""

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
    """문서를 청크로 분할한다."""

    def __init__(self, config: RAGConfig) -> None:
        self._min_chars = config.chunk_min_tokens * _CHARS_PER_TOKEN
        self._max_chars = config.chunk_max_tokens * _CHARS_PER_TOKEN
        self._overlap_ratio = config.chunk_overlap_ratio
        self._logger = get_logger("rag_chunker")

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """파일을 읽어 청크 목록을 반환한다."""
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
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        mtime = os.path.getmtime(file_path)

        if ext in (".py",):
            raw_chunks = self._chunk_code(text, _PY_FUNC_RE)
        elif ext in (".js", ".ts"):
            raw_chunks = self._chunk_code(text, _JS_FUNC_RE)
        elif ext == ".md":
            raw_chunks = self._chunk_markdown(text)
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
        """파일 확장자에 따라 텍스트를 추출한다."""
        if ext in (".md", ".txt", ".py", ".js", ".ts"):
            return path.read_text(encoding="utf-8", errors="replace")
        if ext in (".html", ".htm"):
            return DocumentChunker._load_html(path)
        if ext == ".docx":
            return DocumentChunker._load_docx(path)
        if ext == ".pdf":
            return DocumentChunker._load_pdf(path)
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if ext == ".csv":
            text = path.read_text(encoding="utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            return "\n".join(" | ".join(row) for row in rows)
        # 기타 텍스트 파일
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

    @staticmethod
    def _load_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                pages.append(page_text)
        return "\n\n".join(pages)

    def _chunk_text(self, text: str) -> list[tuple[str, str | None]]:
        """토큰 수 기반 슬라이딩 윈도우 청킹."""
        overlap_chars = int(self._max_chars * self._overlap_ratio)
        # overlap이 max_chars 이상이면 전진 불가 → 상한 보정
        overlap_chars = min(overlap_chars, self._max_chars - 1)
        chunks: list[tuple[str, str | None]] = []
        start = 0
        while start < len(text):
            end = start + self._max_chars
            if end < len(text):
                # 문장 경계에서 자르기
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
                next_start = start + 1  # 전진 보장
            start = next_start
        return chunks

    def _chunk_markdown(self, text: str) -> list[tuple[str, str | None]]:
        """마크다운 헤더를 기준으로 섹션 단위 청킹."""
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
                # 큰 섹션은 추가 분할
                sub_chunks = self._chunk_text(section_text)
                for chunk_text, _ in sub_chunks:
                    sections.append((chunk_text, section_title))

        # 첫 번째 헤더 이전 텍스트
        if headers and headers[0].start() > 0:
            preamble = text[: headers[0].start()].strip()
            if preamble:
                sections.insert(0, (preamble, None))

        return sections

    def _chunk_code(
        self, text: str, func_pattern: re.Pattern,
    ) -> list[tuple[str, str | None]]:
        """코드 파일: 함수/클래스 단위 우선 분할."""
        boundaries = list(func_pattern.finditer(text))
        if not boundaries:
            return self._chunk_text(text)

        chunks: list[tuple[str, str | None]] = []

        # 첫 함수/클래스 이전의 import/상수 영역
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
                # 큰 함수/클래스는 라인 기반 분할
                sub_chunks = self._chunk_text(block)
                for j, (chunk_text, _) in enumerate(sub_chunks):
                    label = f"{func_name} (part {j + 1})" if j > 0 else func_name
                    chunks.append((chunk_text, label))

        return chunks

    @staticmethod
    def content_hash(file_path: str) -> str:
        """파일의 content hash를 계산한다."""
        data = Path(file_path).read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
