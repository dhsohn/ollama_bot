"""RAG 파이프라인 통합 테스트."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import zipfile
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from core.config import RAGConfig
from core.rag.chunker import DocumentChunker
from core.rag.context_builder import RAGContextBuilder
from core.rag.indexer import RAGIndexer
from core.rag.pipeline import RAGPipeline
from core.rag.retriever import RAGRetriever
from core.rag.types import ChunkMetadata, RAGResult, RAGTrace, RetrievedItem, Chunk


@pytest.fixture
def rag_config():
    return RAGConfig(
        enabled=True,
        kb_dirs=["./test_kb"],
        chunk_min_tokens=10,
        chunk_max_tokens=100,
        chunk_overlap_ratio=0.1,
        retrieve_k0=5,
        rerank_enabled=False,
        rerank_topk=3,
        retrieval_score_floor=0.0,
    )


class TestDocumentChunker:

    def test_chunk_text(self, rag_config):
        chunker = DocumentChunker(rag_config)
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("Hello world. " * 100)
            f.flush()
            chunks = chunker.chunk_file(f.name)
        os.unlink(f.name)
        assert len(chunks) > 0
        assert all(c.metadata.file_type == ".txt" for c in chunks)

    def test_chunk_python(self, rag_config):
        chunker = DocumentChunker(rag_config)
        code = "import os\n\ndef hello():\n    print('hi')\n\ndef world():\n    print('world')\n"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            f.flush()
            chunks = chunker.chunk_file(f.name)
        os.unlink(f.name)
        assert len(chunks) > 0

    def test_chunk_markdown(self, rag_config):
        chunker = DocumentChunker(rag_config)
        md = "# Title\n\nSome content\n\n## Section\n\nMore content\n"
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write(md)
            f.flush()
            chunks = chunker.chunk_file(f.name)
        os.unlink(f.name)
        assert len(chunks) > 0
        # 마크다운 헤더가 section_title로 보존됨
        titles = [c.metadata.section_title for c in chunks if c.metadata.section_title]
        assert len(titles) > 0

    def test_chunk_html(self, rag_config):
        chunker = DocumentChunker(rag_config)
        html = "<html><body><h1>제목</h1><p>본문 내용입니다.</p></body></html>"
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, encoding="utf-8") as f:
            f.write(html)
            f.flush()
            chunks = chunker.chunk_file(f.name)
        os.unlink(f.name)
        assert len(chunks) > 0
        assert any("본문 내용" in c.text for c in chunks)

    def test_chunk_docx(self, rag_config):
        chunker = DocumentChunker(rag_config)
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>DOCX 테스트 본문</w:t></w:r></w:p>"
            "</w:body>"
            "</w:document>"
        )
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            with zipfile.ZipFile(f.name, mode="w") as zf:
                zf.writestr("word/document.xml", xml)
            chunks = chunker.chunk_file(f.name)
        os.unlink(f.name)
        assert len(chunks) > 0
        assert any("DOCX 테스트 본문" in c.text for c in chunks)

    def test_chunk_empty_file(self, rag_config):
        chunker = DocumentChunker(rag_config)
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("")
            f.flush()
            chunks = chunker.chunk_file(f.name)
        os.unlink(f.name)
        assert chunks == []

    def test_content_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("test content")
            f.flush()
            h1 = DocumentChunker.content_hash(f.name)
            h2 = DocumentChunker.content_hash(f.name)
        os.unlink(f.name)
        assert h1 == h2
        assert len(h1) == 16


class TestRAGIndexer:

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        dim = 10
        client.embed = AsyncMock(
            side_effect=lambda texts, model=None: [
                np.random.randn(dim).tolist() for _ in texts
            ]
        )
        return client

    @pytest_asyncio.fixture
    async def indexer(self, rag_config, mock_client):
        idx = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await idx.initialize(db_path)
            yield idx
            await idx.close()

    @pytest.mark.asyncio
    async def test_index_corpus(self, rag_config, mock_client):
        indexer = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await indexer.initialize(db_path)

            # 테스트 kb 디렉토리 생성
            corpus_dir = Path(tmpdir) / "kb"
            corpus_dir.mkdir()
            (corpus_dir / "test.txt").write_text("Hello world test content " * 20, encoding="utf-8")
            (corpus_dir / "test.md").write_text("# Title\n\nSome markdown content\n", encoding="utf-8")

            result = await indexer.index_corpus(str(corpus_dir))
            assert result["indexed"] >= 1
            assert indexer.chunk_count > 0

            await indexer.close()

    @pytest.mark.asyncio
    async def test_search(self, rag_config, mock_client):
        indexer = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await indexer.initialize(db_path)

            corpus_dir = Path(tmpdir) / "kb"
            corpus_dir.mkdir()
            (corpus_dir / "test.txt").write_text("Hello world " * 50, encoding="utf-8")

            await indexer.index_corpus(str(corpus_dir))

            query_emb = np.random.randn(10).astype(np.float32)
            results = await indexer.search(query_emb, k=3)
            assert len(results) > 0
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

            await indexer.close()

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, rag_config, mock_client):
        indexer = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await indexer.initialize(db_path)

            corpus_dir = Path(tmpdir) / "kb"
            corpus_dir.mkdir()
            (corpus_dir / "test.txt").write_text("Initial content " * 20, encoding="utf-8")

            r1 = await indexer.index_corpus(str(corpus_dir))
            assert r1["indexed"] >= 1

            # 변경 없이 재인덱싱
            r2 = await indexer.index_corpus(str(corpus_dir))
            assert r2["indexed"] == 0
            assert r2["skipped"] >= 1

            await indexer.close()

    @pytest.mark.asyncio
    async def test_index_multiple_kb_dirs_keeps_all_sources(self, rag_config, mock_client):
        indexer = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await indexer.initialize(db_path)

            kb_a = Path(tmpdir) / "kb_a"
            kb_b = Path(tmpdir) / "kb_b"
            kb_a.mkdir()
            kb_b.mkdir()
            file_a = kb_a / "a.txt"
            file_b = kb_b / "b.txt"
            file_a.write_text("A content " * 20, encoding="utf-8")
            file_b.write_text("B content " * 20, encoding="utf-8")

            result = await indexer.index_corpus([str(kb_a), str(kb_b)])
            assert result["indexed"] >= 2
            assert indexer.chunk_count > 0

            assert indexer._db is not None
            async with indexer._db.execute(
                "SELECT DISTINCT source_path FROM rag_chunks"
            ) as cursor:
                rows = await cursor.fetchall()
            sources = {row[0] for row in rows}
            assert os.path.normpath(str(file_a)) in sources
            assert os.path.normpath(str(file_b)) in sources

            result_2 = await indexer.index_corpus([str(kb_a), str(kb_b)])
            assert result_2["indexed"] == 0
            assert result_2["skipped"] >= 2

            await indexer.close()

    @pytest.mark.asyncio
    async def test_index_corpus_skips_large_files(self, rag_config, mock_client):
        rag_config.max_file_size_mb = 1
        indexer = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await indexer.initialize(db_path)

            corpus_dir = Path(tmpdir) / "kb"
            corpus_dir.mkdir()
            (corpus_dir / "small.txt").write_text("small content " * 20, encoding="utf-8")
            (corpus_dir / "large.txt").write_bytes(b"a" * (2 * 1024 * 1024))

            result = await indexer.index_corpus(str(corpus_dir))
            assert result["indexed"] >= 1
            assert result["skipped_large"] >= 1

            assert indexer._db is not None
            async with indexer._db.execute(
                "SELECT DISTINCT source_path FROM rag_chunks"
            ) as cursor:
                rows = await cursor.fetchall()
            sources = {row[0] for row in rows}
            assert os.path.normpath(str(corpus_dir / "small.txt")) in sources
            assert os.path.normpath(str(corpus_dir / "large.txt")) not in sources

            await indexer.close()

    @pytest.mark.asyncio
    async def test_index_corpus_commits_per_file_and_keeps_progress_on_failure(
        self, rag_config,
    ):
        call_count = 0

        async def embed_side_effect(texts, model=None):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("embedding failure")
            dim = 10
            return [np.random.randn(dim).tolist() for _ in texts]

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=embed_side_effect)

        indexer = RAGIndexer(rag_config, mock_client, "test-embed")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            await indexer.initialize(db_path)

            corpus_dir = Path(tmpdir) / "kb"
            corpus_dir.mkdir()
            file_a = corpus_dir / "a.txt"
            file_b = corpus_dir / "b.txt"
            file_a.write_text("alpha content " * 50, encoding="utf-8")
            file_b.write_text("beta content " * 50, encoding="utf-8")

            result = await indexer.index_corpus(str(corpus_dir))
            assert result["indexed"] == 1
            assert result["failed"] == 1

            assert indexer._db is not None
            async with indexer._db.execute(
                "SELECT DISTINCT source_path FROM rag_chunks"
            ) as cursor:
                rows = await cursor.fetchall()
            sources = {row[0] for row in rows}
            assert os.path.normpath(str(file_a)) in sources
            assert os.path.normpath(str(file_b)) not in sources

            await indexer.close()


class TestRAGRetriever:

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return client

    @pytest.mark.asyncio
    async def test_retrieve_skips_adjacent_chunks_in_same_doc(self, mock_client):
        indexer = AsyncMock()
        indexer.chunk_count = 4
        indexer.search = AsyncMock(return_value=[
            (1, 0.95),
            (2, 0.94),
            (3, 0.93),
            (4, 0.92),
        ])
        indexer.get_chunks_by_ids = AsyncMock(return_value=[
            Chunk(
                text="doc-a-10",
                metadata=ChunkMetadata(
                    doc_id="doc-a", source_path="kb/a.md", chunk_id=10,
                ),
            ),
            Chunk(
                text="doc-a-11",
                metadata=ChunkMetadata(
                    doc_id="doc-a", source_path="kb/a.md", chunk_id=11,
                ),
            ),
            Chunk(
                text="doc-a-14",
                metadata=ChunkMetadata(
                    doc_id="doc-a", source_path="kb/a.md", chunk_id=14,
                ),
            ),
            Chunk(
                text="doc-b-0",
                metadata=ChunkMetadata(
                    doc_id="doc-b", source_path="kb/b.md", chunk_id=0,
                ),
            ),
        ])

        retriever = RAGRetriever(indexer, mock_client, "test-embed")
        items = await retriever.retrieve("query", k0=10)

        selected = [(it.chunk.metadata.doc_id, it.chunk.metadata.chunk_id) for it in items]
        assert selected == [("doc-a", 10), ("doc-a", 14), ("doc-b", 0)]

    @pytest.mark.asyncio
    async def test_retrieve_respects_max_chunks_per_doc(self, mock_client):
        indexer = AsyncMock()
        indexer.chunk_count = 4
        indexer.search = AsyncMock(return_value=[
            (1, 0.95),
            (2, 0.94),
            (3, 0.93),
            (4, 0.92),
        ])
        indexer.get_chunks_by_ids = AsyncMock(return_value=[
            Chunk(
                text="doc-a-0",
                metadata=ChunkMetadata(
                    doc_id="doc-a", source_path="kb/a.md", chunk_id=0,
                ),
            ),
            Chunk(
                text="doc-a-2",
                metadata=ChunkMetadata(
                    doc_id="doc-a", source_path="kb/a.md", chunk_id=2,
                ),
            ),
            Chunk(
                text="doc-a-4",
                metadata=ChunkMetadata(
                    doc_id="doc-a", source_path="kb/a.md", chunk_id=4,
                ),
            ),
            Chunk(
                text="doc-b-0",
                metadata=ChunkMetadata(
                    doc_id="doc-b", source_path="kb/b.md", chunk_id=0,
                ),
            ),
        ])

        retriever = RAGRetriever(indexer, mock_client, "test-embed")
        items = await retriever.retrieve("query", k0=10)

        doc_a_chunks = [
            it.chunk.metadata.chunk_id
            for it in items
            if it.chunk.metadata.doc_id == "doc-a"
        ]
        assert doc_a_chunks == [0, 2]


class TestRAGContextBuilder:

    def test_build_context(self):
        builder = RAGContextBuilder()
        items = [
            RetrievedItem(
                chunk=Chunk(
                    text="Test content 1",
                    metadata=ChunkMetadata(
                        doc_id="d1", source_path="kb/test.txt",
                        chunk_id=0, section_title="Section A",
                    ),
                ),
                retrieval_score=0.9,
            ),
            RetrievedItem(
                chunk=Chunk(
                    text="Test content 2",
                    metadata=ChunkMetadata(
                        doc_id="d2", source_path="kb/test2.md",
                        chunk_id=0, section_title=None,
                    ),
                ),
                retrieval_score=0.8,
            ),
        ]
        context, citation_map = builder.build_context(items)
        assert "[#1]" in context
        assert "[#2]" in context
        assert "kb/test.txt" in context
        assert "#1" in citation_map
        assert "#2" in citation_map

    def test_build_context_empty(self):
        builder = RAGContextBuilder()
        context, citation_map = builder.build_context([])
        assert context == ""
        assert citation_map == {}

    def test_build_rag_system_suffix(self):
        suffix = RAGContextBuilder.build_rag_system_suffix("test context")
        assert "[참고 문서]" in suffix
        assert "test context" in suffix
        assert "[#번호]" in suffix
        assert "최종 답변 설명은 한국어로 작성" in suffix


class TestRAGPipeline:

    @pytest.mark.asyncio
    async def test_should_trigger_rag_metadata(self, rag_config):
        retriever = AsyncMock()
        pipeline = RAGPipeline(retriever, None, RAGContextBuilder(), rag_config)
        assert pipeline.should_trigger_rag("test", {"use_rag": True})
        assert not pipeline.should_trigger_rag("안녕하세요")

    @pytest.mark.asyncio
    async def test_should_trigger_rag_keywords(self, rag_config):
        retriever = AsyncMock()
        pipeline = RAGPipeline(retriever, None, RAGContextBuilder(), rag_config)
        assert pipeline.should_trigger_rag("내 프로젝트 문서에서 찾아줘")
        assert pipeline.should_trigger_rag("kb에서 검색해")
        assert not pipeline.should_trigger_rag("안녕하세요")

    @pytest.mark.asyncio
    async def test_execute_no_candidates(self, rag_config):
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[])
        pipeline = RAGPipeline(retriever, None, RAGContextBuilder(), rag_config)
        result = await pipeline.execute("test query")
        assert result.trace.rag_used is False
        assert result.contexts == []

    @pytest.mark.asyncio
    async def test_execute_with_candidates(self, rag_config):
        candidates = [
            RetrievedItem(
                chunk=Chunk(
                    text="Relevant content",
                    metadata=ChunkMetadata(
                        doc_id="d1", source_path="kb/test.txt",
                        chunk_id=0, section_title="Title",
                    ),
                ),
                retrieval_score=0.9,
            ),
        ]
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=candidates)
        pipeline = RAGPipeline(retriever, None, RAGContextBuilder(), rag_config)
        result = await pipeline.execute("test query")
        assert result.trace.rag_used is True
        assert len(result.contexts) == 1
        assert len(result.trace.retrieved_items) == 1


class TestRAGTrace:

    def test_to_dict(self):
        trace = RAGTrace(
            rag_used=True,
            rerank_used=False,
            retrieve_k0=40,
            total_latency_ms=123.456,
        )
        d = trace.to_dict()
        assert d["rag_used"] is True
        assert d["total_latency_ms"] == 123.5
