import pytest
from pydantic import ValidationError

from simple_rag.models import ChunkRecord, DocumentRecord, SourceRow


class TestSourceRow:
    def test_all_required_fields(self):
        row = SourceRow(
            topic="reclamaciones",
            answer="La respuesta.",
            question="¿Cuál es la pregunta?",
            variant="question_1",
        )
        assert row.topic == "reclamaciones"
        assert row.variant == "question_1"

    def test_optional_fields_default_to_none(self):
        row = SourceRow(
            topic="yoga",
            answer="Ans",
            question="Q",
            variant="question_1",
        )
        for i in range(1, 6):
            assert getattr(row, f"context_{i}") is None
            assert getattr(row, f"text_{i}") is None
            assert getattr(row, f"link_{i}") is None

    def test_partial_text_fields(self):
        row = SourceRow(
            topic="seguros",
            answer="Ans",
            question="Q",
            variant="question_2",
            text_1="texto uno",
            link_1="https://example.com",
            context_1="fragmento de contexto",
        )
        assert row.text_1 == "texto uno"
        assert row.text_2 is None
        assert row.context_1 == "fragmento de contexto"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SourceRow(topic="yoga", answer="Ans", question="Q")

    def test_all_five_slots_populated(self):
        row = SourceRow(
            topic="recetas",
            answer="Ans",
            question="Q",
            variant="question_1",
            text_1="t1", text_2="t2", text_3="t3", text_4="t4", text_5="t5",
            link_1="https://a.com", link_2="https://b.com", link_3="https://c.com",
            link_4="https://d.com", link_5="https://e.com",
            context_1="c1", context_2="c2", context_3="c3", context_4="c4", context_5="c5",
        )
        assert row.text_5 == "t5"
        assert row.link_3 == "https://c.com"
        assert row.context_5 == "c5"


class TestDocumentRecord:
    def test_minimal_creation(self):
        doc = DocumentRecord(
            doc_id="abc123",
            topic="seguros",
            text="El texto del documento.",
        )
        assert doc.doc_id == "abc123"
        assert doc.link is None
        assert doc.source_domain is None

    def test_optional_fields_accepted(self):
        doc = DocumentRecord(
            doc_id="abc",
            topic="viajes",
            text="Texto",
            link="https://example.com",
            source_domain="example.com",
        )
        assert doc.source_domain == "example.com"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DocumentRecord(doc_id="x", topic="yoga")


class TestChunkRecord:
    def test_basic_creation(self):
        chunk = ChunkRecord(
            chunk_id="abc123_chunk_0",
            doc_id="abc123",
            topic="recetas",
            text="Fragmento del texto de prueba.",
            chunk_index=0,
            char_length=30,
            token_count=5,
        )
        assert chunk.chunk_id == "abc123_chunk_0"
        assert chunk.chunk_index == 0

    def test_chunk_id_convention(self):
        chunk = ChunkRecord(
            chunk_id="doc_0042_chunk_1",
            doc_id="doc_0042",
            topic="astronomia",
            text="Texto del segundo fragmento.",
            chunk_index=1,
            char_length=28,
            token_count=5,
        )
        assert chunk.chunk_id == "doc_0042_chunk_1"
        assert chunk.doc_id == "doc_0042"
        assert chunk.chunk_index == 1

    def test_optional_link_and_domain(self):
        chunk = ChunkRecord(
            chunk_id="x_0", doc_id="x", topic="t", text="t",
            chunk_index=0, char_length=1, token_count=1,
            link="https://example.com",
            source_domain="example.com",
        )
        assert chunk.source_domain == "example.com"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ChunkRecord(chunk_id="x_0", doc_id="x", topic="t", text="t",
                        chunk_index=0, char_length=1)
