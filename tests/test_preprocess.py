import pytest
import polars as pl

from simple_rag.models import ChunkRecord, DocumentRecord, SourceRow
from simple_rag.settings import settings
from simple_rag.utils.io_helpers import extract_url_domain, generate_cache_key
from simple_rag.utils.llm_helpers import load_tokenizer_only
from simple_rag.data_pipeline.utils.data_transformation_helpers import (
    chunk_documents,
    explode_and_deduplicate,
    load_dataset,
    save_chunks_parquet,
)

SOURCE_PARQUET = settings.ASSETS_DIRPATH / "test-00000-of-00001.parquet"


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer_only()


def _row(**kwargs) -> SourceRow:
    """Construye un SourceRow mínimo; sobrescribe cualquier campo vía kwargs."""
    defaults = {
        "topic": "seguros",
        "answer": "Respuesta de prueba.",
        "question": "¿Cuál es la pregunta de prueba?",
        "variant": "question_1",
        "text_1": "Texto de prueba para el documento fuente único.",
        "link_1": "https://ejemplo.com/pagina",
    }
    defaults.update(kwargs)
    return SourceRow(**defaults)


# ---------------------------------------------------------------------------
# extract_url_domain
# ---------------------------------------------------------------------------

class TestExtractUrlDomain:
    def test_https_url(self):
        assert extract_url_domain("https://airhelp.com/es/policy") == "airhelp.com"

    def test_http_url(self):
        assert extract_url_domain("http://ejemplo.com/") == "ejemplo.com"

    def test_url_with_www(self):
        assert extract_url_domain("https://www.ejemplo.com/path?q=1") == "www.ejemplo.com"

    def test_none_returns_none(self):
        assert extract_url_domain(None) is None

    def test_empty_string_returns_none(self):
        assert extract_url_domain("") is None

    def test_url_without_scheme_returns_none(self):
        assert extract_url_domain("ejemplo.com/path") is None


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_returns_list_of_source_rows(self):
        rows = load_dataset(SOURCE_PARQUET, limit=3)
        assert isinstance(rows, list)
        assert all(isinstance(r, SourceRow) for r in rows)

    def test_limit_caps_row_count(self):
        rows = load_dataset(SOURCE_PARQUET, limit=5)
        assert len(rows) == 5

    def test_full_dataset_has_201_rows(self):
        rows = load_dataset(SOURCE_PARQUET)
        assert len(rows) == 201

    def test_required_fields_are_non_empty(self):
        rows = load_dataset(SOURCE_PARQUET, limit=3)
        for row in rows:
            assert row.topic
            assert row.question
            assert row.variant

    def test_optional_fields_are_none_or_str(self):
        rows = load_dataset(SOURCE_PARQUET, limit=3)
        for row in rows:
            assert row.text_1 is None or isinstance(row.text_1, str)
            assert row.link_1 is None or isinstance(row.link_1, str)


# ---------------------------------------------------------------------------
# explode_and_deduplicate
# ---------------------------------------------------------------------------

class TestExplodeAndDeduplicate:
    def test_empty_input_returns_empty(self):
        assert explode_and_deduplicate([]) == []

    def test_single_row_single_slot_yields_one_doc(self):
        docs = explode_and_deduplicate([_row()])
        assert len(docs) == 1
        assert isinstance(docs[0], DocumentRecord)

    def test_whitespace_only_text_is_skipped(self):
        docs = explode_and_deduplicate([_row(text_1="   \n  ", link_1=None)])
        assert len(docs) == 0

    def test_none_text_is_skipped(self):
        docs = explode_and_deduplicate([_row(text_1=None, link_1=None)])
        assert len(docs) == 0

    def test_two_distinct_slots_yield_two_docs(self):
        row = _row(
            text_1="Primer texto completamente distinto del segundo.",
            link_1="https://a.com",
            text_2="Segundo texto diferente del primero aquí.",
            link_2="https://b.com",
        )
        docs = explode_and_deduplicate([row])
        assert len(docs) == 2

    def test_same_text_across_variants_deduplicates_to_one_doc(self):
        text = "Texto idéntico compartido entre dos variantes de pregunta."
        row1 = _row(variant="question_1", question="¿Primera pregunta?", text_1=text)
        row2 = _row(variant="question_2", question="¿Segunda pregunta?", text_1=text)
        docs = explode_and_deduplicate([row1, row2])
        assert len(docs) == 1

    def test_doc_id_is_sha256_of_text(self):
        text = "Texto único para verificar el identificador del documento."
        docs = explode_and_deduplicate([_row(text_1=text)])
        assert docs[0].doc_id == generate_cache_key(text)

    def test_source_domain_extracted_from_link(self):
        docs = explode_and_deduplicate([_row(link_1="https://airhelp.com/es/")])
        assert docs[0].source_domain == "airhelp.com"

    def test_none_link_gives_none_source_domain(self):
        docs = explode_and_deduplicate([_row(link_1=None)])
        assert docs[0].source_domain is None

    def test_full_dataset_produces_documents(self):
        rows = load_dataset(SOURCE_PARQUET)
        docs = explode_and_deduplicate(rows)
        assert len(docs) > 0
        assert all(isinstance(d, DocumentRecord) for d in docs)

    def test_full_dataset_dedup_reduces_count(self):
        rows = load_dataset(SOURCE_PARQUET)
        docs = explode_and_deduplicate(rows)
        # 201 filas de origen × hasta 5 slots = hasta 1005 registros antes de dedup;
        # la deduplicación debe reducir esto a menos documentos únicos
        assert len(docs) < 201 * 5


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    @pytest.fixture
    def doc(self):
        return DocumentRecord(
            doc_id="test_doc_id",
            topic="yoga",
            text=(
                "Primera oración de prueba en español. "
                "Segunda oración con contenido relevante. "
                "Tercera oración que completa el párrafo de muestra."
            ),
            link="https://yoga.com/articulo",
            source_domain="yoga.com",
        )

    def test_empty_input_returns_empty(self, tok):
        assert chunk_documents([], tok) == []

    def test_returns_chunk_records(self, tok, doc):
        chunks = chunk_documents([doc], tok)
        assert all(isinstance(c, ChunkRecord) for c in chunks)

    def test_at_least_one_chunk_per_doc(self, tok, doc):
        chunks = chunk_documents([doc], tok)
        assert len(chunks) >= 1

    def test_chunk_id_follows_convention(self, tok, doc):
        chunks = chunk_documents([doc], tok)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"test_doc_id_chunk_{i}"

    def test_chunk_index_sequential(self, tok, doc):
        chunks = chunk_documents([doc], tok)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_metadata_inherited_from_doc(self, tok, doc):
        chunk = chunk_documents([doc], tok)[0]
        assert chunk.doc_id == doc.doc_id
        assert chunk.topic == doc.topic
        assert chunk.link == doc.link
        assert chunk.source_domain == doc.source_domain

    def test_char_length_matches_text(self, tok, doc):
        chunk = chunk_documents([doc], tok)[0]
        assert chunk.char_length == len(chunk.text)

    def test_token_count_is_positive(self, tok, doc):
        chunk = chunk_documents([doc], tok)[0]
        assert chunk.token_count > 0

    def test_long_text_produces_multiple_chunks(self, tok):
        long_text = " ".join(
            [f"Oración número {i} con suficiente contenido en español para contar tokens." for i in range(60)]
        )
        doc = DocumentRecord(doc_id="long_doc", topic="test", text=long_text)
        chunks = chunk_documents([doc], tok)
        assert len(chunks) > 1

    def test_full_pipeline_chunk_count(self, tok):
        rows = load_dataset(SOURCE_PARQUET, limit=10)
        docs = explode_and_deduplicate(rows)
        chunks = chunk_documents(docs, tok)
        assert len(chunks) >= len(docs)


# ---------------------------------------------------------------------------
# save_chunks_parquet
# ---------------------------------------------------------------------------

class TestSaveChunksParquet:
    @pytest.fixture
    def sample_chunks(self):
        return [
            ChunkRecord(
                chunk_id="doc1_chunk_0", doc_id="doc1", topic="seguros",
                text="Primer fragmento de texto de prueba.",
                link="https://a.com", source_domain="a.com",
                chunk_index=0, char_length=36, token_count=5,
            ),
            ChunkRecord(
                chunk_id="doc1_chunk_1", doc_id="doc1", topic="seguros",
                text="Segundo fragmento con más contenido.",
                link="https://a.com", source_domain="a.com",
                chunk_index=1, char_length=36, token_count=5,
            ),
        ]

    def test_file_is_created(self, tmp_path, sample_chunks):
        path = tmp_path / "chunks.parquet"
        save_chunks_parquet(sample_chunks, path)
        assert path.exists()

    def test_row_count_matches(self, tmp_path, sample_chunks):
        path = tmp_path / "chunks.parquet"
        save_chunks_parquet(sample_chunks, path)
        assert pl.read_parquet(path).shape[0] == len(sample_chunks)

    def test_all_columns_present(self, tmp_path, sample_chunks):
        path = tmp_path / "chunks.parquet"
        save_chunks_parquet(sample_chunks, path)
        cols = set(pl.read_parquet(path).columns)
        expected = {"chunk_id", "doc_id", "topic", "text", "link", "source_domain",
                    "chunk_index", "char_length", "token_count"}
        assert expected.issubset(cols)

    def test_creates_parent_dirs(self, tmp_path, sample_chunks):
        path = tmp_path / "nested" / "deep" / "chunks.parquet"
        save_chunks_parquet(sample_chunks, path)
        assert path.exists()
