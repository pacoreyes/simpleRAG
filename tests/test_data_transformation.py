import polars as pl
import pytest

from simple_rag.data_pipeline.utils.data_transformation_helpers import (
    build_chunks,
    chunk_text,
    deduplicate_by_priority,
    normalize_and_clean_text,
    split_sentences,
)
from simple_rag.utils.io_helpers import generate_cache_key
from simple_rag.utils.llm_helpers import strip_json_fences


class _WordTokenizer:
    """Cuenta tokens como palabras separadas por espacios — determinístico, sin dependencias externas."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


@pytest.fixture
def tok():
    return _WordTokenizer()


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_empty_string_returns_empty(self):
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty(self):
        assert split_sentences("   \n   \t  ") == []

    def test_single_sentence(self):
        result = split_sentences("Hola mundo.")
        assert len(result) == 1
        assert result[0] == "Hola mundo."

    def test_newline_creates_separate_segment(self):
        result = split_sentences("Primera oración.\nSegunda oración.")
        assert len(result) >= 2

    def test_blank_lines_are_skipped(self):
        result = split_sentences("Una.\n\n\nDos.")
        assert len(result) >= 2

    def test_returns_stripped_sentences(self):
        result = split_sentences("  Texto con espacios.  ")
        assert all(s == s.strip() for s in result)


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_empty_sentences_returns_empty(self, tok):
        assert build_chunks([], tok) == []

    def test_short_text_yields_single_chunk(self, tok):
        sentences = ["Una.", "Dos.", "Tres."]
        chunks = build_chunks(sentences, tok, target_tokens=200, overlap_sentences=1, min_sentences=2)
        assert len(chunks) == 1
        assert chunks[0]["is_last"] is True

    def test_last_chunk_bypasses_min_sentences(self, tok):
        # Solo 2 oraciones pero min_sentences=5 — igual debe producir un chunk
        sentences = ["Una.", "Dos."]
        chunks = build_chunks(sentences, tok, target_tokens=500, overlap_sentences=1, min_sentences=5)
        assert len(chunks) == 1
        assert chunks[0]["is_last"] is True

    def test_long_text_produces_multiple_chunks(self, tok):
        sentences = [f"Palabra_{i} extra relleno para contar tokens." for i in range(30)]
        chunks = build_chunks(sentences, tok, target_tokens=15, overlap_sentences=1, min_sentences=2)
        assert len(chunks) > 1

    def test_overlap_second_chunk_starts_before_first_ends(self, tok):
        sentences = [f"Oración {i}." for i in range(20)]
        chunks = build_chunks(sentences, tok, target_tokens=10, overlap_sentences=2, min_sentences=2)
        if len(chunks) > 1:
            assert chunks[1]["start_idx"] < chunks[0]["end_idx"]

    def test_chunk_dict_has_required_keys(self, tok):
        sentences = ["Una.", "Dos.", "Tres.", "Cuatro.", "Cinco."]
        chunks = build_chunks(sentences, tok, target_tokens=100, overlap_sentences=1, min_sentences=2)
        required = {"sentences", "start_idx", "end_idx", "is_last", "token_count"}
        assert required.issubset(chunks[0].keys())

    def test_token_count_is_positive(self, tok):
        sentences = ["word1 word2 word3.", "word4 word5."]
        chunks = build_chunks(sentences, tok, target_tokens=100, overlap_sentences=0, min_sentences=1)
        assert chunks[0]["token_count"] > 0

    def test_all_sentences_covered(self, tok):
        sentences = [f"S{i}." for i in range(10)]
        chunks = build_chunks(sentences, tok, target_tokens=5, overlap_sentences=1, min_sentences=2)
        last = chunks[-1]
        assert last["is_last"] is True
        assert last["end_idx"] == len(sentences)

    def test_uses_settings_defaults(self, tok):
        from simple_rag.settings import settings
        sentences = ["Una oración.", "Segunda oración.", "Tercera."]
        # Llamar sin parámetros explícitos usa los valores de settings — no debe lanzar excepción
        chunks = build_chunks(sentences, tok)
        assert isinstance(chunks, list)

    def test_no_overlap_on_first_chunk(self, tok):
        sentences = [f"S{i}." for i in range(10)]
        chunks = build_chunks(sentences, tok, target_tokens=5, overlap_sentences=3, min_sentences=2)
        assert chunks[0]["start_idx"] == 0


# ---------------------------------------------------------------------------
# chunk_text  (integration: split → build → prepare)
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_string_returns_empty(self, tok):
        assert chunk_text("", tok) == []

    def test_returns_list_of_strings(self, tok):
        text = "Primera. Segunda. Tercera. Cuarta. Quinta."
        result = chunk_text(text, tok)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_short_text_single_chunk(self, tok):
        text = "Breve texto de prueba con pocas palabras."
        result = chunk_text(text, tok, target_tokens=500, overlap_sentences=1)
        assert len(result) == 1

    def test_chunk_text_is_non_empty(self, tok):
        text = "Texto real. Otra oración. Y una más."
        result = chunk_text(text, tok)
        assert all(len(c) > 0 for c in result)

    def test_uses_settings_defaults(self, tok):
        text = "Una oración completa. Otra más."
        result = chunk_text(text, tok)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# deduplicate_by_priority
# ---------------------------------------------------------------------------

class TestDeduplicateByPriority:
    def _make_df(self, data: dict) -> pl.DataFrame:
        return pl.DataFrame(data)

    def test_removes_duplicate_doc_ids(self):
        df = self._make_df({
            "variant": ["question_1", "question_2", "question_1"],
            "doc_id": ["doc_a", "doc_a", "doc_b"],
            "text": ["texto A", "texto A", "texto B"],
        })
        result = deduplicate_by_priority(df, sort_col="variant", unique_cols=["doc_id"]).collect()
        assert result.shape[0] == 2
        assert set(result["doc_id"].to_list()) == {"doc_a", "doc_b"}

    def test_keeps_first_by_ascending_sort(self):
        df = self._make_df({
            "variant": ["question_2", "question_1"],
            "doc_id": ["doc_a", "doc_a"],
            "text": ["texto v2", "texto v1"],
        })
        result = deduplicate_by_priority(df, sort_col="variant", unique_cols=["doc_id"]).collect()
        assert result.shape[0] == 1
        assert result["variant"][0] == "question_1"

    def test_descending_sort_keeps_last(self):
        df = self._make_df({
            "variant": ["question_1", "question_2"],
            "doc_id": ["doc_a", "doc_a"],
        })
        result = deduplicate_by_priority(
            df, sort_col="variant", unique_cols=["doc_id"], descending=True
        ).collect()
        assert result["variant"][0] == "question_2"

    def test_accepts_lazy_frame(self):
        lf = pl.LazyFrame({"variant": ["q1"], "doc_id": ["d1"], "text": ["t"]})
        result = deduplicate_by_priority(lf, sort_col="variant", unique_cols=["doc_id"]).collect()
        assert result.shape[0] == 1

    def test_no_duplicates_unchanged_count(self):
        df = self._make_df({
            "variant": ["question_1", "question_1"],
            "doc_id": ["doc_a", "doc_b"],
        })
        result = deduplicate_by_priority(df, sort_col="variant", unique_cols=["doc_id"]).collect()
        assert result.shape[0] == 2

    def test_multiple_unique_cols_applied_sequentially(self):
        df = self._make_df({
            "variant": ["question_1", "question_1", "question_2"],
            "doc_id":  ["doc_a",      "doc_a",      "doc_b"],
            "topic":   ["seguros",    "seguros",    "yoga"],
        })
        result = deduplicate_by_priority(
            df, sort_col="variant", unique_cols=["doc_id", "topic"]
        ).collect()
        assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# normalize_and_clean_text
# ---------------------------------------------------------------------------

class TestNormalizeAndCleanText:
    def test_strips_whitespace(self):
        assert normalize_and_clean_text("  hola  ") == "hola"

    def test_none_returns_none(self):
        assert normalize_and_clean_text(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_and_clean_text("") is None

    def test_preserves_internal_content(self):
        text = "  texto con espacios internos  "
        assert normalize_and_clean_text(text) == "texto con espacios internos"


# ---------------------------------------------------------------------------
# strip_json_fences
# ---------------------------------------------------------------------------

class TestStripJsonFences:
    def test_removes_json_fence(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        result = strip_json_fences(text)
        assert result == '{"key": "value"}'

    def test_removes_plain_fence(self):
        text = "```\n[1, 2, 3]\n```"
        result = strip_json_fences(text)
        assert result == "[1, 2, 3]"

    def test_no_fence_passthrough(self):
        text = '{"key": "value"}'
        assert strip_json_fences(text) == text

    def test_extracts_object_from_surrounding_text(self):
        text = 'Some preamble {"key": "val"} some suffix'
        result = strip_json_fences(text)
        assert result == '{"key": "val"}'


# ---------------------------------------------------------------------------
# generate_cache_key (from io_helpers)
# ---------------------------------------------------------------------------

class TestGenerateCacheKey:
    def test_returns_string(self):
        key = generate_cache_key("texto de prueba")
        assert isinstance(key, str)

    def test_deterministic(self):
        assert generate_cache_key("abc") == generate_cache_key("abc")

    def test_different_inputs_produce_different_keys(self):
        assert generate_cache_key("texto A") != generate_cache_key("texto B")

    def test_sha256_length(self):
        assert len(generate_cache_key("cualquier texto")) == 64

    def test_used_as_doc_id(self):
        text = "El texto del documento de prueba."
        doc_id = generate_cache_key(text)
        assert len(doc_id) == 64
        assert generate_cache_key(text) == doc_id
