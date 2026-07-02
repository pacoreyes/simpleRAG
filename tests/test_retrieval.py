"""Tests unitarios para retrieval.py — sin llamadas a APIs."""

from simple_rag.rag.utils.retrieval_helpers import cap_chunks_per_doc


def _chunk(doc_id: str, score: float) -> dict:
    return {"doc_id": doc_id, "score": score}


class TestCapChunksPerDoc:
    def test_caps_repeated_doc_id(self):
        chunks = [
            _chunk("a", 0.9),
            _chunk("a", 0.88),
            _chunk("a", 0.87),
            _chunk("a", 0.86),
            _chunk("b", 0.85),
        ]
        result = cap_chunks_per_doc(chunks, max_per_doc=2, limit=5)
        assert [c["doc_id"] for c in result] == ["a", "a", "b"]

    def test_preserves_rank_order(self):
        chunks = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("a", 0.7), _chunk("c", 0.6)]
        result = cap_chunks_per_doc(chunks, max_per_doc=1, limit=10)
        assert [c["doc_id"] for c in result] == ["a", "b", "c"]

    def test_stops_at_limit(self):
        chunks = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7), _chunk("d", 0.6)]
        result = cap_chunks_per_doc(chunks, max_per_doc=2, limit=2)
        assert len(result) == 2
        assert [c["doc_id"] for c in result] == ["a", "b"]

    def test_no_capping_needed(self):
        chunks = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
        result = cap_chunks_per_doc(chunks, max_per_doc=2, limit=5)
        assert result == chunks

    def test_empty_input(self):
        assert cap_chunks_per_doc([], max_per_doc=2, limit=5) == []

    def test_missing_doc_id_treated_as_own_bucket(self):
        chunks = [{"score": 0.9}, {"score": 0.8}]
        result = cap_chunks_per_doc(chunks, max_per_doc=1, limit=5)
        # ambos tienen doc_id="" (default) así que comparten un bucket, limitado a 1
        assert len(result) == 1
