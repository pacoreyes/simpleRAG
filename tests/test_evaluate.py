"""Tests unitarios para evaluate.py — solo funciones puras, sin llamadas a APIs."""

import math

import pytest

from simple_rag.rag.utils.evaluation_helpers import (
    cosine_similarity,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_hit_at_rank_1(self):
        assert recall_at_k(["doc_a", "doc_b", "doc_c"], {"doc_a"}, k=3) == 1.0

    def test_hit_at_rank_3(self):
        assert recall_at_k(["doc_x", "doc_y", "doc_a"], {"doc_a"}, k=3) == 1.0

    def test_miss_beyond_k(self):
        assert recall_at_k(["doc_x", "doc_y", "doc_z", "doc_a"], {"doc_a"}, k=3) == 0.0

    def test_miss_entirely(self):
        assert recall_at_k(["doc_x", "doc_y", "doc_z"], {"doc_a"}, k=3) == 0.0

    def test_empty_gold(self):
        assert recall_at_k(["doc_a", "doc_b"], set(), k=3) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"doc_a"}, k=3) == 0.0

    def test_multiple_gold_partial_hit(self):
        # 1 de 2 docs gold encontrados → 0.5
        result = recall_at_k(["doc_a", "doc_x", "doc_y"], {"doc_a", "doc_b"}, k=3)
        assert result == pytest.approx(0.5)

    def test_multiple_gold_full_hit(self):
        result = recall_at_k(["doc_a", "doc_b", "doc_c"], {"doc_a", "doc_b"}, k=3)
        assert result == pytest.approx(1.0)

    def test_k_larger_than_list(self):
        assert recall_at_k(["doc_a"], {"doc_a"}, k=10) == 1.0

    def test_k_equals_1(self):
        assert recall_at_k(["doc_a", "doc_b", "doc_c"], {"doc_a"}, k=1) == 1.0
        assert recall_at_k(["doc_x", "doc_a"], {"doc_a"}, k=1) == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_hit_at_rank_1(self):
        assert reciprocal_rank(["doc_a", "doc_b"], {"doc_a"}) == pytest.approx(1.0)

    def test_hit_at_rank_2(self):
        assert reciprocal_rank(["doc_x", "doc_a"], {"doc_a"}) == pytest.approx(0.5)

    def test_hit_at_rank_3(self):
        assert reciprocal_rank(["doc_x", "doc_y", "doc_a"], {"doc_a"}) == pytest.approx(1 / 3)

    def test_miss(self):
        assert reciprocal_rank(["doc_x", "doc_y"], {"doc_a"}) == 0.0

    def test_first_of_multiple_gold(self):
        # doc_b está en rank 2, doc_a está en rank 1 → el RR debería ser 1.0
        assert reciprocal_rank(["doc_a", "doc_b"], {"doc_a", "doc_b"}) == pytest.approx(1.0)

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"doc_a"}) == 0.0


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.5]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_known_value(self):
        # [1,1] vs [1,0]: cos = 1/sqrt(2) ≈ 0.7071
        result = cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert result == pytest.approx(1 / math.sqrt(2), rel=1e-5)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_all_rank_1(self):
        assert mean_reciprocal_rank([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_mixed(self):
        # MRR de [1.0, 0.5, 0.333] = 1.833/3 ≈ 0.611
        result = mean_reciprocal_rank([1.0, 0.5, 1 / 3])
        assert result == pytest.approx((1.0 + 0.5 + 1 / 3) / 3, rel=1e-5)

    def test_all_misses(self):
        assert mean_reciprocal_rank([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_empty_list(self):
        assert mean_reciprocal_rank([]) == 0.0
