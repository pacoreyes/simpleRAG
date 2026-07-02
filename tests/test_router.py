"""Tests unitarios para router.py — sin llamadas a APIs."""

import pytest

from simple_rag.rag.router import KNOWN_TOPICS, RouterDecision
from simple_rag.rag.utils.routing_helpers import build_pinecone_filter

# ---------------------------------------------------------------------------
# RouterDecision model validation
# ---------------------------------------------------------------------------

class TestRouterDecision:
    def test_valid_exact(self):
        d = RouterDecision(
            topics=["yoga"],
            filter_mode="exact",
            query_rewrite="beneficios del yoga",
            confidence=0.95,
            reasoning="Yoga question",
        )
        assert d.topics == ["yoga"]
        assert d.filter_mode == "exact"
        assert d.confidence == 0.95

    def test_valid_multi(self):
        d = RouterDecision(
            topics=["yoga", "primeros auxilios"],
            filter_mode="multi",
            query_rewrite="ejercicio para lesiones",
            confidence=0.75,
            reasoning="Overlaps two domains",
        )
        assert len(d.topics) == 2

    def test_valid_none(self):
        d = RouterDecision(
            topics=[],
            filter_mode="none",
            query_rewrite="hola",
            confidence=0.0,
            reasoning="Ambiguous greeting",
        )
        assert d.topics == []
        assert d.filter_mode == "none"

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            RouterDecision(topics=["yoga"], filter_mode="exact")  # faltan campos

    def test_topics_list_is_isolated(self):
        d1 = RouterDecision(topics=["yoga"], filter_mode="exact", query_rewrite="q", confidence=0.9, reasoning="r")
        d2 = RouterDecision(topics=["seguros"], filter_mode="exact", query_rewrite="q", confidence=0.9, reasoning="r")
        assert d1.topics != d2.topics


# ---------------------------------------------------------------------------
# build_pinecone_filter
# ---------------------------------------------------------------------------

class TestBuildPineconeFilter:
    def test_exact_single_topic(self):
        d = RouterDecision(topics=["yoga"], filter_mode="exact", query_rewrite="q", confidence=0.9, reasoning="r")
        f = build_pinecone_filter(d)
        assert f == {"topic": {"$eq": "yoga"}}

    def test_multi_topics(self):
        d = RouterDecision(
            topics=["yoga", "primeros auxilios"],
            filter_mode="multi",
            query_rewrite="q",
            confidence=0.8,
            reasoning="r",
        )
        f = build_pinecone_filter(d)
        assert f == {"topic": {"$in": ["yoga", "primeros auxilios"]}}

    def test_none_filter_mode(self):
        d = RouterDecision(topics=[], filter_mode="none", query_rewrite="q", confidence=0.0, reasoning="r")
        assert build_pinecone_filter(d) is None

    def test_exact_empty_topics_returns_none(self):
        d = RouterDecision(topics=[], filter_mode="exact", query_rewrite="q", confidence=0.9, reasoning="r")
        assert build_pinecone_filter(d) is None

    def test_multi_single_topic_falls_through(self):
        # multi con un solo topic se trata como sin filtro (routing ambiguo)
        d = RouterDecision(topics=["yoga"], filter_mode="multi", query_rewrite="q", confidence=0.8, reasoning="r")
        assert build_pinecone_filter(d) is None

    def test_exact_uses_first_topic(self):
        d = RouterDecision(
            topics=["seguros", "reclamaciones"],
            filter_mode="exact",
            query_rewrite="q",
            confidence=0.9,
            reasoning="r",
        )
        f = build_pinecone_filter(d)
        assert f == {"topic": {"$eq": "seguros"}}


# ---------------------------------------------------------------------------
# KNOWN_TOPICS list integrity
# ---------------------------------------------------------------------------

class TestKnownTopics:
    def test_non_empty(self):
        assert len(KNOWN_TOPICS) > 0

    def test_no_empty_strings(self):
        assert all(t for t in KNOWN_TOPICS)

    def test_no_duplicates(self):
        assert len(KNOWN_TOPICS) == len(set(KNOWN_TOPICS))

    def test_contains_key_topics(self):
        required = {"yoga", "seguros", "veterinaria", "reclamaciones", "primeros auxilios"}
        assert required <= set(KNOWN_TOPICS)
