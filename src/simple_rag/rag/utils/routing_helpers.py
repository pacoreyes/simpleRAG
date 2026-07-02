# -----------------------------------------------------------
# Simple RAG Demo - Routing Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Lógica pura para convertir una decisión del router en un filtro de Pinecone."""

from simple_rag.rag.router import RouterDecision


def build_pinecone_filter(decision: RouterDecision) -> dict | None:
    """Convierte una RouterDecision en un dict de filtro de metadata de Pinecone, o None si no hay filtro."""
    if decision.filter_mode == "exact" and decision.topics:
        return {"topic": {"$eq": decision.topics[0]}}
    if decision.filter_mode == "multi" and len(decision.topics) > 1:
        return {"topic": {"$in": decision.topics}}
    return None
