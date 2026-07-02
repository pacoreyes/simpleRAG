# -----------------------------------------------------------
# Simple RAG Demo - Evaluation Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Métricas puras de IR/matemática y parsing de respuestas de LLM-judge para la evaluación del RAG."""

import json
import logging
import math

from simple_rag.utils.llm_helpers import strip_json_fences

log = logging.getLogger(__name__)


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: set[str], k: int) -> float:
    """Fracción de doc_ids gold encontrados entre los top-k doc_ids recuperados.

    Devuelve 1.0 si alguno de los IDs recuperados en el top-k es un ID gold,
    0.0 en caso contrario (recall binario). Para preguntas multi-contexto con
    N IDs gold, devuelve la fracción de IDs gold cubiertos en el top-k.

    Args:
        retrieved_doc_ids: Lista ordenada de doc_ids de Pinecone (posición 0 = rank 1).
        gold_doc_ids: Set de doc_ids que corresponden a contextos gold.
        k: Rank de corte.

    Returns:
        Score de recall en [0.0, 1.0].
    """
    if not gold_doc_ids:
        return 0.0
    top_k_ids = set(retrieved_doc_ids[:k])
    hits = top_k_ids & gold_doc_ids
    return len(hits) / len(gold_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: set[str]) -> float:
    """Reciprocal rank del primer doc_id gold en la lista recuperada.

    Devuelve 0.0 si ningún doc_id gold aparece en la lista recuperada.

    Args:
        retrieved_doc_ids: Lista ordenada de doc_ids de Pinecone (posición 0 = rank 1).
        gold_doc_ids: Set de doc_ids gold.

    Returns:
        1/rank del primer hit, o 0.0 si no hay hit.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mean_reciprocal_rank(rr_scores: list[float]) -> float:
    """Calcula el MRR a partir de una lista de scores de reciprocal rank."""
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


def parse_judge_response(raw: str, error_prefix: str) -> dict:
    """Parsea una respuesta JSON de un LLM-judge, usando score 0 como fallback si hay error.

    Compartida por faithfulness() y answer_correctness(), que le piden al
    modelo judge `{"score": <1-5>, "reason": "..."}`.

    Args:
        raw: Texto crudo de la respuesta del LLM, posiblemente envuelto en fences JSON de markdown.
        error_prefix: Prefijo del mensaje de log que identifica qué llamada al
            judge falló (p. ej. "Faithfulness", "Correctness").

    Returns:
        Dict con 'score' (int) y 'reason' (str); score es 0 si falla el parsing.
    """
    try:
        return json.loads(strip_json_fences(raw))
    except json.JSONDecodeError as e:
        log.warning("Error de parsing JSON en %s: %s", error_prefix, str(e)[:200])
        return {"score": 0, "reason": "error de parsing"}
