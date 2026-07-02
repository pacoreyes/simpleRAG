# -----------------------------------------------------------
# Simple RAG Demo - Retrieval Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Lógica pura del lado de retrieval: reordenamiento por diversidad de fuentes y normalización de chunks."""


def cap_chunks_per_doc(
    chunks: list[dict],
    max_per_doc: int,
    limit: int,
) -> list[dict]:
    """Impone diversidad de fuentes limitando los chunks por doc_id, preservando el orden de rank.

    Recorre la lista de chunks ordenados (mejor score primero) y conserva un
    chunk solo si su doc_id todavía no llegó a max_per_doc, deteniéndose al
    juntar limit chunks. Esto evita que un único documento muy relevante
    ocupe todo el top-k con chunks casi duplicados.

    Args:
        chunks: Dicts de chunks ordenados (mejor primero), cada uno con una clave 'doc_id'.
        max_per_doc: Máximo de chunks permitidos de un mismo doc_id.
        limit: Máximo total de chunks a devolver.

    Returns:
        Lista filtrada, preservando el orden de rank, de a lo sumo `limit` chunks.
    """
    seen_counts: dict[str, int] = {}
    capped: list[dict] = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        count = seen_counts.get(doc_id, 0)
        if count >= max_per_doc:
            continue
        seen_counts[doc_id] = count + 1
        capped.append(chunk)
        if len(capped) >= limit:
            break
    return capped


def match_to_chunk(match) -> dict:
    """Normaliza un match de query de Pinecone al formato plano de chunk dict de este pipeline.

    Args:
        match: Un único objeto match de una respuesta `index.query()` de Pinecone.

    Returns:
        Dict con 'chunk_id', 'doc_id', 'topic', 'text', 'source_domain',
        'link', 'chunk_index', y 'score'.
    """
    meta = match.metadata or {}
    return {
        "chunk_id": match.id,
        "doc_id": meta.get("doc_id", ""),
        "topic": meta.get("topic", ""),
        "text": meta.get("text", ""),
        "source_domain": meta.get("source_domain", ""),
        "link": meta.get("link", ""),
        "chunk_index": meta.get("chunk_index", 0),
        "score": match.score,
    }
