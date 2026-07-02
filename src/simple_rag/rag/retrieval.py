# -----------------------------------------------------------
# Simple RAG Demo - Retrieval
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import logging
from typing import Optional

from pinecone import Pinecone

from simple_rag.rag.utils.retrieval_helpers import cap_chunks_per_doc, match_to_chunk
from simple_rag.settings import settings
from simple_rag.utils.pinecone_helpers import generate_embeddings_pinecone_async

log = logging.getLogger(__name__)


async def retrieve(
    query: str,
    pc: Pinecone,
    pinecone_filter: Optional[dict] = None,
    top_k: int = settings.RAG_TOP_K,
    candidate_pool_size: int = settings.RAG_CANDIDATE_POOL_SIZE,
    max_chunks_per_doc: int = settings.RAG_MAX_CHUNKS_PER_DOC,
    index_name: str = settings.PINECONE_INDEX_CHUNKS,
    model: str = settings.DEFAULT_EMBEDDINGS_MODEL_NAME,
) -> list[dict]:
    """Embebe la query, consulta Pinecone, y reordena por diversidad de fuentes.

    Obtiene un pool de candidatos más grande que top_k para que, al aplicar
    el límite de chunks por doc_id (reordenamiento por diversidad de fuentes),
    todavía queden suficientes chunks para completar el top_k final, en lugar
    de truncar una lista de top_k ya de por sí acotada.

    Args:
        query: La query del usuario (reescrita) a embeber.
        pc: Cliente de Pinecone inicializado.
        pinecone_filter: Dict de filtro de metadata opcional (de build_pinecone_filter).
        top_k: Cantidad de chunks a devolver tras el reordenamiento.
        candidate_pool_size: Cantidad de candidatos a obtener de Pinecone antes de aplicar el límite.
        max_chunks_per_doc: Máximo de chunks permitidos de un mismo doc_id.
        index_name: Índice de Pinecone a consultar.
        model: Nombre del modelo de embedding.

    Returns:
        Lista de dicts de metadata de match, cada uno incluyendo 'text', 'doc_id',
        'topic', 'source_domain', 'link', 'chunk_index', y 'score'.
    """
    embeddings = await generate_embeddings_pinecone_async(
        pc=pc,
        texts=[query],
        model=model,
        input_type="query",
        batch_size=1,
        description="Embebiendo la query",
    )
    query_vector = embeddings[0]

    index = pc.Index(index_name)
    result = index.query(
        vector=query_vector,
        top_k=max(candidate_pool_size, top_k),
        filter=pinecone_filter,
        include_metadata=True,
    )

    chunks = [match_to_chunk(match) for match in result.matches]
    chunks = cap_chunks_per_doc(chunks, max_per_doc=max_chunks_per_doc, limit=top_k)

    log.debug("Se recuperaron %d chunks (filter=%s)", len(chunks), pinecone_filter)
    return chunks
