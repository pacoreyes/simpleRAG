# -----------------------------------------------------------
# Simple RAG Demo - Gold Data Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Carga datos gold de evaluación (doc_ids, answer, topic) desde el parquet de RagQuAS.

Compartido por el test suite de integración y la app de Chainlit para que
ambos puedan adjuntar gold_topic / gold_doc_ids / gold_answer a una query
cada vez que coincide textualmente con una de las 201 preguntas del dataset.
"""

from pathlib import Path

import polars as pl

from simple_rag.settings import settings
from simple_rag.utils.io_helpers import generate_cache_key

GOLD_PARQUET: Path = settings.ASSETS_DIRPATH / "test-00000-of-00001.parquet"


def load_gold_data(parquet_path: Path = GOLD_PARQUET) -> dict[str, dict]:
    """Lee el parquet de RagQuAS y devuelve metadata gold indexada por el texto de la pregunta.

    Los doc_ids gold se derivan de los campos text_i (lo que se indexó en
    Pinecone), no de los campos context_i. context_i es un extracto de text_i;
    el doc_id de Pinecone es generate_cache_key(text_i). Para cada context_i
    buscamos el text_i que lo contiene (substring) y usamos el hash de ese
    texto como el doc_id gold.

    Args:
        parquet_path: Ruta al parquet de origen de RagQuAS.

    Returns:
        Dict indexado por el string exacto de la pregunta; cada valor tiene
        gold_doc_ids (set[str]), gold_answer (str), y gold_topic (str).
    """
    df = pl.read_parquet(parquet_path)
    lookup: dict[str, dict] = {}
    for row in df.to_dicts():
        q = row["question"]
        # Construye el mapa (hash de text_i → text_i) para esta fila
        text_hashes: list[tuple[str, str]] = []
        for i in range(1, 6):
            txt = row.get(f"text_{i}", "") or ""
            if txt.strip():
                text_hashes.append((generate_cache_key(txt), txt))

        gold_doc_ids: set[str] = set()
        for i in range(1, 6):
            ctx = row.get(f"context_{i}", "") or ""
            if not ctx.strip():
                continue
            # Busca el text_i que contiene este extracto de contexto
            ctx_snippet = ctx[:60]  # usa los primeros 60 caracteres como ancla confiable
            matched = False
            for doc_id, txt in text_hashes:
                if ctx_snippet in txt:
                    gold_doc_ids.add(doc_id)
                    matched = True
                    break
            if not matched:
                # Fallback: hashea el contexto directamente (no debería pasar con datos limpios)
                gold_doc_ids.add(generate_cache_key(ctx))

        lookup[q] = {
            "gold_doc_ids": gold_doc_ids,
            "gold_answer": row["answer"],
            "gold_topic": row["topic"],
        }
    return lookup
