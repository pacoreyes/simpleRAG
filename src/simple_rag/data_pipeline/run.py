# -----------------------------------------------------------
# Simple RAG Demo - Data Pipeline Runner
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""
CLI del data pipeline.

Tres subcomandos dan control independiente sobre cada fase del pipeline:

    run-pipeline preprocess          # Pasos 1–3: cargar, deduplicar, hacer chunking
    run-pipeline load                # Pasos 4–5: embeber chunks, upsert a Pinecone
    run-pipeline all                 # Pipeline completo (Pasos 1–5 en secuencia)

El archivo Parquet escrito por `preprocess` es el punto de entrega entre las
dos fases, así que cada fase puede volver a ejecutarse sin re-ejecutar la otra.
"""

import logging

import click

from simple_rag.data_pipeline.utils.data_transformation_helpers import (
    chunk_documents,
    explode_and_deduplicate,
    load_chunks_parquet,
    load_dataset,
    save_chunks_parquet,
)
from simple_rag.settings import settings
from simple_rag.utils.llm_helpers import load_tokenizer_only
from simple_rag.utils.pinecone_helpers import (
    ensure_index,
    generate_embeddings_pinecone,
    get_pinecone_client,
    upsert_vectors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOURCE_PARQUET = settings.ASSETS_DIRPATH / "test-00000-of-00001.parquet"
OUTPUT_PARQUET = settings.ASSETS_DIRPATH / "chunks-processed.parquet"


# ---------------------------------------------------------------------------
# Implementación de las fases (llamadas por los comandos de Click y por `all`)
# ---------------------------------------------------------------------------

def _run_preprocess(limit: int | None = None) -> None:
    """Pasos 1–3: cargar → explode/deduplicar → hacer chunking → guardar Parquet."""
    log.info("── PREPROCESS ──────────────────────────────────────────────")
    log.info("Origen : %s", SOURCE_PARQUET)
    log.info("Salida : %s", OUTPUT_PARQUET)
    if limit is not None:
        log.info("Límite de filas: %d (modo de prueba)", limit)

    # ------------------------------------------------------------------
    # Paso 1 — Cargar el dataset
    # ------------------------------------------------------------------
    log.info("Paso 1/3 — Cargando dataset (%s)", SOURCE_PARQUET.name)
    rows = load_dataset(SOURCE_PARQUET, limit=limit)
    log.info("          Se cargaron %d filas de origen", len(rows))

    # ------------------------------------------------------------------
    # Paso 2 — Explode y deduplicación de documentos
    # ------------------------------------------------------------------
    log.info("Paso 2/3 — Aplicando explode y deduplicando documentos")
    docs = explode_and_deduplicate(rows)
    log.info("          %d documentos únicos tras la deduplicación", len(docs))

    # ------------------------------------------------------------------
    # Paso 3 — Chunking de documentos → guardar Parquet intermedio
    # ------------------------------------------------------------------
    log.info("Paso 3/3 — Chunking de documentos → %s", OUTPUT_PARQUET.name)
    tokenizer = load_tokenizer_only()
    chunks = chunk_documents(docs, tokenizer)
    log.info("          %d chunks generados", len(chunks))
    save_chunks_parquet(chunks, OUTPUT_PARQUET)
    log.info("          Guardado en %s", OUTPUT_PARQUET)

    log.info("Preprocess completo.")


def _run_load() -> None:
    """Pasos 4–5: embeber chunks → upsert a Pinecone.

    Lee OUTPUT_PARQUET escrito por la fase de preprocess.
    Lanza FileNotFoundError si el preprocess aún no se ejecutó.
    """
    if not OUTPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"No se encontró el Parquet preprocesado: {OUTPUT_PARQUET}\n"
            "Ejecutá `run-pipeline preprocess` primero."
        )
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY no está configurada. Agregala a tu archivo .env.")

    log.info("── PINECONE LOAD ────────────────────────────────────────────")
    log.info("Entrada : %s", OUTPUT_PARQUET)
    log.info("Índice  : %s", settings.PINECONE_INDEX_CHUNKS)
    log.info("Modelo  : %s", settings.DEFAULT_EMBEDDINGS_MODEL_NAME)

    # ------------------------------------------------------------------
    # Paso 4 — Generar embeddings vía la Pinecone Inference API
    # ------------------------------------------------------------------
    log.info("Paso 1/2 — Cargando chunks y generando embeddings")
    chunks = load_chunks_parquet(OUTPUT_PARQUET)
    log.info("          Se cargaron %d chunks", len(chunks))

    pc = get_pinecone_client(settings.PINECONE_API_KEY)
    texts = [c.text for c in chunks]
    embeddings = generate_embeddings_pinecone(
        pc=pc,
        texts=texts,
        model=settings.DEFAULT_EMBEDDINGS_MODEL_NAME,
        input_type="passage",
        batch_size=settings.PINECONE_INFERENCE_BATCH_SIZE,
        delay_seconds=settings.PINECONE_INFERENCE_DELAY_SECONDS,
        description="Generando embeddings de los chunks",
    )
    log.info("          Se generaron %d embeddings", len(embeddings))

    # ------------------------------------------------------------------
    # Paso 5 — Upsert a Pinecone
    # ------------------------------------------------------------------
    log.info("Paso 2/2 — Haciendo upsert al índice de Pinecone '%s'", settings.PINECONE_INDEX_CHUNKS)
    ensure_index(
        pc=pc,
        index_name=settings.PINECONE_INDEX_CHUNKS,
        dimension=settings.DEFAULT_EMBEDDING_DIMENSIONS,
    )
    vectors = [
        {
            "id": chunk.chunk_id,
            "values": emb,
            "metadata": {
                "topic": chunk.topic,
                "doc_id": chunk.doc_id,
                "link": chunk.link or "",
                "source_domain": chunk.source_domain or "",
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            },
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    upsert_vectors(pc=pc, index_name=settings.PINECONE_INDEX_CHUNKS, vectors=vectors)

    log.info("Load completo.")


# ---------------------------------------------------------------------------
# CLI de Click
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Simple RAG Demo — data pipeline."""


@cli.command()
@click.option(
    "--limit", default=None, type=int, metavar="N",
    help="Procesa solo las primeras N filas de origen (útil para pruebas).",
)
def preprocess(limit: int | None) -> None:
    """Pasos 1–3: cargar dataset, explode/deduplicar, chunking, guardar Parquet."""
    _run_preprocess(limit=limit)


@cli.command(name="load")
def load_pinecone() -> None:
    """Pasos 4–5: extraer keywords, embeber chunks, upsert a Pinecone."""
    _run_load()


@cli.command(name="all")
@click.option(
    "--limit", default=None, type=int, metavar="N",
    help="Procesa solo las primeras N filas de origen (útil para pruebas).",
)
def run_all(limit: int | None) -> None:
    """Pasos 1–5: pipeline completo (preprocess y luego load)."""
    _run_preprocess(limit=limit)
    _run_load()


if __name__ == "__main__":
    cli()
