"""
Data pipeline CLI.

Three subcommands give independent control over each pipeline phase:

    run-pipeline preprocess          # Steps 1–3: load, deduplicate, chunk
    run-pipeline load                # Steps 4–5: embed chunks, upsert to Pinecone
    run-pipeline all                 # Full pipeline (Steps 1–5 in sequence)

The Parquet file written by `preprocess` is the hand-off point between the
two phases, so each phase can be rerun without re-executing the other.
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
# Phase implementations (called by Click commands and by `all`)
# ---------------------------------------------------------------------------

def _run_preprocess(limit: int | None = None) -> None:
    """Steps 1–3: load → explode/deduplicate → chunk → save Parquet."""
    log.info("── PREPROCESS ──────────────────────────────────────────────")
    log.info("Source : %s", SOURCE_PARQUET)
    log.info("Output : %s", OUTPUT_PARQUET)
    if limit is not None:
        log.info("Row limit: %d (test mode)", limit)

    # ------------------------------------------------------------------
    # Step 1 — Load dataset
    # ------------------------------------------------------------------
    log.info("Step 1/3 — Loading dataset (%s)", SOURCE_PARQUET.name)
    rows = load_dataset(SOURCE_PARQUET, limit=limit)
    log.info("          Loaded %d source rows", len(rows))

    # ------------------------------------------------------------------
    # Step 2 — Explode & deduplicate documents
    # ------------------------------------------------------------------
    log.info("Step 2/3 — Exploding and deduplicating documents")
    docs = explode_and_deduplicate(rows)
    log.info("          %d unique documents after deduplication", len(docs))

    # ------------------------------------------------------------------
    # Step 3 — Chunk documents → save intermediate Parquet
    # ------------------------------------------------------------------
    log.info("Step 3/3 — Chunking documents → %s", OUTPUT_PARQUET.name)
    tokenizer = load_tokenizer_only()
    chunks = chunk_documents(docs, tokenizer)
    log.info("          %d chunks produced", len(chunks))
    save_chunks_parquet(chunks, OUTPUT_PARQUET)
    log.info("          Saved to %s", OUTPUT_PARQUET)

    log.info("Preprocess complete.")


def _run_load() -> None:
    """Steps 4–5: embed chunks → upsert to Pinecone.

    Reads OUTPUT_PARQUET written by the preprocess phase.
    Raises FileNotFoundError if preprocess has not been run yet.
    """
    if not OUTPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Preprocessed Parquet not found: {OUTPUT_PARQUET}\n"
            "Run `run-pipeline preprocess` first."
        )
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")

    log.info("── PINECONE LOAD ────────────────────────────────────────────")
    log.info("Input  : %s", OUTPUT_PARQUET)
    log.info("Index  : %s", settings.PINECONE_INDEX_CHUNKS)
    log.info("Model  : %s", settings.DEFAULT_EMBEDDINGS_MODEL_NAME)

    # ------------------------------------------------------------------
    # Step 4 — Generate embeddings via Pinecone Inference API
    # ------------------------------------------------------------------
    log.info("Step 1/2 — Loading chunks and generating embeddings")
    chunks = load_chunks_parquet(OUTPUT_PARQUET)
    log.info("          Loaded %d chunks", len(chunks))

    pc = get_pinecone_client(settings.PINECONE_API_KEY)
    texts = [c.text for c in chunks]
    embeddings = generate_embeddings_pinecone(
        pc=pc,
        texts=texts,
        model=settings.DEFAULT_EMBEDDINGS_MODEL_NAME,
        input_type="passage",
        batch_size=settings.PINECONE_INFERENCE_BATCH_SIZE,
        delay_seconds=settings.PINECONE_INFERENCE_DELAY_SECONDS,
        description="Embedding chunks",
    )
    log.info("          Generated %d embeddings", len(embeddings))

    # ------------------------------------------------------------------
    # Step 5 — Upsert to Pinecone
    # ------------------------------------------------------------------
    log.info("Step 2/2 — Upserting to Pinecone index '%s'", settings.PINECONE_INDEX_CHUNKS)
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

    log.info("Load complete.")


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Simple RAG Demo — data pipeline."""


@cli.command()
@click.option(
    "--limit", default=None, type=int, metavar="N",
    help="Process only the first N source rows (useful for testing).",
)
def preprocess(limit: int | None) -> None:
    """Steps 1–3: load dataset, explode/deduplicate, chunk, save Parquet."""
    _run_preprocess(limit=limit)


@cli.command(name="load")
def load_pinecone() -> None:
    """Steps 4–5: extract keywords, embed chunks, upsert to Pinecone."""
    _run_load()


@cli.command(name="all")
@click.option(
    "--limit", default=None, type=int, metavar="N",
    help="Process only the first N source rows (useful for testing).",
)
def run_all(limit: int | None) -> None:
    """Steps 1–5: full pipeline (preprocess then load)."""
    _run_preprocess(limit=limit)
    _run_load()


if __name__ == "__main__":
    cli()
