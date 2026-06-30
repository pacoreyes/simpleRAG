# -----------------------------------------------------------
# Pinecone Vector Database Helpers
# simple_rag — shared utilities
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import asyncio
import logging
import time
from typing import List, Optional

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_pinecone_client(api_key: str) -> Pinecone:
    """Returns an initialized Pinecone client."""
    return Pinecone(api_key=api_key)


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = 1024,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Creates the Pinecone index if it does not already exist."""
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        log.info("Creating Pinecone index '%s' (%d dims, %s)", index_name, dimension, metric)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        log.info("Index '%s' created.", index_name)
    else:
        log.info("Index '%s' already exists.", index_name)


def upsert_vectors(
    pc: Pinecone,
    index_name: str,
    vectors: List[dict],
    batch_size: int = 100,
) -> None:
    """Upserts pre-built vector dicts into a Pinecone index in batches.

    Each dict must follow the Pinecone upsert format:
    ``{"id": str, "values": List[float], "metadata": dict}``.
    The caller is responsible for building the dicts (including which metadata
    fields to include), keeping domain knowledge out of this helper.

    Args:
        pc: Initialized Pinecone client.
        index_name: Target index name.
        vectors: List of vector dicts ready for upsert.
        batch_size: Vectors per upsert call.
    """
    index = pc.Index(index_name)

    with tqdm(total=len(vectors), desc="Upserting vectors", unit="vec") as pbar:
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start: start + batch_size]
            index.upsert(vectors=batch)
            pbar.update(len(batch))

    log.info("Upserted %d vectors to index '%s'.", len(vectors), index_name)


async def clear_index(pc: Pinecone, index_name: str) -> None:
    """Deletes all vectors from a Pinecone index.

    Handles the case where the index is already empty or has no default
    namespace (Pinecone serverless returns 404 in this case).

    Args:
        pc: Initialized Pinecone client.
        index_name: Name of the Pinecone index to clear.
    """
    from pinecone.exceptions import NotFoundException

    try:
        index = pc.Index(index_name)
        index.delete(delete_all=True)
    except NotFoundException:
        pass


async def generate_embeddings_pinecone_async(
    pc: Pinecone,
    texts: List[str],
    model: str,
    input_type: str = "passage",
    batch_size: int = 30,
    dimensions: Optional[int] = None,
    concurrency: int = 10,
    retry_count: int = 5,
    backoff_factor: float = 2.0,
    description: str = "Generating embeddings",
) -> List[List[float]]:
    """Generates embeddings using Pinecone Inference API with asyncio concurrency.

    Intended for the RAG query path where multiple queries may arrive concurrently.
    Uses ``asyncio.to_thread`` to wrap the synchronous Pinecone SDK call.

    Args:
        pc: Initialized Pinecone client.
        texts: List of strings to embed.
        model: Pinecone Inference model name (e.g. 'multilingual-e5-large').
        input_type: Pinecone input type ('passage' or 'query').
        batch_size: Number of texts per API request.
        dimensions: Optional truncation dimension.
        concurrency: Maximum simultaneous API requests.
        retry_count: Maximum retry attempts per batch.
        backoff_factor: Exponential backoff multiplier.
        description: Progress bar label.

    Returns:
        List of embedding vectors in the same order as input texts.
    """
    semaphore = asyncio.Semaphore(concurrency)
    all_results: List[Optional[List[float]]] = [None] * len(texts)

    async def _process_batch(start_idx: int, batch_texts: List[str], pbar: tqdm) -> None:
        async with semaphore:
            for attempt in range(retry_count):
                try:
                    response = await asyncio.to_thread(
                        pc.inference.embed,
                        model=model,
                        inputs=batch_texts,
                        parameters={"input_type": input_type, "truncate": "END"},
                    )
                    for i, item in enumerate(response.data):
                        vec = item.values
                        if dimensions:
                            vec = vec[:dimensions]
                        all_results[start_idx + i] = vec
                    pbar.update(len(batch_texts))
                    return
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = (backoff_factor ** attempt) * 2
                        await asyncio.sleep(wait_time)
                    else:
                        raise

    with tqdm(total=len(texts), desc=description, leave=False) as pbar:
        tasks = [
            _process_batch(start, texts[start: start + batch_size], pbar)
            for start in range(0, len(texts), batch_size)
        ]
        await asyncio.gather(*tasks)

    return all_results


def generate_embeddings_pinecone(
    pc: Pinecone,
    texts: List[str],
    model: str,
    input_type: str = "passage",
    batch_size: int = 30,
    dimensions: Optional[int] = None,
    delay_seconds: float = 2.0,
    retry_count: int = 5,
    backoff_factor: float = 2.0,
    description: str = "Generating embeddings",
) -> List[List[float]]:
    """Generates embeddings using Pinecone Inference API with retry logic.

    Includes exponential backoff for rate-limit errors (HTTP 429) and optional
    vector truncation.

    Args:
        pc: Initialized Pinecone client.
        texts: List of strings to embed.
        model: Pinecone Inference model name (e.g. 'multilingual-e5-large').
        input_type: Pinecone input type ('passage' or 'query').
        batch_size: Number of texts per API request.
        dimensions: Optional truncation dimension.
        delay_seconds: Pause between successful batches to stay under TPM limits.
        retry_count: Maximum retry attempts per batch.
        backoff_factor: Exponential backoff multiplier.
        description: Progress bar label.

    Returns:
        List of embedding vectors in the same order as input texts.
    """
    all_embeddings: List[List[float]] = []

    with tqdm(total=len(texts), desc=description, leave=False) as pbar:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            attempt = 0
            success = False

            while not success and attempt < retry_count:
                try:
                    response = pc.inference.embed(
                        model=model,
                        inputs=batch_texts,
                        parameters={"input_type": input_type, "truncate": "END"},
                    )
                    for item in response.data:
                        vec = item.values
                        if dimensions:
                            vec = vec[:dimensions]
                        all_embeddings.append(vec)

                    success = True
                    pbar.update(len(batch_texts))

                    if len(texts) > batch_size:
                        time.sleep(delay_seconds)

                except Exception as e:
                    attempt += 1
                    error_msg = str(e).upper()
                    is_rate_limit = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

                    if is_rate_limit and attempt < retry_count:
                        wait_time = (backoff_factor**attempt) * 5
                        log.warning(
                            "Pinecone rate limit (429). Retrying in %.1fs (attempt %d/%d).",
                            wait_time, attempt, retry_count,
                        )
                        time.sleep(wait_time)
                    else:
                        log.error("Pinecone Inference failed: %s", e)
                        raise

    return all_embeddings
