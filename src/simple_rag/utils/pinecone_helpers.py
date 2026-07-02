# -----------------------------------------------------------
# Simple RAG Demo - Pinecone Vector Database Helpers
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
    """Devuelve un cliente de Pinecone inicializado."""
    return Pinecone(api_key=api_key)


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = 1024,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Crea el índice de Pinecone si aún no existe."""
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        log.info("Creando índice de Pinecone '%s' (%d dims, %s)", index_name, dimension, metric)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        log.info("Índice '%s' creado.", index_name)
    else:
        log.info("El índice '%s' ya existe.", index_name)


def upsert_vectors(
    pc: Pinecone,
    index_name: str,
    vectors: List[dict],
    batch_size: int = 100,
) -> None:
    """Hace upsert de dicts de vectores ya construidos en un índice de Pinecone, por batches.

    Cada dict debe seguir el formato de upsert de Pinecone:
    ``{"id": str, "values": List[float], "metadata": dict}``.
    Quien llama esta función es responsable de construir los dicts (incluyendo
    qué campos de metadata incluir), manteniendo el conocimiento de dominio
    fuera de este helper.

    Args:
        pc: Cliente de Pinecone inicializado.
        index_name: Nombre del índice destino.
        vectors: Lista de dicts de vectores listos para el upsert.
        batch_size: Vectores por llamada de upsert.
    """
    index = pc.Index(index_name)

    with tqdm(total=len(vectors), desc="Haciendo upsert de vectores", unit="vec") as pbar:
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start: start + batch_size]
            index.upsert(vectors=batch)
            pbar.update(len(batch))

    log.info("Se hizo upsert de %d vectores al índice '%s'.", len(vectors), index_name)


async def clear_index(pc: Pinecone, index_name: str) -> None:
    """Elimina todos los vectores de un índice de Pinecone.

    Maneja el caso en que el índice ya está vacío o no tiene un namespace
    por defecto (Pinecone serverless devuelve 404 en ese caso).

    Args:
        pc: Cliente de Pinecone inicializado.
        index_name: Nombre del índice de Pinecone a vaciar.
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
    description: str = "Generando embeddings",
) -> List[List[float]]:
    """Genera embeddings usando la Pinecone Inference API con concurrencia de asyncio.

    Pensado para el flujo de consulta del RAG, donde pueden llegar varias
    queries de forma concurrente. Usa ``asyncio.to_thread`` para envolver
    la llamada síncrona del SDK de Pinecone.

    Args:
        pc: Cliente de Pinecone inicializado.
        texts: Lista de strings a embeber.
        model: Nombre del modelo de Pinecone Inference (p. ej. 'multilingual-e5-large').
        input_type: Tipo de input de Pinecone ('passage' o 'query').
        batch_size: Cantidad de textos por request a la API.
        dimensions: Dimensión de truncado opcional.
        concurrency: Máximo de requests simultáneos a la API.
        retry_count: Máximo de reintentos por batch.
        backoff_factor: Multiplicador del backoff exponencial.
        description: Etiqueta de la barra de progreso.

    Returns:
        Lista de vectores de embedding en el mismo orden que los textos de entrada.
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
    description: str = "Generando embeddings",
) -> List[List[float]]:
    """Genera embeddings usando la Pinecone Inference API con lógica de reintentos.

    Incluye backoff exponencial para errores de rate limit (HTTP 429) y
    truncado de vector opcional.

    Args:
        pc: Cliente de Pinecone inicializado.
        texts: Lista de strings a embeber.
        model: Nombre del modelo de Pinecone Inference (p. ej. 'multilingual-e5-large').
        input_type: Tipo de input de Pinecone ('passage' o 'query').
        batch_size: Cantidad de textos por request a la API.
        dimensions: Dimensión de truncado opcional.
        delay_seconds: Pausa entre batches exitosos para mantenerse bajo los límites de TPM.
        retry_count: Máximo de reintentos por batch.
        backoff_factor: Multiplicador del backoff exponencial.
        description: Etiqueta de la barra de progreso.

    Returns:
        Lista de vectores de embedding en el mismo orden que los textos de entrada.
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
                            "Rate limit de Pinecone (429). Reintentando en %.1fs (intento %d/%d).",
                            wait_time, attempt, retry_count,
                        )
                        time.sleep(wait_time)
                    else:
                        log.error("Pinecone Inference falló: %s", e)
                        raise

    return all_embeddings
