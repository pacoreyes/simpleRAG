# -----------------------------------------------------------
# Data Transformation Helpers
# Simple RAG — Data Pipeline Utilities
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import json
from pathlib import Path
from typing import Any, Optional

import polars as pl
import spacy
from spacy.language import Language

from simple_rag.models import ChunkRecord, DocumentRecord, SourceRow
from simple_rag.settings import settings
from simple_rag.utils.io_helpers import extract_url_domain, generate_cache_key

# Instancia global del modelo (carga diferida)
_nlp: Optional[Language] = None


def _get_nlp_model() -> Language:
    """Carga de forma diferida un modelo spaCy en blanco con sentencizer basado en reglas para español."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.blank(settings.SPACY_LANGUAGE)
        _nlp.add_pipe("sentencizer")
    return _nlp


def split_sentences(text: str) -> list[str]:
    """
    Divide el texto en oraciones usando pre-división por saltos de línea y el sentencizer de spaCy.

    Primero divide por saltos de línea (para manejar contenido tipo lista,
    como discografías), y luego aplica el sentencizer basado en reglas de
    spaCy a cada párrafo para una detección más fina de límites de oración
    dentro de la prosa.

    Args:
        text: Texto de entrada.

    Returns:
        Lista de oraciones.
    """
    nlp = _get_nlp_model()
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        doc = nlp(paragraph)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                sentences.append(s)
    return sentences


def count_tokens(text: str, tokenizer: Any) -> int:
    """Cuenta los tokens en el texto usando el tokenizer provisto."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_chunks(
    sentences: list[str],
    tokenizer: Any,
    target_tokens: int = settings.CHUNK_TARGET_TOKENS,
    overlap_sentences: int = settings.CHUNK_OVERLAP_SENTENCES,
    min_sentences: int = settings.CHUNK_MIN_SENTENCES,
) -> list[dict[str, Any]]:
    """
    Agrupa oraciones en chunks superpuestos de aproximadamente target_tokens.

    Args:
        sentences: Lista de oraciones de split_sentences().
        tokenizer: Tokenizer del modelo para el conteo de tokens.
        target_tokens: Cantidad objetivo de tokens por núcleo de chunk.
        overlap_sentences: Cantidad de oraciones a superponer entre chunks.
        min_sentences: Mínimo de oraciones para un chunk válido (se omite si hay menos).

    Returns:
        Lista de dicts de chunk con:
          - 'sentences': lista de strings de oraciones
          - 'start_idx': índice de la primera oración en la lista original
          - 'end_idx': índice de la última oración (exclusivo)
          - 'is_last': si este es el último chunk
          - 'token_count': cantidad de tokens en el chunk
    """
    if not sentences:
        return []

    chunks = []
    current_idx = 0

    while current_idx < len(sentences):
        # Empieza con el overlap del chunk anterior (si no es el primer chunk)
        if current_idx > 0:
            start_idx = max(0, current_idx - overlap_sentences)
        else:
            start_idx = 0

        # Acumula oraciones hasta alcanzar el target de tokens
        chunk_sentences = []
        token_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = count_tokens(sentence, tokenizer)

            # Siempre incluye las oraciones de overlap
            if i < current_idx:
                chunk_sentences.append(sentence)
                token_count += sentence_tokens
                end_idx = i + 1
                continue

            # Verifica si agregar esta oración supera el target
            if token_count + sentence_tokens > target_tokens and i > current_idx:
                break

            chunk_sentences.append(sentence)
            token_count += sentence_tokens
            end_idx = i + 1

        # Determina si este es el último chunk
        is_last = end_idx >= len(sentences)

        # Omite chunks demasiado pequeños (a menos que sea el último chunk)
        if len(chunk_sentences) < min_sentences and not is_last:
            # Avanza y vuelve a intentar
            current_idx = end_idx
            continue

        chunks.append({
            'sentences': chunk_sentences,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'is_last': is_last,
            'token_count': token_count,
        })

        # Avanza el índice actual (salta el área de overlap para el siguiente chunk)
        if is_last:
            break

        # El siguiente chunk empieza después del contenido actual (menos el overlap)
        current_idx = end_idx

    return chunks


def prepare_chunks_for_extraction(chunks: list[dict[str, Any]]) -> list[str]:
    """
    Convierte dicts de chunk en strings de texto plano para embedding o extracción con LLM.

    Devuelve el texto completo de cada chunk (incluyendo las oraciones de
    overlap) para preservar el contexto en la capa de almacenamiento.
    """
    return [
        text
        for chunk in chunks
        if (text := " ".join(chunk["sentences"]).strip())
    ]


def chunk_text(
    text: str,
    tokenizer: Any,
    target_tokens: int = settings.CHUNK_TARGET_TOKENS,
    overlap_sentences: int = settings.CHUNK_OVERLAP_SENTENCES,
) -> list[str]:
    """
    Función principal de chunking: divide el texto en chunks listos para extracción.

    Args:
        text: El texto completo de entrada.
        tokenizer: El tokenizer del modelo.
        target_tokens: Tokens objetivo por chunk.
        overlap_sentences: Overlap de oraciones para dar contexto.

    Returns:
        Lista de chunks de texto listos para extracción con LLM.
    """
    # Paso 1: Dividir en oraciones
    sentences = split_sentences(text)

    if not sentences:
        return []

    # Paso 2: Construir chunks superpuestos
    chunks = build_chunks(
        sentences,
        tokenizer,
        target_tokens=target_tokens,
        overlap_sentences=overlap_sentences,
    )

    # Paso 3: Unir las oraciones de cada chunk en un string de texto plano
    extraction_texts = prepare_chunks_for_extraction(chunks)

    return extraction_texts


def deduplicate_by_priority(
    df: pl.DataFrame | pl.LazyFrame,
    sort_col: str,
    unique_cols: list[str],
    descending: bool = False,
) -> pl.LazyFrame:
    """
    Deduplica un DataFrame/LazyFrame según una prioridad (orden de sort).
    """
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    lf = lf.sort(sort_col, descending=descending)

    for col in unique_cols:
        lf = lf.unique(subset=[col], keep="first", maintain_order=True)

    return lf


def normalize_and_clean_text(text: Optional[str]) -> Optional[str]:
    """Elimina espacios al inicio/final; devuelve None si la entrada está vacía o es None."""
    if not text:
        return None
    return text.strip()


# ---------------------------------------------------------------------------
# Funciones de los pasos del pipeline (usadas por data_pipeline/run.py)
# ---------------------------------------------------------------------------

def load_dataset(path: Path, limit: int | None = None) -> list[SourceRow]:
    """Lee el archivo Parquet de RagQuAS en una lista de objetos SourceRow.

    Args:
        path: Ruta al archivo Parquet de origen.
        limit: Si se especifica, lee solo las primeras N filas (para pruebas).

    Returns:
        Lista de instancias de SourceRow.
    """
    df = pl.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    return [SourceRow(**row) for row in df.to_dicts()]


def explode_and_deduplicate(rows: list[SourceRow]) -> list[DocumentRecord]:
    """Aplica explode a los slots de texto y deduplica por doc_id.

    Cada SourceRow puede referenciar hasta cinco textos de origen
    (text_1..text_5). Esta función:
      1. Emite un registro plano por cada slot (text_i, link_i) no vacío.
      2. Usa el SHA256 del texto como doc_id estable.
      3. Deduplica por doc_id, conservando la metadata de la ocurrencia de
         variante más temprana (question_1 ordena antes que question_2, etc.).

    Args:
        rows: Lista de objetos SourceRow de load_dataset().

    Returns:
        Lista de objetos DocumentRecord — uno por cada texto de origen único.
    """
    if not rows:
        return []

    records: list[dict] = []
    for row in rows:
        for i in range(1, 6):
            text: str | None = getattr(row, f"text_{i}")
            link: str | None = getattr(row, f"link_{i}")
            if not text or not text.strip():
                continue
            records.append({
                "doc_id": generate_cache_key(text),
                "topic": row.topic,
                "text": text,
                "link": link,
                "source_domain": extract_url_domain(link),
                "variant": row.variant,
            })

    if not records:
        return []

    result = (
        pl.DataFrame(records)
        .sort("variant")
        .unique(subset=["doc_id"], keep="first")
        .drop("variant")
    )

    return [
        DocumentRecord(
            doc_id=r["doc_id"],
            topic=r["topic"],
            text=r["text"],
            link=r["link"],
            source_domain=r["source_domain"],
        )
        for r in result.to_dicts()
    ]


def chunk_documents(docs: list[DocumentRecord], tokenizer: Any) -> list[ChunkRecord]:
    """Divide cada DocumentRecord en objetos ChunkRecord.

    Usa chunk_text() para que los parámetros de chunking se lean desde settings.

    Args:
        docs: Lista de objetos DocumentRecord de explode_and_deduplicate().
        tokenizer: Instancia de TiktokenTokenizer (cargada una vez por quien llama).

    Returns:
        Lista de objetos ChunkRecord en orden de documento, y orden de chunk
        dentro de cada documento.
    """
    chunks: list[ChunkRecord] = []
    for doc in docs:
        texts = chunk_text(doc.text, tokenizer)
        for i, text in enumerate(texts):
            chunks.append(ChunkRecord(
                chunk_id=f"{doc.doc_id}_chunk_{i}",
                doc_id=doc.doc_id,
                topic=doc.topic,
                text=text,
                link=doc.link,
                source_domain=doc.source_domain,
                chunk_index=i,
                char_length=len(text),
                token_count=len(tokenizer.encode(text, add_special_tokens=False)),
            ))
    return chunks


def save_chunks_parquet(chunks: list[ChunkRecord], path: Path) -> None:
    """Guarda una lista de objetos ChunkRecord en un archivo Parquet.

    Args:
        chunks: Lista de objetos ChunkRecord de chunk_documents().
        path: Ruta de destino del Parquet (se crean los directorios padre si hace falta).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([c.model_dump() for c in chunks]).write_parquet(path)


def load_chunks_parquet(path: Path) -> list[ChunkRecord]:
    """Carga objetos ChunkRecord desde un archivo Parquet escrito por save_chunks_parquet.

    Args:
        path: Ruta al archivo Parquet.

    Returns:
        Lista de instancias de ChunkRecord.
    """
    return [ChunkRecord.model_validate(row) for row in pl.read_parquet(path).to_dicts()]
