# -----------------------------------------------------------
# Data Transformation Helpers
# simple_rag — data_pipeline specific utilities
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

# Global model instance (lazy loaded)
_nlp: Optional[Language] = None


def _get_nlp_model() -> Language:
    """Lazy-loads a blank spaCy model with rule-based sentencizer for Spanish."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.blank(settings.SPACY_LANGUAGE)
        _nlp.add_pipe("sentencizer")
    return _nlp


def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using newline pre-splitting and spaCy's sentencizer.

    First splits on newlines (to handle list-style content like discographies),
    then applies spaCy's rule-based sentencizer to each paragraph for further
    sentence boundary detection within prose.

    Args:
        text: Input text.

    Returns:
        List of sentences.
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
    """Counts tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_chunks(
    sentences: list[str],
    tokenizer: Any,
    target_tokens: int = settings.CHUNK_TARGET_TOKENS,
    overlap_sentences: int = settings.CHUNK_OVERLAP_SENTENCES,
    min_sentences: int = settings.CHUNK_MIN_SENTENCES,
) -> list[dict[str, Any]]:
    """
    Groups sentences into overlapping chunks of approximately target_tokens.

    Args:
        sentences: List of sentences from split_sentences().
        tokenizer: Model tokenizer for token counting.
        target_tokens: Target number of tokens per chunk core.
        overlap_sentences: Number of sentences to overlap between chunks.
        min_sentences: Minimum sentences for a valid chunk (skip if less).

    Returns:
        List of chunk dicts with:
          - 'sentences': list of sentence strings
          - 'start_idx': index of first sentence in original list
          - 'end_idx': index of last sentence (exclusive)
          - 'is_last': whether this is the last chunk
          - 'token_count': number of tokens in the chunk
    """
    if not sentences:
        return []

    chunks = []
    current_idx = 0

    while current_idx < len(sentences):
        # Start with overlap from previous chunk (if not first chunk)
        if current_idx > 0:
            start_idx = max(0, current_idx - overlap_sentences)
        else:
            start_idx = 0

        # Accumulate sentences until we reach target tokens
        chunk_sentences = []
        token_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = count_tokens(sentence, tokenizer)

            # Always include overlap sentences
            if i < current_idx:
                chunk_sentences.append(sentence)
                token_count += sentence_tokens
                end_idx = i + 1
                continue

            # Check if adding this sentence exceeds target
            if token_count + sentence_tokens > target_tokens and i > current_idx:
                break

            chunk_sentences.append(sentence)
            token_count += sentence_tokens
            end_idx = i + 1

        # Determine if this is the last chunk
        is_last = end_idx >= len(sentences)

        # Skip chunks that are too small (unless it's the last chunk)
        if len(chunk_sentences) < min_sentences and not is_last:
            # Move forward and try again
            current_idx = end_idx
            continue

        chunks.append({
            'sentences': chunk_sentences,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'is_last': is_last,
            'token_count': token_count,
        })

        # Move current index forward (skip overlap area for next chunk)
        if is_last:
            break

        # Next chunk starts after current content (minus overlap)
        current_idx = end_idx

    return chunks


def prepare_chunks_for_extraction(chunks: list[dict[str, Any]]) -> list[str]:
    """
    Converts chunk dicts into plain text strings for embedding or LLM extraction.

    Returns the full text of each chunk (including overlap sentences) to
    preserve context in the storage layer.
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
    Main chunking function: splits text into extraction-ready chunks.

    Args:
        text: The full input text.
        tokenizer: The model's tokenizer.
        target_tokens: Target tokens per chunk.
        overlap_sentences: Sentence overlap for context.

    Returns:
        List of text chunks ready for LLM extraction.
    """
    # Step 1: Split into sentences
    sentences = split_sentences(text)

    if not sentences:
        return []

    # Step 2: Build overlapping chunks
    chunks = build_chunks(
        sentences,
        tokenizer,
        target_tokens=target_tokens,
        overlap_sentences=overlap_sentences,
    )

    # Step 3: Join each chunk's sentences into a plain text string
    extraction_texts = prepare_chunks_for_extraction(chunks)

    return extraction_texts


def deduplicate_by_priority(
    df: pl.DataFrame | pl.LazyFrame,
    sort_col: str,
    unique_cols: list[str],
    descending: bool = False,
) -> pl.LazyFrame:
    """
    Deduplicates a DataFrame/LazyFrame based on priority (sort order).
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
    """Strips leading/trailing whitespace; returns None for empty or None input."""
    if not text:
        return None
    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline step functions (used by data_pipeline/run.py)
# ---------------------------------------------------------------------------

def load_dataset(path: Path, limit: int | None = None) -> list[SourceRow]:
    """Reads the RagQuAS Parquet file into a list of SourceRow objects.

    Args:
        path: Path to the source Parquet file.
        limit: If set, read only the first N rows (for testing).

    Returns:
        List of SourceRow instances.
    """
    df = pl.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    return [SourceRow(**row) for row in df.to_dicts()]


def explode_and_deduplicate(rows: list[SourceRow]) -> list[DocumentRecord]:
    """Explodes text slots and deduplicates by doc_id.

    Each SourceRow can reference up to five source texts (text_1..text_5).
    This function:
      1. Emits one flat record per non-empty (text_i, link_i) slot.
      2. Uses SHA256 of text as a stable doc_id.
      3. Deduplicates by doc_id, keeping metadata from the earliest variant
         occurrence (question_1 sorts before question_2, etc.).

    Args:
        rows: List of SourceRow objects from load_dataset().

    Returns:
        List of DocumentRecord objects — one per unique source text.
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
    """Splits each DocumentRecord into ChunkRecord objects.

    Uses chunk_text() so chunking parameters are read from settings.

    Args:
        docs: List of DocumentRecord objects from explode_and_deduplicate().
        tokenizer: TiktokenTokenizer instance (loaded once by the caller).

    Returns:
        List of ChunkRecord objects in doc order, chunk order within each doc.
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
    """Saves a list of ChunkRecord objects to a Parquet file.

    Args:
        chunks: List of ChunkRecord objects from chunk_documents().
        path: Destination Parquet path (parent dirs created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([c.model_dump() for c in chunks]).write_parquet(path)


def load_chunks_parquet(path: Path) -> list[ChunkRecord]:
    """Loads ChunkRecord objects from a Parquet file written by save_chunks_parquet.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of ChunkRecord instances.
    """
    return [ChunkRecord.model_validate(row) for row in pl.read_parquet(path).to_dicts()]