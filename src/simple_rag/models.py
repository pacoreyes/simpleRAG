# -----------------------------------------------------------
# Simple RAG Demo - Models
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from typing import Optional

from pydantic import BaseModel


class SourceRow(BaseModel):
    """Fila cruda del dataset Parquet de RagQuAS (201 filas, 19 columnas)."""
    topic: str
    answer: str
    question: str
    variant: str
    context_1: Optional[str] = None
    context_2: Optional[str] = None
    context_3: Optional[str] = None
    context_4: Optional[str] = None
    context_5: Optional[str] = None
    link_1: Optional[str] = None
    link_2: Optional[str] = None
    link_3: Optional[str] = None
    link_4: Optional[str] = None
    link_5: Optional[str] = None
    text_1: Optional[str] = None
    text_2: Optional[str] = None
    text_3: Optional[str] = None
    text_4: Optional[str] = None
    text_5: Optional[str] = None


class DocumentRecord(BaseModel):
    """Documento fuente único tras aplicar explode y deduplicar los SourceRows."""
    doc_id: str
    topic: str
    text: str
    link: Optional[str] = None
    source_domain: Optional[str] = None


class ChunkRecord(BaseModel):
    """Un chunk de un DocumentRecord, listo para el upsert en Pinecone."""
    chunk_id: str
    doc_id: str
    topic: str
    text: str
    link: Optional[str] = None
    source_domain: Optional[str] = None
    chunk_index: int
    char_length: int
    token_count: int
