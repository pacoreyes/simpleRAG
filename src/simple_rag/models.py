from typing import Optional

from pydantic import BaseModel


class SourceRow(BaseModel):
    """Raw row from the RagQuAS Parquet dataset (201 rows, 19 columns)."""
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
    """One unique source document after exploding and deduplicating SourceRows."""
    doc_id: str
    topic: str
    text: str
    link: Optional[str] = None
    source_domain: Optional[str] = None


class ChunkRecord(BaseModel):
    """One chunk of a DocumentRecord, ready for Pinecone upsert."""
    chunk_id: str
    doc_id: str
    topic: str
    text: str
    link: Optional[str] = None
    source_domain: Optional[str] = None
    chunk_index: int
    char_length: int
    token_count: int
