# -----------------------------------------------------------
# Simple RAG Demo Settings and Configuration
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==============================================================================
    #  CORE PATHS
    # ==============================================================================

    # The 'app' directory, which is the root for Python imports
    APP_DIR: Path = Path(__file__).resolve().parent
    # For accessing top-level project resources (Repo Root)
    PROJECT_DIR: Path = APP_DIR.parent.parent

    # ==============================================================================
    #  LOCAL DATA PATHS
    # ==============================================================================

    # Top-level directory for all data, caches, temp, databases, and datasets
    DATA_DIR: Path = PROJECT_DIR / "data_volume"

    ASSETS_DIRPATH: Path = Field(default_factory=lambda: Path(".."), init=False)
    CHUNKS_CACHE_DIRPATH: Path = Field(default_factory=lambda: Path(".."), init=False)

    # ==============================================================================
    #  VECTOR DATABASE SETTINGS
    # ==============================================================================

    # Pinecone Inference Embeddings (multilingual, used via pc.inference.embed)
    DEFAULT_EMBEDDINGS_MODEL_NAME: str = "multilingual-e5-large"
    DEFAULT_EMBEDDING_DIMENSIONS: int = 1024

    # Pinecone
    PINECONE_INDEX_CHUNKS: str = "ragchunks"
    PINECONE_INFERENCE_BATCH_SIZE: int = 30
    PINECONE_INFERENCE_DELAY_SECONDS: float = 2.0

    # ==============================================================================
    #  TEXT PROCESSING / RAG SETTINGS
    # ==============================================================================
    # Language
    SPACY_LANGUAGE: str = "es"   # BCP-47 language code for spaCy tokenizer

    # Chunking
    CHUNK_TARGET_TOKENS: int = 300
    CHUNK_OVERLAP_SENTENCES: int = 1
    CHUNK_MIN_SENTENCES: int = 2

    LLM_MAX_OUTPUT_TOKENS: int = 16384
    LLM_RETRY_COUNT: int = 5
    LLM_RETRY_BACKOFF_FACTOR: float = 2.0
    GEMINI_MODEL: str = "models/gemini-2.5-flash-lite"

    # ==============================================================================
    #  AUTO-CREATION DIRS
    # ==============================================================================
    @model_validator(mode='after')
    def _compute_and_create_paths(self) -> "Settings":
        """
        Computes derived paths and creates necessary directories.

        Also constructs the User-Agent string and default headers.

        Returns:
            The Settings instance (self).
        """
        # Assign directory values
        self.ASSETS_DIRPATH = self.DATA_DIR / "assets"
        self.CHUNKS_CACHE_DIRPATH = self.DATA_DIR / ".cache" / "chunks"

        # Create directories if they don't exist
        dirs_to_create = [
            self.CHUNKS_CACHE_DIRPATH,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

        return self

    # ==============================================================================
    #  API KEYS (read from .env)
    # ==============================================================================
    GEMINI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""

    # ==============================================================================
    #  ENVIRONMENT VARIABLES
    # ==============================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
