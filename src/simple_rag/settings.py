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
    #  RUTAS PRINCIPALES
    # ==============================================================================

    # El directorio 'app', que es la raíz para las importaciones de Python
    APP_DIR: Path = Path(__file__).resolve().parent
    # Para acceder a los recursos de nivel superior del proyecto (raíz del repositorio)
    PROJECT_DIR: Path = APP_DIR.parent.parent

    # ==============================================================================
    #  RUTAS DE DATOS LOCALES
    # ==============================================================================

    # Directorio de nivel superior para todos los datos, cachés, archivos temporales,
    # bases de datos y datasets
    DATA_DIR: Path = PROJECT_DIR / "data_volume"

    ASSETS_DIRPATH: Path = Field(default_factory=lambda: Path(".."), init=False)
    CHUNKS_CACHE_DIRPATH: Path = Field(default_factory=lambda: Path(".."), init=False)

    # ==============================================================================
    #  CONFIGURACIÓN DE BASE DE DATOS VECTORIAL
    # ==============================================================================

    # Embeddings de Pinecone Inference (multilingüe, usado vía pc.inference.embed)
    DEFAULT_EMBEDDINGS_MODEL_NAME: str = "multilingual-e5-large"
    DEFAULT_EMBEDDING_DIMENSIONS: int = 1024

    # Pinecone
    PINECONE_INDEX_CHUNKS: str = "ragchunks"
    PINECONE_INFERENCE_BATCH_SIZE: int = 30
    PINECONE_INFERENCE_DELAY_SECONDS: float = 2.0

    # ==============================================================================
    #  PROCESAMIENTO DE TEXTO / CONFIGURACIÓN RAG
    # ==============================================================================
    # Idioma
    SPACY_LANGUAGE: str = "es"   # Código de idioma BCP-47 para el tokenizer de spaCy

    # Chunking
    CHUNK_TARGET_TOKENS: int = 300
    CHUNK_OVERLAP_SENTENCES: int = 1
    CHUNK_MIN_SENTENCES: int = 2

    LLM_MAX_OUTPUT_TOKENS: int = 16384
    LLM_RETRY_COUNT: int = 5
    LLM_RETRY_BACKOFF_FACTOR: float = 2.0
    GEMINI_MODEL: str = "models/gemini-2.5-flash-lite"
    ROUTER_MODEL: str = "models/gemini-2.5-flash"
    ROUTER_THINKING_BUDGET: int = 1024
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.6
    RAG_TOP_K: int = 5
    RAG_CANDIDATE_POOL_SIZE: int = 15  # chunks obtenidos de Pinecone antes de aplicar el límite de diversidad
    RAG_MAX_CHUNKS_PER_DOC: int = 2    # máximo de chunks por doc_id tras el reordenamiento por diversidad de fuentes

    # ==============================================================================
    #  DIRECTORIOS DE CREACIÓN AUTOMÁTICA
    # ==============================================================================
    @model_validator(mode='after')
    def _compute_and_create_paths(self) -> "Settings":
        """
        Calcula las rutas derivadas y crea los directorios necesarios.

        También construye la cadena User-Agent y los headers por defecto.

        Returns:
            La instancia de Settings (self).
        """
        # Asigna los valores de directorio
        self.ASSETS_DIRPATH = self.DATA_DIR / "assets"
        self.CHUNKS_CACHE_DIRPATH = self.DATA_DIR / ".cache" / "chunks"

        # Crea los directorios si no existen
        dirs_to_create = [
            self.CHUNKS_CACHE_DIRPATH,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

        return self

    # ==============================================================================
    #  CLAVES DE API (leídas desde .env)
    # ==============================================================================
    GEMINI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""

    # ==============================================================================
    #  VARIABLES DE ENTORNO
    # ==============================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
