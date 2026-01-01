"""
Configuration module for Sentinel.
"""

import os
from typing import Optional


class Config:
    """Base configuration class."""

    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = "/api/v1"

    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # =========================================================================
    # Vector Store Configuration (ChromaDB)
    # =========================================================================
    # ChromaDB is used for semantic search across:
    # - Runbooks and documentation
    # - Past incident records
    # - Knowledge base articles
    #
    # WHY VECTOR STORAGE OVER KEYWORD SEARCH?
    # ----------------------------------------
    # 1. Semantic understanding: "pod crash" matches "container restart"
    # 2. Synonym handling: "OOM" matches "out of memory"
    # 3. Context-aware: Understands query intent, not just keywords
    # 4. Natural language: Handles messy, real-world incident descriptions
    #
    # See memory/vector_store.py for detailed explanation.
    # =========================================================================
    
    # Directory where ChromaDB stores its persistent data
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chromadb")
    
    # Default collection name for the knowledge base
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "sentinel_knowledge")
    
    # Embedding model for converting text to vectors
    # all-MiniLM-L6-v2: Good balance of quality and speed (384 dimensions)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # OpenAI embeddings (optional, requires OPENAI_API_KEY)
    USE_OPENAI_EMBEDDINGS: bool = os.getenv("USE_OPENAI_EMBEDDINGS", "False").lower() == "true"
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    LOG_LEVEL = "DEBUG"
    CHROMA_PERSIST_DIRECTORY = "./data/chromadb_dev"


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    LOG_LEVEL = "WARNING"
    CHROMA_PERSIST_DIRECTORY = "/var/lib/sentinel/chromadb"


def get_config() -> Config:
    """
    Get configuration based on environment.

    Returns:
        Configuration object.
    """
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()


def get_vector_store_config():
    """
    Get VectorStoreConfig from application config.
    
    Creates a VectorStoreConfig instance using application settings.
    
    Returns:
        VectorStoreConfig: Configuration for vector store initialization.
    """
    from memory.vector_store import VectorStoreConfig
    
    config = get_config()
    
    return VectorStoreConfig(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_dimension=config.EMBEDDING_DIMENSION,
        use_openai_embeddings=config.USE_OPENAI_EMBEDDINGS,
        openai_model=config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
