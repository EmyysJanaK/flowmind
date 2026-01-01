"""
Memory module for Sentinel multi-agent system.

Handles conversation history, context management, state persistence,
and vector-based semantic search for knowledge retrieval.

Components:
- MemoryStore: Conversation history and shared state
- VectorStore: ChromaDB-based semantic search
- IncidentVectorStore: Past incidents for similarity matching
- RunbookVectorStore: Documentation and runbooks search
"""

from .store import MemoryStore
from .vector_store import (
    # Core classes
    VectorStore,
    VectorStoreConfig,
    Document,
    SearchResult,
    # Specialized stores
    IncidentVectorStore,
    RunbookVectorStore,
    # Singleton accessors
    get_vector_store,
    get_incident_store,
    get_runbook_store,
    # Utilities
    generate_document_id,
    bulk_import_documents,
)
from .runbook_loader import (
    # Core loader
    RunbookLoader,
    LoadResult,
    # Data classes
    RunbookMetadata,
    RunbookChunk,
    # Parsers
    MarkdownParser,
    PlainTextParser,
    TextChunker,
    # Convenience functions
    load_runbooks,
    create_runbook,
)

__all__ = [
    # Memory Store
    "MemoryStore",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "Document",
    "SearchResult",
    # Specialized stores
    "IncidentVectorStore",
    "RunbookVectorStore",
    # Singleton accessors
    "get_vector_store",
    "get_incident_store",
    "get_runbook_store",
    # Utilities
    "generate_document_id",
    "bulk_import_documents",
    # Runbook Loader
    "RunbookLoader",
    "LoadResult",
    "RunbookMetadata",
    "RunbookChunk",
    "MarkdownParser",
    "PlainTextParser",
    "TextChunker",
    "load_runbooks",
    "create_runbook",
]
