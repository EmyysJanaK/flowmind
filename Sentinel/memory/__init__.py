"""
Memory module for Sentinel multi-agent system.

Handles conversation history, context management, state persistence,
and vector-based semantic search for knowledge retrieval.

Components:
- MemoryStore: Conversation history and shared state
- VectorStore: ChromaDB-based semantic search
- IncidentStore: Past incidents with resolutions for similarity matching
- RunbookVectorStore: Documentation and runbooks search
- RunbookLoader: Ingestion of markdown/text runbooks
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
    get_incident_store as get_incident_vector_store,  # Legacy alias
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
from .incident_store import (
    # Core store
    IncidentStore,
    # Data classes
    IncidentRecord,
    IncidentTimeline,
    ResolutionStep,
    IncidentSearchResult,
    # Enums
    Severity,
    IncidentStatus,
    # Singleton accessor
    get_incident_store,
    # Convenience functions
    store_incident,
    store_resolved_incident,
    find_similar_past_incidents,
)
from .rag_retriever import (
    # Core retriever
    RAGRetriever,
    # Data classes
    IncidentContext,
    RetrievalResult,
    RetrievedDocument,
    RankingExplanation,
    # Enums
    ContentType,
    RankingFactor,
    # Ranking engine
    RankingEngine,
    # Singleton accessor
    get_rag_retriever,
    # Convenience functions
    retrieve_for_incident,
    get_grounding_context,
)

__all__ = [
    # Memory Store
    "MemoryStore",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "Document",
    "SearchResult",
    # Specialized stores (legacy)
    "IncidentVectorStore",
    "RunbookVectorStore",
    # Singleton accessors
    "get_vector_store",
    "get_incident_store",
    "get_incident_vector_store",
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
    # Incident Store
    "IncidentStore",
    "IncidentRecord",
    "IncidentTimeline",
    "ResolutionStep",
    "IncidentSearchResult",
    "Severity",
    "IncidentStatus",
    "store_incident",
    "store_resolved_incident",
    "find_similar_past_incidents",
    # RAG Retriever
    "RAGRetriever",
    "IncidentContext",
    "RetrievalResult",
    "RetrievedDocument",
    "RankingExplanation",
    "ContentType",
    "RankingFactor",
    "RankingEngine",
    "get_rag_retriever",
    "retrieve_for_incident",
    "get_grounding_context",
]
