"""
Vector Store Module for Sentinel

This module provides ChromaDB-based vector storage for semantic search
across knowledge bases, incident history, and runbooks.

============================================================================
WHY VECTOR STORAGE INSTEAD OF KEYWORD SEARCH?
============================================================================

Traditional keyword search (TF-IDF, BM25, exact match) has fundamental
limitations for incident resolution:

1. VOCABULARY MISMATCH PROBLEM
   ────────────────────────────
   User searches: "database connection timing out"
   Runbook says:  "PostgreSQL socket timeout exceeded"
   
   Keywords don't match → Runbook not found ❌
   
   With vector search:
   - "database connection timing out" → embedding [0.23, 0.87, ...]
   - "PostgreSQL socket timeout exceeded" → embedding [0.24, 0.85, ...]
   - Cosine similarity: 0.94 → Match found! ✓
   
   Vector search understands MEANING, not just WORDS.

2. SYNONYMS AND DOMAIN KNOWLEDGE
   ─────────────────────────────
   Keyword search fails with:
   - "OOM" vs "out of memory" vs "memory exhausted"
   - "pod crash" vs "container restart" vs "CrashLoopBackOff"
   - "latency spike" vs "slow response" vs "high p99"
   
   Vector embeddings capture semantic equivalence:
   - All variations map to similar vector space regions
   - Model has learned from millions of documents
   - Domain-specific relationships preserved

3. CONTEXT-AWARE RETRIEVAL
   ────────────────────────
   Keyword: "restart" returns:
   - How to restart a service
   - How to restart a server
   - Restart policies in Kubernetes
   - Application restart loops
   - ... (too many irrelevant results)
   
   Vector search with context:
   Query: "restart pod to fix memory leak"
   Returns: Kubernetes pod restart runbook (not server restart)
   
   The embedding captures INTENT, not just keywords.

4. HANDLING NATURAL LANGUAGE QUERIES
   ──────────────────────────────────
   Real incident descriptions are messy:
   - "The thing is slow again, like last Tuesday"
   - "Database doing that weird thing with connections"
   - "Same error as the AWS migration issue"
   
   Keyword search: ❌ No useful matches
   Vector search: ✓ Finds semantically similar past incidents

5. CROSS-LINGUAL AND CROSS-DOMAIN TRANSFER
   ─────────────────────────────────────────
   Modern embedding models can match concepts across:
   - Different phrasing styles (formal vs casual)
   - Different experience levels (junior vs senior)
   - Different teams (DevOps vs SRE vs Platform)
   
   This enables knowledge sharing without perfect terminology alignment.

6. INCREMENTAL LEARNING
   ─────────────────────
   As new incidents are resolved, their embeddings can be:
   - Added to the vector store
   - Used to find similar past incidents
   - Clustered to identify patterns
   
   No need to manually curate keyword lists or taxonomies.

============================================================================
WHEN TO USE KEYWORD SEARCH (Hybrid Approach)
============================================================================

Keyword search is still valuable for:
- Exact matches (error codes, service names, ticket IDs)
- Known-item searches (user knows exactly what they want)
- Filtering by metadata (date ranges, severity levels)

Sentinel uses a HYBRID approach:
1. Vector search for semantic retrieval
2. Keyword filters for precision
3. Metadata filtering for scoping

============================================================================
CHROMADB: WHY THIS CHOICE?
============================================================================

ChromaDB was chosen for Sentinel because:

1. EMBEDDED MODE
   - Runs in-process (no external service required)
   - Persistence to local disk
   - Easy deployment (single Python package)

2. SIMPLE API
   - Add documents with embeddings
   - Query by similarity
   - Filter by metadata
   - Update and delete

3. GOOD ENOUGH FOR MOST USE CASES
   - Scales to millions of documents
   - Fast approximate nearest neighbor search
   - Production-ready for medium-scale deployments

4. SWAP-OUT CAPABILITY
   - Can migrate to Pinecone, Weaviate, Milvus later
   - Interface abstraction allows easy replacement
   - Start simple, scale when needed

For enterprise scale (billions of vectors), consider:
- Pinecone (managed, high scale)
- Weaviate (GraphQL interface, hybrid search)
- Milvus (open source, high performance)
- Qdrant (Rust-based, fast)

============================================================================
"""

import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# ChromaDB imports (optional dependency)
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Document:
    """
    Represents a document to be stored in the vector database.
    
    Attributes:
        id: Unique identifier for the document
        content: The text content to be embedded
        metadata: Additional information about the document
        embedding: Pre-computed embedding (optional, computed if not provided)
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """
    Represents a search result from the vector database.
    
    Attributes:
        id: Document ID
        content: Document text content
        metadata: Document metadata
        score: Similarity score (higher = more similar)
        distance: Distance in vector space (lower = more similar)
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: float


# ============================================================================
# VECTOR STORE CONFIGURATION
# ============================================================================

@dataclass
class VectorStoreConfig:
    """
    Configuration for the vector store.
    
    Attributes:
        persist_directory: Where to store the database files
        collection_name: Name of the default collection
        embedding_model: Which embedding model to use
        embedding_dimension: Dimension of embeddings (depends on model)
        distance_metric: How to measure similarity (cosine, l2, ip)
    """
    persist_directory: str = "./data/chromadb"
    collection_name: str = "sentinel_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence Transformers model
    embedding_dimension: int = 384  # Dimension for MiniLM
    distance_metric: str = "cosine"  # cosine, l2, or ip (inner product)
    
    # Performance settings
    batch_size: int = 100  # Documents to process at once
    max_results: int = 10  # Default number of results to return
    
    # OpenAI embedding option (if using OpenAI)
    use_openai_embeddings: bool = False
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"


# ============================================================================
# VECTOR STORE IMPLEMENTATION
# ============================================================================

class VectorStore:
    """
    ChromaDB-based vector store for semantic search.
    
    This class provides:
    - Document storage with automatic embedding
    - Semantic similarity search
    - Metadata filtering
    - Persistence to disk
    - Collection management
    
    Usage:
        # Initialize
        store = VectorStore()
        await store.initialize()
        
        # Add documents
        await store.add_documents([
            Document(id="doc1", content="How to restart a pod", metadata={"type": "runbook"}),
            Document(id="doc2", content="Troubleshooting memory leaks", metadata={"type": "guide"})
        ])
        
        # Search
        results = await store.search("pod keeps crashing", n_results=5)
        for result in results:
            print(f"{result.id}: {result.score:.2f} - {result.content[:50]}...")
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store configuration. Uses defaults if not provided.
        """
        self.config = config or VectorStoreConfig()
        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.embedding_function: Optional[Any] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the ChromaDB client and collection.
        
        Creates the persist directory if it doesn't exist,
        initializes the embedding function, and opens/creates
        the collection.
        
        Raises:
            ImportError: If ChromaDB is not installed
            RuntimeError: If initialization fails
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )
        
        try:
            # Create persist directory
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            # Settings control behavior (anonymized telemetry, etc.)
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=settings
            )
            
            # Set up embedding function
            self.embedding_function = self._create_embedding_function()
            
            # Get or create collection
            # Collections are like tables - they group related documents
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Sentinel knowledge base for incident resolution",
                    "created_at": datetime.now().isoformat(),
                    "distance_metric": self.config.distance_metric
                }
            )
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {e}")
    
    def _create_embedding_function(self) -> Any:
        """
        Create the embedding function based on configuration.
        
        Supports:
        - Sentence Transformers (default, runs locally)
        - OpenAI embeddings (requires API key)
        
        Returns:
            Embedding function for ChromaDB
        """
        if self.config.use_openai_embeddings:
            # OpenAI embeddings (better quality, requires API key)
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required for OpenAI embeddings. "
                    "Set OPENAI_API_KEY environment variable or pass in config."
                )
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=self.config.openai_model
            )
        else:
            # Sentence Transformers (runs locally, no API key needed)
            # all-MiniLM-L6-v2 is a good balance of quality and speed
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
    
    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before operations."""
        if not self._initialized:
            raise RuntimeError(
                "Vector store not initialized. Call initialize() first."
            )
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    async def add_documents(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Documents are automatically embedded using the configured
        embedding function. Existing documents with the same ID
        are updated (upsert behavior).
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process at once
        
        Returns:
            Dict with 'added' count and any errors
        
        Example:
            docs = [
                Document(
                    id="runbook-001",
                    content="To restart a Kubernetes pod, run kubectl delete pod...",
                    metadata={"type": "runbook", "service": "kubernetes"}
                )
            ]
            result = await store.add_documents(docs)
            print(f"Added {result['added']} documents")
        """
        self._ensure_initialized()
        
        batch_size = batch_size or self.config.batch_size
        added = 0
        errors = []
        
        # Process in batches to avoid memory issues with large datasets
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Prepare batch data
                ids = [doc.id for doc in batch]
                contents = [doc.content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                # Add embeddings if pre-computed
                embeddings = None
                if all(doc.embedding is not None for doc in batch):
                    embeddings = [doc.embedding for doc in batch]
                
                # Upsert to collection (add or update)
                self.collection.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                added += len(batch)
                
            except Exception as e:
                errors.append({
                    "batch_start": i,
                    "batch_end": i + len(batch),
                    "error": str(e)
                })
        
        return {
            "added": added,
            "total": len(documents),
            "errors": errors
        }
    
    async def add_document(self, document: Document) -> bool:
        """
        Add a single document to the vector store.
        
        Convenience wrapper around add_documents for single docs.
        
        Args:
            document: Document to add
        
        Returns:
            True if successful
        """
        result = await self.add_documents([document])
        return result["added"] == 1
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: ID of document to delete
        
        Returns:
            True if deleted (or didn't exist)
        """
        self._ensure_initialized()
        
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception:
            return False
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: ID of document to retrieve
        
        Returns:
            Document if found, None otherwise
        """
        self._ensure_initialized()
        
        try:
            result = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result["ids"]:
                return Document(
                    id=result["ids"][0],
                    content=result["documents"][0] if result["documents"] else "",
                    metadata=result["metadatas"][0] if result["metadatas"] else {},
                    embedding=result["embeddings"][0] if result.get("embeddings") else None
                )
            return None
            
        except Exception:
            return None
    
    # ========================================================================
    # SEARCH OPERATIONS
    # ========================================================================
    
    async def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        This is the primary search method. It:
        1. Embeds the query using the same embedding function
        2. Finds nearest neighbors in vector space
        3. Returns ranked results with similarity scores
        
        Args:
            query: Natural language query string
            n_results: Maximum number of results to return
            where: Metadata filter (e.g., {"type": "runbook"})
            where_document: Document content filter (e.g., {"$contains": "kubernetes"})
        
        Returns:
            List of SearchResult objects, ordered by similarity
        
        Example:
            # Basic search
            results = await store.search("pod keeps crashing")
            
            # With metadata filter
            results = await store.search(
                "memory issues",
                where={"service": "web-api"}
            )
            
            # With document content filter
            results = await store.search(
                "database timeout",
                where_document={"$contains": "PostgreSQL"}
            )
        """
        self._ensure_initialized()
        
        n_results = n_results or self.config.max_results
        
        try:
            # Query the collection
            # ChromaDB automatically embeds the query and finds nearest neighbors
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Distance is returned; convert to similarity score
                    # For cosine distance: similarity = 1 - distance
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = 1 - distance  # Convert distance to similarity
                    
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            # Log error but return empty results
            print(f"Search error: {e}")
            return []
    
    async def search_by_embedding(
        self,
        embedding: List[float],
        n_results: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search using a pre-computed embedding vector.
        
        Useful when you want to reuse an embedding or have
        computed it externally.
        
        Args:
            embedding: Query embedding vector
            n_results: Maximum number of results
            where: Metadata filter
        
        Returns:
            List of SearchResult objects
        """
        self._ensure_initialized()
        
        n_results = n_results or self.config.max_results
        
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = 1 - distance
                    
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Embedding search error: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        keyword_filter: Optional[str] = None,
        n_results: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity with keyword filtering.
        
        This approach gives you the best of both worlds:
        - Vector search for semantic understanding
        - Keyword filtering for precision
        
        Args:
            query: Semantic query for vector search
            keyword_filter: Keyword that must appear in results
            n_results: Maximum number of results
            where: Metadata filter
        
        Returns:
            List of SearchResult objects matching both criteria
        
        Example:
            # Find semantically similar docs that mention "PostgreSQL"
            results = await store.hybrid_search(
                query="database connection timeout",
                keyword_filter="PostgreSQL"
            )
        """
        # Build document filter for keyword
        where_document = None
        if keyword_filter:
            where_document = {"$contains": keyword_filter}
        
        return await self.search(
            query=query,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
    
    # ========================================================================
    # COLLECTION MANAGEMENT
    # ========================================================================
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict with count, metadata, and other stats
        """
        self._ensure_initialized()
        
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
            "persist_directory": self.config.persist_directory
        }
    
    async def clear_collection(self) -> bool:
        """
        Delete all documents from the collection.
        
        WARNING: This is destructive and cannot be undone!
        
        Returns:
            True if successful
        """
        self._ensure_initialized()
        
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Sentinel knowledge base for incident resolution",
                    "created_at": datetime.now().isoformat(),
                    "distance_metric": self.config.distance_metric
                }
            )
            return True
        except Exception:
            return False
    
    async def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new collection.
        
        Collections allow you to organize documents by category
        (e.g., runbooks, incidents, alerts).
        
        Args:
            name: Collection name
            metadata: Optional metadata for the collection
        
        Returns:
            True if created successfully
        """
        self._ensure_initialized()
        
        try:
            self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata=metadata or {}
            )
            return True
        except Exception:
            return False
    
    async def switch_collection(self, name: str) -> bool:
        """
        Switch to a different collection.
        
        Args:
            name: Collection name to switch to
        
        Returns:
            True if switched successfully
        """
        self._ensure_initialized()
        
        try:
            self.collection = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            self.config.collection_name = name
            return True
        except Exception:
            return False
    
    async def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        self._ensure_initialized()
        
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception:
            return []


# ============================================================================
# SPECIALIZED COLLECTIONS FOR SENTINEL
# ============================================================================

class IncidentVectorStore(VectorStore):
    """
    Specialized vector store for past incidents.
    
    Stores incident descriptions with resolution information,
    enabling "similar incident" searches for faster resolution.
    
    Metadata includes:
    - severity: critical, high, medium, low
    - service: affected service name
    - resolved: whether incident was resolved
    - resolution_time: time to resolution
    - root_cause: identified root cause
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        config = config or VectorStoreConfig()
        config.collection_name = "sentinel_incidents"
        super().__init__(config)
    
    async def add_incident(
        self,
        incident_id: str,
        description: str,
        severity: str,
        service: str,
        resolution: Optional[str] = None,
        root_cause: Optional[str] = None,
        resolution_time_minutes: Optional[int] = None
    ) -> bool:
        """
        Add an incident to the store.
        
        Args:
            incident_id: Unique incident identifier
            description: Full incident description
            severity: critical, high, medium, or low
            service: Affected service name
            resolution: How the incident was resolved (if known)
            root_cause: Identified root cause (if known)
            resolution_time_minutes: Time to resolution in minutes
        
        Returns:
            True if added successfully
        """
        # Combine description and resolution for better semantic matching
        content = description
        if resolution:
            content += f"\n\nResolution: {resolution}"
        if root_cause:
            content += f"\n\nRoot Cause: {root_cause}"
        
        document = Document(
            id=incident_id,
            content=content,
            metadata={
                "type": "incident",
                "severity": severity,
                "service": service,
                "resolved": resolution is not None,
                "root_cause": root_cause or "",
                "resolution_time_minutes": resolution_time_minutes or 0,
                "created_at": datetime.now().isoformat()
            }
        )
        
        return await self.add_document(document)
    
    async def find_similar_incidents(
        self,
        incident_description: str,
        service: Optional[str] = None,
        severity: Optional[str] = None,
        only_resolved: bool = True,
        n_results: int = 5
    ) -> List[SearchResult]:
        """
        Find incidents similar to the current one.
        
        This is the key method for incident resolution - it finds
        past incidents with similar symptoms that were already resolved.
        
        Args:
            incident_description: Description of current incident
            service: Filter by specific service
            severity: Filter by severity level
            only_resolved: Only return resolved incidents
            n_results: Number of results to return
        
        Returns:
            List of similar incidents, ranked by similarity
        """
        # Build metadata filter
        where = {"type": "incident"}
        
        if service:
            where["service"] = service
        if severity:
            where["severity"] = severity
        if only_resolved:
            where["resolved"] = True
        
        return await self.search(
            query=incident_description,
            n_results=n_results,
            where=where
        )


class RunbookVectorStore(VectorStore):
    """
    Specialized vector store for runbooks and documentation.
    
    Enables semantic search across operational documentation
    to find relevant procedures for incident resolution.
    
    Metadata includes:
    - category: runbook, guide, playbook, sop
    - service: related service
    - last_updated: when the doc was last updated
    - verified: whether the doc has been verified recently
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        config = config or VectorStoreConfig()
        config.collection_name = "sentinel_runbooks"
        super().__init__(config)
    
    async def add_runbook(
        self,
        runbook_id: str,
        title: str,
        content: str,
        service: str,
        category: str = "runbook",
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a runbook to the store.
        
        Args:
            runbook_id: Unique runbook identifier
            title: Runbook title
            content: Full runbook content (procedures, commands, etc.)
            service: Related service
            category: runbook, guide, playbook, or sop
            tags: Optional tags for filtering
        
        Returns:
            True if added successfully
        """
        # Include title in content for better matching
        full_content = f"{title}\n\n{content}"
        
        document = Document(
            id=runbook_id,
            content=full_content,
            metadata={
                "type": category,
                "title": title,
                "service": service,
                "tags": json.dumps(tags or []),
                "last_updated": datetime.now().isoformat(),
                "verified": True
            }
        )
        
        return await self.add_document(document)
    
    async def find_relevant_runbooks(
        self,
        problem_description: str,
        service: Optional[str] = None,
        category: Optional[str] = None,
        n_results: int = 5
    ) -> List[SearchResult]:
        """
        Find runbooks relevant to a problem.
        
        Args:
            problem_description: Description of the problem
            service: Filter by specific service
            category: Filter by category (runbook, guide, etc.)
            n_results: Number of results to return
        
        Returns:
            List of relevant runbooks, ranked by relevance
        """
        where = {}
        
        if service:
            where["service"] = service
        if category:
            where["type"] = category
        
        return await self.search(
            query=problem_description,
            n_results=n_results,
            where=where if where else None
        )


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

_vector_store: Optional[VectorStore] = None
_incident_store: Optional[IncidentVectorStore] = None
_runbook_store: Optional[RunbookVectorStore] = None


async def get_vector_store(config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """
    Get the global vector store instance.
    
    Creates and initializes the store on first call.
    
    Args:
        config: Optional configuration (only used on first call)
    
    Returns:
        Initialized VectorStore instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = VectorStore(config)
        await _vector_store.initialize()
    
    return _vector_store


async def get_incident_store(config: Optional[VectorStoreConfig] = None) -> IncidentVectorStore:
    """Get the global incident vector store instance."""
    global _incident_store
    
    if _incident_store is None:
        _incident_store = IncidentVectorStore(config)
        await _incident_store.initialize()
    
    return _incident_store


async def get_runbook_store(config: Optional[VectorStoreConfig] = None) -> RunbookVectorStore:
    """Get the global runbook vector store instance."""
    global _runbook_store
    
    if _runbook_store is None:
        _runbook_store = RunbookVectorStore(config)
        await _runbook_store.initialize()
    
    return _runbook_store


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_document_id(content: str, prefix: str = "doc") -> str:
    """
    Generate a deterministic document ID from content.
    
    Useful for deduplication - same content = same ID.
    
    Args:
        content: Document content
        prefix: Optional prefix for the ID
    
    Returns:
        Deterministic ID string
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}_{content_hash}"


async def bulk_import_documents(
    store: VectorStore,
    documents: List[Dict[str, Any]],
    id_field: str = "id",
    content_field: str = "content",
    metadata_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Bulk import documents from a list of dictionaries.
    
    Useful for importing from JSON files or databases.
    
    Args:
        store: VectorStore to import into
        documents: List of document dictionaries
        id_field: Field name for document ID
        content_field: Field name for content
        metadata_fields: Fields to include as metadata
    
    Returns:
        Import statistics
    """
    docs = []
    
    for doc_dict in documents:
        doc_id = doc_dict.get(id_field) or generate_document_id(doc_dict.get(content_field, ""))
        content = doc_dict.get(content_field, "")
        
        metadata = {}
        if metadata_fields:
            for field in metadata_fields:
                if field in doc_dict:
                    metadata[field] = doc_dict[field]
        
        docs.append(Document(
            id=doc_id,
            content=content,
            metadata=metadata
        ))
    
    return await store.add_documents(docs)
