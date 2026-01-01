"""
Runbook Loader for Sentinel

This module ingests markdown and text runbooks into ChromaDB for semantic search.
It supports metadata extraction, document chunking, and embedding storage.

============================================================================
HOW THIS SUPPORTS RETRIEVAL AUGMENTED GENERATION (RAG)
============================================================================

RAG is a technique that enhances LLM responses by retrieving relevant
context from a knowledge base before generating answers. Here's how
this runbook loader fits into Sentinel's RAG pipeline:

┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE IN SENTINEL                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. INDEXING PHASE (This Module)                                        │
│  ────────────────────────────────                                       │
│                                                                         │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│     │   Runbooks   │───▶│   Loader     │───▶│   ChromaDB           │   │
│     │  (MD/TXT)    │    │  (Parse +    │    │   Vector Store       │   │
│     │              │    │   Chunk)     │    │                      │   │
│     └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                         │
│     - Parse markdown/text files                                         │
│     - Extract metadata (service, severity, error type)                  │
│     - Chunk into retrievable segments                                   │
│     - Embed each chunk using sentence-transformers                      │
│     - Store embeddings + metadata in ChromaDB                           │
│                                                                         │
│  2. RETRIEVAL PHASE (At Incident Time)                                  │
│  ─────────────────────────────────────                                  │
│                                                                         │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│     │   Incident   │───▶│   Embed      │───▶│   Similarity         │   │
│     │  Description │    │   Query      │    │   Search             │   │
│     └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│                         ┌──────────────────────┐                       │
│                         │   Top-K Relevant     │                       │
│                         │   Runbook Chunks     │                       │
│                         └──────────────────────┘                       │
│                                                                         │
│     - Incident description is embedded                                  │
│     - ChromaDB finds semantically similar runbook chunks                │
│     - Returns top-K most relevant procedures                            │
│                                                                         │
│  3. GENERATION PHASE (LLM Response)                                     │
│  ──────────────────────────────────                                     │
│                                                                         │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│     │   Retrieved  │───▶│   Prompt     │───▶│   LLM                │   │
│     │   Context    │    │   Template   │    │   (GPT-4)            │   │
│     │              │    │              │    │                      │   │
│     └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                         │
│     Prompt: "Given these runbooks:                                      │
│              {retrieved_context}                                        │
│                                                                         │
│              The incident is: {incident_description}                    │
│                                                                         │
│              What steps should we take to resolve this?"                │
│                                                                         │
│     - Retrieved runbook chunks are injected into the prompt             │
│     - LLM generates response grounded in actual documentation           │
│     - Reduces hallucination by providing factual context                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

WHY RAG INSTEAD OF FINE-TUNING?
───────────────────────────────

1. REAL-TIME UPDATES
   - Add new runbooks instantly (no retraining)
   - Update procedures without model changes
   - Knowledge is always current

2. TRANSPARENCY
   - We know exactly which runbooks were used
   - Can audit and verify recommendations
   - Source attribution for compliance

3. DOMAIN SPECIFICITY
   - Works with YOUR organization's documentation
   - Handles internal terminology and procedures
   - Respects your security and access policies

4. COST EFFICIENCY
   - No expensive fine-tuning runs
   - Pay only for inference
   - Scales with document count, not model size

5. HALLUCINATION REDUCTION
   - LLM generates based on real documentation
   - Can't invent procedures that don't exist
   - Grounded in verified runbooks

============================================================================
RUNBOOK FORMAT SPECIFICATION
============================================================================

Runbooks can be in Markdown or plain text format. Markdown files support
YAML frontmatter for structured metadata.

MARKDOWN FORMAT (Recommended):
──────────────────────────────

```markdown
---
title: Restart Kubernetes Pod
service: kubernetes
error_type: CrashLoopBackOff
severity: high
tags:
  - kubernetes
  - pods
  - restart
author: SRE Team
last_updated: 2024-01-15
---

# Restart Kubernetes Pod

## Symptoms
- Pod shows CrashLoopBackOff status
- Container exits immediately after start

## Resolution Steps
1. Check pod logs: `kubectl logs <pod-name>`
2. Check events: `kubectl describe pod <pod-name>`
3. If memory issue, increase limits
4. Delete pod to force restart: `kubectl delete pod <pod-name>`
```

PLAIN TEXT FORMAT:
──────────────────

For plain text files, metadata is extracted from the filename and content:
- Filename: `kubernetes_crashloopbackoff_high.txt`
- Format: `{service}_{error_type}_{severity}.txt`

Or include metadata in the first lines:
```
SERVICE: kubernetes
ERROR_TYPE: CrashLoopBackOff
SEVERITY: high
---
# Restart Kubernetes Pod
...
```

============================================================================
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Try to import yaml for frontmatter parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class Severity(str, Enum):
    """Severity levels for runbooks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RunbookType(str, Enum):
    """Types of runbooks."""
    RUNBOOK = "runbook"
    PLAYBOOK = "playbook"
    GUIDE = "guide"
    SOP = "sop"  # Standard Operating Procedure
    TROUBLESHOOTING = "troubleshooting"


# Default chunk settings
# ──────────────────────
# Chunks should be:
# - Large enough to contain meaningful context
# - Small enough for efficient retrieval
# - Overlapping to avoid losing context at boundaries
DEFAULT_CHUNK_SIZE = 1000  # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # Overlap between chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size (avoid tiny fragments)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RunbookMetadata:
    """
    Metadata for a runbook document.
    
    This metadata is stored alongside the embedding in ChromaDB,
    enabling filtered searches (e.g., "find runbooks for kubernetes with high severity").
    
    Attributes:
        service: The service this runbook applies to (e.g., "kubernetes", "postgresql")
        error_type: Type of error/issue addressed (e.g., "CrashLoopBackOff", "OOM")
        severity: How critical the issue is (critical, high, medium, low, info)
        title: Human-readable title
        runbook_type: Type of document (runbook, playbook, guide, sop, troubleshooting)
        tags: Additional tags for categorization
        author: Who wrote/maintains this runbook
        last_updated: When the runbook was last updated
        source_file: Original file path
        version: Document version
    """
    service: str
    error_type: str = "general"
    severity: str = "medium"
    title: str = ""
    runbook_type: str = "runbook"
    tags: List[str] = field(default_factory=list)
    author: str = ""
    last_updated: str = ""
    source_file: str = ""
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        return {
            "service": self.service,
            "error_type": self.error_type,
            "severity": self.severity,
            "title": self.title,
            "runbook_type": self.runbook_type,
            "tags": ",".join(self.tags),  # ChromaDB doesn't support list values
            "author": self.author,
            "last_updated": self.last_updated or datetime.now().isoformat(),
            "source_file": self.source_file,
            "version": self.version,
        }


@dataclass
class RunbookChunk:
    """
    A chunk of a runbook for embedding.
    
    Large runbooks are split into chunks for better retrieval.
    Each chunk maintains reference to its parent document.
    
    Attributes:
        id: Unique identifier for this chunk
        content: The text content of this chunk
        metadata: Runbook metadata
        chunk_index: Position of this chunk in the document
        total_chunks: Total number of chunks in the document
        parent_id: ID of the parent runbook
    """
    id: str
    content: str
    metadata: RunbookMetadata
    chunk_index: int = 0
    total_chunks: int = 1
    parent_id: str = ""
    
    def to_document(self):
        """Convert to Document for vector store."""
        from .vector_store import Document
        
        # Merge chunk info with metadata
        full_metadata = self.metadata.to_dict()
        full_metadata["chunk_index"] = self.chunk_index
        full_metadata["total_chunks"] = self.total_chunks
        full_metadata["parent_id"] = self.parent_id
        
        return Document(
            id=self.id,
            content=self.content,
            metadata=full_metadata
        )


@dataclass
class LoadResult:
    """
    Result of loading runbooks.
    
    Attributes:
        loaded: Number of runbooks successfully loaded
        chunks: Total number of chunks created
        errors: List of errors encountered
        files: List of files processed
    """
    loaded: int = 0
    chunks: int = 0
    errors: List[Dict[str, str]] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


# ============================================================================
# MARKDOWN PARSER
# ============================================================================

class MarkdownParser:
    """
    Parses markdown files with optional YAML frontmatter.
    
    Supports extracting:
    - YAML frontmatter for metadata
    - Markdown content
    - Headers and sections
    """
    
    # Regex patterns
    FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    @classmethod
    def parse(cls, content: str, source_file: str = "") -> Tuple[RunbookMetadata, str]:
        """
        Parse markdown content and extract metadata.
        
        Args:
            content: Raw markdown content
            source_file: Path to source file (for fallback metadata)
        
        Returns:
            Tuple of (metadata, content without frontmatter)
        """
        metadata_dict = {}
        body = content
        
        # Try to extract YAML frontmatter
        frontmatter_match = cls.FRONTMATTER_PATTERN.match(content)
        if frontmatter_match and YAML_AVAILABLE:
            try:
                frontmatter = frontmatter_match.group(1)
                metadata_dict = yaml.safe_load(frontmatter) or {}
                body = content[frontmatter_match.end():]
            except yaml.YAMLError:
                # If YAML parsing fails, continue without frontmatter
                pass
        
        # Extract title from first H1 if not in frontmatter
        if "title" not in metadata_dict:
            title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
            if title_match:
                metadata_dict["title"] = title_match.group(1).strip()
        
        # Build metadata object
        metadata = RunbookMetadata(
            service=metadata_dict.get("service", cls._extract_service_from_path(source_file)),
            error_type=metadata_dict.get("error_type", metadata_dict.get("error", "general")),
            severity=metadata_dict.get("severity", "medium"),
            title=metadata_dict.get("title", cls._extract_title_from_path(source_file)),
            runbook_type=metadata_dict.get("type", metadata_dict.get("runbook_type", "runbook")),
            tags=cls._normalize_tags(metadata_dict.get("tags", [])),
            author=metadata_dict.get("author", ""),
            last_updated=metadata_dict.get("last_updated", ""),
            source_file=source_file,
            version=str(metadata_dict.get("version", "1.0")),
        )
        
        return metadata, body.strip()
    
    @classmethod
    def _extract_service_from_path(cls, path: str) -> str:
        """Extract service name from file path."""
        if not path:
            return "unknown"
        
        # Try to get from directory name or filename
        parts = Path(path).parts
        filename = Path(path).stem
        
        # Check if filename contains underscore-separated parts
        if "_" in filename:
            return filename.split("_")[0]
        
        # Use parent directory name
        if len(parts) > 1:
            return parts[-2]
        
        return filename
    
    @classmethod
    def _extract_title_from_path(cls, path: str) -> str:
        """Extract title from file path."""
        if not path:
            return "Untitled Runbook"
        
        # Use filename without extension
        return Path(path).stem.replace("_", " ").replace("-", " ").title()
    
    @classmethod
    def _normalize_tags(cls, tags: Any) -> List[str]:
        """Normalize tags to a list of strings."""
        if isinstance(tags, list):
            return [str(t).lower() for t in tags]
        elif isinstance(tags, str):
            return [t.strip().lower() for t in tags.split(",")]
        return []


class PlainTextParser:
    """
    Parses plain text runbooks with simple header-based metadata.
    
    Expected format:
    ```
    SERVICE: kubernetes
    ERROR_TYPE: CrashLoopBackOff
    SEVERITY: high
    ---
    Content here...
    ```
    """
    
    # Pattern for key: value metadata lines
    METADATA_PATTERN = re.compile(r'^([A-Z_]+):\s*(.+)$', re.MULTILINE)
    SEPARATOR_PATTERN = re.compile(r'^---+\s*$', re.MULTILINE)
    
    @classmethod
    def parse(cls, content: str, source_file: str = "") -> Tuple[RunbookMetadata, str]:
        """
        Parse plain text content and extract metadata.
        
        Args:
            content: Raw text content
            source_file: Path to source file
        
        Returns:
            Tuple of (metadata, content without header)
        """
        metadata_dict = {}
        body = content
        
        # Check for separator
        separator_match = cls.SEPARATOR_PATTERN.search(content)
        if separator_match:
            header = content[:separator_match.start()]
            body = content[separator_match.end():]
            
            # Extract metadata from header
            for match in cls.METADATA_PATTERN.finditer(header):
                key = match.group(1).lower()
                value = match.group(2).strip()
                metadata_dict[key] = value
        
        # Try to extract from filename if no metadata found
        if not metadata_dict and source_file:
            metadata_dict = cls._parse_filename(source_file)
        
        # Build metadata object
        metadata = RunbookMetadata(
            service=metadata_dict.get("service", "unknown"),
            error_type=metadata_dict.get("error_type", metadata_dict.get("error", "general")),
            severity=metadata_dict.get("severity", "medium"),
            title=metadata_dict.get("title", Path(source_file).stem if source_file else "Untitled"),
            runbook_type=metadata_dict.get("type", "runbook"),
            tags=metadata_dict.get("tags", "").split(",") if metadata_dict.get("tags") else [],
            author=metadata_dict.get("author", ""),
            source_file=source_file,
        )
        
        return metadata, body.strip()
    
    @classmethod
    def _parse_filename(cls, path: str) -> Dict[str, str]:
        """
        Extract metadata from filename.
        
        Expected format: {service}_{error_type}_{severity}.txt
        """
        filename = Path(path).stem
        parts = filename.split("_")
        
        result = {}
        if len(parts) >= 1:
            result["service"] = parts[0]
        if len(parts) >= 2:
            result["error_type"] = parts[1]
        if len(parts) >= 3:
            result["severity"] = parts[2]
        
        return result


# ============================================================================
# TEXT CHUNKER
# ============================================================================

class TextChunker:
    """
    Splits text into overlapping chunks for embedding.
    
    WHY CHUNKING IS IMPORTANT FOR RAG:
    ──────────────────────────────────
    
    1. EMBEDDING MODELS HAVE TOKEN LIMITS
       - Most models have 512 token max
       - Long documents must be split
    
    2. SMALLER CHUNKS = BETTER RETRIEVAL PRECISION
       - Retrieval finds the MOST relevant section
       - Large documents dilute relevance signal
    
    3. OVERLAP PRESERVES CONTEXT
       - Without overlap, you lose context at boundaries
       - Overlap ensures continuous information
    
    4. CHUNKS BECOME RETRIEVAL UNITS
       - Each chunk is independently searchable
       - LLM receives only the most relevant chunks
    """
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (avoid tiny fragments)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Uses a smart splitting strategy:
        1. Try to split at paragraph boundaries
        2. Fall back to sentence boundaries
        3. Fall back to word boundaries
        4. Last resort: character-level split
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk = text[start:].strip()
                if len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk)
                break
            
            # Find a good break point
            chunk_text = text[start:end]
            break_point = self._find_break_point(chunk_text)
            
            if break_point > 0:
                end = start + break_point
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end  # Prevent infinite loop
        
        return chunks
    
    def _find_break_point(self, text: str) -> int:
        """
        Find the best break point in text.
        
        Priority:
        1. Paragraph boundary (double newline)
        2. Section header (markdown header)
        3. Sentence boundary (. ! ?)
        4. Clause boundary (, ;)
        5. Word boundary (space)
        """
        # Try paragraph boundary (prefer last one in the chunk)
        para_match = re.search(r'\n\n(?=[^\n])', text[::-1])
        if para_match:
            return len(text) - para_match.start()
        
        # Try section header
        header_match = re.search(r'\n(?=#{1,6}\s)', text[::-1])
        if header_match:
            return len(text) - header_match.start()
        
        # Try sentence boundary
        sentence_match = re.search(r'[.!?]\s+(?=[A-Z])', text[::-1])
        if sentence_match:
            return len(text) - sentence_match.start() - 1
        
        # Try clause boundary
        clause_match = re.search(r'[,;]\s+', text[::-1])
        if clause_match:
            return len(text) - clause_match.start()
        
        # Try word boundary
        space_match = re.search(r'\s+', text[::-1])
        if space_match:
            return len(text) - space_match.start()
        
        # No good break point found
        return 0
    
    def chunk_by_sections(self, text: str, metadata: RunbookMetadata) -> List[RunbookChunk]:
        """
        Chunk by markdown sections, then by size.
        
        This preserves logical document structure while ensuring
        chunks don't exceed the size limit.
        
        Args:
            text: Markdown text to chunk
            metadata: Runbook metadata
        
        Returns:
            List of RunbookChunk objects
        """
        # Split by headers
        sections = self._split_by_headers(text)
        
        chunks = []
        parent_id = self._generate_id(text)
        
        for section in sections:
            if len(section) <= self.chunk_size:
                if section.strip():
                    chunks.append(section)
            else:
                # Further chunk large sections
                chunks.extend(self.chunk_text(section))
        
        # Convert to RunbookChunk objects
        result = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{parent_id}_chunk_{i}"
            result.append(RunbookChunk(
                id=chunk_id,
                content=chunk_content,
                metadata=metadata,
                chunk_index=i,
                total_chunks=len(chunks),
                parent_id=parent_id
            ))
        
        return result
    
    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by markdown headers."""
        # Pattern matches headers and captures content until next header
        pattern = r'(?=^#{1,6}\s+.+$)'
        sections = re.split(pattern, text, flags=re.MULTILINE)
        return [s for s in sections if s.strip()]
    
    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]


# ============================================================================
# RUNBOOK LOADER
# ============================================================================

class RunbookLoader:
    """
    Loads runbooks from files and directories into ChromaDB.
    
    This is the main entry point for ingesting runbooks into Sentinel's
    knowledge base for RAG-based incident resolution.
    
    Usage:
        # Initialize
        loader = RunbookLoader()
        await loader.initialize()
        
        # Load a single file
        result = await loader.load_file("runbooks/kubernetes/pod-restart.md")
        
        # Load a directory
        result = await loader.load_directory("runbooks/")
        
        # Check results
        print(f"Loaded {result.loaded} runbooks ({result.chunks} chunks)")
    """
    
    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Initialize the runbook loader.
        
        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.vector_store = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the vector store connection.
        
        Must be called before loading runbooks.
        """
        from .vector_store import RunbookVectorStore
        
        self.vector_store = RunbookVectorStore()
        await self.vector_store.initialize()
        self._initialized = True
    
    def _ensure_initialized(self) -> None:
        """Ensure loader is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RunbookLoader not initialized. Call initialize() first."
            )
    
    async def load_file(self, file_path: Union[str, Path]) -> LoadResult:
        """
        Load a single runbook file.
        
        Supports markdown (.md, .markdown) and plain text (.txt, .text) files.
        
        Args:
            file_path: Path to the runbook file
        
        Returns:
            LoadResult with loading statistics
        """
        self._ensure_initialized()
        
        result = LoadResult()
        path = Path(file_path)
        
        # Validate file
        if not path.exists():
            result.errors.append({"file": str(path), "error": "File not found"})
            return result
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            result.errors.append({
                "file": str(path),
                "error": f"Unsupported file type: {path.suffix}"
            })
            return result
        
        try:
            # Read file content
            content = path.read_text(encoding="utf-8")
            
            # Parse based on file type
            if path.suffix.lower() in {".md", ".markdown"}:
                metadata, body = MarkdownParser.parse(content, str(path))
            else:
                metadata, body = PlainTextParser.parse(content, str(path))
            
            # Chunk the content
            chunks = self.chunker.chunk_by_sections(body, metadata)
            
            if not chunks:
                result.errors.append({
                    "file": str(path),
                    "error": "No content to index"
                })
                return result
            
            # Convert to documents and store
            documents = [chunk.to_document() for chunk in chunks]
            store_result = await self.vector_store.add_documents(documents)
            
            result.loaded = 1
            result.chunks = store_result["added"]
            result.files.append(str(path))
            
        except Exception as e:
            result.errors.append({"file": str(path), "error": str(e)})
        
        return result
    
    async def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True
    ) -> LoadResult:
        """
        Load all runbooks from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
        
        Returns:
            LoadResult with combined statistics
        """
        self._ensure_initialized()
        
        result = LoadResult()
        directory = Path(directory_path)
        
        if not directory.exists():
            result.errors.append({
                "file": str(directory),
                "error": "Directory not found"
            })
            return result
        
        if not directory.is_dir():
            result.errors.append({
                "file": str(directory),
                "error": "Not a directory"
            })
            return result
        
        # Find all matching files
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        
        # Load each file
        for file_path in files:
            file_result = await self.load_file(file_path)
            result.loaded += file_result.loaded
            result.chunks += file_result.chunks
            result.files.extend(file_result.files)
            result.errors.extend(file_result.errors)
        
        return result
    
    async def load_from_string(
        self,
        content: str,
        metadata: RunbookMetadata,
        runbook_id: Optional[str] = None
    ) -> LoadResult:
        """
        Load a runbook from a string.
        
        Useful for loading runbooks from databases or APIs.
        
        Args:
            content: Runbook content
            metadata: Pre-defined metadata
            runbook_id: Optional ID (generated if not provided)
        
        Returns:
            LoadResult with loading statistics
        """
        self._ensure_initialized()
        
        result = LoadResult()
        
        try:
            # Set source file to indicate in-memory loading
            metadata.source_file = f"memory://{runbook_id or 'inline'}"
            
            # Chunk the content
            chunks = self.chunker.chunk_by_sections(content, metadata)
            
            if not chunks:
                result.errors.append({
                    "file": metadata.source_file,
                    "error": "No content to index"
                })
                return result
            
            # Convert to documents and store
            documents = [chunk.to_document() for chunk in chunks]
            store_result = await self.vector_store.add_documents(documents)
            
            result.loaded = 1
            result.chunks = store_result["added"]
            
        except Exception as e:
            result.errors.append({
                "file": metadata.source_file,
                "error": str(e)
            })
        
        return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded runbooks.
        
        Returns:
            Dictionary with collection statistics
        """
        self._ensure_initialized()
        return await self.vector_store.get_collection_stats()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def load_runbooks(
    path: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> LoadResult:
    """
    Convenience function to load runbooks from a file or directory.
    
    Args:
        path: Path to file or directory
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        LoadResult with loading statistics
    
    Example:
        # Load a single file
        result = await load_runbooks("runbook.md")
        
        # Load a directory
        result = await load_runbooks("runbooks/")
        
        print(f"Loaded {result.loaded} runbooks ({result.chunks} chunks)")
    """
    loader = RunbookLoader(chunk_size, chunk_overlap)
    await loader.initialize()
    
    path = Path(path)
    if path.is_dir():
        return await loader.load_directory(path)
    else:
        return await loader.load_file(path)


async def create_runbook(
    title: str,
    content: str,
    service: str,
    error_type: str = "general",
    severity: str = "medium",
    tags: Optional[List[str]] = None
) -> LoadResult:
    """
    Convenience function to create and index a runbook from code.
    
    Args:
        title: Runbook title
        content: Runbook content
        service: Service name
        error_type: Type of error addressed
        severity: Severity level
        tags: Optional tags
    
    Returns:
        LoadResult with loading statistics
    
    Example:
        result = await create_runbook(
            title="Restart Kubernetes Pod",
            content="1. Check pod status\\n2. Delete pod\\n3. Verify restart",
            service="kubernetes",
            error_type="CrashLoopBackOff",
            severity="high",
            tags=["kubernetes", "pods"]
        )
    """
    metadata = RunbookMetadata(
        title=title,
        service=service,
        error_type=error_type,
        severity=severity,
        tags=tags or []
    )
    
    loader = RunbookLoader()
    await loader.initialize()
    
    return await loader.load_from_string(content, metadata)
