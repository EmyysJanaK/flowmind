"""
RAG Retriever for Sentinel

This module implements Retrieval Augmented Generation (RAG) for incident resolution.
It searches across runbooks and past incidents to find relevant context for LLM grounding.

============================================================================
RAG ARCHITECTURE IN SENTINEL
============================================================================

The RAG system follows this flow:

┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG RETRIEVAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Current Incident Context                                        │
│  ────────────────────────────────                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ • Description: "Pod keeps crashing with exit code 137"          │    │
│  │ • Service: "kubernetes"                                          │    │
│  │ • Severity: "high"                                               │    │
│  │ • Error Type: "OOMKilled" (optional)                             │    │
│  │ • Context: Additional symptoms/logs                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                     │                                   │
│                                     ▼                                   │
│  RETRIEVAL PHASE: Multi-Source Search                                   │
│  ──────────────────────────────────────                                 │
│                                                                         │
│  ┌─────────────────┐           ┌─────────────────────────────────┐      │
│  │   RUNBOOKS      │           │     PAST INCIDENTS              │      │
│  │   SEARCH        │           │       SEARCH                    │      │
│  └─────────────────┘           └─────────────────────────────────┘      │
│          │                                     │                        │
│          ▼                                     ▼                        │
│  Vector similarity              Vector similarity                       │
│  search in RunbookStore         search in IncidentStore                │
│                                                                         │
│  Returns:                       Returns:                                │
│  • Relevant procedures          • Similar past cases                   │
│  • Step-by-step guides          • Root causes                          │
│  • Configuration fixes          • Resolution steps                     │
│  • Troubleshooting steps        • What worked before                   │
│                                                                         │
│                                     │                                   │
│                                     ▼                                   │
│  RANKING PHASE: Unified Scoring                                         │
│  ─────────────────────────────────                                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ MULTI-FACTOR RANKING:                                           │    │
│  │                                                                 │    │
│  │ 1. SEMANTIC SIMILARITY (40%)                                    │    │
│  │    ├── Vector distance from ChromaDB                           │    │
│  │    └── Higher score = more similar content                     │    │
│  │                                                                 │    │
│  │ 2. EXACT MATCHES (25%)                                          │    │
│  │    ├── Service name exact match                                │    │
│  │    ├── Error type exact match                                  │    │
│  │    └── Keyword presence bonus                                  │    │
│  │                                                                 │    │
│  │ 3. METADATA ALIGNMENT (20%)                                     │    │
│  │    ├── Severity level match                                    │    │
│  │    ├── Recency (newer = better)                                │    │
│  │    └── Resolution success rate                                 │    │
│  │                                                                 │    │
│  │ 4. CONTENT QUALITY (15%)                                        │    │
│  │    ├── Has resolution steps                                    │    │
│  │    ├── Has root cause analysis                                 │    │
│  │    └── Content completeness                                    │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                     │                                   │
│                                     ▼                                   │
│  OUTPUT: Ranked Results                                                  │
│  ─────────────────────                                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ RetrievalResult {                                               │    │
│  │   runbooks: [                                                  │    │
│  │     {                                                          │    │
│  │       title: "Kubernetes Pod OOMKilled Resolution",           │    │
│  │       content: "Step 1: Check memory usage...",               │    │
│  │       score: 0.89,                                            │    │
│  │       reasoning: "Exact error type match + high similarity"   │    │
│  │     }                                                          │    │
│  │   ],                                                           │    │
│  │   incidents: [                                                 │    │
│  │     {                                                          │    │
│  │       title: "API pods crashing with OOM",                    │    │
│  │       root_cause: "Memory limit too low",                     │    │
│  │       resolution: "Increased memory from 512Mi to 1Gi",       │    │
│  │       score: 0.87,                                            │    │
│  │       reasoning: "Similar symptoms + same service"            │    │
│  │     }                                                          │    │
│  │   ],                                                           │    │
│  │   confidence: 0.88,                                           │    │
│  │   total_results: 8                                            │    │
│  │ }                                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

============================================================================
SCORING ALGORITHM DETAILS
============================================================================

SEMANTIC SIMILARITY (40% weight)
─────────────────────────────────
- Uses cosine similarity from ChromaDB embeddings
- Score = 1 - distance (higher = more similar)
- Captures meaning beyond exact keywords
- Example: "OOM" matches "out of memory" matches "memory exhausted"

EXACT MATCHES (25% weight)
────────────────────────────
- Service name match: +0.2 bonus
- Error type match: +0.15 bonus
- Critical keyword presence: +0.05 each
- Prevents semantic drift from exact requirements

METADATA ALIGNMENT (20% weight)
──────────────────────────────
- Severity level match: +0.1 (critical=critical, high=high, etc.)
- Recency bonus: newer content gets slight boost
- Resolution success: resolved incidents score higher
- Quality indicators: verified runbooks score higher

CONTENT QUALITY (15% weight)
───────────────────────────
- Has step-by-step resolution: +0.08
- Has root cause analysis: +0.05
- Has prevention advice: +0.02
- Content completeness: based on text length and structure

FINAL SCORE CALCULATION:
score = (semantic_score * 0.4) + (exact_matches * 0.25) + 
        (metadata_alignment * 0.2) + (content_quality * 0.15)

============================================================================
WHY THIS RANKING APPROACH?
============================================================================

1. SEMANTIC SIMILARITY IS PRIMARY
   - Captures the "meaning" of the incident
   - Finds relevant content even with different wording
   - But weighted to avoid false positives

2. EXACT MATCHES PREVENT DRIFT
   - Ensures service-specific results
   - Locks onto known error patterns
   - Prevents generic responses

3. METADATA PROVIDES CONTEXT
   - Recent incidents may have better solutions
   - Severity helps prioritize urgent vs. routine
   - Resolution success indicates quality

4. CONTENT QUALITY ENSURES UTILITY
   - Detailed runbooks are more helpful
   - Complete incident records provide better learning
   - Prevents returning stub/incomplete content

This multi-factor approach balances semantic understanding with practical
constraints, ensuring retrieved content is both relevant and actionable.

============================================================================
"""

import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .vector_store import SearchResult
from .runbook_loader import RunbookVectorStore
from .incident_store import IncidentStore, IncidentSearchResult


# ============================================================================
# CONFIGURATION AND ENUMS
# ============================================================================

class ContentType(str, Enum):
    """Type of retrieved content."""
    RUNBOOK = "runbook"
    INCIDENT = "incident"


class RankingFactor(str, Enum):
    """Factors used in ranking algorithm."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    EXACT_MATCHES = "exact_matches"
    METADATA_ALIGNMENT = "metadata_alignment"
    CONTENT_QUALITY = "content_quality"


# Default weights for ranking factors
DEFAULT_RANKING_WEIGHTS = {
    RankingFactor.SEMANTIC_SIMILARITY: 0.40,
    RankingFactor.EXACT_MATCHES: 0.25,
    RankingFactor.METADATA_ALIGNMENT: 0.20,
    RankingFactor.CONTENT_QUALITY: 0.15,
}

# Keywords that indicate quality content
QUALITY_INDICATORS = {
    "resolution_keywords": ["fix", "resolve", "solution", "steps", "procedure"],
    "root_cause_keywords": ["cause", "reason", "due to", "because", "triggered by"],
    "prevention_keywords": ["prevent", "avoid", "monitor", "alert", "check"],
    "diagnostic_keywords": ["check", "verify", "examine", "investigate", "analyze"],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class IncidentContext:
    """
    Context about the current incident for RAG retrieval.
    
    This is the input to the RAG system - all available information
    about the current incident that needs resolution.
    
    Attributes:
        description: Full description of the incident symptoms
        service: Affected service name
        severity: Incident severity (critical, high, medium, low)
        error_type: Specific error type if known (e.g., "OOMKilled")
        additional_context: Extra symptoms, logs, or observations
        constraints: Any constraints on solutions (e.g., "no downtime")
    """
    description: str
    service: Optional[str] = None
    severity: Optional[str] = None
    error_type: Optional[str] = None
    additional_context: str = ""
    constraints: str = ""
    
    def to_search_query(self) -> str:
        """Convert to a search query string."""
        parts = [self.description]
        
        if self.error_type:
            parts.append(f"error type: {self.error_type}")
        
        if self.additional_context:
            parts.append(self.additional_context)
        
        return " ".join(parts)
    
    def get_keywords(self) -> List[str]:
        """Extract important keywords from the context."""
        keywords = []
        
        if self.service:
            keywords.append(self.service)
        
        if self.error_type:
            keywords.append(self.error_type)
        
        if self.severity:
            keywords.append(self.severity)
        
        # Extract technical terms from description
        technical_terms = re.findall(r'\b(?:OOM|CPU|memory|disk|network|timeout|error|crash|restart|pod|container|database|connection)\w*\b', 
                                   self.description, re.IGNORECASE)
        keywords.extend(technical_terms)
        
        return list(set(keywords))


@dataclass
class RankingExplanation:
    """
    Explanation of why a result received its ranking score.
    
    Provides transparency into the ranking algorithm for debugging
    and helping users understand why certain results were chosen.
    
    Attributes:
        total_score: Final composite score
        factor_scores: Breakdown by ranking factor
        reasoning: Human-readable explanation
        match_details: Specific matches found
    """
    total_score: float
    factor_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    match_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    """
    A single retrieved document with ranking information.
    
    Contains the original content plus all metadata needed
    for LLM grounding and user understanding.
    
    Attributes:
        id: Unique identifier
        title: Human-readable title
        content: Full text content
        content_type: Whether this is a runbook or incident
        score: Composite ranking score
        semantic_score: Raw semantic similarity score
        explanation: Why this was ranked highly
        metadata: Original metadata from the document
        relevant_sections: Most relevant parts of the content
    """
    id: str
    title: str
    content: str
    content_type: str
    score: float
    semantic_score: float
    explanation: RankingExplanation
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevant_sections: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """
    Complete result from RAG retrieval.
    
    Contains all retrieved documents organized by type,
    plus aggregate statistics and confidence scores.
    
    Attributes:
        runbooks: Retrieved runbook documents
        incidents: Retrieved incident documents
        all_documents: All documents sorted by score
        confidence: Overall confidence in the retrieval (0-1)
        total_searched: Total number of documents searched
        total_retrieved: Number of documents retrieved
        search_time_ms: Time taken for retrieval
        context: Original incident context
    """
    runbooks: List[RetrievedDocument]
    incidents: List[RetrievedDocument]
    all_documents: List[RetrievedDocument]
    confidence: float
    total_searched: int
    total_retrieved: int
    search_time_ms: int
    context: IncidentContext
    
    def get_top_k(self, k: int) -> List[RetrievedDocument]:
        """Get the top K documents across all types."""
        return self.all_documents[:k]
    
    def get_grounding_context(self, max_length: int = 4000) -> str:
        """
        Get formatted context for LLM grounding.
        
        Combines the most relevant documents into a single context string
        suitable for injecting into an LLM prompt.
        
        Args:
            max_length: Maximum length of the context string
        
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for doc in self.all_documents:
            # Format each document
            doc_context = f"[{doc.content_type.upper()}] {doc.title}\n{doc.content}\n"
            
            # Check if we have room
            if current_length + len(doc_context) > max_length:
                break
            
            context_parts.append(doc_context)
            current_length += len(doc_context)
        
        header = f"RELEVANT KNOWLEDGE BASE ({len(context_parts)} documents):\n\n"
        return header + "\n---\n\n".join(context_parts)


# ============================================================================
# RANKING ENGINE
# ============================================================================

class RankingEngine:
    """
    Multi-factor ranking engine for retrieved documents.
    
    Implements the composite scoring algorithm that combines
    semantic similarity with metadata-based scoring.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ranking engine.
        
        Args:
            weights: Custom weights for ranking factors
        """
        self.weights = weights or DEFAULT_RANKING_WEIGHTS
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Ranking weights must sum to 1.0, got {weight_sum}")
    
    def score_document(
        self,
        semantic_score: float,
        metadata: Dict[str, Any],
        content: str,
        context: IncidentContext
    ) -> RankingExplanation:
        """
        Score a single document using multi-factor ranking.
        
        Args:
            semantic_score: Semantic similarity score from ChromaDB
            metadata: Document metadata
            content: Document content
            context: Current incident context
        
        Returns:
            RankingExplanation with score and reasoning
        """
        factor_scores = {}
        match_details = {}
        reasoning_parts = []
        
        # Factor 1: Semantic Similarity (40%)
        factor_scores[RankingFactor.SEMANTIC_SIMILARITY] = semantic_score
        if semantic_score > 0.8:
            reasoning_parts.append("high semantic similarity")
        elif semantic_score > 0.6:
            reasoning_parts.append("moderate semantic similarity")
        else:
            reasoning_parts.append("low semantic similarity")
        
        # Factor 2: Exact Matches (25%)
        exact_score = self._calculate_exact_matches(metadata, content, context, match_details)
        factor_scores[RankingFactor.EXACT_MATCHES] = exact_score
        
        # Factor 3: Metadata Alignment (20%)
        metadata_score = self._calculate_metadata_alignment(metadata, context, match_details)
        factor_scores[RankingFactor.METADATA_ALIGNMENT] = metadata_score
        
        # Factor 4: Content Quality (15%)
        quality_score = self._calculate_content_quality(content, metadata, match_details)
        factor_scores[RankingFactor.CONTENT_QUALITY] = quality_score
        
        # Calculate weighted total
        total_score = sum(
            score * self.weights[factor]
            for factor, score in factor_scores.items()
        )
        
        # Build reasoning
        reasoning = self._build_reasoning(reasoning_parts, match_details, factor_scores)
        
        return RankingExplanation(
            total_score=total_score,
            factor_scores=factor_scores,
            reasoning=reasoning,
            match_details=match_details
        )
    
    def _calculate_exact_matches(
        self,
        metadata: Dict[str, Any],
        content: str,
        context: IncidentContext,
        match_details: Dict[str, Any]
    ) -> float:
        """Calculate exact match bonus score."""
        score = 0.0
        matches = []
        
        # Service name exact match
        if context.service and metadata.get("service") == context.service:
            score += 0.20
            matches.append("service name")
        
        # Error type exact match
        if (context.error_type and 
            (metadata.get("error_type") == context.error_type or 
             context.error_type.lower() in content.lower())):
            score += 0.15
            matches.append("error type")
        
        # Severity match
        if context.severity and metadata.get("severity") == context.severity:
            score += 0.10
            matches.append("severity level")
        
        # Keyword presence
        keywords = context.get_keywords()
        content_lower = content.lower()
        matched_keywords = [kw for kw in keywords if kw.lower() in content_lower]
        keyword_score = min(len(matched_keywords) * 0.02, 0.10)  # Max 0.10
        score += keyword_score
        
        if matched_keywords:
            matches.append(f"{len(matched_keywords)} keywords")
        
        match_details["exact_matches"] = matches
        match_details["matched_keywords"] = matched_keywords
        
        return min(score, 1.0)
    
    def _calculate_metadata_alignment(
        self,
        metadata: Dict[str, Any],
        context: IncidentContext,
        match_details: Dict[str, Any]
    ) -> float:
        """Calculate metadata alignment score."""
        score = 0.0
        alignment_factors = []
        
        # Recency bonus (newer is better)
        if "last_updated" in metadata or "resolved_at" in metadata:
            date_str = metadata.get("resolved_at") or metadata.get("last_updated", "")
            if date_str:
                try:
                    doc_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    days_old = (datetime.now().replace(tzinfo=doc_date.tzinfo) - doc_date).days
                    
                    if days_old < 30:
                        score += 0.10
                        alignment_factors.append("recent (< 30 days)")
                    elif days_old < 90:
                        score += 0.05
                        alignment_factors.append("fairly recent (< 90 days)")
                except (ValueError, TypeError):
                    pass
        
        # Resolution success bonus
        if metadata.get("is_resolved") or metadata.get("status") == "resolved":
            score += 0.08
            alignment_factors.append("has resolution")
        
        # Quality indicators
        if metadata.get("has_root_cause"):
            score += 0.05
            alignment_factors.append("has root cause")
        
        if metadata.get("verified") or metadata.get("step_count", 0) > 0:
            score += 0.03
            alignment_factors.append("detailed steps")
        
        match_details["alignment_factors"] = alignment_factors
        
        return min(score, 1.0)
    
    def _calculate_content_quality(
        self,
        content: str,
        metadata: Dict[str, Any],
        match_details: Dict[str, Any]
    ) -> float:
        """Calculate content quality score."""
        score = 0.0
        quality_factors = []
        content_lower = content.lower()
        
        # Check for resolution steps
        if any(keyword in content_lower for keyword in QUALITY_INDICATORS["resolution_keywords"]):
            score += 0.08
            quality_factors.append("has resolution steps")
        
        # Check for root cause analysis
        if any(keyword in content_lower for keyword in QUALITY_INDICATORS["root_cause_keywords"]):
            score += 0.05
            quality_factors.append("explains root cause")
        
        # Check for prevention advice
        if any(keyword in content_lower for keyword in QUALITY_INDICATORS["prevention_keywords"]):
            score += 0.02
            quality_factors.append("includes prevention")
        
        # Content completeness (length and structure)
        if len(content) > 500:
            score += 0.03
            quality_factors.append("comprehensive content")
        
        # Structured content (has headers or numbered steps)
        if re.search(r'^\d+\.|\#{1,6}|^-', content, re.MULTILINE):
            score += 0.02
            quality_factors.append("well-structured")
        
        match_details["quality_factors"] = quality_factors
        
        return min(score, 1.0)
    
    def _build_reasoning(
        self,
        reasoning_parts: List[str],
        match_details: Dict[str, Any],
        factor_scores: Dict[str, float]
    ) -> str:
        """Build human-readable reasoning explanation."""
        main_reasoning = reasoning_parts[0] if reasoning_parts else "generic match"
        
        # Add specific match details
        details = []
        
        if match_details.get("exact_matches"):
            details.append(f"exact matches: {', '.join(match_details['exact_matches'])}")
        
        if match_details.get("alignment_factors"):
            details.append(f"alignment: {', '.join(match_details['alignment_factors'])}")
        
        if match_details.get("quality_factors"):
            details.append(f"quality: {', '.join(match_details['quality_factors'])}")
        
        # Build final reasoning
        if details:
            return f"{main_reasoning} ({'; '.join(details)})"
        else:
            return main_reasoning


# ============================================================================
# RAG RETRIEVER
# ============================================================================

class RAGRetriever:
    """
    Main RAG retriever that searches across runbooks and incidents.
    
    This is the primary interface for retrieving grounding documents
    for incident resolution. It orchestrates searches across multiple
    knowledge bases and returns ranked, explained results.
    
    Usage:
        # Initialize
        retriever = RAGRetriever()
        await retriever.initialize()
        
        # Define incident context
        context = IncidentContext(
            description="Pod keeps crashing with exit code 137",
            service="kubernetes",
            severity="high",
            error_type="OOMKilled"
        )
        
        # Retrieve relevant documents
        result = await retriever.retrieve(context, top_k=10)
        
        # Use for LLM grounding
        grounding_context = result.get_grounding_context()
        llm_prompt = f'''
        Given this context from our knowledge base:
        
        {grounding_context}
        
        Current incident: {context.description}
        
        What steps should we take to resolve this?
        '''
    """
    
    def __init__(
        self,
        runbook_store: Optional[RunbookVectorStore] = None,
        incident_store: Optional[IncidentStore] = None,
        ranking_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            runbook_store: Pre-initialized runbook store
            incident_store: Pre-initialized incident store
            ranking_weights: Custom ranking weights
        """
        self.runbook_store = runbook_store
        self.incident_store = incident_store
        self.ranking_engine = RankingEngine(ranking_weights)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector stores."""
        if not self.runbook_store:
            from .vector_store import RunbookVectorStore
            self.runbook_store = RunbookVectorStore()
            await self.runbook_store.initialize()
        
        if not self.incident_store:
            self.incident_store = IncidentStore()
            await self.incident_store.initialize()
        
        self._initialized = True
    
    def _ensure_initialized(self) -> None:
        """Ensure retriever is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RAGRetriever not initialized. Call initialize() first."
            )
    
    async def retrieve(
        self,
        context: IncidentContext,
        top_k: int = 10,
        runbook_k: int = 5,
        incident_k: int = 5
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for an incident context.
        
        This is the main RAG retrieval function that:
        1. Searches runbooks for relevant procedures
        2. Searches incidents for similar past cases
        3. Ranks all results using multi-factor scoring
        4. Returns structured results with explanations
        
        Args:
            context: Current incident context
            top_k: Total number of documents to return
            runbook_k: Maximum runbooks to retrieve
            incident_k: Maximum incidents to retrieve
        
        Returns:
            RetrievalResult with ranked documents and metadata
        """
        self._ensure_initialized()
        
        start_time = datetime.now()
        
        # Parallel search across both stores
        search_query = context.to_search_query()
        
        runbook_task = self._search_runbooks(search_query, context, runbook_k)
        incident_task = self._search_incidents(search_query, context, incident_k)
        
        runbook_results, incident_results = await asyncio.gather(
            runbook_task, incident_task
        )
        
        # Rank and combine all results
        all_documents = []
        all_documents.extend(runbook_results)
        all_documents.extend(incident_results)
        
        # Sort by composite score
        all_documents.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to top_k
        top_documents = all_documents[:top_k]
        
        # Separate by type
        runbooks = [doc for doc in top_documents if doc.content_type == ContentType.RUNBOOK]
        incidents = [doc for doc in top_documents if doc.content_type == ContentType.INCIDENT]
        
        # Calculate confidence
        confidence = self._calculate_confidence(top_documents, context)
        
        # Calculate timing
        end_time = datetime.now()
        search_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return RetrievalResult(
            runbooks=runbooks,
            incidents=incidents,
            all_documents=top_documents,
            confidence=confidence,
            total_searched=len(runbook_results) + len(incident_results),
            total_retrieved=len(top_documents),
            search_time_ms=search_time_ms,
            context=context
        )
    
    async def _search_runbooks(
        self,
        query: str,
        context: IncidentContext,
        k: int
    ) -> List[RetrievedDocument]:
        """Search runbooks and return ranked documents."""
        # Build search filters
        where = {}
        if context.service:
            where["service"] = context.service
        
        # Search
        results = await self.runbook_store.search(
            query=query,
            n_results=k,
            where=where if where else None
        )
        
        # Convert to RetrievedDocument with ranking
        documents = []
        for result in results:
            explanation = self.ranking_engine.score_document(
                semantic_score=result.score,
                metadata=result.metadata,
                content=result.content,
                context=context
            )
            
            documents.append(RetrievedDocument(
                id=result.id,
                title=result.metadata.get("title", "Untitled Runbook"),
                content=result.content,
                content_type=ContentType.RUNBOOK,
                score=explanation.total_score,
                semantic_score=result.score,
                explanation=explanation,
                metadata=result.metadata,
                relevant_sections=self._extract_relevant_sections(result.content)
            ))
        
        return documents
    
    async def _search_incidents(
        self,
        query: str,
        context: IncidentContext,
        k: int
    ) -> List[RetrievedDocument]:
        """Search incidents and return ranked documents."""
        # Use incident store's search
        results = await self.incident_store.find_similar_incidents(
            description=query,
            service=context.service,
            severity=context.severity,
            error_type=context.error_type,
            only_resolved=True,
            n_results=k
        )
        
        # Convert to RetrievedDocument with ranking
        documents = []
        for result in results:
            # Get metadata from incident record
            incident = result.incident
            metadata = incident.to_metadata()
            
            # Build content from incident
            content = incident.to_searchable_content()
            
            explanation = self.ranking_engine.score_document(
                semantic_score=result.score,
                metadata=metadata,
                content=content,
                context=context
            )
            
            documents.append(RetrievedDocument(
                id=incident.incident_id,
                title=incident.title,
                content=content,
                content_type=ContentType.INCIDENT,
                score=explanation.total_score,
                semantic_score=result.score,
                explanation=explanation,
                metadata=metadata,
                relevant_sections=self._extract_incident_sections(incident)
            ))
        
        return documents
    
    def _extract_relevant_sections(self, content: str) -> List[str]:
        """Extract the most relevant sections from runbook content."""
        sections = []
        
        # Split by headers and take first few sections
        parts = re.split(r'\n#{1,6}\s+', content)
        for part in parts[:3]:  # Take first 3 sections
            if part.strip() and len(part) > 100:
                sections.append(part.strip()[:500])  # Limit length
        
        return sections
    
    def _extract_incident_sections(self, incident) -> List[str]:
        """Extract relevant sections from incident record."""
        sections = []
        
        if incident.root_cause:
            sections.append(f"Root Cause: {incident.root_cause}")
        
        if incident.resolution_summary:
            sections.append(f"Resolution: {incident.resolution_summary}")
        
        if incident.resolution_steps:
            steps = "\n".join([f"{i+1}. {step.action}" for i, step in enumerate(incident.resolution_steps[:3])])
            sections.append(f"Steps Taken:\n{steps}")
        
        return sections
    
    def _calculate_confidence(
        self,
        documents: List[RetrievedDocument],
        context: IncidentContext
    ) -> float:
        """Calculate overall confidence in the retrieval results."""
        if not documents:
            return 0.0
        
        # Average of top 3 scores
        top_scores = [doc.score for doc in documents[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Boost if we have both runbooks and incidents
        content_types = set(doc.content_type for doc in documents)
        diversity_bonus = 0.1 if len(content_types) > 1 else 0.0
        
        # Boost if we have service-specific results
        service_matches = sum(1 for doc in documents[:5] 
                            if doc.metadata.get("service") == context.service)
        service_bonus = min(service_matches * 0.05, 0.15)
        
        confidence = min(avg_score + diversity_bonus + service_bonus, 1.0)
        return confidence


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance
_rag_retriever: Optional[RAGRetriever] = None


async def get_rag_retriever() -> RAGRetriever:
    """Get the global RAG retriever instance."""
    global _rag_retriever
    
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
        await _rag_retriever.initialize()
    
    return _rag_retriever


async def retrieve_for_incident(
    description: str,
    service: Optional[str] = None,
    severity: Optional[str] = None,
    error_type: Optional[str] = None,
    additional_context: str = "",
    top_k: int = 10
) -> RetrievalResult:
    """
    Convenience function for RAG retrieval.
    
    Args:
        description: Incident description
        service: Affected service
        severity: Incident severity
        error_type: Specific error type
        additional_context: Additional symptoms/context
        top_k: Number of documents to retrieve
    
    Returns:
        RetrievalResult with relevant documents
    
    Example:
        # Retrieve relevant knowledge
        result = await retrieve_for_incident(
            description="Pod keeps restarting with exit code 137",
            service="kubernetes",
            severity="high",
            error_type="OOMKilled",
            top_k=8
        )
        
        # Get grounding context for LLM
        context = result.get_grounding_context()
        
        # Print top results
        for doc in result.get_top_k(3):
            print(f"[{doc.score:.2f}] {doc.title}")
            print(f"  Type: {doc.content_type}")
            print(f"  Reasoning: {doc.explanation.reasoning}")
    """
    retriever = await get_rag_retriever()
    
    context = IncidentContext(
        description=description,
        service=service,
        severity=severity,
        error_type=error_type,
        additional_context=additional_context
    )
    
    return await retriever.retrieve(context, top_k=top_k)


async def get_grounding_context(
    description: str,
    service: Optional[str] = None,
    severity: Optional[str] = None,
    max_length: int = 4000
) -> str:
    """
    Get formatted grounding context for LLM prompts.
    
    This is the simplest interface - just provide incident details
    and get back formatted context ready for LLM injection.
    
    Args:
        description: Incident description
        service: Affected service
        severity: Incident severity
        max_length: Maximum context length
    
    Returns:
        Formatted context string for LLM grounding
    
    Example:
        # Get grounding context
        context = await get_grounding_context(
            "API response times are very slow",
            service="api-gateway",
            severity="high"
        )
        
        # Use in LLM prompt
        prompt = f'''
        You are an expert SRE helping resolve an incident.
        
        {context}
        
        Current incident: API response times are very slow
        
        Based on the knowledge base above, what are the most likely
        causes and what steps should we take to resolve this?
        '''
    """
    result = await retrieve_for_incident(
        description=description,
        service=service,
        severity=severity,
        top_k=5
    )
    
    return result.get_grounding_context(max_length=max_length)