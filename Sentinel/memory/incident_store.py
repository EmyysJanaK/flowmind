"""
Incident Store for Sentinel

This module provides comprehensive storage and retrieval of past incidents
and their resolutions using ChromaDB vector storage.

============================================================================
WHY STORE PAST INCIDENTS?
============================================================================

Storing past incidents with their resolutions enables:

1. PATTERN RECOGNITION
   ────────────────────
   "We've seen this before" - Find similar past incidents to:
   - Accelerate diagnosis
   - Avoid repeating mistakes
   - Identify recurring issues
   
2. KNOWLEDGE TRANSFER
   ───────────────────
   When an on-call engineer encounters a new issue:
   - Search semantically similar past incidents
   - See how experienced engineers resolved them
   - Learn from documented resolution steps
   
3. INCIDENT ANALYTICS
   ───────────────────
   Track patterns over time:
   - Which services have the most incidents?
   - What's the average resolution time?
   - Are certain error types becoming more frequent?
   
4. POST-MORTEM REFERENCE
   ──────────────────────
   During post-mortems:
   - Find all related past incidents
   - Identify systemic issues
   - Track improvement over time

============================================================================
INCIDENT DATA MODEL
============================================================================

Each incident record contains:

┌─────────────────────────────────────────────────────────────────────────┐
│ INCIDENT RECORD                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ IDENTIFICATION                                                          │
│ ├── incident_id: Unique identifier (e.g., "INC-2024-001234")          │
│ ├── title: Short description                                          │
│ └── description: Full incident details                                 │
│                                                                         │
│ CLASSIFICATION                                                          │
│ ├── severity: critical | high | medium | low                          │
│ ├── service: Affected service name                                    │
│ ├── error_type: Category of error                                     │
│ └── tags: Additional categorization                                   │
│                                                                         │
│ TIMELINE                                                                │
│ ├── detected_at: When incident was detected                           │
│ ├── acknowledged_at: When someone started working on it               │
│ ├── resolved_at: When incident was resolved                           │
│ └── duration_minutes: Total time to resolution                        │
│                                                                         │
│ RESOLUTION                                                              │
│ ├── root_cause: What caused the incident                              │
│ ├── resolution_steps: List of steps taken to resolve                  │
│ ├── resolution_summary: Brief summary of resolution                   │
│ └── prevented_by: What could have prevented this                      │
│                                                                         │
│ CONTEXT                                                                 │
│ ├── responders: Who worked on the incident                            │
│ ├── related_incidents: Links to related incidents                     │
│ ├── runbooks_used: Runbooks that were helpful                         │
│ └── external_links: Tickets, docs, etc.                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

============================================================================
"""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    Document,
    SearchResult,
)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class Severity(str, Enum):
    """Incident severity levels."""
    CRITICAL = "critical"  # Service down, customer impact
    HIGH = "high"          # Major degradation, some customer impact
    MEDIUM = "medium"      # Minor issue, limited impact
    LOW = "low"            # Cosmetic or minor issue


class IncidentStatus(str, Enum):
    """Incident lifecycle status."""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResolutionStep:
    """
    A single step in the incident resolution process.
    
    Captures what was done, when, and by whom.
    """
    step_number: int
    action: str
    timestamp: str = ""
    performed_by: str = ""
    outcome: str = ""
    duration_minutes: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class IncidentTimeline:
    """
    Timeline of an incident from detection to resolution.
    
    All timestamps are ISO 8601 format strings.
    """
    detected_at: str = ""
    acknowledged_at: str = ""
    investigation_started_at: str = ""
    root_cause_identified_at: str = ""
    mitigated_at: str = ""
    resolved_at: str = ""
    closed_at: str = ""
    
    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()
    
    @property
    def time_to_acknowledge_minutes(self) -> Optional[int]:
        """Time from detection to acknowledgment."""
        return self._diff_minutes(self.detected_at, self.acknowledged_at)
    
    @property
    def time_to_resolve_minutes(self) -> Optional[int]:
        """Time from detection to resolution."""
        return self._diff_minutes(self.detected_at, self.resolved_at)
    
    @property
    def time_to_mitigate_minutes(self) -> Optional[int]:
        """Time from detection to mitigation."""
        return self._diff_minutes(self.detected_at, self.mitigated_at)
    
    def _diff_minutes(self, start: str, end: str) -> Optional[int]:
        """Calculate difference between two ISO timestamps in minutes."""
        if not start or not end:
            return None
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return int((end_dt - start_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            return None


@dataclass
class IncidentRecord:
    """
    Complete record of an incident.
    
    This is the primary data structure for storing incidents.
    Contains all information needed to learn from past incidents.
    
    Attributes:
        incident_id: Unique identifier (e.g., "INC-2024-001234")
        title: Short description of the incident
        description: Full incident details
        service: Affected service name
        severity: Incident severity level
        status: Current status
        error_type: Category of error (e.g., "OOMKilled", "connection_timeout")
        
        timeline: Timestamps for incident lifecycle
        resolution_steps: List of steps taken to resolve
        root_cause: What caused the incident
        resolution_summary: Brief summary of how it was resolved
        prevented_by: What could have prevented this incident
        
        responders: List of people who worked on the incident
        tags: Additional categorization tags
        related_incidents: IDs of related incidents
        runbooks_used: Runbooks that were helpful
        external_links: Links to tickets, docs, etc.
        
        metadata: Additional custom metadata
    """
    # Identification
    incident_id: str
    title: str
    description: str
    
    # Classification
    service: str
    severity: str = "medium"
    status: str = "detected"
    error_type: str = "general"
    
    # Timeline
    timeline: IncidentTimeline = field(default_factory=IncidentTimeline)
    
    # Resolution
    resolution_steps: List[ResolutionStep] = field(default_factory=list)
    root_cause: str = ""
    resolution_summary: str = ""
    prevented_by: str = ""
    
    # Context
    responders: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)
    runbooks_used: List[str] = field(default_factory=list)
    external_links: Dict[str, str] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_searchable_content(self) -> str:
        """
        Convert incident to searchable text content.
        
        This content is embedded for semantic search.
        Includes all relevant text fields for matching.
        """
        parts = [
            f"Title: {self.title}",
            f"Service: {self.service}",
            f"Severity: {self.severity}",
            f"Error Type: {self.error_type}",
            "",
            "Description:",
            self.description,
        ]
        
        if self.root_cause:
            parts.extend(["", "Root Cause:", self.root_cause])
        
        if self.resolution_summary:
            parts.extend(["", "Resolution:", self.resolution_summary])
        
        if self.resolution_steps:
            parts.extend(["", "Resolution Steps:"])
            for step in self.resolution_steps:
                parts.append(f"{step.step_number}. {step.action}")
                if step.outcome:
                    parts.append(f"   Outcome: {step.outcome}")
        
        if self.prevented_by:
            parts.extend(["", "Prevention:", self.prevented_by])
        
        if self.tags:
            parts.extend(["", f"Tags: {', '.join(self.tags)}"])
        
        return "\n".join(parts)
    
    def to_metadata(self) -> Dict[str, Any]:
        """
        Convert incident to ChromaDB metadata.
        
        ChromaDB metadata must be flat (no nested objects).
        Lists are converted to comma-separated strings.
        """
        # Calculate duration if resolved
        duration = self.timeline.time_to_resolve_minutes
        
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "service": self.service,
            "severity": self.severity,
            "status": self.status,
            "error_type": self.error_type,
            "is_resolved": self.status in ["resolved", "closed"],
            "has_root_cause": bool(self.root_cause),
            "has_resolution": bool(self.resolution_summary or self.resolution_steps),
            "detected_at": self.timeline.detected_at,
            "resolved_at": self.timeline.resolved_at or "",
            "duration_minutes": duration or 0,
            "responders": ",".join(self.responders),
            "tags": ",".join(self.tags),
            "runbooks_used": ",".join(self.runbooks_used),
            "step_count": len(self.resolution_steps),
        }
    
    def to_document(self) -> Document:
        """Convert to Document for vector storage."""
        return Document(
            id=self.incident_id,
            content=self.to_searchable_content(),
            metadata=self.to_metadata()
        )


@dataclass
class IncidentSearchResult:
    """
    Search result with full incident details.
    
    Extends SearchResult with parsed incident data.
    """
    incident: IncidentRecord
    score: float
    distance: float
    
    @classmethod
    def from_search_result(cls, result: SearchResult) -> "IncidentSearchResult":
        """Create from a raw SearchResult."""
        # Parse metadata back to incident
        meta = result.metadata
        
        incident = IncidentRecord(
            incident_id=meta.get("incident_id", result.id),
            title=meta.get("title", ""),
            description=result.content,  # Full content as description
            service=meta.get("service", "unknown"),
            severity=meta.get("severity", "medium"),
            status=meta.get("status", "unknown"),
            error_type=meta.get("error_type", "general"),
            timeline=IncidentTimeline(
                detected_at=meta.get("detected_at", ""),
                resolved_at=meta.get("resolved_at", "")
            ),
            responders=meta.get("responders", "").split(",") if meta.get("responders") else [],
            tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            runbooks_used=meta.get("runbooks_used", "").split(",") if meta.get("runbooks_used") else [],
        )
        
        return cls(
            incident=incident,
            score=result.score,
            distance=result.distance
        )


# ============================================================================
# INCIDENT STORE
# ============================================================================

class IncidentStore:
    """
    Comprehensive store for past incidents and resolutions.
    
    Provides:
    - Storage of complete incident records with timestamps
    - Semantic search for similar incidents
    - Filtering by service, severity, status
    - Resolution step tracking
    - Analytics and statistics
    
    Usage:
        # Initialize
        store = IncidentStore()
        await store.initialize()
        
        # Create and store an incident
        incident = IncidentRecord(
            incident_id="INC-2024-001234",
            title="API latency spike",
            description="Users reporting slow response times...",
            service="api-gateway",
            severity="high",
            error_type="latency"
        )
        await store.store_incident(incident)
        
        # Add resolution after fixing
        await store.add_resolution(
            incident_id="INC-2024-001234",
            root_cause="Database connection pool exhausted",
            resolution_summary="Increased pool size and added circuit breaker",
            resolution_steps=[
                ResolutionStep(1, "Identified slow queries in logs"),
                ResolutionStep(2, "Found connection pool at max capacity"),
                ResolutionStep(3, "Increased pool size from 10 to 50"),
                ResolutionStep(4, "Added circuit breaker to prevent cascade"),
            ]
        )
        
        # Find similar incidents
        similar = await store.find_similar_incidents(
            "database connections timing out",
            service="api-gateway"
        )
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the incident store.
        
        Args:
            config: Vector store configuration. Uses defaults if not provided.
        """
        self.config = config or VectorStoreConfig()
        self.config.collection_name = "sentinel_incidents"
        self.vector_store = VectorStore(self.config)
        self._initialized = False
        
        # In-memory cache for full incident records
        # ChromaDB metadata has size limits, so we cache full records
        self._incident_cache: Dict[str, IncidentRecord] = {}
    
    async def initialize(self) -> None:
        """Initialize the vector store connection."""
        await self.vector_store.initialize()
        self._initialized = True
    
    def _ensure_initialized(self) -> None:
        """Ensure store is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "IncidentStore not initialized. Call initialize() first."
            )
    
    # ========================================================================
    # INCIDENT STORAGE
    # ========================================================================
    
    async def store_incident(self, incident: IncidentRecord) -> bool:
        """
        Store an incident record.
        
        Creates embeddings for semantic search and stores metadata
        for filtering. Full record is cached for retrieval.
        
        Args:
            incident: Complete incident record
        
        Returns:
            True if stored successfully
        """
        self._ensure_initialized()
        
        # Cache full record
        self._incident_cache[incident.incident_id] = incident
        
        # Store in vector DB
        document = incident.to_document()
        return await self.vector_store.add_document(document)
    
    async def store_incidents(self, incidents: List[IncidentRecord]) -> Dict[str, Any]:
        """
        Store multiple incidents.
        
        Args:
            incidents: List of incident records
        
        Returns:
            Dict with 'stored' count and any errors
        """
        self._ensure_initialized()
        
        # Cache all records
        for incident in incidents:
            self._incident_cache[incident.incident_id] = incident
        
        # Store in vector DB
        documents = [inc.to_document() for inc in incidents]
        return await self.vector_store.add_documents(documents)
    
    async def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        """
        Retrieve an incident by ID.
        
        Args:
            incident_id: Incident identifier
        
        Returns:
            IncidentRecord if found, None otherwise
        """
        self._ensure_initialized()
        
        # Check cache first
        if incident_id in self._incident_cache:
            return self._incident_cache[incident_id]
        
        # Fetch from vector store
        document = await self.vector_store.get_document(incident_id)
        if not document:
            return None
        
        # Reconstruct from metadata (limited data)
        meta = document.metadata
        return IncidentRecord(
            incident_id=incident_id,
            title=meta.get("title", ""),
            description=document.content,
            service=meta.get("service", "unknown"),
            severity=meta.get("severity", "medium"),
            status=meta.get("status", "unknown"),
            error_type=meta.get("error_type", "general"),
            timeline=IncidentTimeline(
                detected_at=meta.get("detected_at", ""),
                resolved_at=meta.get("resolved_at", "")
            ),
        )
    
    async def delete_incident(self, incident_id: str) -> bool:
        """
        Delete an incident.
        
        Args:
            incident_id: Incident identifier
        
        Returns:
            True if deleted successfully
        """
        self._ensure_initialized()
        
        # Remove from cache
        self._incident_cache.pop(incident_id, None)
        
        # Remove from vector store
        return await self.vector_store.delete_document(incident_id)
    
    # ========================================================================
    # RESOLUTION MANAGEMENT
    # ========================================================================
    
    async def add_resolution(
        self,
        incident_id: str,
        root_cause: str,
        resolution_summary: str,
        resolution_steps: Optional[List[ResolutionStep]] = None,
        prevented_by: str = "",
        resolved_at: Optional[str] = None
    ) -> bool:
        """
        Add resolution information to an existing incident.
        
        Updates the incident with root cause, resolution details,
        and marks it as resolved.
        
        Args:
            incident_id: Incident to update
            root_cause: What caused the incident
            resolution_summary: Brief description of the fix
            resolution_steps: Detailed steps taken
            prevented_by: What could prevent recurrence
            resolved_at: Resolution timestamp (defaults to now)
        
        Returns:
            True if updated successfully
        """
        self._ensure_initialized()
        
        # Get existing incident
        incident = await self.get_incident(incident_id)
        if not incident:
            return False
        
        # Update fields
        incident.root_cause = root_cause
        incident.resolution_summary = resolution_summary
        incident.resolution_steps = resolution_steps or []
        incident.prevented_by = prevented_by
        incident.status = "resolved"
        incident.timeline.resolved_at = resolved_at or datetime.now().isoformat()
        
        # Re-store with updated information
        return await self.store_incident(incident)
    
    async def add_resolution_step(
        self,
        incident_id: str,
        action: str,
        performed_by: str = "",
        outcome: str = ""
    ) -> bool:
        """
        Add a single resolution step to an incident.
        
        Useful for real-time tracking during incident response.
        
        Args:
            incident_id: Incident to update
            action: What was done
            performed_by: Who did it
            outcome: Result of the action
        
        Returns:
            True if added successfully
        """
        self._ensure_initialized()
        
        incident = await self.get_incident(incident_id)
        if not incident:
            return False
        
        step_number = len(incident.resolution_steps) + 1
        step = ResolutionStep(
            step_number=step_number,
            action=action,
            performed_by=performed_by,
            outcome=outcome
        )
        
        incident.resolution_steps.append(step)
        return await self.store_incident(incident)
    
    # ========================================================================
    # SEARCH AND RETRIEVAL
    # ========================================================================
    
    async def find_similar_incidents(
        self,
        description: str,
        service: Optional[str] = None,
        severity: Optional[str] = None,
        error_type: Optional[str] = None,
        only_resolved: bool = True,
        n_results: int = 5
    ) -> List[IncidentSearchResult]:
        """
        Find incidents similar to a description.
        
        This is the primary search method for incident resolution.
        Finds past incidents with similar symptoms that have been resolved.
        
        Args:
            description: Description of current incident/symptoms
            service: Filter by specific service
            severity: Filter by severity level
            error_type: Filter by error type
            only_resolved: Only return resolved incidents (default True)
            n_results: Number of results to return
        
        Returns:
            List of similar incidents with similarity scores
        
        Example:
            results = await store.find_similar_incidents(
                "Pod keeps restarting with exit code 137",
                service="kubernetes",
                only_resolved=True
            )
            for r in results:
                print(f"[{r.score:.2f}] {r.incident.title}")
                print(f"  Resolution: {r.incident.resolution_summary}")
        """
        self._ensure_initialized()
        
        # Build metadata filter
        where = {}
        if service:
            where["service"] = service
        if severity:
            where["severity"] = severity
        if error_type:
            where["error_type"] = error_type
        if only_resolved:
            where["is_resolved"] = True
        
        # Search
        results = await self.vector_store.search(
            query=description,
            n_results=n_results,
            where=where if where else None
        )
        
        # Convert to IncidentSearchResult
        incident_results = []
        for result in results:
            # Try to get full record from cache
            incident_id = result.metadata.get("incident_id", result.id)
            if incident_id in self._incident_cache:
                incident = self._incident_cache[incident_id]
            else:
                # Parse from search result
                incident_result = IncidentSearchResult.from_search_result(result)
                incident = incident_result.incident
            
            incident_results.append(IncidentSearchResult(
                incident=incident,
                score=result.score,
                distance=result.distance
            ))
        
        return incident_results
    
    async def search_by_service(
        self,
        service: str,
        n_results: int = 10
    ) -> List[IncidentSearchResult]:
        """
        Get all incidents for a service.
        
        Args:
            service: Service name
            n_results: Maximum results
        
        Returns:
            List of incidents for the service
        """
        # Use a generic query with service filter
        return await self.find_similar_incidents(
            description=f"incidents for {service}",
            service=service,
            only_resolved=False,
            n_results=n_results
        )
    
    async def search_by_error_type(
        self,
        error_type: str,
        n_results: int = 10
    ) -> List[IncidentSearchResult]:
        """
        Get all incidents of a specific error type.
        
        Args:
            error_type: Error type (e.g., "OOMKilled", "connection_timeout")
            n_results: Maximum results
        
        Returns:
            List of incidents with that error type
        """
        return await self.find_similar_incidents(
            description=f"{error_type} error",
            error_type=error_type,
            only_resolved=False,
            n_results=n_results
        )
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get incident store statistics.
        
        Returns:
            Dict with counts and analytics
        """
        self._ensure_initialized()
        
        collection_stats = await self.vector_store.get_collection_stats()
        
        # Aggregate from cache
        total = len(self._incident_cache)
        resolved = sum(1 for i in self._incident_cache.values() if i.status in ["resolved", "closed"])
        
        services = {}
        severities = {}
        
        for incident in self._incident_cache.values():
            services[incident.service] = services.get(incident.service, 0) + 1
            severities[incident.severity] = severities.get(incident.severity, 0) + 1
        
        return {
            "total_incidents": collection_stats["count"],
            "cached_incidents": total,
            "resolved_incidents": resolved,
            "incidents_by_service": services,
            "incidents_by_severity": severities,
            "collection": collection_stats
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance
_incident_store: Optional[IncidentStore] = None


async def get_incident_store(config: Optional[VectorStoreConfig] = None) -> IncidentStore:
    """
    Get the global incident store instance.
    
    Creates and initializes the store on first call.
    
    Args:
        config: Optional configuration (only used on first call)
    
    Returns:
        Initialized IncidentStore instance
    """
    global _incident_store
    
    if _incident_store is None:
        _incident_store = IncidentStore(config)
        await _incident_store.initialize()
    
    return _incident_store


async def store_incident(
    incident_id: str,
    title: str,
    description: str,
    service: str,
    severity: str = "medium",
    error_type: str = "general",
    tags: Optional[List[str]] = None
) -> bool:
    """
    Convenience function to store a new incident.
    
    Args:
        incident_id: Unique identifier
        title: Short description
        description: Full incident details
        service: Affected service
        severity: Severity level
        error_type: Type of error
        tags: Additional tags
    
    Returns:
        True if stored successfully
    
    Example:
        await store_incident(
            incident_id="INC-2024-001234",
            title="API latency spike",
            description="Users reporting 5s+ response times for /api/users endpoint",
            service="api-gateway",
            severity="high",
            error_type="latency",
            tags=["api", "performance"]
        )
    """
    store = await get_incident_store()
    
    incident = IncidentRecord(
        incident_id=incident_id,
        title=title,
        description=description,
        service=service,
        severity=severity,
        error_type=error_type,
        tags=tags or []
    )
    
    return await store.store_incident(incident)


async def store_resolved_incident(
    incident_id: str,
    title: str,
    description: str,
    service: str,
    root_cause: str,
    resolution_summary: str,
    resolution_steps: Optional[List[Dict[str, str]]] = None,
    severity: str = "medium",
    error_type: str = "general",
    detected_at: Optional[str] = None,
    resolved_at: Optional[str] = None,
    prevented_by: str = "",
    tags: Optional[List[str]] = None
) -> bool:
    """
    Store a complete resolved incident with all details.
    
    This is the primary function for populating the knowledge base
    with past incidents and their resolutions.
    
    Args:
        incident_id: Unique identifier
        title: Short description
        description: Full incident details
        service: Affected service
        root_cause: What caused the incident
        resolution_summary: Brief description of the fix
        resolution_steps: List of steps taken (each dict has 'action', optionally 'outcome')
        severity: Severity level
        error_type: Type of error
        detected_at: When incident was detected (ISO format)
        resolved_at: When incident was resolved (ISO format)
        prevented_by: What could prevent recurrence
        tags: Additional tags
    
    Returns:
        True if stored successfully
    
    Example:
        await store_resolved_incident(
            incident_id="INC-2024-001234",
            title="Database connection exhaustion",
            description="API returning 500 errors due to no available DB connections",
            service="api-gateway",
            severity="critical",
            error_type="connection_exhaustion",
            root_cause="Connection pool size too small for traffic spike",
            resolution_summary="Increased pool size from 10 to 50 connections",
            resolution_steps=[
                {"action": "Checked application logs for errors", "outcome": "Found connection timeout errors"},
                {"action": "Verified database was responsive", "outcome": "Database healthy, connections maxed"},
                {"action": "Increased connection pool size", "outcome": "Connections stabilized"},
                {"action": "Added connection pool monitoring", "outcome": "Alert configured at 80%"},
            ],
            prevented_by="Connection pool monitoring and auto-scaling",
            tags=["database", "connection-pool", "scaling"]
        )
    """
    store = await get_incident_store()
    
    # Build resolution steps
    steps = []
    if resolution_steps:
        for i, step_dict in enumerate(resolution_steps, 1):
            steps.append(ResolutionStep(
                step_number=i,
                action=step_dict.get("action", ""),
                outcome=step_dict.get("outcome", ""),
                performed_by=step_dict.get("performed_by", "")
            ))
    
    # Build timeline
    timeline = IncidentTimeline(
        detected_at=detected_at or datetime.now().isoformat(),
        resolved_at=resolved_at or datetime.now().isoformat()
    )
    
    incident = IncidentRecord(
        incident_id=incident_id,
        title=title,
        description=description,
        service=service,
        severity=severity,
        status="resolved",
        error_type=error_type,
        timeline=timeline,
        root_cause=root_cause,
        resolution_summary=resolution_summary,
        resolution_steps=steps,
        prevented_by=prevented_by,
        tags=tags or []
    )
    
    return await store.store_incident(incident)


async def find_similar_past_incidents(
    current_incident_description: str,
    service: Optional[str] = None,
    n_results: int = 5
) -> List[IncidentSearchResult]:
    """
    Find past incidents similar to a current one.
    
    This is the key function for incident resolution - finds
    past incidents with similar symptoms that were already resolved.
    
    Args:
        current_incident_description: Description of current symptoms
        service: Filter by specific service
        n_results: Number of results
    
    Returns:
        List of similar past incidents with resolutions
    
    Example:
        results = await find_similar_past_incidents(
            "Pods are crashing with OOMKilled error",
            service="kubernetes"
        )
        
        for r in results:
            print(f"Similar incident: {r.incident.title}")
            print(f"  Root cause: {r.incident.root_cause}")
            print(f"  Resolution: {r.incident.resolution_summary}")
    """
    store = await get_incident_store()
    
    return await store.find_similar_incidents(
        description=current_incident_description,
        service=service,
        only_resolved=True,
        n_results=n_results
    )
