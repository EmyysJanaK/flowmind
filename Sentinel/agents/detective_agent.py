"""
DetectiveAgent implementation for incident analysis.
"""

import json
from typing import Any, Dict, List
from datetime import datetime
from .base import BaseAgent


class DetectiveAgent(BaseAgent):
    """
    Agent specialized in analyzing incident descriptions and generating hypotheses.

    The DetectiveAgent acts as the investigator in the multi-agent system. It:
    1. Receives incident descriptions from user input or state
    2. Gathers evidence through structured analysis (mocked for now)
    3. Generates multiple competing hypotheses about root causes
    4. Ranks hypotheses by confidence and supporting evidence
    5. Stores findings in shared state for downstream agents to process

    This agent is typically the first responder in an incident workflow,
    preparing analysis for the ResearcherAgent to find solutions.

    Attributes:
        name (str): Identifier for this agent ("detective_agent").
        description (str): Role description in the multi-agent system.
        confidence_threshold (float): Minimum confidence (0-1) to include a hypothesis.
    """

    def __init__(
        self,
        name: str = "detective_agent",
        description: str = "Analyzes incident descriptions and generates root cause hypotheses",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the DetectiveAgent.

        Args:
            name: Agent identifier.
            description: Agent role in the system.
            confidence_threshold: Minimum confidence score for hypotheses (0-1).
                Hypotheses below this are filtered out.
        """
        super().__init__(name, description)
        self.confidence_threshold = confidence_threshold

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze incident description and produce root cause hypotheses.

        The agent extracts the incident description from state['user_input'],
        performs mocked evidence gathering, and generates structured hypotheses
        ranked by confidence. Results are stored in state['detective_findings'].

        Args:
            state (Dict[str, Any]): Workflow state containing:
                - user_input (str): The incident description to analyze
                - conversation_history (List[Dict]): Previous messages (optional)
                - agent_outputs (Dict): Results from other agents (optional)

        Returns:
            Dict[str, Any]: Updated state with:
                - detective_findings (Dict): Contains:
                    - incident_summary (str): Condensed incident description
                    - hypotheses (List[Dict]): Generated root cause hypotheses
                    - evidence (Dict): Mocked evidence gathering results
                    - analysis_timestamp (str): ISO timestamp of analysis
                    - agent_name (str): This agent's identifier

        Example:
            >>> state = {
            ...     "user_input": "Database connection timeout every morning at 9 AM",
            ...     "conversation_history": []
            ... }
            >>> result = await detective.run(state)
            >>> hypotheses = result["detective_findings"]["hypotheses"]
            >>> # Hypotheses ranked by confidence score
        """
        # Extract incident description from state
        incident_description = state.get("user_input", "")
        
        if not incident_description:
            state["detective_findings"] = {
                "agent_name": self.name,
                "error": "No incident description provided in state",
                "analysis_timestamp": datetime.now().isoformat()
            }
            return state

        # Perform incident analysis
        incident_summary = self._summarize_incident(incident_description)
        
        # Gather mocked evidence
        evidence = self._gather_evidence(incident_description)
        
        # Generate hypotheses
        hypotheses = self._generate_hypotheses(incident_description, evidence)
        
        # Filter by confidence threshold
        filtered_hypotheses = [
            h for h in hypotheses
            if h["confidence"] >= self.confidence_threshold
        ]
        
        # Sort by confidence (descending)
        filtered_hypotheses.sort(key=lambda x: x["confidence"], reverse=True)

        # Structure findings for downstream agents
        findings = {
            "agent_name": self.name,
            "incident_summary": incident_summary,
            "hypotheses": filtered_hypotheses,
            "evidence": evidence,
            "total_hypotheses_generated": len(hypotheses),
            "hypotheses_filtered": len(hypotheses) - len(filtered_hypotheses),
            "analysis_timestamp": datetime.now().isoformat()
        }

        # Add to state for downstream agents
        state["detective_findings"] = findings
        
        # Add conversation message
        state.setdefault("conversation_history", []).append({
            "role": "agent",
            "agent_name": self.name,
            "content": f"Generated {len(filtered_hypotheses)} hypotheses for investigation"
        })

        return state

    def _summarize_incident(self, description: str) -> str:
        """
        Summarize the incident description.

        Args:
            description: Full incident description.

        Returns:
            Condensed summary.
        """
        # Mock summarization - in production, use LLM or NLP
        words = description.split()
        if len(words) > 20:
            return " ".join(words[:20]) + "..."
        return description

    def _gather_evidence(self, description: str) -> Dict[str, Any]:
        """
        Gather and structure evidence related to the incident.

        This is mocked for now. Future integration points:
        - Real system logs and monitoring data
        - Metrics from observability platforms
        - Historical incident patterns

        Args:
            description: Incident description to analyze for patterns.

        Returns:
            Dictionary containing mocked evidence data.
        """
        # Mock evidence based on keywords in description
        evidence = {
            "data_sources": ["system_logs", "metrics", "traces"],
            "time_window": "24_hours",
            "affected_services": self._extract_services(description),
            "error_patterns": self._extract_patterns(description),
            "frequency": "recurring" if any(
                keyword in description.lower()
                for keyword in ["every", "every time", "always", "repeatedly"]
            ) else "one_time"
        }
        return evidence

    def _extract_services(self, description: str) -> List[str]:
        """Extract mentioned services/systems from description."""
        keywords = {
            "database": ["db", "database", "sql", "postgres", "mysql"],
            "api": ["api", "endpoint", "request", "http"],
            "cache": ["cache", "redis", "memcached"],
            "queue": ["queue", "message", "kafka", "rabbitmq"],
            "network": ["network", "connection", "timeout", "dns"]
        }
        
        mentioned = []
        desc_lower = description.lower()
        for service, keywords_list in keywords.items():
            if any(keyword in desc_lower for keyword in keywords_list):
                mentioned.append(service)
        
        return mentioned or ["unknown"]

    def _extract_patterns(self, description: str) -> List[str]:
        """Extract error patterns from description."""
        patterns = []
        keywords_patterns = {
            "timeout": ["timeout", "hang", "slow", "delay"],
            "connection_error": ["connection", "refused", "reset", "closed"],
            "resource_exhaustion": ["out of memory", "cpu", "disk", "limit"],
            "permission_denied": ["permission", "unauthorized", "forbidden"],
            "not_found": ["not found", "missing", "does not exist"]
        }
        
        desc_lower = description.lower()
        for pattern, keywords_list in keywords_patterns.items():
            if any(keyword in desc_lower for keyword in keywords_list):
                patterns.append(pattern)
        
        return patterns or ["unclassified"]

    def _generate_hypotheses(
        self,
        description: str,
        evidence: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate root cause hypotheses based on incident and evidence.

        Mocked hypothesis generation using pattern matching. Future improvements:
        - Use LLM for intelligent hypothesis generation
        - Reference historical incident database
        - Apply domain-specific knowledge graphs

        Args:
            description: Incident description.
            evidence: Gathered evidence from analysis.

        Returns:
            List of hypothesis dictionaries with confidence scores.
        """
        hypotheses = []
        desc_lower = description.lower()

        # Define hypothesis templates with triggering patterns and confidence
        hypothesis_templates = [
            {
                "title": "Resource exhaustion or capacity limit reached",
                "description": "System resources (memory, CPU, connections) exceeded threshold",
                "patterns": ["timeout", "hang", "slow", "limit", "out of memory"],
                "confidence": 0.85,
                "affected_areas": ["performance", "availability"]
            },
            {
                "title": "Database connection pool exhaustion",
                "description": "Connection pool depleted, new requests cannot connect",
                "patterns": ["database", "connection", "timeout", "db"],
                "confidence": 0.75,
                "affected_areas": ["database", "connectivity"]
            },
            {
                "title": "Network connectivity issue",
                "description": "Network latency, packet loss, or DNS resolution failure",
                "patterns": ["timeout", "connection", "network", "dns", "refused"],
                "confidence": 0.70,
                "affected_areas": ["network", "connectivity"]
            },
            {
                "title": "Cache invalidation or staleness",
                "description": "Cache returning outdated or inconsistent data",
                "patterns": ["cache", "stale", "inconsistent", "wrong data"],
                "confidence": 0.65,
                "affected_areas": ["data_consistency", "performance"]
            },
            {
                "title": "Scheduled job or batch process interference",
                "description": "Background job consuming resources at specific time",
                "patterns": ["morning", "night", "scheduled", "batch", "9 am"],
                "confidence": 0.60,
                "affected_areas": ["resource_contention", "timing"]
            },
            {
                "title": "Configuration or environment issue",
                "description": "Incorrect settings or environment mismatch",
                "patterns": ["config", "setting", "environment", "deploy"],
                "confidence": 0.55,
                "affected_areas": ["configuration", "deployment"]
            }
        ]

        # Score each hypothesis based on evidence
        for template in hypothesis_templates:
            pattern_matches = sum(
                1 for pattern in template["patterns"]
                if pattern in desc_lower
            )
            
            # Boost confidence if patterns match evidence
            confidence = template["confidence"]
            if evidence["error_patterns"]:
                pattern_overlap = len(
                    set(evidence["error_patterns"]) &
                    set(template.get("affected_areas", []))
                )
                confidence = min(1.0, confidence + (pattern_overlap * 0.05))

            # Only include if there's at least some pattern match
            if pattern_matches > 0 or confidence >= 0.55:
                hypotheses.append({
                    "hypothesis": template["title"],
                    "explanation": template["description"],
                    "confidence": round(confidence, 2),
                    "supporting_evidence": template["patterns"],
                    "pattern_matches": pattern_matches,
                    "affected_components": template["affected_areas"]
                })

        return hypotheses
