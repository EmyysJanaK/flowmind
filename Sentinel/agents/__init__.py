"""
Agents module for Sentinel multi-agent system.
Contains individual agent implementations for specialized tasks.
"""

from .base import BaseAgent
from .detective_agent import DetectiveAgent
from .researcher_agent import ResearcherAgent

__all__ = ["BaseAgent", "DetectiveAgent", "ResearcherAgent"]
