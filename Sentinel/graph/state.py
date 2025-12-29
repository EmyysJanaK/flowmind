"""
Graph state definition for Sentinel workflow.
"""

from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class GraphState:
    """State object for LangGraph workflow."""

    user_input: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def set_agent_output(self, agent_name: str, output: Any) -> None:
        """Set output from an agent."""
        self.agent_outputs[agent_name] = output
