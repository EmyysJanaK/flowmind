"""
Memory store for managing conversation history and state.
"""

from typing import Any, Dict, List
from datetime import datetime


class MemoryStore:
    """In-memory storage for conversation history and shared state."""

    def __init__(self):
        """Initialize the memory store."""
        self.conversation_history: List[Dict[str, Any]] = []
        self.shared_state: Dict[str, Any] = {}
        self.created_at = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role: The role of the speaker (e.g., 'user', 'assistant', 'agent').
            content: The message content.
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.conversation_history

    def set_state(self, key: str, value: Any) -> None:
        """Set a value in shared state."""
        self.shared_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state."""
        return self.shared_state.get(key, default)

    def clear(self) -> None:
        """Clear all history and state."""
        self.conversation_history.clear()
        self.shared_state.clear()
