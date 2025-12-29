"""
Base agent class for Sentinel agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for all Sentinel agents."""

    def __init__(self, name: str, description: str):
        """
        Initialize a base agent.

        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task.

        Args:
            input_data: Input data for the agent.

        Returns:
            The result of the agent's execution.
        """
        pass
