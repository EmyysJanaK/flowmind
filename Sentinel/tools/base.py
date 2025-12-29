"""
Base tool class for Sentinel agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class for all Sentinel tools."""

    def __init__(self, name: str, description: str):
        """
        Initialize a base tool.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with provided arguments.

        Returns:
            The result of tool execution.
        """
        pass
