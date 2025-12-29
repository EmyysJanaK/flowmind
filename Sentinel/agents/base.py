"""
Base agent class for Sentinel multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Abstract base class for all Sentinel agents.

    In the Sentinel multi-agent system, agents are autonomous entities that:
    - Receive state from the LangGraph workflow
    - Process information using their specialized logic
    - Interact with tools to accomplish tasks
    - Return updated state to the graph for coordination

    Each agent specializes in a specific domain or capability, such as analysis,
    decision-making, planning, or execution. Agents communicate through shared
    state passed by the LangGraph orchestrator, enabling complex multi-step
    workflows where agents can delegate tasks or build upon each other's work.

    Attributes:
        name (str): Unique identifier for the agent.
        description (str): Human-readable description of the agent's purpose
            and capabilities in the multi-agent system.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a Sentinel agent.

        Args:
            name: Unique identifier for this agent. Used for logging and
                routing decisions in the workflow graph.
            description: Comprehensive description of the agent's specialized
                role, capabilities, and the types of tasks it handles within
                the multi-agent ecosystem.

        Example:
            >>> agent = MyAgent(
            ...     name="analysis_agent",
            ...     description="Analyzes workflow data and identifies bottlenecks"
            ... )
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's logic on the provided state.

        This method is called by the LangGraph workflow coordinator. The agent
        receives the current state, processes it using its specialized logic
        and tools, and returns an updated state to be passed to other agents
        or returned as the final result.

        The state is a shared data structure containing:
        - Conversation history and user context
        - Results from previously executed agents
        - Workflow metadata and execution tracking
        - Shared memory accessible to all agents

        Args:
            state (Dict[str, Any]): The current workflow state from LangGraph.
                Contains all context needed for decision-making, including
                conversation history, prior agent results, and shared data.

        Returns:
            Dict[str, Any]: Updated state with the agent's contributions.
                Typically includes:
                - New conversation messages (if applicable)
                - Agent-specific results (added to state)
                - Updated metadata or tracking info
                - Recommendations for next workflow steps

        Raises:
            ValueError: If required state fields are missing or invalid.
            RuntimeError: If the agent encounters an unrecoverable error
                during execution.

        Example:
            An analysis agent might:
            1. Extract data from state['workflow_data']
            2. Perform analysis using its specialized logic
            3. Add results to state['analysis_results']
            4. Return the updated state for downstream agents
        """
        pass
