"""
LangGraph workflow definition for Sentinel.
"""

from langgraph.graph import StateGraph
from .state import GraphState


def create_workflow():
    """
    Create the main Sentinel workflow graph.

    Returns:
        A compiled StateGraph for the Sentinel workflow.
    """
    workflow = StateGraph(GraphState)

    # TODO: Add nodes for each agent
    # TODO: Add edges connecting nodes
    # TODO: Set entry and exit points

    return workflow.compile()
