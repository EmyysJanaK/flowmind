"""
LangGraph workflow definition for Sentinel.

Graph Architecture
==================

The Sentinel workflow is a cyclic state graph that implements self-correcting
autonomous agents. The graph structure is designed to loop through investigation
and remediation cycles until the incident is resolved.

Workflow Diagram (Cyclic):

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  ┌──────────────┐         ┌──────────────┐                     │
    │  │  Detective   │────────▶│ Researcher   │                     │
    │  │  (Analyze)   │         │  (Research)  │                     │
    │  └──────────────┘         └──────────────┘                     │
    │         ▲                         │                             │
    │         │                         ▼                             │
    │         │                  ┌──────────────┐                     │
    │         │                  │  Operator    │                     │
    │         │                  │   (Plan)     │                     │
    │         │                  └──────────────┘                     │
    │         │                         │                             │
    │         │                         ▼                             │
    │         │                  ┌──────────────┐                     │
    │         │                  │ Reflection   │                     │
    │         │                  │   (Validate) │                     │
    │         │                  └──────────────┘                     │
    │         │                         │                             │
    │         │        ┌────────────────┼────────────────┐            │
    │         │        │                │                │            │
    │    (Retry?)      │          (Success?)        (Escalate?)       │
    │         │        │                │                │            │
    │         └────────┘          ┌─────▼──────┐   ┌────▼────┐       │
    │                             │  Completed │   │Escalated│       │
    │                             └────────────┘   └─────────┘       │
    └─────────────────────────────────────────────────────────────────┘

Conditional Logic:
- IMMEDIATE_RETRY: No loop, workflow ends successfully
- ADJUSTED_RETRY: Loop back to Detective with retry flag
- ALTERNATIVE_FIX: Loop back to Detective with alternative selected
- ESCALATE_TO_HUMAN: Exit to escalation node
- ABORT: Exit with failure status

State Progression:
1. Initialize state with incident description
2. Detective analyzes and generates hypotheses
3. Researcher researches and generates fixes
4. Operator plans remediation action
5. Execute (would be integrated with ExecutorAgent)
6. Reflect on results
7. Decision: Success/Retry/Escalate
8. If retry: loop back to Detective
9. Otherwise: exit to completion or escalation

Looping Mechanism:
The graph uses conditional_edge() to route based on retry strategy.
This prevents infinite loops by tracking retry_count in state.metadata.
Max retries (default 3) triggers escalation to prevent runaway loops.

Type Annotations:
The workflow uses LangGraph's StateGraph with SentinelState as the state type.
All node functions are async to support concurrent agent execution if needed.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from .state import SentinelState
from .reflection import (
    reflect_on_execution,
    decide_retry_strategy,
    update_state_with_reflection,
    RetryStrategy
)


# Import agent classes
# These would be actual agent instances in production
async def detective_node(state: SentinelState) -> SentinelState:
    """
    Execute the Detective agent.

    The detective analyzes the incident description and generates root cause
    hypotheses. This node runs at the start of the workflow and again if
    the system needs to re-investigate after a failed remediation.

    Args:
        state: Current workflow state

    Returns:
        Updated state with detective_findings
    """
    from ..agents import DetectiveAgent

    # Create detective agent
    detective = DetectiveAgent()

    # Check if this is a re-investigation
    is_reinvestigation = state.metadata.get("needs_reinvestigation", False)
    if is_reinvestigation:
        state.add_message(
            role="agent",
            agent_name="workflow",
            content="Re-investigating incident based on previous results"
        )
        state.metadata["needs_reinvestigation"] = False

    # Execute detective analysis
    state = await detective.run(state)
    state.status = "analysis"

    return state


async def researcher_node(state: SentinelState) -> SentinelState:
    """
    Execute the Researcher agent.

    The researcher takes hypotheses from detective findings and researches
    solutions, generating fix recommendations ranked by effectiveness.

    Args:
        state: Current workflow state with detective_findings

    Returns:
        Updated state with researcher_recommendations
    """
    from ..agents import ResearcherAgent

    # Validate detective findings exist
    if not state.detective_findings:
        state.set_error("Detective findings not found - cannot proceed with research")
        return state

    # Create researcher agent
    researcher = ResearcherAgent()

    # Execute research
    state = await researcher.run(state)
    state.status = "planning"

    return state


async def operator_node(state: SentinelState) -> SentinelState:
    """
    Execute the Operator agent.

    The operator evaluates researcher recommendations and creates a detailed
    action plan for remediation. The plan is NOT executed yet - it's prepared
    for review or handoff to an ExecutorAgent.

    Args:
        state: Current workflow state with researcher_recommendations

    Returns:
        Updated state with operator_action_plan
    """
    from ..agents import OperatorAgent

    # Validate researcher recommendations exist
    if not state.researcher_recommendations:
        state.set_error("Researcher recommendations not found - cannot plan")
        return state

    # Create operator agent
    operator = OperatorAgent()

    # Execute planning
    state = await operator.run(state)

    return state


async def execute_node(state: SentinelState) -> SentinelState:
    """
    Execute the remediation action plan.

    This node would integrate with an ExecutorAgent to actually run the
    planned remediation. For now, it's mocked to simulate execution.

    Args:
        state: Current workflow state with operator_action_plan

    Returns:
        Updated state with execution_result
    """

    # TODO: Integrate with ExecutorAgent
    # For now, mock execution result
    state.execution_result = {
        "agent_name": "executor_agent",
        "status": "success",
        "executed_steps": state.operator_action_plan.get("action_plan", {}).get("phases", []),
        "errors": [],
        "duration_seconds": 45.5
    }

    state.add_message(
        role="agent",
        agent_name="executor_agent",
        content="Remediation action executed"
    )

    return state


async def reflection_node(state: SentinelState) -> SentinelState:
    """
    Execute reflection and validation.

    This node validates whether remediation succeeded and decides whether
    to retry, try an alternative, or escalate. This is where the self-correction
    loop is controlled.

    Args:
        state: Current workflow state with execution_result

    Returns:
        Updated state with reflection results and retry decision
    """

    # Perform reflection on execution results
    reflection_outcome, reflection_data = reflect_on_execution(state)

    # Decide retry strategy
    retry_strategy, retry_decision = decide_retry_strategy(
        reflection_outcome, reflection_data, state
    )

    # Update state with reflection results
    state = update_state_with_reflection(
        state, reflection_outcome, reflection_data, retry_strategy, retry_decision
    )

    return state


async def escalation_node(state: SentinelState) -> SentinelState:
    """
    Handle escalation to human review.

    This node is reached when the system needs human intervention because:
    - Maximum retries exceeded
    - Unknown failure occurred
    - Execution errors require investigation

    Args:
        state: Current workflow state

    Returns:
        Updated state with escalation flag
    """

    state.status = "pending_approval"
    escalation_msg = (
        "Incident escalated to human review. "
        "Reason: " + state.error or "Complex case requiring manual intervention"
    )
    state.add_message(
        role="agent",
        agent_name="workflow",
        content=escalation_msg
    )

    return state


def route_after_reflection(state: SentinelState) -> str:
    """
    Route the workflow based on reflection decision.

    This conditional edge determines whether to:
    - Loop back to detective for re-investigation
    - Complete the workflow (success)
    - Escalate to human review

    Args:
        state: Current workflow state with reflection results

    Returns:
        Node name to route to: "detective", "completion", or "escalation"

    Routing Logic:
    - If retry_strategy is IMMEDIATE_RETRY → completion (success)
    - If retry_strategy is ADJUSTED_RETRY or ALTERNATIVE_FIX → detective (retry)
    - If retry_strategy is ESCALATE_TO_HUMAN → escalation
    - If retry_strategy is ABORT → completion (with error)
    """

    if not state.metadata.get("latest_reflection"):
        return "escalation"

    retry_strategy = state.metadata["latest_reflection"].get("retry_strategy")

    # Success path: end workflow
    if retry_strategy == "success":
        return "completion"

    # Retry paths: loop back to detective
    if retry_strategy in ["immediate_retry", "adjusted_retry", "alternative_fix"]:
        return "detective"

    # Escalation path
    if retry_strategy == "escalate_to_human":
        return "escalation"

    # Abort path
    if retry_strategy == "abort":
        return "completion"

    # Default to escalation if unclear
    return "escalation"


def create_workflow():
    """
    Create the main Sentinel workflow graph.

    This creates a LangGraph StateGraph that implements the cyclic workflow:
    Detective → Researcher → Operator → Execute → Reflection → [Decision]

    The decision point routes based on reflection outcome:
    - Success: Complete
    - Retry: Loop back to Detective
    - Escalate: Go to escalation node

    Returns:
        A compiled StateGraph for the Sentinel workflow.

    Usage:
        >>> workflow = create_workflow()
        >>> state = SentinelState(user_input="Database timeout at 9 AM")
        >>> result = await workflow.ainvoke(state)
    """

    # Create the state graph
    workflow = StateGraph(SentinelState)

    # Add nodes for each agent and step
    workflow.add_node("detective", detective_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("operator", operator_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("completion", lambda state: state)  # Sink node

    # Add edges: linear flow to reflection
    workflow.add_edge("detective", "researcher")
    workflow.add_edge("researcher", "operator")
    workflow.add_edge("operator", "execute")
    workflow.add_edge("execute", "reflection")

    # Add conditional edge: reflection routes to retry or exit
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "detective": "detective",      # Loop back for retry
            "completion": "completion",    # Success or abort
            "escalation": "escalation"     # Need human help
        }
    )

    # Escalation always goes to completion
    workflow.add_edge("escalation", "completion")

    # Set entry point
    workflow.set_entry_point("detective")

    # Set exit point
    workflow.set_finish_point("completion")

    # Compile the graph
    return workflow.compile()


def visualize_workflow() -> str:
    """
    Return a text visualization of the workflow graph.

    This provides an ASCII diagram of the graph structure for documentation.

    Returns:
        String containing the workflow diagram
    """

    diagram = """
    Sentinel Workflow: Cyclic Self-Correcting Investigation

    ┌─────────────────────────────────────────────────────────┐
    │                    START (User Input)                   │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │              DETECTIVE: Analyze Incident                │
    │  - Extract incident details                             │
    │  - Gather evidence                                      │
    │  - Generate root cause hypotheses                       │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │           RESEARCHER: Research Solutions                │
    │  - Retrieve fixes for hypotheses                        │
    │  - Rank by effectiveness & feasibility                  │
    │  - Create implementation plans                          │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │            OPERATOR: Plan Remediation                   │
    │  - Select best fix                                      │
    │  - Create action plan with safety checks                │
    │  - Prepare rollback procedure                           │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │           EXECUTE: Run Remediation                      │
    │  - Execute action plan steps                            │
    │  - Track progress and errors                            │
    │  - Generate execution log                               │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │      REFLECTION: Validate & Decide Next Step            │
    │  - Check if fix succeeded                               │
    │  - Identify any new issues                              │
    │  - Decide: Retry/Alternative/Escalate/Success          │
    └────────────────────────┬────────────────────────────────┘
                             │
                ┌────────────┼────────────┬──────────────┐
                │            │            │              │
                ▼            ▼            ▼              ▼
    ┌──────────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐
    │ LOOP: Retry      │ │ RETRY    │ │ESCALATE  │ │ SUCCESS     │
    │ (Detective)      │ │(Researcher)
    │ Max 3 attempts   │ │(Operator)│ │(Human)   │ │ Complete    │
    │                  │ │          │ │          │ │             │
    └──────────────────┘ └──────────┘ └──────────┘ └─────────────┘
         │                  │             │             │
         │                  │             │             │
         └──────────────────┴─────────────┴─────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   END: Completion                       │
    │  - Status: completed/failed                             │
    │  - Return full analysis and state                       │
    └─────────────────────────────────────────────────────────┘

    Key Features:
    ✓ Cyclic: Can loop back for re-investigation
    ✓ Self-Correcting: Tries adjustments before escalating
    ✓ Safe: Max 3 retries prevents infinite loops
    ✓ Auditable: Complete history in conversation_history
    ✓ Human-in-loop: Escalates when needed
    ✓ Async: All nodes are async-capable
    """

    return diagram

