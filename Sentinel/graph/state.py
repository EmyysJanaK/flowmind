"""
Graph state definition for Sentinel workflow.

State Evolution Flow
====================

The state object flows through agents in sequence, with each agent adding
to and updating specific fields. This creates an audit trail of the entire
incident analysis and remediation process.

Flow Diagram:
    User Input
        ↓
    [DetectiveAgent]
        ├─ Reads: user_input
        ├─ Adds: detective_findings (incident_summary, evidence, hypotheses)
        └─ State Progress: discovery phase
        ↓
    [ResearcherAgent]
        ├─ Reads: detective_findings, conversation_history
        ├─ Adds: researcher_recommendations (fixes, alternatives)
        └─ State Progress: analysis phase
        ↓
    [OperatorAgent]
        ├─ Reads: researcher_recommendations, detective_findings
        ├─ Adds: operator_action_plan (action_plan, risk_assessment)
        └─ State Progress: planning phase
        ↓
    [ExecutorAgent] (future)
        ├─ Reads: operator_action_plan
        ├─ Adds: execution_result, execution_log
        └─ State Progress: execution phase
        ↓
    [ReportAgent] (future)
        ├─ Reads: all previous results
        ├─ Generates: summary, metrics, recommendations
        └─ Final state: complete analysis record

State Immutability Note:
- Agents should NOT delete or modify existing fields
- Only append/add new findings to existing structures
- This preserves the complete history for debugging and auditing
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SentinelState:
    """
    Complete state object for Sentinel multi-agent workflow.

    This dataclass represents the shared state that flows through all agents
    in the Sentinel system. Each agent reads relevant fields, performs its
    analysis, and adds results back to the state for downstream agents.

    The state acts as the "blackboard" in the multi-agent architecture,
    enabling implicit communication between agents without direct coupling.

    Attributes:
        user_input (str): The original incident description from the user.
            This is read by DetectiveAgent to initiate analysis.

        conversation_history (List[Dict]): Complete history of all messages
            exchanged in the workflow. Each entry contains:
            {
                "role": "user|agent",
                "agent_name": str (only if role is "agent"),
                "content": str,
                "timestamp": str (ISO format)
            }
            Used by all agents for context. Enables human-in-the-loop
            interaction and provides audit trail.

        detective_findings (Dict): Output from DetectiveAgent. Contains:
            {
                "agent_name": "detective_agent",
                "incident_summary": str - condensed description,
                "hypotheses": List[Dict] - root cause hypotheses with:
                    - hypothesis: str
                    - explanation: str
                    - confidence: float (0-1)
                    - supporting_evidence: List[str]
                    - affected_components: List[str]
                "evidence": Dict - gathered evidence:
                    - data_sources: List[str]
                    - affected_services: List[str]
                    - error_patterns: List[str]
                    - frequency: str
                "total_hypotheses_generated": int,
                "hypotheses_filtered": int,
                "analysis_timestamp": str (ISO)
            }

        researcher_recommendations (Dict): Output from ResearcherAgent. Contains:
            {
                "agent_name": "researcher_agent",
                "fixes": List[Dict] - suggested fixes ranked by effectiveness,
                  each with:
                    - title: str
                    - description: str
                    - effectiveness_score: float
                    - feasibility_score: float
                    - steps: List[str]
                    - source: str - where fix came from
                    - related_issues: List[str]
                "primary_recommendation": Dict - the top-ranked fix,
                "alternative_approaches": List[Dict] - backup options,
                "knowledge_sources": List[str] - documentation used,
                "implementation_plan": Dict - detailed steps,
                "timestamp": str (ISO)
            }

        operator_action_plan (Dict): Output from OperatorAgent. Contains:
            {
                "agent_name": "operator_agent",
                "selected_action": Dict - the chosen fix with scores,
                "action_plan": Dict - executable procedure:
                    - phases: List[Dict] - logical execution phases,
                    - safety_checkpoints: List[Dict] - pre-flight checks,
                    - rollback_plan: Dict - recovery procedure,
                    - validation_steps: List[Dict] - success criteria,
                "decision_rationale": str - why this action was chosen,
                "risk_assessment": Dict:
                    - overall_risk: str (low|medium|high),
                    - specific_risks: List[Dict],
                    - safety_recommendations: List[str],
                "approval_status": str,
                "timestamp": str (ISO)
            }

        execution_result (Optional[Dict]): Output from ExecutorAgent (future).
            {
                "agent_name": "executor_agent",
                "status": str (success|partial|failed),
                "executed_steps": List[Dict],
                "errors": List[str] if any,
                "duration_seconds": float,
                "timestamp": str (ISO)
            }

        metadata (Dict): Arbitrary metadata about the workflow:
            - workflow_id: unique identifier
            - start_time: when workflow began
            - priority: incident priority level
            - environment: production|staging|development
            - custom_tags: Dict for filtering/categorization

        error (str): Error message if workflow encounters issues.
            If non-empty, workflow should terminate or switch to fallback mode.

        status (str): Current workflow status:
            - "initialized": workflow just started
            - "discovery": DetectiveAgent analyzing
            - "analysis": ResearcherAgent researching
            - "planning": OperatorAgent planning
            - "pending_approval": waiting for human approval
            - "executing": ExecutorAgent running actions
            - "completed": workflow finished successfully
            - "failed": workflow encountered error
            - "rolled_back": changes were reverted
    """

    # Primary inputs
    user_input: str = ""

    # Conversation context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Agent outputs (populated sequentially)
    detective_findings: Dict[str, Any] = field(default_factory=dict)
    researcher_recommendations: Dict[str, Any] = field(default_factory=dict)
    operator_action_plan: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    status: str = "initialized"

    # Helper methods for safe state updates
    def add_message(self, role: str, content: str, agent_name: Optional[str] = None) -> None:
        """
        Add a message to conversation history safely.

        Args:
            role: "user" or "agent"
            content: The message text
            agent_name: Name of agent if role is "agent"
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if agent_name:
            message["agent_name"] = agent_name
        self.conversation_history.append(message)

    def set_detective_findings(self, findings: Dict[str, Any]) -> None:
        """
        Set detective agent findings.

        Args:
            findings: Detection results from DetectiveAgent
        """
        self.detective_findings = findings
        self.status = "analysis"

    def set_researcher_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        Set researcher agent recommendations.

        Args:
            recommendations: Research results from ResearcherAgent
        """
        self.researcher_recommendations = recommendations
        self.status = "planning"

    def set_operator_action_plan(self, action_plan: Dict[str, Any]) -> None:
        """
        Set operator agent action plan.

        Args:
            action_plan: Action plan from OperatorAgent
        """
        self.operator_action_plan = action_plan
        approval_required = action_plan.get("approval_required", True)
        self.status = "pending_approval" if approval_required else "executing"

    def set_execution_result(self, result: Dict[str, Any]) -> None:
        """
        Set execution results.

        Args:
            result: Execution result from ExecutorAgent
        """
        self.execution_result = result
        exec_status = result.get("status", "unknown")
        self.status = "completed" if exec_status == "success" else "failed"

    def set_error(self, error_message: str) -> None:
        """
        Record an error and update status.

        Args:
            error_message: Description of the error
        """
        self.error = error_message
        self.status = "failed"

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata field.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current workflow state.

        Returns:
            Dictionary with workflow progress and key findings
        """
        return {
            "status": self.status,
            "incident": self.user_input[:100] + "..." if len(self.user_input) > 100 else self.user_input,
            "hypotheses_count": len(self.detective_findings.get("hypotheses", [])),
            "fixes_available": len(self.researcher_recommendations.get("fixes", [])),
            "action_planned": bool(self.operator_action_plan),
            "error": self.error if self.error else None,
            "execution_status": self.execution_result.get("status") if self.execution_result else None
        }


# Type alias for convenience
GraphState = SentinelState

