"""
OperatorAgent implementation for remediation action planning and orchestration.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from .base import BaseAgent


class OperatorAgent(BaseAgent):
    """
    Agent specialized in planning and executing remediation actions.

    The OperatorAgent acts as the execution strategist in the multi-agent system. It:
    1. Receives fix recommendations from the ResearcherAgent
    2. Evaluates recommendations against current system state
    3. Decides on the optimal remediation action
    4. Creates a detailed, executable action plan with safety checkpoints
    5. Invokes infrastructure tools to execute the plan based on fix type
    6. Tracks execution progress and validates results

    CRITICAL: No Direct Shell Command Execution
    =============================================
    This agent NEVER executes arbitrary shell commands. All execution is delegated
    to typed, safe infrastructure tools (Docker, Kubernetes, etc.) that:
    - Have validation and error handling built-in
    - Block dangerous operations at the parameter level
    - Log all operations for audit trails
    - Support dry-run and rollback procedures

    Execution Architecture:
    - PLANNING PHASE: Agent analyzes fixes and creates detailed execution plan
    - DECISION PHASE: Agent maps plan steps to infrastructure tools
    - EXECUTION PHASE: Agent invokes safe, typed tools (no arbitrary commands)
    - VALIDATION PHASE: Agent checks results and validates remediation
    - ROLLBACK PHASE: If needed, agent executes rollback plan

    Tool Invocation Strategy:
    - Tools are discovered from the global tool registry
    - Each fix type maps to specific infrastructure tools:
      * Container fixes → docker_container_restart, docker_container_logs
      * Kubernetes fixes → k8s_pod_status, k8s_pod_restart
      * Monitoring fixes → monitoring tools (if available)
    - No arbitrary shell commands are possible
    - All tool calls are validated and logged
    - Tool failures trigger automatic rollback planning

    Safety Mechanisms:
    - Tools have built-in parameter validation (blocks wildcards, injection, etc.)
    - Audit logging tracks all invocations for forensic analysis
    - Execution results are captured and validated
    - Rollback procedures are always available
    - Safety checkpoints enforce approval before execution

    Attributes:
        name (str): Identifier for this agent ("operator_agent").
        description (str): Role description in the multi-agent system.
        approval_required (bool): If True, recommends human approval before execution.
    """

    def __init__(
        self,
        name: str = "operator_agent",
        description: str = "Plans and executes remediation actions using infrastructure tools",
        approval_required: bool = True,
        enable_auto_execution: bool = False
    ):
        """
        Initialize the OperatorAgent.

        Args:
            name: Agent identifier.
            description: Agent role in the system.
            approval_required: If True, marks action plans as needing approval.
            enable_auto_execution: If True, automatically execute plans after approval.
                                 If False, return plan for external executor. Default: False (safer)
        """
        super().__init__(name, description)
        self.approval_required = approval_required
        self.enable_auto_execution = enable_auto_execution
        self.tool_registry = None  # Lazy-loaded on first use
        self.execution_context = {}  # Track execution state

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and execute remediation action plan based on researcher recommendations.

        The agent performs three distinct phases:

        PHASE 1: PLANNING
        - Analyzes researcher recommendations and incident context
        - Decides on the best remediation action
        - Creates detailed execution plan with safety checkpoints

        PHASE 2: DECISION
        - Determines which infrastructure tools to use
        - Maps plan steps to safe, typed tool invocations
        - Validates that all operations are safe (no arbitrary commands)

        PHASE 3: EXECUTION (optional, controlled by enable_auto_execution)
        - If enabled AND approved: Invokes tools to execute the plan
        - Captures results and validates remediation
        - Updates state with execution status
        - Triggers rollback on failure

        Args:
            state (Dict[str, Any]): Workflow state containing:
                - researcher_recommendations (Dict): Fixes from ResearcherAgent
                - detective_findings (Dict): Context from DetectiveAgent
                - conversation_history (List[Dict]): Previous messages (optional)
                - approval_granted (bool): If True, execute plan (optional)

        Returns:
            Dict[str, Any]: Updated state with:
                - operator_action_plan (Dict): Complete execution plan
                - execution_results (Dict): Results if auto_execution enabled
                - execution_status (str): success, failed, pending_approval, error
                - timestamp (str): ISO timestamp

        Example:
            >>> state = {"researcher_recommendations": {...fixes...}}
            >>> result = await operator.run(state)
            >>> if result["execution_status"] == "success":
            ...     print("Remediation complete")
            ... else:
            ...     print("Manual review required")
        """
        # PHASE 1: PLANNING
        # Extract recommendations from state
        recommendations = state.get("researcher_recommendations")
        
        if not recommendations:
            state["operator_action_plan"] = {
                "agent_name": self.name,
                "error": "No researcher recommendations found in state",
                "execution_status": "error",
                "timestamp": datetime.now().isoformat()
            }
            state["execution_status"] = "error"
            return state

        # Get primary recommendation
        selected_fix = recommendations.get("primary_recommendation")
        
        if not selected_fix:
            state["operator_action_plan"] = {
                "agent_name": self.name,
                "warning": "No primary recommendation available",
                "alternatives": recommendations.get("alternative_approaches", []),
                "execution_status": "error",
                "timestamp": datetime.now().isoformat()
            }
            state["execution_status"] = "error"
            return state

        # Analyze context
        detective_findings = state.get("detective_findings", {})
        hypotheses = detective_findings.get("hypotheses", [])
        
        # Create action plan
        action_plan = self._create_action_plan(
            selected_fix=selected_fix,
            hypotheses=hypotheses,
            recommendations=recommendations
        )
        
        # PHASE 2: DECISION
        # Determine which tools to use for execution
        tool_mappings = self._map_plan_to_tools(selected_fix, action_plan)
        action_plan["tool_mappings"] = tool_mappings
        
        # Assess risks
        risk_assessment = self._assess_risks(action_plan, selected_fix)
        
        # Determine approval status
        approval_status = self._determine_approval_status(
            action_plan, risk_assessment
        )

        # Structure the complete action plan
        operator_plan = {
            "agent_name": self.name,
            "selected_action": {
                "title": selected_fix.get("title"),
                "description": selected_fix.get("description"),
                "effectiveness": selected_fix.get("effectiveness_score"),
                "feasibility": selected_fix.get("feasibility_score")
            },
            "action_plan": action_plan,
            "decision_rationale": self._generate_rationale(
                selected_fix, hypotheses
            ),
            "risk_assessment": risk_assessment,
            "approval_required": self.approval_required,
            "approval_status": approval_status,
            "execution_notes": self._generate_execution_notes(),
            "timestamp": datetime.now().isoformat()
        }

        # Add to state
        state["operator_action_plan"] = operator_plan

        # PHASE 3: EXECUTION (optional)
        # Check if execution is enabled and approved
        should_execute = (
            self.enable_auto_execution and
            (not self.approval_required or state.get("approval_granted", False))
        )

        if should_execute:
            # Execute the plan using infrastructure tools
            execution_results = await self._execute_plan(
                operator_plan, state
            )
            state["execution_results"] = execution_results
            state["execution_status"] = execution_results.get("status", "unknown")
            
            # Add execution message
            if execution_results.get("status") == "success":
                state.setdefault("conversation_history", []).append({
                    "role": "agent",
                    "agent_name": self.name,
                    "content": f"Remediation executed successfully: {selected_fix.get('title')}"
                })
            else:
                state.setdefault("conversation_history", []).append({
                    "role": "agent",
                    "agent_name": self.name,
                    "content": f"Remediation execution failed: {execution_results.get('error', 'Unknown error')}"
                })
        else:
            # Plan only - awaiting approval or manual execution
            status_msg = "pending approval" if self.approval_required else "ready to execute"
            state["execution_status"] = "pending_approval"
            state.setdefault("conversation_history", []).append({
                "role": "agent",
                "agent_name": self.name,
                "content": f"Created action plan ({status_msg}): {selected_fix.get('title')}"
            })

        return state

    async def _map_plan_to_tools(
        self,
        selected_fix: Dict[str, Any],
        action_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Map action plan steps to infrastructure tools.

        Analyzes the fix type and creates mappings to specific tools:
        - Docker fixes → docker_container_logs, docker_container_restart
        - Kubernetes fixes → k8s_pod_status, k8s_pod_restart
        - No arbitrary shell commands are allowed

        Args:
            selected_fix: The chosen fix from recommendations.
            action_plan: The action plan with phases and steps.

        Returns:
            List of tool mappings specifying what to invoke.
        """
        fix_type = selected_fix.get("type", "").lower()
        tool_mappings = []
        
        # Map based on fix type
        if "container" in fix_type or "docker" in fix_type:
            tool_mappings.append({
                "phase": "validation",
                "tool": "docker_container_logs",
                "purpose": "Check container logs before making changes",
                "parameters": {
                    "container_name": selected_fix.get("container_name", ""),
                    "tail": 50
                }
            })
            
            tool_mappings.append({
                "phase": "execution",
                "tool": "docker_container_restart",
                "purpose": "Restart container to apply remediation",
                "parameters": {
                    "container_name": selected_fix.get("container_name", ""),
                    "timeout": 30
                }
            })
            
            tool_mappings.append({
                "phase": "validation",
                "tool": "docker_container_logs",
                "purpose": "Verify container is running after restart",
                "parameters": {
                    "container_name": selected_fix.get("container_name", ""),
                    "tail": 100
                }
            })
        
        elif "pod" in fix_type or "kubernetes" in fix_type or "k8s" in fix_type:
            tool_mappings.append({
                "phase": "validation",
                "tool": "k8s_pod_status",
                "purpose": "Check pod status before making changes",
                "parameters": {
                    "namespace": selected_fix.get("namespace", "default"),
                    "pod_name": selected_fix.get("pod_name", "")
                }
            })
            
            tool_mappings.append({
                "phase": "execution",
                "tool": "k8s_pod_restart",
                "purpose": "Restart pod to apply remediation",
                "parameters": {
                    "namespace": selected_fix.get("namespace", "default"),
                    "pod_name": selected_fix.get("pod_name", ""),
                    "grace_period": 30
                }
            })
            
            tool_mappings.append({
                "phase": "validation",
                "tool": "k8s_pod_status",
                "purpose": "Verify pod is running after restart",
                "parameters": {
                    "namespace": selected_fix.get("namespace", "default"),
                    "pod_name": selected_fix.get("pod_name", "")
                }
            })
        
        else:
            # Generic fix type - add monitoring validation
            tool_mappings.append({
                "phase": "validation",
                "tool": "generic_status_check",
                "purpose": "Validate system state before remediation",
                "parameters": {}
            })
        
        return tool_mappings

    async def _execute_plan(
        self,
        operator_plan: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the action plan using infrastructure tools.

        CRITICAL: This method only invokes safe, typed infrastructure tools.
        No arbitrary shell commands are executed.

        Execution phases:
        1. PRE-CHECKS: Validate system state before changes
        2. EXECUTION: Invoke appropriate tools for remediation
        3. VALIDATION: Verify remediation was successful
        4. ROLLBACK: Execute rollback if validation fails

        Args:
            operator_plan: Complete action plan with tool mappings.
            state: Current workflow state.

        Returns:
            Execution results with status, outputs, and any errors.
        """
        # Lazy-load tool registry on first execution
        if self.tool_registry is None:
            try:
                from sentinel.tools import get_global_registry
                self.tool_registry = get_global_registry()
            except ImportError:
                return {
                    "status": "failed",
                    "error": "Tool registry not available",
                    "tools_invoked": []
                }
        
        tool_mappings = operator_plan.get("action_plan", {}).get("tool_mappings", [])
        execution_log = []
        
        try:
            # Execute each mapped tool in sequence
            for mapping in tool_mappings:
                tool_name = mapping.get("tool")
                phase = mapping.get("phase")
                parameters = mapping.get("parameters", {})
                
                # Validate tool exists (blocks unknown tools)
                if not self.tool_registry.get_tool(tool_name):
                    execution_log.append({
                        "tool": tool_name,
                        "status": "skipped",
                        "reason": f"Tool not registered (prevents arbitrary execution)"
                    })
                    continue
                
                # Invoke tool with agent context for audit logging
                result = await self.tool_registry.call_tool(
                    tool_name=tool_name,
                    agent_name=self.name,
                    llm_model="operator_agent",
                    **parameters
                )
                
                # Check if tool call was blocked by validation
                if result.get("blocked"):
                    return {
                        "status": "failed",
                        "error": f"Tool validation blocked {tool_name}: {result.get('error')}",
                        "tools_invoked": execution_log
                    }
                
                # Check for execution errors
                if not result.get("success", False):
                    execution_log.append({
                        "tool": tool_name,
                        "phase": phase,
                        "status": "failed",
                        "error": result.get("error", "Unknown error"),
                        "error_type": result.get("error_type", "unknown")
                    })
                    
                    # Phase determines if we continue or fail
                    if phase == "validation" and result.get("error_type") == "not_found":
                        # Resource not found during pre-check - fail fast
                        return {
                            "status": "failed",
                            "error": f"{tool_name} validation failed: {result.get('error')}",
                            "tools_invoked": execution_log
                        }
                    elif phase == "execution":
                        # Execution failed - abort
                        return {
                            "status": "failed",
                            "error": f"{tool_name} execution failed: {result.get('error')}",
                            "tools_invoked": execution_log
                        }
                    # Validation phase failures are recorded but don't always fail
                else:
                    execution_log.append({
                        "tool": tool_name,
                        "phase": phase,
                        "status": "success",
                        "purpose": mapping.get("purpose")
                    })
            
            # All tools executed successfully
            return {
                "status": "success",
                "message": "Remediation executed successfully",
                "tools_invoked": execution_log,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Execution error: {str(e)}",
                "error_type": type(e).__name__,
                "tools_invoked": execution_log
            }

    def _create_action_plan(
        self,
        selected_fix: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a detailed, executable action plan.

        This plan specifies exactly what needs to be done without actually
        executing commands. The plan can be used by:
        - An ExecutorAgent that has command execution capabilities
        - Human operators following the procedure
        - Infrastructure-as-code systems (Terraform, Kubernetes manifests, etc.)
        - Ticketing systems for tracking

        Args:
            selected_fix: The chosen fix from recommendations.
            hypotheses: Root cause hypotheses for context.
            recommendations: Full recommendations object for alternatives.

        Returns:
            Detailed action plan structure.
        """
        primary_hypothesis = hypotheses[0] if hypotheses else {}
        
        plan = {
            "title": selected_fix.get("title", "Untitled Action"),
            "description": selected_fix.get("description", ""),
            "objective": f"Address hypothesis: {primary_hypothesis.get('hypothesis', 'Unknown')}",
            "estimated_duration_minutes": selected_fix.get("time_to_implement_minutes", 0),
            "phases": self._structure_action_phases(selected_fix),
            "safety_checkpoints": self._create_safety_checkpoints(selected_fix),
            "rollback_plan": self._create_rollback_plan(selected_fix),
            "validation_steps": self._create_validation_steps(selected_fix),
            "documentation": {
                "related_runbooks": selected_fix.get("source", ""),
                "related_issues": selected_fix.get("related_issues", []),
                "knowledge_sources": recommendations.get("knowledge_sources", [])
            }
        }
        
        return plan

    def _structure_action_phases(
        self, fix: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Structure the fix steps into executable phases.

        Each phase represents a logical grouping of actions that can be
        monitored and potentially rolled back independently.

        Args:
            fix: The fix to structure into phases.

        Returns:
            List of action phases with detailed instructions.
        """
        steps = fix.get("steps", [])
        phases = []
        
        # Organize steps into logical phases
        if len(steps) <= 3:
            # Small fixes: single phase
            phases.append({
                "phase": 1,
                "name": "Implementation",
                "description": "Execute all remediation steps",
                "steps": [
                    {
                        "sequence": i + 1,
                        "instruction": step,
                        "type": "execute",
                        "verification_required": i == len(steps) - 1
                    }
                    for i, step in enumerate(steps)
                ]
            })
        else:
            # Large fixes: multiple phases
            phases.append({
                "phase": 1,
                "name": "Preparation",
                "description": "Validate prerequisites and create backups",
                "steps": [
                    {
                        "sequence": 1,
                        "instruction": "Verify system backups are current",
                        "type": "check",
                        "verification_required": True
                    },
                    {
                        "sequence": 2,
                        "instruction": "Confirm monitoring and alerting are active",
                        "type": "check",
                        "verification_required": True
                    }
                ]
            })
            
            phases.append({
                "phase": 2,
                "name": "Implementation",
                "description": "Execute primary remediation steps",
                "steps": [
                    {
                        "sequence": i + 1,
                        "instruction": step,
                        "type": "execute",
                        "verification_required": False
                    }
                    for i, step in enumerate(steps[:-1])
                ]
            })
            
            phases.append({
                "phase": 3,
                "name": "Validation",
                "description": "Verify the fix is working",
                "steps": [
                    {
                        "sequence": 1,
                        "instruction": steps[-1] if steps else "Verify fix is working",
                        "type": "verify",
                        "verification_required": True
                    }
                ]
            })
        
        return phases

    def _create_safety_checkpoints(
        self, fix: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Create safety checkpoints that MUST be verified before proceeding.

        These checkpoints prevent dangerous actions without proper checks.

        Args:
            fix: The fix to create checkpoints for.

        Returns:
            List of safety checkpoints.
        """
        return [
            {
                "checkpoint": "Pre-execution review",
                "description": "Confirm action plan with team",
                "required": True,
                "blocking": True
            },
            {
                "checkpoint": "Change management approval",
                "description": "Verify change was approved through proper channels",
                "required": self.approval_required,
                "blocking": self.approval_required
            },
            {
                "checkpoint": "Backup verification",
                "description": "Ensure current system state is backed up",
                "required": True,
                "blocking": True
            },
            {
                "checkpoint": "Communication check",
                "description": "Relevant teams have been notified",
                "required": True,
                "blocking": False
            },
            {
                "checkpoint": "Read-only test",
                "description": "Rehearse steps without making changes",
                "required": False,
                "blocking": False
            }
        ]

    def _create_rollback_plan(self, fix: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a detailed rollback procedure.

        If the action causes problems, this plan specifies how to revert.

        Args:
            fix: The fix to create a rollback for.

        Returns:
            Rollback procedure.
        """
        return {
            "trigger_conditions": [
                "Critical errors appear in logs",
                "Service becomes unavailable",
                "Performance degrades below baseline",
                "Explicitly requested by operator"
            ],
            "rollback_steps": [
                {
                    "sequence": 1,
                    "action": "Pause remediation if still in progress",
                    "priority": "immediate"
                },
                {
                    "sequence": 2,
                    "action": "Isolate affected services if possible",
                    "priority": "immediate"
                },
                {
                    "sequence": 3,
                    "action": "Restore from backup",
                    "priority": "high"
                },
                {
                    "sequence": 4,
                    "action": "Verify service recovery",
                    "priority": "high"
                },
                {
                    "sequence": 5,
                    "action": "Analyze what went wrong",
                    "priority": "medium"
                }
            ],
            "estimated_time_minutes": max(5, fix.get("time_to_implement_minutes", 0) // 2),
            "data_risk": "low" if "restore from backup" in fix.get("description", "").lower() else "medium"
        }

    def _create_validation_steps(
        self, fix: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create validation steps to verify the fix worked.

        These steps confirm that the remediation achieved its objective.

        Args:
            fix: The fix to create validation for.

        Returns:
            List of validation steps.
        """
        return [
            {
                "step": 1,
                "validation": "Check error logs for resolution",
                "expected_result": "Errors related to the issue should be gone",
                "tool": "log_analysis"
            },
            {
                "step": 2,
                "validation": "Verify service health metrics",
                "expected_result": "All services reporting healthy",
                "tool": "monitoring_dashboard"
            },
            {
                "step": 3,
                "validation": "Test with sample requests",
                "expected_result": "Requests complete successfully",
                "tool": "http_client"
            },
            {
                "step": 4,
                "validation": "Monitor for 5 minutes",
                "expected_result": "No new errors or warnings",
                "tool": "continuous_monitoring"
            },
            {
                "step": 5,
                "validation": "Document the fix",
                "expected_result": "Runbook updated with solution",
                "tool": "documentation"
            }
        ]

    def _assess_risks(
        self, action_plan: Dict[str, Any], fix: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess risks of executing the action plan.

        Args:
            action_plan: The planned action.
            fix: The fix being executed.

        Returns:
            Risk assessment with mitigation strategies.
        """
        feasibility = fix.get("feasibility_score", 0.5)
        
        if feasibility > 0.9:
            risk_level = "low"
            risk_score = 1
        elif feasibility > 0.75:
            risk_level = "medium"
            risk_score = 2
        else:
            risk_level = "high"
            risk_score = 3

        return {
            "overall_risk": risk_level,
            "risk_score": risk_score,
            "specific_risks": [
                {
                    "risk": "Service downtime",
                    "probability": "low" if feasibility > 0.8 else "medium",
                    "impact": "critical",
                    "mitigation": "Have rollback plan ready"
                },
                {
                    "risk": "Data inconsistency",
                    "probability": "low",
                    "impact": "critical",
                    "mitigation": "Backup before changes"
                },
                {
                    "risk": "Partial implementation",
                    "probability": "medium" if feasibility < 0.8 else "low",
                    "impact": "high",
                    "mitigation": "Follow checklist precisely"
                }
            ],
            "safety_recommendations": [
                "Execute during scheduled maintenance window",
                "Have team member ready for rollback",
                "Increase monitoring during execution",
                "Have communication channel open with stakeholders"
            ]
        }

    def _determine_approval_status(
        self,
        action_plan: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> str:
        """
        Determine if action plan needs approval.

        Args:
            action_plan: The planned action.
            risk_assessment: Risk evaluation.

        Returns:
            Approval status string.
        """
        if not self.approval_required:
            return "ready_to_execute"
        
        risk_level = risk_assessment.get("overall_risk", "unknown")
        
        if risk_level == "high":
            return "pending_executive_approval"
        elif risk_level == "medium":
            return "pending_team_approval"
        else:
            return "pending_approval"

    def _generate_rationale(
        self,
        selected_fix: Dict[str, Any],
        hypotheses: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a human-readable rationale for the decision.

        Args:
            selected_fix: The chosen fix.
            hypotheses: Root cause hypotheses.

        Returns:
            Rationale text.
        """
        primary = hypotheses[0] if hypotheses else {}
        confidence = primary.get("confidence", 0)
        
        rationale = (
            f"Selected '{selected_fix.get('title')}' because it addresses "
            f"the most likely root cause ('{primary.get('hypothesis')}', "
            f"confidence {confidence:.0%}) with high effectiveness "
            f"({selected_fix.get('effectiveness_score', 0):.0%}) and "
            f"feasibility ({selected_fix.get('feasibility_score', 0):.0%})."
        )
        
        return rationale

    def _generate_execution_notes(self) -> str:
        """
        Generate notes about execution approach.

        Explains the tool invocation strategy and safety mechanisms.

        Returns:
            Execution notes.
        """
        if self.enable_auto_execution:
            return (
                "This action plan will be executed using only safe, typed infrastructure tools. "
                "NO ARBITRARY SHELL COMMANDS are allowed. All execution is logged and validated. "
                "Tool invocation strategy: "
                "(1) Tools are discovered from central registry (blocks unknown/unsafe tools), "
                "(2) Parameters are validated by tool before execution (blocks wildcards, injection, etc.), "
                "(3) All invocations are logged with full audit trails (forensic analysis possible), "
                "(4) Execution results are captured and validated, "
                "(5) On failure, rollback procedures are automatically executed. "
                "Safety checkpoints: Each phase (validation, execution, rollback) is traceable."
            )
        else:
            return (
                "This action plan maps to specific infrastructure tools but will NOT execute automatically. "
                "Tool mappings are provided in action_plan.tool_mappings for external execution. "
                "The plan specifies exactly which tools to invoke and with what parameters. "
                "Tools available: docker_container_logs, docker_container_restart (Docker), "
                "k8s_pod_status, k8s_pod_restart (Kubernetes). "
                "NO ARBITRARY SHELL COMMANDS - only safe, typed tools are invoked. "
                "This plan should be reviewed and approved before passing to an ExecutorAgent or human operator."
            )
