"""
Reflection and self-correction module for Sentinel workflow.

Self-Correcting Agent Architecture
===================================

Sentinel agents are designed to be self-correcting through a reflection loop:

    Execution → Validation → Reflection → Decision
       ↑                                      ↓
       └──────── Retry or Escalate ←────────┘

Instead of following a linear workflow, the system can loop back to
re-investigate if initial remediation fails. This creates a cycle of:
1. Execute a fix
2. Measure results
3. Reflect on what went wrong
4. Decide: retry with adjustment or escalate to human review

Benefits:
- Resilience: Failed fixes trigger automatic retry attempts
- Learning: Each failure provides data to improve future recommendations
- Adaptation: Agent can suggest different fixes based on feedback
- Transparency: Each retry adds to audit trail
- Safety: After N failures, escalates to human rather than looping infinitely
"""

from typing import Any, Dict, List, Tuple
from datetime import datetime
from enum import Enum


class ReflectionOutcome(Enum):
    """Possible outcomes of reflection analysis."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"


class RetryStrategy(Enum):
    """Strategy for handling failed remediation."""
    IMMEDIATE_RETRY = "immediate_retry"
    ADJUSTED_RETRY = "adjusted_retry"
    ALTERNATIVE_FIX = "alternative_fix"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    ABORT = "abort"


def reflect_on_execution(state: "SentinelState") -> Tuple[ReflectionOutcome, Dict[str, Any]]:
    """
    Reflect on whether remediation was successful.

    This function performs post-execution analysis by comparing actual
    results against expected success criteria. It's the decision point
    for whether to consider the incident resolved, retry, or escalate.

    Self-Correction Logic:
    ----------------------
    If execution was unsuccessful, reflection answers critical questions:
    - Why did it fail? (root cause of failure)
    - Should we retry? (can success be achieved with adjustments?)
    - What should we try differently? (which alternative makes sense?)
    - Do we need human help? (has complexity exceeded automation?)

    The reflection mechanism enables the system to:
    1. Learn from failures without human intervention
    2. Attempt multiple strategies automatically
    3. Provide detailed diagnostic info when escalating
    4. Build an audit trail of what was attempted

    Args:
        state: The current SentinelState containing execution results

    Returns:
        Tuple of:
        - ReflectionOutcome: success, partial_success, failure, or inconclusive
        - Dict containing:
            - "analysis": str - what happened
            - "validation_results": List[Dict] - each success criterion and its status
            - "failure_reasons": List[str] - why it failed (if applicable)
            - "contributing_factors": List[str] - conditions that prevented success
            - "confidence_score": float - how confident in the analysis (0-1)
    """

    # Check if execution results exist
    if not state.execution_result:
        return ReflectionOutcome.INCONCLUSIVE, {
            "analysis": "No execution results found - cannot reflect",
            "validation_results": [],
            "failure_reasons": ["Execution never occurred"],
            "confidence_score": 0.0
        }

    execution_status = state.execution_result.get("status", "unknown")
    
    # If execution reported success, validate it further
    if execution_status == "success":
        validation_results = _validate_fix(state)
        
        # Check if all validations passed
        all_passed = all(v.get("passed", False) for v in validation_results)
        
        if all_passed:
            return ReflectionOutcome.SUCCESS, {
                "analysis": "Remediation executed successfully and validation passed",
                "validation_results": validation_results,
                "failure_reasons": [],
                "confidence_score": 0.95
            }
        else:
            # Execution succeeded but validation found issues
            failed_validations = [v for v in validation_results if not v.get("passed")]
            return ReflectionOutcome.PARTIAL_SUCCESS, {
                "analysis": "Execution reported success but post-execution validation found issues",
                "validation_results": validation_results,
                "failure_reasons": [v.get("failure_reason", "") for v in failed_validations],
                "confidence_score": 0.6
            }
    
    elif execution_status == "partial":
        validation_results = _validate_fix(state)
        failed_validations = [v for v in validation_results if not v.get("passed")]
        
        return ReflectionOutcome.PARTIAL_SUCCESS, {
            "analysis": "Remediation partially executed. Some steps failed or were incomplete.",
            "validation_results": validation_results,
            "failure_reasons": [v.get("failure_reason", "") for v in failed_validations],
            "confidence_score": 0.5
        }
    
    else:  # execution_status == "failed" or other
        execution_errors = state.execution_result.get("errors", [])
        
        return ReflectionOutcome.FAILURE, {
            "analysis": "Remediation execution failed",
            "validation_results": [],
            "failure_reasons": execution_errors or ["Execution encountered unrecoverable error"],
            "confidence_score": 0.8
        }


def decide_retry_strategy(
    reflection_outcome: ReflectionOutcome,
    reflection_data: Dict[str, Any],
    state: "SentinelState",
    max_retry_count: int = 3
) -> Tuple[RetryStrategy, Dict[str, Any]]:
    """
    Decide whether and how to retry remediation.

    This function uses reflection results to make strategic decisions:
    - If likely to succeed with retry: IMMEDIATE_RETRY
    - If needs adjustment: ADJUSTED_RETRY
    - If needs different approach: ALTERNATIVE_FIX
    - If beyond automation capability: ESCALATE_TO_HUMAN
    - If too risky or too many attempts: ABORT

    Self-Correction Decision Tree:
    ==============================
    1. Success? → Done, no retry needed
    2. Too many retries already? → Escalate (prevent infinite loops)
    3. Partial success? → Try alternative fix (current approach partially works)
    4. Clear root cause found? → Adjusted retry (apply fix to root cause)
    5. Unknown failure? → Escalate (insufficient data to continue)
    6. Resource exhaustion? → Abort (system at limits)

    Args:
        reflection_outcome: Result from reflect_on_execution()
        reflection_data: Analysis data from reflect_on_execution()
        state: The current SentinelState
        max_retry_count: Maximum attempts before escalating

    Returns:
        Tuple of:
        - RetryStrategy: action to take
        - Dict containing:
            - "recommendation": str - explanation of strategy
            - "next_steps": List[str] - what to do next
            - "estimated_success_probability": float - confidence in strategy (0-1)
    """

    # Count how many times we've already retried
    retry_count = state.metadata.get("retry_count", 0)

    # Success path: no retry needed
    if reflection_outcome == ReflectionOutcome.SUCCESS:
        return RetryStrategy.IMMEDIATE_RETRY, {
            "recommendation": "No retry needed - incident resolved",
            "next_steps": ["Close incident", "Document lessons learned"],
            "estimated_success_probability": 1.0
        }

    # Too many retries: escalate to prevent loops
    if retry_count >= max_retry_count:
        return RetryStrategy.ESCALATE_TO_HUMAN, {
            "recommendation": f"Reached maximum retry attempts ({max_retry_count}). Escalating to human review.",
            "next_steps": [
                "Create escalation ticket",
                "Provide human with complete analysis",
                "Request manual intervention"
            ],
            "estimated_success_probability": 0.0
        }

    # Analyze failure reasons to decide retry approach
    failure_reasons = reflection_data.get("failure_reasons", [])
    confidence = reflection_data.get("confidence_score", 0.5)

    # Partial success: try alternative fix
    if reflection_outcome == ReflectionOutcome.PARTIAL_SUCCESS:
        return RetryStrategy.ALTERNATIVE_FIX, {
            "recommendation": "Current fix partially addressed the issue. Try alternative approach.",
            "next_steps": [
                "Review alternative fixes from researcher",
                "Select fix addressing remaining issues",
                "Re-execute with new approach"
            ],
            "estimated_success_probability": 0.65
        }

    # Clear failure with identifiable reason: retry with adjustment
    if reflection_outcome == ReflectionOutcome.FAILURE and confidence > 0.8:
        adjustment_suggestion = _analyze_failure_for_adjustment(
            failure_reasons, state
        )
        
        return RetryStrategy.ADJUSTED_RETRY, {
            "recommendation": f"Failure reason identified: {failure_reasons[0] if failure_reasons else 'unknown'}. Adjusting approach.",
            "next_steps": [
                f"Adjustment: {adjustment_suggestion}",
                "Re-run remediation with adjustment",
                "Validate results"
            ],
            "estimated_success_probability": 0.6
        }

    # Inconclusive or unclear failure: need more investigation
    if reflection_outcome == ReflectionOutcome.INCONCLUSIVE:
        return RetryStrategy.ESCALATE_TO_HUMAN, {
            "recommendation": "Cannot determine if remediation succeeded. Need human investigation.",
            "next_steps": [
                "Review execution logs manually",
                "Check system state directly",
                "Request human investigation"
            ],
            "estimated_success_probability": 0.3
        }

    # Default: escalate uncertain failures
    return RetryStrategy.ESCALATE_TO_HUMAN, {
        "recommendation": "Remediation failed with unclear cause. Escalating for investigation.",
        "next_steps": [
            "Provide human with full context",
            "Include execution logs and errors",
            "Request investigation and guidance"
        ],
        "estimated_success_probability": 0.2
    }


def update_state_with_reflection(
    state: "SentinelState",
    reflection_outcome: ReflectionOutcome,
    reflection_data: Dict[str, Any],
    retry_strategy: RetryStrategy,
    retry_decision: Dict[str, Any]
) -> "SentinelState":
    """
    Update the state based on reflection and retry decision.

    This modifies the state to record what was learned and what action
    to take next. It enables the workflow to loop back for re-investigation
    or escalation as needed.

    Args:
        state: The current SentinelState
        reflection_outcome: Result from reflect_on_execution()
        reflection_data: Analysis data from reflect_on_execution()
        retry_strategy: Decision from decide_retry_strategy()
        retry_decision: Details from decide_retry_strategy()

    Returns:
        Updated SentinelState with reflection results recorded
    """

    # Record this reflection cycle
    reflection_record = {
        "cycle": state.metadata.get("reflection_cycle", 0) + 1,
        "timestamp": datetime.now().isoformat(),
        "outcome": reflection_outcome.value,
        "analysis": reflection_data.get("analysis", ""),
        "failure_reasons": reflection_data.get("failure_reasons", []),
        "retry_strategy": retry_strategy.value,
        "recommendation": retry_decision.get("recommendation", "")
    }

    # Add to reflection history
    if "reflection_history" not in state.metadata:
        state.metadata["reflection_history"] = []
    state.metadata["reflection_history"].append(reflection_record)

    # Update retry count
    if retry_strategy in [
        RetryStrategy.IMMEDIATE_RETRY,
        RetryStrategy.ADJUSTED_RETRY,
        RetryStrategy.ALTERNATIVE_FIX
    ]:
        state.metadata["retry_count"] = state.metadata.get("retry_count", 0) + 1

    # Update status based on strategy
    if retry_strategy == RetryStrategy.IMMEDIATE_RETRY:
        state.status = "completed"
        state.error = ""
    elif retry_strategy in [RetryStrategy.ADJUSTED_RETRY, RetryStrategy.ALTERNATIVE_FIX]:
        # Trigger re-investigation cycle
        state.status = "discovery"  # Go back to start of pipeline
        state.metadata["needs_reinvestigation"] = True
    elif retry_strategy == RetryStrategy.ESCALATE_TO_HUMAN:
        state.status = "pending_approval"
        state.metadata["needs_escalation"] = True
        state.error = f"Escalation needed: {retry_decision.get('recommendation', '')}"
    elif retry_strategy == RetryStrategy.ABORT:
        state.status = "failed"
        state.error = "Remediation aborted after analysis"

    # Store reflection data for downstream use
    state.metadata["latest_reflection"] = {
        "outcome": reflection_outcome.value,
        "analysis": reflection_data.get("analysis", ""),
        "validation_results": reflection_data.get("validation_results", []),
        "next_steps": retry_decision.get("next_steps", [])
    }

    # Add to conversation history
    state.add_message(
        role="agent",
        agent_name="reflection_agent",
        content=f"Reflection: {reflection_outcome.value} - {retry_decision.get('recommendation', '')}"
    )

    return state


def _validate_fix(state: "SentinelState") -> List[Dict[str, Any]]:
    """
    Validate fix by checking success criteria from action plan.

    Args:
        state: The current state

    Returns:
        List of validation results, each with:
        - "validation": str - what was checked
        - "passed": bool - did it pass?
        - "expected": str - what was expected
        - "actual": str - what was found
        - "failure_reason": str - if failed, why?
    """

    validation_steps = (
        state.operator_action_plan.get("action_plan", {})
        .get("validation_steps", [])
    )
    
    execution_errors = state.execution_result.get("errors", []) if state.execution_result else []
    execution_log = state.execution_result.get("execution_log", "") if state.execution_result else ""

    results = []

    for step in validation_steps:
        validation_name = step.get("validation", "")
        expected = step.get("expected_result", "")

        # Mock validation based on execution results
        if execution_errors:
            results.append({
                "validation": validation_name,
                "passed": False,
                "expected": expected,
                "actual": f"Errors occurred: {execution_errors[0]}",
                "failure_reason": "Execution had errors"
            })
        elif "error" in execution_log.lower():
            results.append({
                "validation": validation_name,
                "passed": False,
                "expected": expected,
                "actual": "Errors detected in logs",
                "failure_reason": "Log shows errors"
            })
        else:
            results.append({
                "validation": validation_name,
                "passed": True,
                "expected": expected,
                "actual": expected,
                "failure_reason": ""
            })

    return results


def _analyze_failure_for_adjustment(
    failure_reasons: List[str],
    state: "SentinelState"
) -> str:
    """
    Analyze failure reasons and suggest an adjustment.

    Args:
        failure_reasons: List of identified failure reasons
        state: Current state

    Returns:
        Suggested adjustment string
    """

    if not failure_reasons:
        return "Increase resource limits and retry"

    reason = failure_reasons[0].lower()

    # Map common failure patterns to adjustments
    adjustments = {
        "timeout": "Increase timeout values and retry",
        "connection": "Check connectivity and retry",
        "permission": "Verify credentials and permissions, then retry",
        "resource": "Free up resources or increase limits, then retry",
        "already": "Skip already-applied steps and continue",
        "not found": "Verify resource exists before attempting fix",
        "conflict": "Resolve conflicts and retry with conflict resolution enabled"
    }

    for pattern, adjustment in adjustments.items():
        if pattern in reason:
            return adjustment

    return "Review logs, adjust approach, and retry"
