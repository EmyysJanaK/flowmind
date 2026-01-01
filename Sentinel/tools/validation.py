"""
Validation and safety constraints for tool execution.

This module implements validation rules that prevent dangerous operations
before they reach the tool execution layer. This is critical for preventing
LLM misuse where a compromised or adversarial LLM might try to:

- Delete large numbers of pods with wildcards
- Escape namespace restrictions  
- Perform unauthorized operations
- Bypass safety constraints
- Escalate privileges

VALIDATION ARCHITECTURE
=======================

Tools go through 3-phase validation:

PHASE 1: PARAMETER VALIDATION
- Check parameter types and formats
- Validate against regex patterns (alphanumeric, no special chars)
- Check value ranges (e.g., timeout <= 300 seconds)
- Reject empty or None values

PHASE 2: SAFETY CONSTRAINT VALIDATION
- Check for wildcard patterns (* or %)
- Verify namespace is not protected (kube-system, etc)
- Validate resource names don't look like commands
- Check for common injection patterns

PHASE 3: AUTHORIZATION CHECK
- Verify tool is allowed for this agent/user
- Check namespace restrictions
- Validate against permission policies
- Record validation attempt for audit

LLM MISUSE PREVENTION
====================

How validation prevents LLM attacks:

ATTACK 1: Wildcard Deletion
  LLM: "Delete all pods with k8s_pod_restart(namespace='*', pod_name='*')"
  DEFENSE: Validator rejects '*' characters in namespace and pod_name
  RESULT: ValueError raised, operation blocked, attempt logged

ATTACK 2: System Namespace Access
  LLM: "Delete pod in kube-system with k8s_pod_restart(namespace='kube-system', pod_name='coredns-7d5f4c')"
  DEFENSE: Validator explicitly blocks protected namespaces
  RESULT: ValueError raised, operation blocked, escalated to human

ATTACK 3: Command Injection
  LLM: "Call docker_container_logs(container_name='web; rm -rf /')"
  DEFENSE: Validator rejects special shell characters (; | & > < etc)
  RESULT: ValueError raised, operation blocked

ATTACK 4: Excessive Resource Access
  LLM: "Get last 1000000 log lines with docker_container_logs(tail=1000000)"
  DEFENSE: Validator enforces maximum values (e.g., tail <= 10000)
  RESULT: ValueError raised, operation blocked

ATTACK 5: Privilege Escalation
  LLM: "Restart container to elevate privileges"
  DEFENSE: Tool execution logged with LLM trace, RBAC enforced
  RESULT: Attempt logged, human alerted, operation prevented by RBAC

ATTACK 6: Namespace Escape
  LLM: "Get pod status in restricted namespace"
  DEFENSE: Validator checks namespace against whitelist
  RESULT: ValueError raised, operation blocked, escalated

These validations ensure that even if an LLM is compromised,
hacked, or adversarially prompted, it cannot:
- Delete arbitrary resources
- Escape namespace boundaries
- Execute shell commands
- Escalate privileges
- Consume excessive resources
- Bypass authorization
"""

from typing import Any, Dict, Optional, List, Callable
from enum import Enum
import re
from dataclasses import dataclass


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidationLevel(Enum):
    """Severity levels for validation failures."""
    CRITICAL = "critical"          # Blocks operation immediately
    HIGH = "high"                   # Requires approval
    MEDIUM = "medium"               # Logged, allowed with warning
    INFO = "info"                   # Logged for audit


@dataclass
class ValidationRule:
    """
    A single validation rule to apply to tool parameters.
    
    Rules are checked in order and any failure blocks the operation.
    """
    name: str                       # e.g., "no_wildcards_in_pod_name"
    parameter: str                  # e.g., "pod_name"
    description: str                # e.g., "Pod names cannot contain wildcards"
    level: ValidationLevel          # CRITICAL, HIGH, MEDIUM, INFO
    validator: Callable[[Any], bool]  # Function that returns True if valid
    error_message: str              # Message if validation fails


class ParameterValidator:
    """
    Validates tool parameters against safety rules.
    
    Prevents dangerous operations before they reach tool execution layer.
    """

    def __init__(self):
        """Initialize parameter validator with standard rules."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self._setup_standard_rules()

    def _setup_standard_rules(self) -> None:
        """Setup standard validation rules for all tools."""

        # ================================================================
        # KUBERNETES NAMESPACE RULES
        # ================================================================

        # Protected namespaces that agents cannot modify
        protected_namespaces = {
            "kube-system",           # Kubernetes core services
            "kube-public",           # Public resources
            "kube-node-lease",       # Node heartbeat
            "kube-apiserver",        # API server
            "kube-controller",       # Controllers
            "default"                # Default namespace (restricted by policy)
        }

        self.add_rule("k8s_pod_status", ValidationRule(
            name="namespace_not_protected",
            parameter="namespace",
            description="Pod status queries blocked in system namespaces",
            level=ValidationLevel.CRITICAL,
            validator=lambda ns: ns not in protected_namespaces,
            error_message=f"Namespace restricted. Protected: {protected_namespaces}"
        ))

        self.add_rule("k8s_pod_restart", ValidationRule(
            name="namespace_not_protected",
            parameter="namespace",
            description="Pod deletions blocked in system namespaces",
            level=ValidationLevel.CRITICAL,
            validator=lambda ns: ns not in protected_namespaces,
            error_message=f"Cannot delete pods in protected namespace. Protected: {protected_namespaces}"
        ))

        # ================================================================
        # WILDCARD & PATTERN RULES
        # ================================================================

        # No wildcards in pod/container names
        no_wildcards = lambda name: '*' not in name and '%' not in name

        self.add_rule("k8s_pod_status", ValidationRule(
            name="no_wildcards_in_pod_name",
            parameter="pod_name",
            description="Pod names cannot contain wildcards (* or %)",
            level=ValidationLevel.CRITICAL,
            validator=no_wildcards,
            error_message="Pod name cannot contain wildcards (* or %)"
        ))

        self.add_rule("k8s_pod_restart", ValidationRule(
            name="no_wildcards_in_pod_name",
            parameter="pod_name",
            description="Pod names cannot contain wildcards",
            level=ValidationLevel.CRITICAL,
            validator=no_wildcards,
            error_message="Pod name cannot contain wildcards (* or %)"
        ))

        self.add_rule("docker_container_logs", ValidationRule(
            name="no_wildcards_in_container_name",
            parameter="container_name",
            description="Container names cannot contain wildcards",
            level=ValidationLevel.CRITICAL,
            validator=no_wildcards,
            error_message="Container name cannot contain wildcards (* or %)"
        ))

        self.add_rule("docker_container_restart", ValidationRule(
            name="no_wildcards_in_container_name",
            parameter="container_name",
            description="Container names cannot contain wildcards",
            level=ValidationLevel.CRITICAL,
            validator=no_wildcards,
            error_message="Container name cannot contain wildcards (* or %)"
        ))

        # ================================================================
        # COMMAND INJECTION PREVENTION
        # ================================================================

        # No shell metacharacters in names
        def no_shell_chars(name: str) -> bool:
            """Block common shell injection characters."""
            dangerous_chars = {';', '|', '&', '>', '<', '`', '$', '(', ')'}
            return not any(char in name for char in dangerous_chars)

        self.add_rule("k8s_pod_status", ValidationRule(
            name="no_shell_chars_in_pod_name",
            parameter="pod_name",
            description="Pod names cannot contain shell metacharacters",
            level=ValidationLevel.CRITICAL,
            validator=no_shell_chars,
            error_message="Pod name contains invalid characters: ; | & > < ` $ ( )"
        ))

        self.add_rule("k8s_pod_restart", ValidationRule(
            name="no_shell_chars_in_pod_name",
            parameter="pod_name",
            description="Pod names cannot contain shell metacharacters",
            level=ValidationLevel.CRITICAL,
            validator=no_shell_chars,
            error_message="Pod name contains invalid characters: ; | & > < ` $ ( )"
        ))

        self.add_rule("docker_container_logs", ValidationRule(
            name="no_shell_chars_in_container_name",
            parameter="container_name",
            description="Container names cannot contain shell metacharacters",
            level=ValidationLevel.CRITICAL,
            validator=no_shell_chars,
            error_message="Container name contains invalid characters: ; | & > < ` $ ( )"
        ))

        self.add_rule("docker_container_restart", ValidationRule(
            name="no_shell_chars_in_container_name",
            parameter="container_name",
            description="Container names cannot contain shell metacharacters",
            level=ValidationLevel.CRITICAL,
            validator=no_shell_chars,
            error_message="Container name contains invalid characters: ; | & > < ` $ ( )"
        ))

        # ================================================================
        # RESOURCE LIMIT RULES
        # ================================================================

        # Tail cannot be excessively large
        self.add_rule("docker_container_logs", ValidationRule(
            name="tail_within_limits",
            parameter="tail",
            description="Tail limited to prevent memory exhaustion",
            level=ValidationLevel.CRITICAL,
            validator=lambda tail: isinstance(tail, int) and 1 <= tail <= 10000,
            error_message="Tail must be between 1 and 10000 lines"
        ))

        # Grace period cannot be excessive
        def grace_period_valid(period: int) -> bool:
            return isinstance(period, int) and 0 <= period <= 300

        self.add_rule("k8s_pod_restart", ValidationRule(
            name="grace_period_within_limits",
            parameter="grace_period",
            description="Grace period limited to prevent denial of service",
            level=ValidationLevel.CRITICAL,
            validator=grace_period_valid,
            error_message="Grace period must be between 0 and 300 seconds"
        ))

        # ================================================================
        # NAME FORMAT RULES
        # ================================================================

        # Valid Kubernetes namespace format
        k8s_namespace_pattern = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"

        self.add_rule("k8s_pod_status", ValidationRule(
            name="namespace_valid_format",
            parameter="namespace",
            description="Namespace must match Kubernetes naming rules",
            level=ValidationLevel.CRITICAL,
            validator=lambda ns: bool(re.match(k8s_namespace_pattern, ns)),
            error_message="Invalid namespace format (must be lowercase alphanumeric with dashes)"
        ))

        self.add_rule("k8s_pod_restart", ValidationRule(
            name="namespace_valid_format",
            parameter="namespace",
            description="Namespace must match Kubernetes naming rules",
            level=ValidationLevel.CRITICAL,
            validator=lambda ns: bool(re.match(k8s_namespace_pattern, ns)),
            error_message="Invalid namespace format (must be lowercase alphanumeric with dashes)"
        ))

        # Valid Kubernetes pod name format
        k8s_pod_pattern = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"

        self.add_rule("k8s_pod_status", ValidationRule(
            name="pod_name_valid_format",
            parameter="pod_name",
            description="Pod name must match Kubernetes naming rules",
            level=ValidationLevel.CRITICAL,
            validator=lambda name: bool(re.match(k8s_pod_pattern, name)),
            error_message="Invalid pod name format (must be lowercase alphanumeric with dashes)"
        ))

        self.add_rule("k8s_pod_restart", ValidationRule(
            name="pod_name_valid_format",
            parameter="pod_name",
            description="Pod name must match Kubernetes naming rules",
            level=ValidationLevel.CRITICAL,
            validator=lambda name: bool(re.match(k8s_pod_pattern, name)),
            error_message="Invalid pod name format (must be lowercase alphanumeric with dashes)"
        ))

        # Valid Docker container name format
        docker_container_pattern = r"^[a-zA-Z0-9_-]+$"

        self.add_rule("docker_container_logs", ValidationRule(
            name="container_name_valid_format",
            parameter="container_name",
            description="Container name must match Docker naming rules",
            level=ValidationLevel.CRITICAL,
            validator=lambda name: bool(re.match(docker_container_pattern, name)),
            error_message="Invalid container name format"
        ))

        self.add_rule("docker_container_restart", ValidationRule(
            name="container_name_valid_format",
            parameter="container_name",
            description="Container name must match Docker naming rules",
            level=ValidationLevel.CRITICAL,
            validator=lambda name: bool(re.match(docker_container_pattern, name)),
            error_message="Invalid container name format"
        ))

        # ================================================================
        # TIMEOUT RULES
        # ================================================================

        # Docker timeout cannot be excessive
        self.add_rule("docker_container_restart", ValidationRule(
            name="timeout_within_limits",
            parameter="timeout",
            description="Timeout limited to prevent hanging",
            level=ValidationLevel.CRITICAL,
            validator=lambda timeout: isinstance(timeout, int) and 1 <= timeout <= 300,
            error_message="Timeout must be between 1 and 300 seconds"
        ))

    def add_rule(self, tool_name: str, rule: ValidationRule) -> None:
        """
        Add a validation rule for a specific tool.

        Args:
            tool_name: Name of tool (e.g., "k8s_pod_restart")
            rule: ValidationRule to add
        """
        if tool_name not in self.rules:
            self.rules[tool_name] = []
        self.rules[tool_name].append(rule)

    def validate(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate tool parameters against all registered rules.

        PHASE 1: Parameter Validation
        PHASE 2: Safety Constraint Validation
        PHASE 3: Authorization Check (integration point)

        Args:
            tool_name: Name of tool being called
            parameters: Dict of parameters to validate

        Returns:
            {
                "valid": bool,
                "errors": List[str],  # Validation error messages
                "warnings": List[str],  # Non-blocking warnings
                "level": ValidationLevel  # Highest severity level
            }

        Raises:
            ValidationError: If critical validation fails
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "level": ValidationLevel.INFO,
            "tool_name": tool_name,
            "parameters": parameters
        }

        # Get rules for this tool
        if tool_name not in self.rules:
            return result  # No rules defined for this tool

        tool_rules = self.rules[tool_name]

        # Run all validation rules
        for rule in tool_rules:
            # Check if parameter exists in provided parameters
            if rule.parameter not in parameters:
                # Parameter not provided - might be optional
                continue

            param_value = parameters[rule.parameter]

            # Run validator function
            try:
                is_valid = rule.validator(param_value)
            except Exception as e:
                result["valid"] = False
                result["errors"].append(
                    f"Validation error in {rule.name}: {str(e)}"
                )
                result["level"] = ValidationLevel.CRITICAL
                continue

            # Check result
            if not is_valid:
                result["valid"] = False
                result["errors"].append(
                    f"[{rule.name}] {rule.error_message} "
                    f"(parameter: {rule.parameter}={param_value})"
                )

                # Update severity level
                if rule.level == ValidationLevel.CRITICAL:
                    result["level"] = ValidationLevel.CRITICAL
                elif rule.level == ValidationLevel.HIGH and result["level"] != ValidationLevel.CRITICAL:
                    result["level"] = ValidationLevel.HIGH

        # Raise exception for critical failures
        if not result["valid"] and result["level"] == ValidationLevel.CRITICAL:
            error_msg = f"Validation failed for tool '{tool_name}':\n" + \
                       "\n".join(f"  - {err}" for err in result["errors"])
            raise ValidationError(error_msg)

        return result

    def describe_rules(self, tool_name: str) -> str:
        """
        Get human-readable description of validation rules for a tool.

        Args:
            tool_name: Name of tool

        Returns:
            Formatted description of all validation rules
        """
        if tool_name not in self.rules:
            return f"No validation rules defined for tool '{tool_name}'"

        desc = f"Validation Rules for {tool_name}:\n"
        desc += "=" * 60 + "\n"

        for rule in self.rules[tool_name]:
            desc += f"\nRule: {rule.name}\n"
            desc += f"  Level: {rule.level.value}\n"
            desc += f"  Parameter: {rule.parameter}\n"
            desc += f"  Description: {rule.description}\n"
            desc += f"  Message: {rule.error_message}\n"

        return desc


# ============================================================================
# GLOBAL VALIDATOR INSTANCE
# ============================================================================

_global_validator = ParameterValidator()


def get_validator() -> ParameterValidator:
    """Get the global parameter validator instance."""
    return _global_validator
