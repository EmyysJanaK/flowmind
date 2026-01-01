"""
Audit logging for tool execution.

This module logs all tool invocations for:
- Security auditing: Who called what tool and when?
- Compliance: Evidence that operations were authorized and logged
- Forensics: Understanding what happened if something goes wrong
- Threat detection: Identifying suspicious patterns
- LLM monitoring: Tracking LLM tool usage and potential misuse

AUDIT LOG ARCHITECTURE
======================

Every tool invocation goes through these logging phases:

PHASE 1: PRE-EXECUTION LOGGING
- Record tool name and parameters
- Record validation results
- Record agent/user context
- Record timestamp and request ID
- Record LLM context (if available)

PHASE 2: EXECUTION LOGGING
- Record execution start time
- Record any system calls or operations
- Record authorization checks
- Log permission enforcement

PHASE 3: POST-EXECUTION LOGGING
- Record execution result (success/failure)
- Record duration and resource usage
- Record error details (if any)
- Record data returned

LOGS INCLUDE:
- Timestamp (ISO format)
- Tool name
- Parameters (with sensitive data masked)
- Agent/User identifier
- LLM model and context
- Request ID (for tracing)
- Validation status
- Execution status
- Duration
- Error details
- Authorization checks

PREVENTING LLM MISUSE THROUGH LOGGING
=====================================

How audit logs prevent LLM attacks:

1. DETECTION
   - Logs show which tools LLM tried to call
   - Logs show parameter values (attempts to inject)
   - Logs show validation failures
   - Logs show success/failure rate
   - Can detect patterns of repeated failures

2. ALERT TRIGGERS
   - Multiple validation failures in short time
   - Suspicious parameter values
   - Accessing restricted namespaces
   - High volume of tool calls
   - Tool calls at unusual times

3. FORENSIC ANALYSIS
   - Replay exactly what LLM did
   - Understand attack progression
   - Identify compromised LLM models
   - Trace data leakage paths
   - Verify compliance

4. PROOF OF SAFETY
   - Logs prove dangerous operations were blocked
   - Logs prove validation worked
   - Logs prove RBAC was enforced
   - Evidence for security audits
   - Compliance certification

Example Detection Scenarios:

ATTACK: LLM tries multiple wildcard deletions
  Log entries show:
  - tool=k8s_pod_restart, pod_name="*", validation=FAILED
  - tool=k8s_pod_restart, pod_name="%.pods", validation=FAILED
  - tool=k8s_pod_restart, pod_name="*; rm -rf /", validation=FAILED
  ALERT: Possible compromised LLM attempting injection

ATTACK: LLM tries to access restricted namespace
  Log entries show:
  - tool=k8s_pod_status, namespace="kube-system", validation=FAILED
  - tool=k8s_pod_status, namespace="kube-public", validation=FAILED
  - tool=k8s_pod_status, namespace="kube-node-lease", validation=FAILED
  ALERT: LLM attempting unauthorized access

ATTACK: LLM causes resource exhaustion
  Log entries show:
  - tool=docker_container_logs, tail=1000000, validation=FAILED
  - tool=docker_container_logs, tail=999999, validation=FAILED
  - tool=docker_container_logs, tail=500000, validation=FAILED
  ALERT: LLM attempting denial of service
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import uuid
from pathlib import Path


class AuditLevel(Enum):
    """Audit log severity levels."""
    CRITICAL = "CRITICAL"          # Security-critical event
    WARNING = "WARNING"            # Suspicious or blocked operation
    INFO = "INFO"                  # Normal operation
    DEBUG = "DEBUG"                # Detailed debugging info


class AuditAction(Enum):
    """Types of actions logged."""
    TOOL_CALL_REQUESTED = "tool_call_requested"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    AUTHORIZATION_CHECK = "authorization_check"
    AUTHORIZATION_PASSED = "authorization_passed"
    AUTHORIZATION_FAILED = "authorization_failed"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"


@dataclass
class AuditLogEntry:
    """
    Single audit log entry for a tool invocation.
    
    Records all details needed for security auditing and forensics.
    """
    timestamp: str                          # ISO format timestamp
    request_id: str                         # Unique request ID for tracing
    action: AuditAction                     # Type of action
    level: AuditLevel                       # Severity level
    tool_name: str                          # Tool being called
    agent_name: Optional[str] = None        # Agent calling tool
    llm_model: Optional[str] = None         # LLM model (if applicable)
    parameters: Optional[Dict[str, Any]] = None  # Tool parameters (sensitive data masked)
    validation_result: Optional[Dict] = None    # Validation result
    authorization_result: Optional[Dict] = None  # Authorization result
    execution_result: Optional[Dict] = None  # Execution result
    duration_ms: Optional[float] = None     # Execution duration
    error: Optional[str] = None             # Error message (if failed)
    error_type: Optional[str] = None        # Error category
    details: Optional[str] = None           # Additional details

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        data['action'] = data['action'].value
        data['level'] = data['level'].value
        return json.dumps(data)


class AuditLogger:
    """
    Centralized audit logging for all tool invocations.
    
    Logs are written to:
    - Console (INFO level and above)
    - File (all levels)
    - Memory (recent entries)
    - External systems (SIEM, logs aggregation)
    """

    def __init__(self, log_file: Optional[str] = None, max_memory_entries: int = 10000):
        """
        Initialize audit logger.

        Args:
            log_file: Path to log file (default: sentinel_audit.log)
            max_memory_entries: Max entries to keep in memory (default: 10000)
        """
        self.log_file = log_file or "sentinel_audit.log"
        self.max_memory_entries = max_memory_entries
        self.memory_log: List[AuditLogEntry] = []
        self.request_id_stack: List[str] = []

    def get_request_id(self) -> str:
        """Get current request ID for tracing."""
        if self.request_id_stack:
            return self.request_id_stack[-1]
        return str(uuid.uuid4())

    def push_request_context(self, request_id: Optional[str] = None) -> str:
        """
        Push a request context for nested operations.

        Args:
            request_id: Request ID (generated if not provided)

        Returns:
            The request ID
        """
        request_id = request_id or str(uuid.uuid4())
        self.request_id_stack.append(request_id)
        return request_id

    def pop_request_context(self) -> Optional[str]:
        """Pop a request context."""
        if self.request_id_stack:
            return self.request_id_stack.pop()
        return None

    def log(
        self,
        action: AuditAction,
        level: AuditLevel,
        tool_name: str,
        agent_name: Optional[str] = None,
        llm_model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict] = None,
        authorization_result: Optional[Dict] = None,
        execution_result: Optional[Dict] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Log a tool invocation event.

        Args:
            action: Type of action (e.g., TOOL_CALL_REQUESTED)
            level: Severity level
            tool_name: Name of tool
            agent_name: Name of agent calling tool
            llm_model: LLM model (if called by LLM)
            parameters: Tool parameters (will be masked)
            validation_result: Result of validation
            authorization_result: Result of authorization check
            execution_result: Result of execution
            duration_ms: Execution duration
            error: Error message if failed
            error_type: Category of error
            details: Additional details

        Returns:
            The audit log entry that was created
        """
        # Create log entry
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            request_id=self.get_request_id(),
            action=action,
            level=level,
            tool_name=tool_name,
            agent_name=agent_name,
            llm_model=llm_model,
            parameters=self._mask_sensitive_data(parameters),
            validation_result=validation_result,
            authorization_result=authorization_result,
            execution_result=self._mask_sensitive_data(execution_result),
            duration_ms=duration_ms,
            error=error,
            error_type=error_type,
            details=details
        )

        # Store in memory
        self.memory_log.append(entry)
        if len(self.memory_log) > self.max_memory_entries:
            self.memory_log.pop(0)

        # Write to file
        self._write_to_file(entry)

        # Log to console for important events
        self._log_to_console(entry)

        return entry

    def _mask_sensitive_data(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Mask sensitive data in logged parameters/results.

        Hides actual values for sensitive fields while keeping structure:
        - Passwords: "***MASKED***"
        - API keys: "key_***...***"
        - Tokens: "token_***...***"
        - Credentials: "***MASKED***"

        Args:
            data: Dictionary to mask

        Returns:
            Masked dictionary or None
        """
        if not data:
            return data

        masked = {}
        sensitive_keys = {
            "password", "passwd", "pwd",
            "api_key", "apikey", "secret",
            "token", "auth", "authorization",
            "credential", "key", "private_key"
        }

        for key, value in data.items():
            key_lower = key.lower()

            # Check if this is a sensitive field
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 10:
                    masked[key] = f"{value[:3]}...{value[-3:]}***MASKED***"
                else:
                    masked[key] = "***MASKED***"
            else:
                masked[key] = value

        return masked

    def _write_to_file(self, entry: AuditLogEntry) -> None:
        """
        Write audit log entry to file.

        Args:
            entry: Log entry to write
        """
        try:
            with open(self.log_file, 'a') as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            # Don't let logging failure break the application
            print(f"WARNING: Failed to write audit log: {e}")

    def _log_to_console(self, entry: AuditLogEntry) -> None:
        """
        Log entry to console for visibility.

        Args:
            entry: Log entry
        """
        # Only log WARNING and CRITICAL to console
        if entry.level in {AuditLevel.CRITICAL, AuditLevel.WARNING}:
            print(
                f"[{entry.level.value}] {entry.action.value}: "
                f"tool={entry.tool_name}, agent={entry.agent_name}"
            )
            if entry.error:
                print(f"  Error: {entry.error}")

    def get_logs(
        self,
        tool_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """
        Query audit logs from memory.

        Args:
            tool_name: Filter by tool name
            agent_name: Filter by agent name
            level: Filter by log level
            limit: Max entries to return

        Returns:
            List of matching audit log entries
        """
        results = self.memory_log

        # Apply filters
        if tool_name:
            results = [e for e in results if e.tool_name == tool_name]
        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]
        if level:
            results = [e for e in results if e.level == level]

        # Return most recent first
        return list(reversed(results))[-limit:]

    def get_failed_validations(self, limit: int = 100) -> List[AuditLogEntry]:
        """
        Get all validation failures (attempts to block dangerous operations).

        Useful for detecting LLM misuse or attacks.

        Args:
            limit: Max entries to return

        Returns:
            List of validation failure entries
        """
        results = [
            e for e in self.memory_log
            if e.action == AuditAction.VALIDATION_FAILED
        ]
        return list(reversed(results))[-limit:]

    def get_suspicious_patterns(self) -> Dict[str, Any]:
        """
        Analyze logs for suspicious patterns that indicate attacks.

        Returns:
            Dict with detected suspicious patterns
        """
        patterns = {
            "repeated_failures": [],
            "wildcard_attempts": [],
            "namespace_escape_attempts": [],
            "injection_attempts": [],
            "resource_exhaustion_attempts": []
        }

        # Get failed validations
        failures = self.get_failed_validations(limit=1000)

        # Analyze patterns
        for entry in failures:
            if not entry.parameters:
                continue

            params = entry.parameters

            # Check for wildcard attempts
            for key, value in params.items():
                if isinstance(value, str) and ('*' in value or '%' in value):
                    patterns["wildcard_attempts"].append({
                        "timestamp": entry.timestamp,
                        "tool": entry.tool_name,
                        "parameter": key,
                        "value": value
                    })

                # Check for injection
                if any(char in str(value) for char in [';', '|', '&', '>', '<']):
                    patterns["injection_attempts"].append({
                        "timestamp": entry.timestamp,
                        "tool": entry.tool_name,
                        "parameter": key
                    })

            # Check for protected namespace access
            if "namespace" in params:
                namespace = params["namespace"]
                if namespace in {"kube-system", "kube-public", "kube-node-lease"}:
                    patterns["namespace_escape_attempts"].append({
                        "timestamp": entry.timestamp,
                        "tool": entry.tool_name,
                        "namespace": namespace
                    })

            # Check for excessive resource requests
            if "tail" in params and isinstance(params["tail"], int):
                if params["tail"] > 10000:
                    patterns["resource_exhaustion_attempts"].append({
                        "timestamp": entry.timestamp,
                        "tool": entry.tool_name,
                        "tail": params["tail"]
                    })

        # Remove empty patterns
        return {k: v for k, v in patterns.items() if v}

    def generate_report(self) -> str:
        """
        Generate audit report for compliance and security review.

        Returns:
            Formatted audit report
        """
        report = "SENTINEL AUDIT REPORT\n"
        report += "=" * 70 + "\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += f"Total log entries: {len(self.memory_log)}\n"
        report += f"Log file: {self.log_file}\n\n"

        # Summary by tool
        tools = {}
        for entry in self.memory_log:
            if entry.tool_name not in tools:
                tools[entry.tool_name] = {"count": 0, "failures": 0}
            tools[entry.tool_name]["count"] += 1
            if entry.level == AuditLevel.CRITICAL:
                tools[entry.tool_name]["failures"] += 1

        report += "TOOLS SUMMARY:\n"
        for tool_name, stats in sorted(tools.items()):
            report += f"  {tool_name}: {stats['count']} calls, {stats['failures']} blocked\n"

        # Suspicious patterns
        patterns = self.get_suspicious_patterns()
        if patterns:
            report += "\nSUSPICIOUS PATTERNS DETECTED:\n"
            for pattern_type, entries in patterns.items():
                report += f"  {pattern_type}: {len(entries)} incidents\n"

        # Recent failures
        failures = self.get_failed_validations(limit=10)
        if failures:
            report += "\nRECENT VALIDATION FAILURES:\n"
            for entry in failures:
                report += f"  {entry.timestamp} | {entry.tool_name} | {entry.error}\n"

        return report


# ============================================================================
# GLOBAL AUDIT LOGGER INSTANCE
# ============================================================================

_global_audit_logger = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return _global_audit_logger
