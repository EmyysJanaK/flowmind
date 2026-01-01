"""
VALIDATION & AUDIT LOGGING - LLM SAFETY ARCHITECTURE

Comprehensive guide to how Sentinel prevents LLM misuse through validation
and audit logging.

============================================================================
PROBLEM: LLM MISUSE RISKS
============================================================================

Large language models (LLMs) can be:
1. COMPROMISED: Adversarial prompts or jailbreaks
2. CONFUSED: Misunderstanding user intent
3. CORRUPTED: Fine-tuning on malicious data
4. CONTROLLED: Hostile actors redirecting outputs

Risks when LLMs have tool access:
- Wildcard deletions: "Delete all pods with pod_name='*'"
- Namespace escape: Access to system namespaces (kube-system)
- Command injection: "container_name='web; rm -rf /'"
- Resource exhaustion: "tail=1000000000" causing memory OOM
- Privilege escalation: Repeatedly calling restricted tools
- Authorization bypass: Trying various permutations to escape restrictions

SOLUTION: MULTI-LAYER DEFENSE
- LAYER 1: Parameter validation (blocks dangerous values)
- LAYER 2: Audit logging (records all attempts)
- LAYER 3: Threat detection (identifies attack patterns)
- LAYER 4: Human escalation (when danger detected)

============================================================================
LAYER 1: PARAMETER VALIDATION
============================================================================

Validation is the FIRST line of defense. It blocks dangerous operations
before they ever reach tool execution.

VALIDATION PHASES:

PHASE 1: PARAMETER VALIDATION
- Check parameter types (string, int, etc.)
- Check parameter format (regex patterns)
- Check value ranges (min/max)
- Reject None/empty values

PHASE 2: SAFETY CONSTRAINT VALIDATION
- No wildcards (* or %) in names
- No shell metacharacters (; | & > < ` $ )
- No SQL injection patterns
- No path traversal sequences (../)
- Valid Kubernetes/Docker naming conventions

PHASE 3: RESOURCE LIMIT VALIDATION
- Max tail lines <= 10000
- Max grace period <= 300 seconds
- Max timeout <= 300 seconds
- Prevent memory exhaustion

VALIDATION RULES BY TOOL:

docker_container_logs:
  ✓ container_name: Must match ^[a-zA-Z0-9_-]+$
  ✓ container_name: Cannot contain *, %, ;, |, &, >, <, `, $
  ✓ tail: Must be 1-10000 (prevents memory exhaustion)

docker_container_restart:
  ✓ container_name: Must match ^[a-zA-Z0-9_-]+$
  ✓ container_name: Cannot contain *, %, ;, |, &, >, <, `, $
  ✓ timeout: Must be 1-300 seconds

k8s_pod_status:
  ✓ namespace: Must match ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$
  ✓ namespace: Cannot be in {kube-system, kube-public, kube-node-lease}
  ✓ pod_name: Must match ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$
  ✓ pod_name: Cannot contain *, %, ;, |, &, >, <, `, $

k8s_pod_restart:
  ✓ namespace: Must match ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$
  ✓ namespace: Cannot be in protected namespaces (kube-system, etc)
  ✓ pod_name: Must match ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$
  ✓ pod_name: Cannot contain *, %, ;, |, &, >, <, `, $
  ✓ grace_period: Must be 0-300 seconds

ATTACK EXAMPLES & BLOCKING:

ATTACK 1: Wildcard Deletion
  LLM attempts: k8s_pod_restart(namespace="*", pod_name="*")
  Validation check: No wildcards in pod_name
  Result: ❌ BLOCKED - "Pod name cannot contain wildcards"
  Log: VALIDATION_FAILED

ATTACK 2: Shell Injection
  LLM attempts: docker_container_logs(container_name="web; rm -rf /")
  Validation check: No shell metacharacters
  Result: ❌ BLOCKED - "Container name contains invalid characters: ; | &"
  Log: VALIDATION_FAILED

ATTACK 3: System Namespace Access
  LLM attempts: k8s_pod_status(namespace="kube-system", pod_name="coredns")
  Validation check: Protected namespaces list
  Result: ❌ BLOCKED - "Namespace restricted"
  Log: VALIDATION_FAILED

ATTACK 4: Resource Exhaustion
  LLM attempts: docker_container_logs(container_name="web", tail=1000000)
  Validation check: tail <= 10000
  Result: ❌ BLOCKED - "Tail must be between 1 and 10000 lines"
  Log: VALIDATION_FAILED

ATTACK 5: Timeout Amplification
  LLM attempts: docker_container_restart(container_name="db", timeout=600)
  Validation check: timeout <= 300
  Result: ❌ BLOCKED - "Timeout must be between 1 and 300 seconds"
  Log: VALIDATION_FAILED

============================================================================
LAYER 2: AUDIT LOGGING
============================================================================

When an operation is blocked by validation, it's logged. When an operation
executes, it's logged. All logs are stored for forensic analysis.

WHAT GETS LOGGED:

Every tool invocation generates audit logs at these points:

1. TOOL_CALL_REQUESTED
   - Tool name
   - Parameters (masked for sensitive data)
   - Agent name
   - LLM model
   - Timestamp
   - Request ID

2. VALIDATION_STARTED
   - Which validation phase
   - Parameters being checked

3. VALIDATION_PASSED / VALIDATION_FAILED
   - Specific validation rules that failed
   - Error messages
   - Severity level (CRITICAL, HIGH, MEDIUM, INFO)

4. AUTHORIZATION_CHECK
   - Service account being checked
   - Namespace restrictions
   - RBAC enforcement

5. EXECUTION_STARTED
   - System calls being made
   - Actual subprocess arguments

6. EXECUTION_COMPLETED / EXECUTION_FAILED
   - Result returned
   - Duration
   - Error details (if any)

LOG EXAMPLE: Blocked Attack

[2024-01-15T10:23:45.123Z] CRITICAL | VALIDATION_FAILED
  tool=k8s_pod_restart
  agent=detective_agent
  llm_model=gpt-4
  parameters={"namespace": "kube-system", "pod_name": "coredns"}
  error=Namespace restricted. Protected: {kube-system, ...}
  error_type=validation_failed
  request_id=abc-123-def

LOG EXAMPLE: Successful Operation

[2024-01-15T10:24:30.456Z] INFO | TOOL_CALL_REQUESTED
  tool=docker_container_logs
  agent=detective_agent
  parameters={"container_name": "web-service", "tail": 100}
  request_id=ghi-789-jkl

[2024-01-15T10:24:30.500Z] INFO | VALIDATION_PASSED
  tool=docker_container_logs
  request_id=ghi-789-jkl

[2024-01-15T10:24:30.600Z] INFO | EXECUTION_STARTED
  tool=docker_container_logs
  request_id=ghi-789-jkl

[2024-01-15T10:24:31.200Z] INFO | EXECUTION_COMPLETED
  tool=docker_container_logs
  duration_ms=600
  request_id=ghi-789-jkl

SENSITIVE DATA MASKING:

Logs automatically mask sensitive data:
- Passwords: "***MASKED***"
- API keys: "key_abc123...***MASKED***"
- Tokens: "token_xyz789...***MASKED***"
- JWTs: "eyJ0...***MASKED***"

This allows logging parameter values for debugging without exposing secrets.

============================================================================
LAYER 3: THREAT DETECTION
============================================================================

Audit logs are analyzed for suspicious patterns that indicate attacks.

DETECTED PATTERNS:

1. REPEATED_FAILURES
   Description: Same validation failing many times in short period
   Indicates: LLM trying different attack variations
   Response: Alert security team, consider blocking agent

2. WILDCARD_ATTEMPTS
   Description: Parameters containing * or % wildcards
   Indicates: Attempting to delete/affect multiple resources
   Response: Block immediately, escalate to human

3. NAMESPACE_ESCAPE_ATTEMPTS
   Description: Repeated attempts to access protected namespaces
   Indicates: LLM trying different namespace names
   Response: Block, investigate agent permissions

4. INJECTION_ATTEMPTS
   Description: Parameters containing shell metacharacters
   Indicates: Command injection attack
   Response: Block, escalate as security incident

5. RESOURCE_EXHAUSTION_ATTEMPTS
   Description: Repeatedly requesting excessive resources (tail>10000)
   Indicates: Denial of service attack
   Response: Block, rate-limit agent

SECURITY REPORT:

Get security report:
  registry.get_security_report()

Returns:
  {
    "audit_logs_total": 1523,
    "validation_failures": 47,
    "suspicious_patterns": {
      "wildcard_attempts": [
        {
          "timestamp": "2024-01-15T10:23:45Z",
          "tool": "k8s_pod_restart",
          "parameter": "pod_name",
          "value": "*"
        }
      ],
      "injection_attempts": [...]
    },
    "audit_report": "..."
  }

============================================================================
LAYER 4: HUMAN ESCALATION
============================================================================

When threat detection identifies suspicious activity:

1. ALERTS GENERATED
   - Suspicious patterns detected
   - Validation failures above threshold
   - Unknown LLM models attempting access
   - Repeated failures on same tool

2. HUMAN REVIEW
   - Security team reviews audit logs
   - Determines if attack or legitimate use
   - Investigates compromised agents/LLMs
   - Updates validation rules if needed

3. REMEDIATION
   - Block problematic agents
   - Disable LLM models showing attack patterns
   - Update RBAC permissions
   - Add new validation rules
   - Escalate to incident response

============================================================================
PREVENTING SPECIFIC LLM ATTACKS
============================================================================

ATTACK SCENARIO 1: Compromised LLM trying to delete everything

LLM Objective: Delete all pods in production
LLM Attempts:
  1. k8s_pod_restart(namespace="prod", pod_name="*")
  2. k8s_pod_restart(namespace="prod", pod_name="%.pod")
  3. k8s_pod_restart(namespace="prod", pod_name="*-*-*")

Defense:
  ✓ Validation blocks all attempts (wildcard detection)
  ✓ Each attempt logged as VALIDATION_FAILED
  ✓ Pattern detected: "wildcard_attempts"
  ✓ Alert: "Possible injection attack detected"
  ✓ Human: Investigates and disables compromised LLM

Result: Attack blocked, no resources deleted, evidence collected

---

ATTACK SCENARIO 2: Jailbroken LLM trying privilege escalation

LLM Objective: Get access to system namespaces
LLM Attempts:
  1. k8s_pod_status(namespace="kube-system", pod_name="coredns")
  2. k8s_pod_status(namespace="kube-public", pod_name="metrics")
  3. k8s_pod_status(namespace="kube-node-lease", pod_name="node-1")

Defense:
  ✓ Validation checks protected namespace list
  ✓ All attempts blocked (namespace restriction)
  ✓ All attempts logged with CRITICAL level
  ✓ Pattern detected: "namespace_escape_attempts"
  ✓ Alert: "Repeated unauthorized namespace access"
  ✓ Human: Reviews LLM behavior, considers disabling

Result: Access denied, pattern identified, containment possible

---

ATTACK SCENARIO 3: Confused LLM causing resource exhaustion

LLM Objective: Get last 500000 log lines (mistake, not attack)
LLM Attempt:
  docker_container_logs(container_name="web-service", tail=500000)

Defense:
  ✓ Validation checks tail <= 10000 limit
  ✓ Attempt blocked (resource limit)
  ✓ Logged as VALIDATION_FAILED with HIGH level
  ✓ If repeated: Pattern "resource_exhaustion_attempts"
  ✓ Alert: "Potential DoS attack or misconfiguration"
  ✓ Human: Checks LLM model, may adjust parameter

Result: Resource protected, failure logged, no impact

============================================================================
IMPLEMENTATION IN CODE
============================================================================

Using validation and logging in agents:

```python
from sentinel.tools import get_registry, get_audit_logger

class DiagnosisAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        audit_logger = get_audit_logger()
        
        # Call tool with agent name and LLM info
        result = await registry.call_tool(
            "docker_container_logs",
            agent_name=self.name,
            llm_model="gpt-4",
            container_name=state["target_container"],
            tail=100
        )
        
        # Check if operation was blocked
        if result.get("blocked"):
            # Validation failed - log and escalate
            state["escalate"] = True
            state["error"] = result["error"]
        
        elif result.get("success"):
            # Operation succeeded
            state["logs"] = result["logs"]
        
        return state

# Check security status
registry = get_registry()
security_report = registry.get_security_report()

if security_report["suspicious_patterns"]:
    # Suspicious activity detected
    print("⚠️  SECURITY ALERT:")
    for pattern, incidents in security_report["suspicious_patterns"].items():
        print(f"  - {pattern}: {len(incidents)} incidents")

# Generate compliance report
audit_logger = get_audit_logger()
report = audit_logger.generate_report()
print(report)
```

============================================================================
SECURITY BENEFITS SUMMARY
============================================================================

Validation + Audit Logging provides:

1. PREVENTION
   - Blocks dangerous operations at validation layer
   - No resource can be deleted by wildcard
   - No system namespace can be accessed
   - No shell commands can be injected

2. DETECTION
   - Every attempt logged (blocked or not)
   - Suspicious patterns identified automatically
   - Anomalies detected and reported
   - Attack progression visible in logs

3. INVESTIGATION
   - Full audit trail for forensics
   - Understand what LLM tried to do
   - Trace attack progression
   - Identify root cause

4. COMPLIANCE
   - Logs prove dangerous operations were blocked
   - Logs prove validation was enforced
   - Audit trail for security reviews
   - Evidence for compliance certifications

5. IMPROVEMENT
   - Identify new attack patterns
   - Update validation rules
   - Enhance threat detection
   - Continuous security improvement

============================================================================
"""

# This is comprehensive documentation
# Refer to validation.py and audit_logger.py for implementation
