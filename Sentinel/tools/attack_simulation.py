"""
ATTACK SIMULATION & DEFENSE DEMONSTRATION

This module shows how Sentinel's validation and logging blocks actual attack
scenarios while keeping them logged for forensics.
"""

from typing import Dict, Any
import asyncio
from sentinel.tools import (
    get_registry,
    get_validator,
    get_audit_logger,
    ValidationError
)


async def demonstrate_validation_blocking() -> None:
    """
    Demonstrate how validation blocks dangerous operations.
    
    Shows 5 attack scenarios and how Sentinel prevents them.
    """
    
    print("=" * 80)
    print("SENTINEL VALIDATION - ATTACK PREVENTION DEMONSTRATION")
    print("=" * 80)
    print()
    
    registry = get_registry()
    validator = get_validator()
    
    # ========================================================================
    # ATTACK 1: WILDCARD POD DELETION
    # ========================================================================
    
    print("ATTACK 1: WILDCARD POD DELETION")
    print("-" * 80)
    print("Objective: Delete all pods with pod_name='*'")
    print("Attack code: k8s_pod_restart(namespace='production', pod_name='*')")
    print()
    
    try:
        result = await registry.call_tool(
            "k8s_pod_restart",
            agent_name="malicious_llm",
            llm_model="jailbroken-gpt",
            namespace="production",
            pod_name="*"
        )
        print(f"Result: {result}")
    except ValidationError as e:
        print(f"✅ BLOCKED: {e}")
    
    print()
    
    # ========================================================================
    # ATTACK 2: COMMAND INJECTION IN CONTAINER NAME
    # ========================================================================
    
    print("ATTACK 2: COMMAND INJECTION")
    print("-" * 80)
    print("Objective: Execute shell command via container name")
    print("Attack code: docker_container_restart(container_name='web; rm -rf /')")
    print()
    
    try:
        result = await registry.call_tool(
            "docker_container_restart",
            agent_name="compromised_llm",
            llm_model="gpt-4",
            container_name="web; rm -rf /"
        )
        print(f"Result: {result}")
    except ValidationError as e:
        print(f"✅ BLOCKED: {e}")
    
    print()
    
    # ========================================================================
    # ATTACK 3: SYSTEM NAMESPACE ACCESS
    # ========================================================================
    
    print("ATTACK 3: SYSTEM NAMESPACE ESCAPE")
    print("-" * 80)
    print("Objective: Access kube-system namespace to modify core components")
    print("Attack code: k8s_pod_status(namespace='kube-system', pod_name='coredns')")
    print()
    
    try:
        result = await registry.call_tool(
            "k8s_pod_status",
            agent_name="unauthorized_llm",
            llm_model="gpt-4",
            namespace="kube-system",
            pod_name="coredns"
        )
        print(f"Result: {result}")
    except ValidationError as e:
        print(f"✅ BLOCKED: {e}")
    
    print()
    
    # ========================================================================
    # ATTACK 4: RESOURCE EXHAUSTION (LOG BOMB)
    # ========================================================================
    
    print("ATTACK 4: RESOURCE EXHAUSTION")
    print("-" * 80)
    print("Objective: Cause memory exhaustion by requesting huge log volume")
    print("Attack code: docker_container_logs(container_name='web', tail=1000000)")
    print()
    
    try:
        result = await registry.call_tool(
            "docker_container_logs",
            agent_name="dos_llm",
            llm_model="gpt-4",
            container_name="web",
            tail=1000000
        )
        print(f"Result: {result}")
    except ValidationError as e:
        print(f"✅ BLOCKED: {e}")
    
    print()
    
    # ========================================================================
    # ATTACK 5: MULTIPLE INJECTION ATTEMPTS
    # ========================================================================
    
    print("ATTACK 5: MULTIPLE INJECTION ATTEMPTS (ESCALATING)")
    print("-" * 80)
    print("Objective: Try multiple injection patterns to bypass validation")
    print()
    
    injection_attempts = [
        "web; echo hack",
        "web | cat /etc/passwd",
        "web && rm -rf /",
        "web > /tmp/pwned",
        "web `whoami`",
        "web $(whoami)"
    ]
    
    blocked_count = 0
    for attempt in injection_attempts:
        print(f"  Attempt: docker_container_logs(container_name='{attempt}')")
        try:
            result = await registry.call_tool(
                "docker_container_logs",
                agent_name="injection_llm",
                llm_model="gpt-4",
                container_name=attempt
            )
            print(f"    Result: {result}")
        except ValidationError:
            print(f"    ✅ BLOCKED")
            blocked_count += 1
    
    print(f"\n  Total attempts: {len(injection_attempts)}, Blocked: {blocked_count}")
    print()


async def demonstrate_audit_logging() -> None:
    """
    Demonstrate how audit logs capture all attempts (blocked and successful).
    
    Shows what gets logged and how it enables forensics.
    """
    
    print("=" * 80)
    print("SENTINEL AUDIT LOGGING - FORENSIC EVIDENCE")
    print("=" * 80)
    print()
    
    audit_logger = get_audit_logger()
    registry = get_registry()
    
    # Make some legitimate calls
    print("Making legitimate tool calls...")
    print()
    
    await registry.call_tool(
        "docker_container_logs",
        agent_name="detective_agent",
        llm_model="gpt-4",
        container_name="web-service",
        tail=50
    )
    
    # Make some blocked calls
    print("Attempting dangerous operations...")
    print()
    
    dangerous_ops = [
        {
            "tool": "k8s_pod_restart",
            "params": {
                "namespace": "kube-system",
                "pod_name": "coredns"
            }
        },
        {
            "tool": "docker_container_logs",
            "params": {
                "container_name": "web; rm -rf /",
                "tail": 100
            }
        }
    ]
    
    for op in dangerous_ops:
        try:
            await registry.call_tool(op["tool"], **op["params"])
        except:
            pass
    
    # Show audit logs
    print("RECENT AUDIT LOG ENTRIES:")
    print("-" * 80)
    
    recent_logs = audit_logger.get_logs(limit=10)
    for entry in recent_logs:
        print(f"\n[{entry.action.value}]")
        print(f"  Tool: {entry.tool_name}")
        print(f"  Agent: {entry.agent_name}")
        print(f"  LLM: {entry.llm_model}")
        print(f"  Level: {entry.level.value}")
        if entry.error:
            print(f"  Error: {entry.error}")
    
    print()
    
    # Show failed validations
    print("VALIDATION FAILURES (BLOCKED ATTACKS):")
    print("-" * 80)
    
    failures = audit_logger.get_failed_validations(limit=10)
    if failures:
        for failure in failures:
            print(f"\n{failure.timestamp}")
            print(f"  Tool: {failure.tool_name}")
            print(f"  Agent: {failure.agent_name}")
            print(f"  Error: {failure.error}")
    
    print()


async def demonstrate_threat_detection() -> None:
    """
    Demonstrate how audit logs reveal attack patterns.
    
    Shows suspicious pattern detection from multiple failed attempts.
    """
    
    print("=" * 80)
    print("SENTINEL THREAT DETECTION - ATTACK PATTERN IDENTIFICATION")
    print("=" * 80)
    print()
    
    audit_logger = get_audit_logger()
    registry = get_registry()
    
    # Simulate an attacker trying different namespaces
    print("Simulating namespace escape attack...")
    print()
    
    protected_namespaces = ["kube-system", "kube-public", "kube-node-lease"]
    
    for ns in protected_namespaces:
        try:
            await registry.call_tool(
                "k8s_pod_status",
                agent_name="attack_llm",
                llm_model="adversarial",
                namespace=ns,
                pod_name="pod-1"
            )
        except:
            pass
    
    # Detect patterns
    print("\nDETECTED SUSPICIOUS PATTERNS:")
    print("-" * 80)
    
    patterns = audit_logger.get_suspicious_patterns()
    
    if patterns:
        print("✅ Attack patterns detected:\n")
        for pattern_type, incidents in patterns.items():
            print(f"{pattern_type}:")
            print(f"  Count: {len(incidents)}")
            for incident in incidents:
                if "namespace" in incident:
                    print(f"  - {incident['timestamp']}: {incident['namespace']}")
                elif "parameter" in incident:
                    print(f"  - {incident['timestamp']}: {incident['parameter']}")
            print()
    else:
        print("No suspicious patterns detected")
    
    print()


async def demonstrate_security_report() -> None:
    """
    Demonstrate security reporting for compliance and monitoring.
    
    Shows what information is available for security teams.
    """
    
    print("=" * 80)
    print("SENTINEL SECURITY REPORT - COMPLIANCE & MONITORING")
    print("=" * 80)
    print()
    
    registry = get_registry()
    audit_logger = get_audit_logger()
    
    # Get security report
    report = registry.get_security_report()
    
    print("SECURITY METRICS:")
    print("-" * 80)
    print(f"Total audit log entries: {report['audit_logs_total']}")
    print(f"Validation failures (blocked): {report['validation_failures']}")
    
    if report['suspicious_patterns']:
        print("\nSuspicious patterns detected:")
        for pattern, incidents in report['suspicious_patterns'].items():
            print(f"  - {pattern}: {len(incidents)}")
    
    print()
    print("AUDIT REPORT:")
    print("-" * 80)
    print(report['audit_report'])


def demonstrate_validation_rules() -> None:
    """
    Show what validation rules are applied to each tool.
    """
    
    print("=" * 80)
    print("SENTINEL VALIDATION RULES - WHAT GETS CHECKED")
    print("=" * 80)
    print()
    
    validator = get_validator()
    
    tools = [
        "docker_container_logs",
        "docker_container_restart",
        "k8s_pod_status",
        "k8s_pod_restart"
    ]
    
    for tool_name in tools:
        print(validator.describe_rules(tool_name))
        print()


async def main() -> None:
    """Run all demonstrations."""
    
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  SENTINEL - LLM SAFETY & ATTACK PREVENTION DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Show validation rules
    demonstrate_validation_rules()
    
    # Show attack prevention
    await demonstrate_validation_blocking()
    
    # Show audit logging
    await demonstrate_audit_logging()
    
    # Show threat detection
    await demonstrate_threat_detection()
    
    # Show security report
    await demonstrate_security_report()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Sentinel's multi-layer defense protects against LLM misuse:

1. VALIDATION: Dangerous operations blocked at parameter validation layer
2. LOGGING: All attempts (blocked and successful) recorded for forensics
3. DETECTION: Suspicious patterns identified automatically
4. ESCALATION: Human security team alerted for investigation

Result: LLMs cannot delete arbitrary resources, access restricted namespaces,
inject commands, or exhaust resources - all attempts are logged and analyzed.
    """)


if __name__ == "__main__":
    asyncio.run(main())
