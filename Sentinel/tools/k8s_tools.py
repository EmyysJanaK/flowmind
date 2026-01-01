"""
Kubernetes tools for Sentinel agents.

Kubernetes Tools Architecture
=============================

These tools provide the ONLY interface for agents to interact with Kubernetes.
Agents never execute kubectl commands directly - they must use these tools.

This enforces:
1. Controlled execution: All K8s operations logged and monitored
2. Error handling: Failures wrapped in tool-specific error handling
3. State tracking: Each operation records what was done for rollback
4. Authorization: Can check RBAC and agent permissions
5. Safety: Can add request limits, timeout enforcement, etc.

Usage:
    tool = KubernetesDeploymentTool()
    result = await tool.execute(
        action="restart",
        deployment="my-service",
        namespace="production"
    )
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import re
import subprocess
import json
from .base import BaseTool


class KubernetesDeploymentTool(BaseTool):
    """
    Tool for Kubernetes deployment operations.

    Agents use this tool to:
    - Create/update deployments
    - Restart deployments
    - Check deployment status
    - Scale deployments
    - Rollback deployments
    """

    def __init__(self):
        """Initialize Kubernetes deployment tool."""
        super().__init__(
            name="k8s_deployment",
            description="Manages Kubernetes deployments (create, update, restart, status, rollback)"
        )
        self.operation_history: List[Dict[str, Any]] = []

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a Kubernetes deployment operation.

        Args:
            action (str): Operation to perform
                - "apply": Create or update deployment from manifest
                - "restart": Restart all pods in deployment
                - "status": Get deployment status
                - "rollback": Rollback to previous revision
                - "get": Get deployment details
            deployment (str): Deployment name
            namespace (str): Kubernetes namespace (default: "default")
            manifest (dict, optional): Deployment manifest (for "apply")
            revision (int, optional): Revision to rollback to

        Returns:
            Dict with:
            - "success": bool - operation succeeded
            - "action": str - what was done
            - "deployment": str - target deployment
            - "namespace": str - namespace
            - "result": Any - operation result
            - "timestamp": str - ISO timestamp
            - "error": str - error message if failed
        """

        action = kwargs.get("action")
        deployment = kwargs.get("deployment")
        namespace = kwargs.get("namespace", "default")

        if not action or not deployment:
            return {
                "success": False,
                "error": "Missing required arguments: action and deployment",
                "timestamp": datetime.now().isoformat()
            }

        # Mock Kubernetes operations
        operations = {
            "apply": self._mock_apply_deployment,
            "restart": self._mock_restart_deployment,
            "status": self._mock_deployment_status,
            "rollback": self._mock_rollback_deployment,
            "get": self._mock_get_deployment
        }

        if action not in operations:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "namespace": namespace,
                "timestamp": datetime.now().isoformat()
            }

        try:
            result = await operations[action](deployment, namespace, kwargs)
            
            # Log the operation
            operation_record = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "deployment": deployment,
                "namespace": namespace,
                "success": result.get("success", False)
            }
            self.operation_history.append(operation_record)

            return result

        except Exception as e:
            return {
                "success": False,
                "action": action,
                "deployment": deployment,
                "namespace": namespace,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _mock_apply_deployment(
        self, deployment: str, namespace: str, kwargs: Dict
    ) -> Dict[str, Any]:
        """Mock applying a deployment."""
        return {
            "success": True,
            "action": "apply",
            "deployment": deployment,
            "namespace": namespace,
            "result": f"Deployment {deployment} applied in {namespace}",
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_restart_deployment(
        self, deployment: str, namespace: str, kwargs: Dict
    ) -> Dict[str, Any]:
        """Mock restarting a deployment."""
        return {
            "success": True,
            "action": "restart",
            "deployment": deployment,
            "namespace": namespace,
            "result": {
                "message": f"Deployment {deployment} restarted",
                "pods_restarted": 3,
                "restart_time_seconds": 15
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_deployment_status(
        self, deployment: str, namespace: str, kwargs: Dict
    ) -> Dict[str, Any]:
        """Mock getting deployment status."""
        return {
            "success": True,
            "action": "status",
            "deployment": deployment,
            "namespace": namespace,
            "result": {
                "replicas_desired": 3,
                "replicas_ready": 3,
                "replicas_updated": 3,
                "replicas_available": 3,
                "status": "Deployment has minimum availability",
                "conditions": [
                    {"type": "Progressing", "status": "True", "reason": "NewReplicaSetAvailable"},
                    {"type": "Available", "status": "True", "reason": "MinimumReplicasAvailable"}
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_rollback_deployment(
        self, deployment: str, namespace: str, kwargs: Dict
    ) -> Dict[str, Any]:
        """Mock rolling back a deployment."""
        revision = kwargs.get("revision", "previous")
        return {
            "success": True,
            "action": "rollback",
            "deployment": deployment,
            "namespace": namespace,
            "result": f"Rolled back {deployment} to revision {revision}",
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_get_deployment(
        self, deployment: str, namespace: str, kwargs: Dict
    ) -> Dict[str, Any]:
        """Mock getting deployment details."""
        return {
            "success": True,
            "action": "get",
            "deployment": deployment,
            "namespace": namespace,
            "result": {
                "name": deployment,
                "namespace": namespace,
                "replicas": 3,
                "image": f"{deployment}:v1.2.3",
                "age": "5 days"
            },
            "timestamp": datetime.now().isoformat()
        }


class KubernetesScaleTool(BaseTool):
    """
    Tool for Kubernetes scaling operations.

    Agents use this tool to:
    - Scale deployments up/down
    - Check current replica count
    - Set auto-scaling policies
    - Get scaling metrics
    """

    def __init__(self):
        """Initialize Kubernetes scale tool."""
        super().__init__(
            name="k8s_scale",
            description="Manages Kubernetes scaling (scale, autoscale, metrics)"
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a Kubernetes scaling operation.

        Args:
            action (str): Operation to perform
                - "scale": Change replica count
                - "autoscale": Set up horizontal pod autoscaling
                - "metrics": Get current scaling metrics
            deployment (str): Deployment name
            namespace (str): Kubernetes namespace (default: "default")
            replicas (int, optional): Target replica count (for "scale")
            min_replicas (int, optional): Min replicas for autoscaling
            max_replicas (int, optional): Max replicas for autoscaling
            cpu_threshold (int, optional): CPU threshold percentage

        Returns:
            Dict with scaling operation result
        """

        action = kwargs.get("action")
        deployment = kwargs.get("deployment")
        namespace = kwargs.get("namespace", "default")
        replicas = kwargs.get("replicas")

        if action == "scale":
            return {
                "success": True,
                "action": "scale",
                "deployment": deployment,
                "namespace": namespace,
                "result": {
                    "message": f"Scaled {deployment} to {replicas} replicas",
                    "previous_replicas": 3,
                    "current_replicas": replicas,
                    "scale_time_seconds": 45
                },
                "timestamp": datetime.now().isoformat()
            }
        elif action == "metrics":
            return {
                "success": True,
                "action": "metrics",
                "deployment": deployment,
                "namespace": namespace,
                "result": {
                    "current_replicas": 3,
                    "cpu_usage_percent": 65,
                    "memory_usage_percent": 45,
                    "requests_per_second": 1250
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "timestamp": datetime.now().isoformat()
            }


class KubernetesHealthTool(BaseTool):
    """
    Tool for Kubernetes health monitoring and diagnostics.

    Agents use this tool to:
    - Check pod health and status
    - Get cluster status
    - View events and logs
    - Diagnose failures
    - Monitor resource availability
    """

    def __init__(self):
        """Initialize Kubernetes health tool."""
        super().__init__(
            name="k8s_health",
            description="Monitors Kubernetes cluster and pod health"
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Check Kubernetes health status.

        Args:
            check_type (str): Type of health check
                - "cluster": Check cluster status
                - "pod": Check pod status
                - "namespace": Check namespace status
                - "events": Get recent events
            namespace (str, optional): Namespace to check
            pod (str, optional): Pod name to check

        Returns:
            Dict with health status and diagnostics
        """

        check_type = kwargs.get("check_type", "cluster")
        namespace = kwargs.get("namespace", "default")

        if check_type == "cluster":
            return {
                "success": True,
                "check_type": "cluster",
                "result": {
                    "status": "healthy",
                    "nodes": 3,
                    "nodes_ready": 3,
                    "api_server": "ready",
                    "controllers": "ready",
                    "scheduler": "ready"
                },
                "timestamp": datetime.now().isoformat()
            }
        elif check_type == "namespace":
            return {
                "success": True,
                "check_type": "namespace",
                "namespace": namespace,
                "result": {
                    "status": "active",
                    "pods_running": 12,
                    "pods_pending": 0,
                    "pods_failed": 0,
                    "cpu_usage_cores": 2.5,
                    "memory_usage_gb": 8.3
                },
                "timestamp": datetime.now().isoformat()
            }
        elif check_type == "events":
            return {
                "success": True,
                "check_type": "events",
                "namespace": namespace,
                "result": {
                    "recent_events": [
                        {
                            "timestamp": datetime.now().isoformat(),
                            "type": "Normal",
                            "reason": "Scheduled",
                            "message": "Successfully assigned pod"
                        },
                        {
                            "timestamp": datetime.now().isoformat(),
                            "type": "Normal",
                            "reason": "Pulled",
                            "message": "Container image already present"
                        }
                    ]
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Unknown check type: {check_type}",
                "timestamp": datetime.now().isoformat()
            }


def get_k8s_tools() -> List[BaseTool]:
    """
    Get all available Kubernetes tools as a collection.

    This is useful for agents that need to work with multiple K8s tools.

    Returns:
        List of initialized Kubernetes tool instances
    """
    return [
        KubernetesDeploymentTool(),
        KubernetesScaleTool(),
        KubernetesHealthTool()
    ]


async def get_pod_status(
    namespace: str,
    pod_name: str,
    use_real_kubectl: bool = False
) -> Dict[str, Any]:
    """
    Safely retrieve Kubernetes pod status with read-only access.

    READ-ONLY ACCESS DESIGN RATIONALE
    ===================================

    This tool is READ-ONLY (no write/delete permissions) by design:

    1. SAFETY THROUGH CONSTRAINTS
       At the current stage of Sentinel development, agents are not
       trusted with modification capabilities. Read-only access allows:
       - Safe investigation of cluster state
       - No risk of accidental deletion or corruption
       - Testing and validation of agent decision-making
       - Building trust in agent behavior

    2. SEPARATION OF CONCERNS
       Read operations (investigation) are separated from write operations
       (execution). This follows the Sentinel principle:
       - DetectiveAgent: Investigates using READ tools
       - ResearcherAgent: Recommends solutions (no cluster access)
       - OperatorAgent: Plans execution (READ + PLAN, no execution)
       - ExecutorAgent: Would have WRITE tools for execution (not yet implemented)

    3. INCREMENTAL CAPABILITY ESCALATION
       As agents prove their reliability:
       - Phase 1 (Current): Read-only investigation
       - Phase 2: Write operations with human approval
       - Phase 3: Write operations with automatic approval (high confidence)
       - Phase 4: Full autonomous cluster management

    4. AUDIT & RECOVERY
       Keeping reads separate from writes simplifies:
       - Audit trail analysis
       - Understanding what was investigated vs executed
       - Rolling back just the writes if needed
       - Forensic analysis of agent decisions

    5. THREAT MITIGATION
       If agent is compromised or behaves unexpectedly:
       - Limited to reading cluster state
       - Cannot delete resources
       - Cannot modify configurations
       - Cannot escalate to other clusters

    FUTURE WRITE OPERATIONS
    When write capabilities are added, they will:
    - Require explicit agent permissions (RBAC)
    - Need approval for production operations
    - Be rate-limited and logged
    - Support dry-run for safety validation
    - Track exact changes for rollback

    Args:
        namespace (str): Kubernetes namespace. Must be alphanumeric with
            dashes only. Validated to prevent injection.
            Examples: "default", "production", "kube-system"
        pod_name (str): Name of the pod to query. Must be alphanumeric
            with dashes only. Validated to prevent injection.
            Examples: "web-service-7d9f4c", "db-primary-0"
        use_real_kubectl (bool): Whether to use real kubectl or mock mode
            (default: False). In production, set to True with proper
            kubectl configuration and RBAC.

    Returns:
        Dict with:
        - "success" (bool): Query succeeded
        - "namespace" (str): Kubernetes namespace
        - "pod_name" (str): Pod name
        - "status" (Dict): Detailed pod status:
            - "phase": str (Pending, Running, Succeeded, Failed, Unknown)
            - "ready": bool - all containers ready?
            - "conditions": List[Dict] - pod conditions
            - "containers": List[Dict] - container statuses
            - "restart_count": int - total restarts
            - "age": str - how long pod has existed
            - "cpu_usage": str - current CPU usage
            - "memory_usage": str - current memory usage
        - "timestamp": str - ISO timestamp
        - "error": str - error message if failed
        - "error_type": str - category of error

    Example:
        >>> result = await get_pod_status("production", "web-service-7d9f4c")
        >>> if result["success"]:
        ...     status = result["status"]
        ...     print(f"Pod phase: {status['phase']}")
        ...     print(f"Ready: {status['ready']}")
        ... else:
        ...     print(f"Error: {result['error']}")
    """

    # =========================================================================
    # PHASE 1: INPUT VALIDATION & SANITIZATION
    # =========================================================================

    # Validate namespace: Kubernetes namespace naming rules
    # Must be alphanumeric and dash, max 63 characters
    namespace_pattern = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"
    if not namespace:
        return {
            "success": False,
            "error": "Namespace cannot be empty",
            "error_type": "invalid_input",
            "timestamp": datetime.now().isoformat()
        }

    if not re.match(namespace_pattern, namespace):
        return {
            "success": False,
            "namespace": namespace,
            "error": "Invalid namespace format",
            "error_type": "invalid_format",
            "timestamp": datetime.now().isoformat(),
            "security_reason": "Namespace must be lowercase alphanumeric with dashes"
        }

    # Validate pod name: Kubernetes pod naming rules
    # Must be alphanumeric and dash, max 63 characters
    pod_pattern = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"
    if not pod_name:
        return {
            "success": False,
            "error": "Pod name cannot be empty",
            "error_type": "invalid_input",
            "timestamp": datetime.now().isoformat()
        }

    if not re.match(pod_pattern, pod_name):
        return {
            "success": False,
            "pod_name": pod_name,
            "error": "Invalid pod name format",
            "error_type": "invalid_format",
            "timestamp": datetime.now().isoformat(),
            "security_reason": "Pod name must be lowercase alphanumeric with dashes"
        }

    # =========================================================================
    # PHASE 2: PERMISSION & AUTHORIZATION CHECK
    # =========================================================================

    # In production, verify read-only permission
    # Example:
    # if not await check_agent_permissions(agent_id, namespace, pod_name, "get_status"):
    #     return {
    #         "success": False,
    #         "error": "Permission denied",
    #         "error_type": "unauthorized",
    #         "timestamp": datetime.now().isoformat()
    #     }

    # =========================================================================
    # PHASE 3: EXECUTE KUBECTL QUERY
    # =========================================================================

    if use_real_kubectl:
        return await _get_pod_status_real(namespace, pod_name)
    else:
        return await _get_pod_status_mock(namespace, pod_name)


async def _get_pod_status_real(
    namespace: str,
    pod_name: str
) -> Dict[str, Any]:
    """
    Execute real kubectl pod status query.

    SECURITY: Uses subprocess with shell=False and argument list to prevent
    command injection. Arguments are passed separately, no shell interpolation.
    Only supports READ operations (--output json for queries).

    Args:
        namespace: Validated Kubernetes namespace
        pod_name: Validated pod name

    Returns:
        Dict with pod status or error information
    """

    try:
        # SECURITY: subprocess with shell=False + argument list prevents injection
        # Command structure: kubectl get pod <pod_name> -n <namespace> -o json
        command = [
            "kubectl",                                  # kubectl executable
            "get",                                      # Subcommand (read-only)
            "pod",                                      # Resource type
            pod_name,                                   # Validated pod name
            "-n", namespace,                            # Validated namespace (argument style)
            "-o", "json"                                # Output format (safe, structured)
        ]

        # Execute with timeout to prevent hanging
        result = subprocess.run(
            command,
            capture_output=True,                        # Capture output
            text=True,                                  # Return strings
            timeout=30,                                 # 30 second timeout
            check=False                                 # Don't raise on non-zero exit
        )

        # =========================================================================
        # PHASE 4: PARSE & STRUCTURE RESULTS
        # =========================================================================

        if result.returncode != 0:
            # kubectl command failed - parse error
            error_type, error_msg = _analyze_kubectl_error(result.stderr)

            return {
                "success": False,
                "namespace": namespace,
                "pod_name": pod_name,
                "error": error_msg,
                "error_type": error_type,
                "timestamp": datetime.now().isoformat()
            }

        # Parse JSON output
        try:
            pod_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "success": False,
                "namespace": namespace,
                "pod_name": pod_name,
                "error": "Failed to parse kubectl output",
                "error_type": "parse_error",
                "timestamp": datetime.now().isoformat()
            }

        # Extract status information
        status_dict = _extract_pod_status(pod_data)

        return {
            "success": True,
            "namespace": namespace,
            "pod_name": pod_name,
            "status": status_dict,
            "timestamp": datetime.now().isoformat(),
            "raw_data": pod_data  # Include full data for debugging
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "namespace": namespace,
            "pod_name": pod_name,
            "error": "kubectl query timed out",
            "error_type": "timeout",
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError:
        return {
            "success": False,
            "namespace": namespace,
            "pod_name": pod_name,
            "error": "kubectl executable not found",
            "error_type": "kubectl_not_available",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "success": False,
            "namespace": namespace,
            "pod_name": pod_name,
            "error": "Failed to get pod status due to system error",
            "error_type": "system_error",
            "timestamp": datetime.now().isoformat()
            # Internal logging: logger.error(f"kubectl get failed: {e}", exc_info=True)
        }


async def _get_pod_status_mock(
    namespace: str,
    pod_name: str
) -> Dict[str, Any]:
    """
    Mock Kubernetes pod status query for testing without real cluster.

    Args:
        namespace: Namespace name
        pod_name: Pod name

    Returns:
        Mock pod status data
    """

    status_dict = {
        "phase": "Running",
        "ready": True,
        "ready_containers": 1,
        "total_containers": 1,
        "restart_count": 0,
        "age": "5 days",
        "conditions": [
            {
                "type": "Initialized",
                "status": "True",
                "reason": "PodInitialized"
            },
            {
                "type": "Ready",
                "status": "True",
                "reason": "ContainersReady"
            },
            {
                "type": "ContainersReady",
                "status": "True",
                "reason": "ContainersReady"
            },
            {
                "type": "PodScheduled",
                "status": "True",
                "reason": "Successfully assigned"
            }
        ],
        "containers": [
            {
                "name": "web-service",
                "image": "web-service:v1.2.3",
                "status": "Running",
                "ready": True,
                "restarts": 0
            }
        ],
        "node": "worker-node-2",
        "cpu_usage": "125m",
        "memory_usage": "256Mi"
    }

    return {
        "success": True,
        "namespace": namespace,
        "pod_name": pod_name,
        "status": status_dict,
        "timestamp": datetime.now().isoformat(),
        "note": "Mock status - real Kubernetes not connected"
    }


def _analyze_kubectl_error(error_output: str) -> tuple[str, str]:
    """
    Analyze kubectl error and categorize it safely.

    Maps kubectl errors to error types:
    - not_found: Pod or namespace doesn't exist
    - connection_error: Cannot connect to cluster
    - permission_error: Insufficient permissions
    - configuration_error: kubectl config issue
    - unknown: Unclassified error

    Args:
        error_output: Raw kubectl error message

    Returns:
        Tuple of (error_type: str, error_message: str)
    """

    error_lower = error_output.lower()

    # Detect specific error types
    if "not found" in error_lower or "does not exist" in error_lower:
        if "namespace" in error_lower:
            return ("not_found", "Namespace not found")
        else:
            return ("not_found", "Pod not found in namespace")

    if "connection refused" in error_lower or "unable to connect" in error_lower:
        return (
            "connection_error",
            "Cannot connect to Kubernetes cluster - API server may be down"
        )

    if "permission denied" in error_lower or "forbidden" in error_lower:
        return (
            "permission_error",
            "Permission denied - check RBAC permissions"
        )

    if "config" in error_lower or "kubeconfig" in error_lower:
        return (
            "configuration_error",
            "kubectl configuration issue - check kubeconfig"
        )

    # Generic error with sanitization
    sanitized = _sanitize_kubectl_error(error_output)
    return ("unknown_error", f"kubectl error: {sanitized}")


def _sanitize_kubectl_error(error_msg: str) -> str:
    """
    Sanitize kubectl error messages to remove sensitive information.

    Removes:
    - File system paths
    - IP addresses and hostnames
    - API server URLs
    - Token references

    Args:
        error_msg: Raw kubectl error message

    Returns:
        Sanitized error message
    """

    # Remove file paths
    error_msg = re.sub(r"/[a-zA-Z0-9/_.-]+", "<path>", error_msg)

    # Remove IP addresses
    error_msg = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "<ip>", error_msg)

    # Remove URLs
    error_msg = re.sub(r"https?://[^\s]+", "<url>", error_msg)

    # Remove token references
    error_msg = re.sub(r"token[^\s]*", "<token>", error_msg, flags=re.IGNORECASE)

    return error_msg


def _extract_pod_status(pod_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant status information from kubectl JSON output.

    Args:
        pod_data: Full pod object from kubectl JSON output

    Returns:
        Structured status dictionary
    """

    metadata = pod_data.get("metadata", {})
    spec = pod_data.get("spec", {})
    status = pod_data.get("status", {})

    # Extract phase
    phase = status.get("phase", "Unknown")

    # Calculate age
    creation_time = metadata.get("creationTimestamp", "")
    age = _calculate_pod_age(creation_time) if creation_time else "Unknown"

    # Extract container statuses
    container_statuses = status.get("containerStatuses", [])
    ready_containers = sum(
        1 for c in container_statuses if c.get("ready", False)
    )
    total_containers = len(container_statuses)

    # Extract conditions
    conditions = []
    for cond in status.get("conditions", []):
        conditions.append({
            "type": cond.get("type", ""),
            "status": cond.get("status", ""),
            "reason": cond.get("reason", ""),
            "message": cond.get("message", "")
        })

    # Calculate total restarts
    total_restarts = sum(
        c.get("restartCount", 0) for c in container_statuses
    )

    # Extract node assignment
    node_name = spec.get("nodeName", "Unassigned")

    return {
        "phase": phase,
        "ready": ready_containers == total_containers,
        "ready_containers": ready_containers,
        "total_containers": total_containers,
        "restart_count": total_restarts,
        "age": age,
        "conditions": conditions,
        "containers": [
            {
                "name": c.get("name", ""),
                "image": c.get("image", ""),
                "status": c.get("state", {}),
                "ready": c.get("ready", False),
                "restarts": c.get("restartCount", 0)
            }
            for c in container_statuses
        ],
        "node": node_name
    }


def _calculate_pod_age(creation_time: str) -> str:
    """
    Calculate pod age from creation timestamp.

    Args:
        creation_time: ISO format creation timestamp

    Returns:
        Human-readable age string
    """

    try:
        from datetime import datetime as dt
        created = dt.fromisoformat(creation_time.replace("Z", "+00:00"))
        now = dt.now(created.tzinfo)
        delta = now - created

        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        if days > 0:
            return f"{days} days"
        elif hours > 0:
            return f"{hours} hours"
        else:
            return f"{minutes} minutes"
    except Exception:
        return "Unknown"
