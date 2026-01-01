"""
SENTINEL TOOLS - QUICK REFERENCE

A quick guide to available tools for agents and developers.

============================================================================
REGISTERED TOOLS
============================================================================

DOCKER TOOLS:
  • docker_container_logs
    - Get logs from container with sensitive data masked
    - Read-only operation
    - Input: container_name (str), tail (int, default 100)
    
  • docker_container_restart  
    - Restart Docker container with graceful shutdown
    - Allows timeout customization
    - Input: container_name (str), timeout (int, default 30)

KUBERNETES TOOLS:
  • k8s_pod_status
    - Query pod status (READ-ONLY)
    - Returns phase, conditions, container status
    - Input: namespace (str), pod_name (str)
    
  • k8s_pod_restart
    - Delete pod (triggers controller to restart)
    - RBAC restricted to specific namespaces
    - Input: namespace (str), pod_name (str), grace_period (int, default 30)

============================================================================
QUICK START - AGENT USAGE
============================================================================

1. ACCESS THE REGISTRY:
   ```python
   from sentinel.tools import get_registry
   registry = get_registry()
   ```

2. LIST AVAILABLE TOOLS:
   ```python
   all_tools = registry.list_tools()
   for tool in all_tools:
       print(f"{tool['name']}: {tool['description']}")
   ```

3. CALL A TOOL:
   ```python
   result = await registry.call_tool(
       "docker_container_logs",
       container_name="web-service",
       tail=100
   )
   
   if result["success"]:
       print(f"Got {result['lines_returned']} lines")
   else:
       print(f"Error: {result['error']} ({result['error_type']})")
   ```

4. GET TOOL DOCUMENTATION:
   ```python
   tool = registry.get_tool("k8s_pod_status")
   
   print(f"Name: {tool.name}")
   print(f"Description: {tool.description}")
   print(f"Input schema: {tool.input_schema}")
   print(f"Restrictions: {tool.restrictions}")
   print(f"Examples: {tool.examples}")
   ```

5. FILTER BY CATEGORY:
   ```python
   from sentinel.tools import ToolCategory
   
   docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
   k8s_tools = registry.get_tools_by_category(ToolCategory.KUBERNETES)
   ```

============================================================================
TOOL DETAILS
============================================================================

docker_container_logs
---
Purpose: Fetch Docker container logs with automatic sensitive data masking
Category: DOCKER
Read-only: YES

Parameters:
  container_name (required): Name of container to get logs from
  tail (optional): Number of lines to retrieve (default: 100, max: 10000)

Returns:
  {
    "success": bool,
    "container_name": str,
    "logs": str (masked for sensitive data),
    "lines_returned": int,
    "timestamp": str,
    "error": str (if failed),
    "error_type": str (if failed)
  }

Restrictions:
  - Cannot read logs from privileged containers
  - Sensitive data (passwords, keys, tokens) automatically masked
  - Maximum 10,000 lines per request
  - Read-only operation

Examples:
  await registry.call_tool("docker_container_logs", container_name="web-service")
  await registry.call_tool("docker_container_logs", container_name="db-primary", tail=500)

---

docker_container_restart
---
Purpose: Safely restart Docker container with graceful shutdown
Category: DOCKER  
Read-only: NO

Parameters:
  container_name (required): Name of container to restart
  timeout (optional): Seconds to gracefully shutdown (default: 30, max: 300)

Returns:
  {
    "success": bool,
    "container_name": str,
    "status": str (restarted, not_found, permission_denied, etc),
    "duration_seconds": float,
    "timestamp": str,
    "error": str (if failed),
    "error_type": str (if failed)
  }

Restrictions:
  - Cannot restart privileged containers without approval
  - Cannot restart system containers
  - Operations logged for audit trail
  - Maximum timeout of 300 seconds

Examples:
  await registry.call_tool("docker_container_restart", container_name="web-service")
  await registry.call_tool("docker_container_restart", container_name="api", timeout=60)

---

k8s_pod_status
---
Purpose: Query Kubernetes pod status and conditions (READ-ONLY)
Category: KUBERNETES
Read-only: YES

Parameters:
  namespace (required): Kubernetes namespace (e.g. "production", "staging")
  pod_name (required): Name of pod to query
  use_real_kubectl (optional): Use real kubectl (default: False)

Returns:
  {
    "success": bool,
    "namespace": str,
    "pod_name": str,
    "status": {
      "phase": str (Running, Pending, Failed, etc),
      "ready": bool,
      "ready_containers": int,
      "total_containers": int,
      "restart_count": int,
      "age": str,
      "conditions": List[Dict],
      "containers": List[Dict],
      "node": str
    },
    "timestamp": str,
    "error": str (if failed),
    "error_type": str (if failed)
  }

Restrictions:
  - READ-ONLY: Cannot modify cluster state
  - Cannot query protected namespaces without override
  - No logs or events (separate tools needed)

Examples:
  await registry.call_tool("k8s_pod_status", namespace="production", pod_name="web-service-0")
  await registry.call_tool("k8s_pod_status", namespace="staging", pod_name="app-1")

---

k8s_pod_restart
---
Purpose: Restart Kubernetes pod by deletion (pod controller recreates it)
Category: KUBERNETES
Read-only: NO

Parameters:
  namespace (required): Kubernetes namespace containing pod
  pod_name (required): Name of pod to delete/restart
  grace_period (optional): Seconds for graceful shutdown (default: 30, max: 300)
  use_real_kubectl (optional): Use real kubectl (default: False)

Returns:
  {
    "success": bool,
    "namespace": str,
    "pod_name": str,
    "status": str (deleted, not_found, permission_error, etc),
    "grace_period_used": int,
    "duration_seconds": float,
    "timestamp": str,
    "error": str (if failed),
    "error_type": str (if failed)
  }

Restrictions:
  - RESTRICTED: Only pod deletion allowed (no deployment/service changes)
  - PROTECTED: Cannot delete pods in system namespaces
  - RBAC ENFORCED: Service account must have 'delete pods' permission
  - AUDIT LOGGED: All deletions recorded for compliance
  - Grace period must be >= 0 and <= 300

Examples:
  await registry.call_tool("k8s_pod_restart", namespace="production", pod_name="web-service-0")
  await registry.call_tool("k8s_pod_restart", namespace="prod", pod_name="api-1", grace_period=60)

============================================================================
ERROR TYPES
============================================================================

Common error_type values returned by tools:

DOCKER:
  not_found          - Container doesn't exist
  permission        - Permission denied
  timeout           - Operation timed out
  docker_connection_error - Cannot connect to Docker daemon
  operation_in_progress  - Operation already running
  invalid_input     - Bad parameters

KUBERNETES:
  not_found         - Pod or namespace not found
  namespace_not_found - Namespace doesn't exist
  permission_error  - RBAC permission denied
  connection_error  - Cannot connect to cluster
  configuration_error - kubectl config issue
  timeout           - Operation timed out
  termination_timeout - Pod didn't terminate in time
  operation_in_progress - Another operation running
  invalid_input     - Bad parameters

GENERAL:
  system_error      - System-level error
  tool_execution_error - Tool execution failed
  restricted_namespace - Protected namespace (kube-system, etc)

============================================================================
TOOL DISCOVERY EXAMPLES
============================================================================

List all tools:
```python
from sentinel.tools import get_registry
registry = get_registry()
tools = registry.list_tools()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
```

Get all Docker tools:
```python
from sentinel.tools import get_registry, ToolCategory
registry = get_registry()
docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
for tool in docker_tools:
    print(tool.name)
```

Get tool documentation:
```python
registry = get_registry()
description = registry.describe_tool("docker_container_logs")
print(description)
```

Check if tool exists:
```python
registry = get_registry()
tool = registry.get_tool("docker_container_logs")
if tool:
    print(f"Tool found: {tool.name}")
else:
    print("Tool not found")
```

============================================================================
AGENT PATTERNS
============================================================================

PATTERN: Simple tool call
```python
registry = get_registry()
result = await registry.call_tool("docker_container_logs", container_name="web")
if result["success"]:
    state["logs"] = result["logs"]
else:
    state["error"] = result["error"]
```

PATTERN: Tool with error handling
```python
registry = get_registry()
result = await registry.call_tool("k8s_pod_status", namespace="prod", pod_name="web-0")
if not result["success"]:
    if result["error_type"] == "not_found":
        state["status"] = "pod_missing"
    elif result["error_type"] == "permission_error":
        state["escalate"] = True
    else:
        state["error"] = result["error"]
else:
    state["pod_phase"] = result["status"]["phase"]
```

PATTERN: Conditional tool selection
```python
registry = get_registry()
if state["infra"] == "docker":
    tool = "docker_container_logs"
    result = await registry.call_tool(tool, container_name=state["target"])
else:
    tool = "k8s_pod_status"
    result = await registry.call_tool(tool, namespace="prod", pod_name=state["target"])
```

PATTERN: Tool pipeline
```python
registry = get_registry()
logs = await registry.call_tool("docker_container_logs", container_name="web")
if "ERROR" in logs["logs"]:
    restart = await registry.call_tool("docker_container_restart", container_name="web")
    state["restarted"] = restart["success"]
state["logs"] = logs
```

============================================================================
"""

# This is a quick reference module - can be viewed as markdown or executed for info
if __name__ == "__main__":
    print(__doc__)
