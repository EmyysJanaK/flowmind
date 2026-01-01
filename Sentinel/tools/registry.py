"""
Tools registry for Sentinel agents.

This module provides a centralized registry of all available tools that agents
can use. Tools are registered with LangChain/LangGraph compatible formats,
including names, descriptions, and schemas for LLM usage.

TOOL REGISTRATION ARCHITECTURE
===============================

Tools are registered in three layers:

1. TOOL DEFINITIONS (this module)
   - Friendly names and descriptions for LLMs
   - Input/output schemas
   - Documentation and usage examples
   - Categorization (container, orchestration, monitoring, etc.)

2. TOOL IMPLEMENTATIONS (docker_tools.py, k8s_tools.py)
   - Async functions that perform actual work
   - Error handling and validation
   - Security controls and constraints
   - Audit logging integration points

3. AGENT ACCESS (agents use registry.get_tools())
   - Agents request tools by category
   - Registry provides list of tools with documentation
   - Agents invoke tools and handle results
   - LangGraph orchestrates agent-tool interactions

DESIGN RATIONALE
================

Why a tool registry?
- DISCOVERABILITY: Agents and LLMs can discover available tools
- DOCUMENTATION: Each tool has clear name, description, schema
- CONTROL: Single point to audit all agent-accessible operations
- FLEXIBILITY: Easy to enable/disable tools without code changes
- LLM INTEGRATION: OpenAI/Anthropic can understand tool signatures
- TESTING: Mock tools can replace real tools for testing

LLM COMPATIBILITY
================

Tool schemas are compatible with:
- OpenAI Function Calling API
- Anthropic Tool Use (Claude)
- LangChain StructuredTool
- LangGraph tool_node with bind_tools()

LLMs receive:
- Tool name (e.g., "docker_container_logs")
- Tool description (clear English explanation)
- Input schema (JSON Schema for parameters)
- Output schema (what the tool returns)

Example:
    {
        "name": "docker_container_logs",
        "description": "Fetch container logs with sensitive data masking",
        "input_schema": {
            "type": "object",
            "properties": {
                "container_name": {
                    "type": "string",
                    "description": "Name of container to get logs from"
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of log lines to retrieve (default: 100)"
                }
            },
            "required": ["container_name"]
        }
    }
"""

from typing import Any, Dict, List, Callable, Optional
from enum import Enum
import inspect
from .docker_tools import (
    fetch_container_logs,
    restart_container
)
from .k8s_tools import (
    get_pod_status,
    restart_pod
)


class ToolCategory(Enum):
    """Categories of tools available to agents."""
    DOCKER = "docker"              # Docker container operations
    KUBERNETES = "kubernetes"      # Kubernetes cluster operations
    MONITORING = "monitoring"      # Monitoring and observability
    NETWORK = "network"            # Network diagnostics
    DATABASE = "database"          # Database operations


class ToolRegistration:
    """
    Registration record for a single tool.

    Stores all metadata needed for agents and LLMs to discover and use a tool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        async_func: Callable,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        restrictions: Optional[List[str]] = None,
        examples: Optional[List[str]] = None
    ):
        """
        Initialize tool registration.

        Args:
            name: LLM-friendly tool name (e.g., "docker_container_logs")
            description: Clear English description for LLMs
            category: Tool category for organization
            async_func: The actual async function to execute
            input_schema: JSON Schema for input parameters
            output_schema: JSON Schema for output
            restrictions: Safety restrictions (what tool CANNOT do)
            examples: Usage examples for LLMs

        Example:
            >>> tool_reg = ToolRegistration(
            ...     name="docker_container_logs",
            ...     description="Fetch Docker container logs with sensitive data masking",
            ...     category=ToolCategory.DOCKER,
            ...     async_func=fetch_container_logs,
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "container_name": {
            ...                 "type": "string",
            ...                 "description": "Container name or ID"
            ...             },
            ...             "tail": {
            ...                 "type": "integer",
            ...                 "description": "Number of lines (default: 100)"
            ...             }
            ...         },
            ...         "required": ["container_name"]
            ...     },
            ...     output_schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "success": {"type": "boolean"},
            ...             "logs": {"type": "string"},
            ...             "error": {"type": "string"}
            ...         }
            ...     },
            ...     restrictions=[
            ...         "Cannot read logs from privileged containers",
            ...         "Sensitive data is automatically masked"
            ...     ],
            ...     examples=[
            ...         "Get last 50 lines from web-service: "
            ...         "docker_container_logs(container_name='web-service', tail=50)"
            ...     ]
            ... )
        """
        self.name = name
        self.description = description
        self.category = category
        self.async_func = async_func
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.restrictions = restrictions or []
        self.examples = examples or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registration to dictionary for LLM consumption.

        Returns:
            Dict with tool metadata for LLMs
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "restrictions": self.restrictions,
            "examples": self.examples
        }


class ToolRegistry:
    """
    Central registry of all available tools for agents.

    Provides:
    - Tool discovery (get tools by category)
    - Tool invocation (async call with validation)
    - Tool documentation (for LLMs and agents)
    - Tool filtering (by permissions, category, etc.)
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, ToolRegistration] = {}

    def register(self, tool: ToolRegistration) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: ToolRegistration object

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolRegistration]:
        """
        Get a registered tool by name.

        Args:
            name: Tool name (e.g., "docker_container_logs")

        Returns:
            ToolRegistration or None if not found
        """
        return self._tools.get(name)

    def get_tools_by_category(
        self, category: ToolCategory
    ) -> List[ToolRegistration]:
        """
        Get all tools in a category.

        Args:
            category: ToolCategory to filter by

        Returns:
            List of ToolRegistration objects
        """
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools with metadata.

        Returns:
            List of tool dictionaries (for LLM consumption)
        """
        return [tool.to_dict() for tool in self._tools.values()]

    def list_tools_for_agent(
        self, categories: Optional[List[ToolCategory]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools appropriate for agent usage.

        Returns only tools matching requested categories with full
        descriptions and schemas for agent decision-making.

        Args:
            categories: Tool categories to include (all if None)

        Returns:
            List of tool dictionaries formatted for agent use
        """
        if not categories:
            categories = list(ToolCategory)

        tools = []
        for category in categories:
            tools.extend(self.get_tools_by_category(category))

        return [tool.to_dict() for tool in tools]

    async def call_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call a registered tool with validation.

        SAFETY: Validates that tool is registered before executing.
        This prevents agents from calling arbitrary functions.

        Args:
            tool_name: Name of registered tool
            **kwargs: Arguments to pass to tool

        Returns:
            Result from tool execution

        Raises:
            ValueError: If tool not found or execution fails
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        try:
            # Call the async function
            result = await tool.async_func(**kwargs)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "tool_execution_error"
            }

    def describe_tool(self, tool_name: str) -> str:
        """
        Get human-readable description of a tool.

        Args:
            tool_name: Name of tool

        Returns:
            Formatted description including name, purpose, restrictions, examples
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found"

        desc = f"Tool: {tool.name}\n"
        desc += f"Category: {tool.category.value}\n"
        desc += f"Description: {tool.description}\n"

        if tool.restrictions:
            desc += "Restrictions:\n"
            for restriction in tool.restrictions:
                desc += f"  - {restriction}\n"

        if tool.examples:
            desc += "Examples:\n"
            for example in tool.examples:
                desc += f"  - {example}\n"

        return desc


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry


# ============================================================================
# DOCKER TOOLS REGISTRATION
# ============================================================================

_global_registry.register(
    ToolRegistration(
        name="docker_container_logs",
        description=(
            "Fetch logs from a Docker container with automatic sensitive data masking. "
            "Returns the last N log lines with passwords, API keys, and tokens redacted. "
            "Safe to use on any container without risk of exposing credentials."
        ),
        category=ToolCategory.DOCKER,
        async_func=fetch_container_logs,
        input_schema={
            "type": "object",
            "properties": {
                "container_name": {
                    "type": "string",
                    "description": (
                        "Name or ID of Docker container. Must be alphanumeric with dashes. "
                        "Examples: 'web-service', 'db-primary', 'redis_cache'"
                    )
                },
                "tail": {
                    "type": "integer",
                    "description": (
                        "Number of most recent log lines to retrieve. "
                        "Default: 100. Higher values may return more context but slower."
                    ),
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 100
                }
            },
            "required": ["container_name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether log retrieval succeeded"
                },
                "container_name": {
                    "type": "string",
                    "description": "Name of the container"
                },
                "logs": {
                    "type": "string",
                    "description": "Container logs with sensitive data masked"
                },
                "lines_returned": {
                    "type": "integer",
                    "description": "Number of log lines returned"
                },
                "timestamp": {
                    "type": "string",
                    "description": "ISO timestamp when logs were fetched"
                },
                "error": {
                    "type": "string",
                    "description": "Error message if retrieval failed"
                },
                "error_type": {
                    "type": "string",
                    "description": "Category of error (not_found, permission, timeout, etc.)"
                }
            }
        },
        restrictions=[
            "Cannot fetch logs from system containers (docker, containerd, etc.)",
            "Sensitive data is automatically masked (passwords, API keys, JWT tokens)",
            "Maximum 10,000 lines per request to prevent memory issues",
            "Read-only operation - does not modify container state"
        ],
        examples=[
            "Get last 100 lines from web-service: docker_container_logs(container_name='web-service')",
            "Get last 500 lines from database: docker_container_logs(container_name='postgres-db', tail=500)",
            "Check error logs in production container: docker_container_logs(container_name='api-server', tail=200)"
        ]
    )
)

_global_registry.register(
    ToolRegistration(
        name="docker_container_restart",
        description=(
            "Safely restart a Docker container with graceful shutdown. "
            "Container is given time to shut down cleanly before forced termination. "
            "Returns status of restart operation and time taken."
        ),
        category=ToolCategory.DOCKER,
        async_func=restart_container,
        input_schema={
            "type": "object",
            "properties": {
                "container_name": {
                    "type": "string",
                    "description": (
                        "Name or ID of Docker container to restart. "
                        "Must be alphanumeric with dashes."
                    )
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Seconds to wait for graceful shutdown before force kill. "
                        "Default: 30. Higher values allow more time for cleanup."
                    ),
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30
                }
            },
            "required": ["container_name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether restart succeeded"
                },
                "container_name": {
                    "type": "string",
                    "description": "Name of restarted container"
                },
                "status": {
                    "type": "string",
                    "description": "Result status (restarted, not_found, permission_denied, etc.)"
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "How long the restart operation took"
                },
                "timestamp": {
                    "type": "string",
                    "description": "ISO timestamp when restart was initiated"
                },
                "error": {
                    "type": "string",
                    "description": "Error message if restart failed"
                },
                "error_type": {
                    "type": "string",
                    "description": "Category of error (not_found, permission, timeout, etc.)"
                }
            }
        },
        restrictions=[
            "Cannot restart privileged containers without approval",
            "Cannot restart system containers (docker, containerd)",
            "Operations logged for audit trail",
            "Maximum timeout of 300 seconds to prevent hanging"
        ],
        examples=[
            "Restart web-service gracefully: docker_container_restart(container_name='web-service')",
            "Restart with 60 second timeout: docker_container_restart(container_name='api-server', timeout=60)",
            "Quick restart with 10 second timeout: docker_container_restart(container_name='cache', timeout=10)"
        ]
    )
)

# ============================================================================
# KUBERNETES TOOLS REGISTRATION
# ============================================================================

_global_registry.register(
    ToolRegistration(
        name="k8s_pod_status",
        description=(
            "Query Kubernetes pod status and state information. "
            "READ-ONLY operation that returns current pod conditions, phase, "
            "container statuses, restart counts, and resource usage. "
            "No modifications to cluster state."
        ),
        category=ToolCategory.KUBERNETES,
        async_func=get_pod_status,
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": (
                        "Kubernetes namespace containing the pod. "
                        "Must be alphanumeric with dashes. "
                        "Examples: 'production', 'staging', 'kube-system'"
                    )
                },
                "pod_name": {
                    "type": "string",
                    "description": (
                        "Name of the pod to query. "
                        "Must be alphanumeric with dashes. "
                        "Examples: 'web-service-7d9f4c', 'db-primary-0'"
                    )
                },
                "use_real_kubectl": {
                    "type": "boolean",
                    "description": (
                        "Whether to use real kubectl (True) or mock mode (False). "
                        "Default: False. Set True only with kubectl configured."
                    ),
                    "default": False
                }
            },
            "required": ["namespace", "pod_name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether query succeeded"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace"
                },
                "pod_name": {
                    "type": "string",
                    "description": "Pod name"
                },
                "status": {
                    "type": "object",
                    "description": "Pod status details (phase, ready, conditions, etc.)"
                },
                "timestamp": {
                    "type": "string",
                    "description": "ISO timestamp when status was queried"
                },
                "error": {
                    "type": "string",
                    "description": "Error message if query failed"
                },
                "error_type": {
                    "type": "string",
                    "description": "Error category (not_found, namespace_not_found, etc.)"
                }
            }
        },
        restrictions=[
            "READ-ONLY: Cannot modify cluster state",
            "Cannot query pods in protected namespaces without explicit override",
            "Returns pod phase, conditions, and container status",
            "No access to pod logs or events (use separate tools)"
        ],
        examples=[
            "Check if web-service pod is ready: k8s_pod_status(namespace='production', pod_name='web-service-7d9f4c')",
            "Query pod in staging: k8s_pod_status(namespace='staging', pod_name='app-0')",
            "Check database pod status: k8s_pod_status(namespace='production', pod_name='postgres-0')"
        ]
    )
)

_global_registry.register(
    ToolRegistration(
        name="k8s_pod_restart",
        description=(
            "Restart a Kubernetes pod by deleting it (pod controller will recreate it). "
            "Pod is given grace period to shut down gracefully before forced termination. "
            "Typically used to recover from hung or stuck pods. "
            "RESTRICTED: Only allows pod deletion in whitelisted namespaces via RBAC."
        ),
        category=ToolCategory.KUBERNETES,
        async_func=restart_pod,
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": (
                        "Kubernetes namespace containing the pod. "
                        "Must be alphanumeric with dashes. "
                        "RBAC restricts to whitelisted namespaces only."
                    )
                },
                "pod_name": {
                    "type": "string",
                    "description": (
                        "Name of the pod to restart/delete. "
                        "Must be alphanumeric with dashes."
                    )
                },
                "grace_period": {
                    "type": "integer",
                    "description": (
                        "Seconds to gracefully terminate pod before forced kill. "
                        "Default: 30. Must match pod's terminationGracePeriodSeconds."
                    ),
                    "minimum": 0,
                    "maximum": 300,
                    "default": 30
                },
                "use_real_kubectl": {
                    "type": "boolean",
                    "description": (
                        "Whether to use real kubectl (True) or mock mode (False). "
                        "Default: False."
                    ),
                    "default": False
                }
            },
            "required": ["namespace", "pod_name"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether deletion succeeded"
                },
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace"
                },
                "pod_name": {
                    "type": "string",
                    "description": "Pod that was deleted"
                },
                "status": {
                    "type": "string",
                    "description": "Result (deleted, not_found, permission_error, etc.)"
                },
                "grace_period_used": {
                    "type": "integer",
                    "description": "Grace period that was applied"
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "How long deletion took"
                },
                "timestamp": {
                    "type": "string",
                    "description": "ISO timestamp when deletion was initiated"
                },
                "error": {
                    "type": "string",
                    "description": "Error message if deletion failed"
                },
                "error_type": {
                    "type": "string",
                    "description": "Error category (not_found, permission_error, timeout, etc.)"
                }
            }
        },
        restrictions=[
            "RESTRICTED: Only pod deletion allowed (no deployment/service/config modifications)",
            "PROTECTED: Cannot delete pods in system namespaces (kube-system, kube-public, etc.)",
            "RBAC ENFORCED: Service account must have 'delete pods' permission",
            "AUDIT LOGGED: All pod deletions recorded for compliance",
            "Grace period must be >= 0 and <= 300 seconds"
        ],
        examples=[
            "Restart stuck web-service: k8s_pod_restart(namespace='production', pod_name='web-service-7d9f4c')",
            "Restart with 60s grace: k8s_pod_restart(namespace='production', pod_name='api-0', grace_period=60)",
            "Quick restart with 10s: k8s_pod_restart(namespace='staging', pod_name='worker-0', grace_period=10)"
        ]
    )
)


def initialize_registry() -> ToolRegistry:
    """
    Initialize and return the global tool registry.

    Called at application startup to ensure all tools are registered.

    Returns:
        The initialized global ToolRegistry
    """
    return get_registry()
