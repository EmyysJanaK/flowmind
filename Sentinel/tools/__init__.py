"""
Tools module for Sentinel multi-agent system.

Tools are the ONLY way agents interact with external infrastructure.

Architecture Principle:
======================
Agents MUST NOT directly execute commands, call APIs, or modify infrastructure.
All agent-infrastructure interactions must go through tools defined in this module.

This separation of concerns provides:
1. Safety: All infrastructure interactions can be monitored/controlled
2. Auditability: Every action has a tool invocation record
3. Testability: Tools can be mocked for testing without real infrastructure
4. Flexibility: Can swap implementations (e.g., Docker → Kubernetes)
5. Authorization: Central point to check permissions before action
6. Rollback: Easy to track what tools were called for recovery

Tool Categories:
- Container tools: Docker operations (run, stop, logs, etc.)
- Orchestration tools: Kubernetes operations (apply, patch, scale, etc.)
- Monitoring tools: Query metrics and logs (future)
- Network tools: DNS, connectivity checks (future)
- Database tools: Query, backup, restore (future)
"""

from .base import BaseTool
from .docker_tools import (
    DockerContainerTool,
    DockerImageTool,
    DockerHealthTool,
    get_docker_tools
)
from .k8s_tools import (
    KubernetesDeploymentTool,
    KubernetesScaleTool,
    KubernetesHealthTool,
    get_k8s_tools
)

__all__ = [
    "BaseTool",
    "DockerContainerTool",
    "DockerImageTool",
    "DockerHealthTool",
    "get_docker_tools",
    "KubernetesDeploymentTool",
    "KubernetesScaleTool",
    "KubernetesHealthTool",
    "get_k8s_tools"
]

