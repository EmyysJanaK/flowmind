"""
Docker container tools for Sentinel agents.

Docker Tools Architecture
=========================

These tools provide the ONLY interface for agents to interact with Docker.
Agents never execute docker commands directly - they must use these tools.

This enforces:
1. Controlled execution: All docker operations logged and monitored
2. Error handling: Failures wrapped in tool-specific error handling
3. State tracking: Each operation records what was done for rollback
4. Authorization: Can check if agent has permission for operation
5. Safety: Can add circuit breakers, rate limits, etc.

Usage:
    tool = DockerContainerTool()
    result = await tool.execute(
        action="restart",
        container_id="my-service",
        timeout=30
    )
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .base import BaseTool


class DockerContainerTool(BaseTool):
    """
    Tool for Docker container lifecycle operations.

    Agents use this tool to:
    - Start/stop/restart containers
    - Check container status
    - View container logs
    - Inspect container configuration

    All operations are logged and tracked for auditability.
    """

    def __init__(self):
        """Initialize Docker container tool."""
        super().__init__(
            name="docker_container",
            description="Manages Docker container lifecycle (start, stop, restart, logs, status)"
        )
        self.last_operations: List[Dict[str, Any]] = []

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a Docker container operation.

        Args:
            action (str): Operation to perform
                - "start": Start a stopped container
                - "stop": Stop a running container
                - "restart": Restart a container
                - "status": Get container status
                - "logs": Retrieve container logs
                - "remove": Remove a container
            container_id (str): Container name or ID
            timeout (int, optional): Operation timeout in seconds
            tail (int, optional): Number of log lines (for "logs" action)

        Returns:
            Dict with:
            - "success": bool - operation succeeded
            - "action": str - what was done
            - "container_id": str - target container
            - "result": Any - operation result
            - "timestamp": str - ISO timestamp
            - "error": str - error message if failed
        """

        action = kwargs.get("action")
        container_id = kwargs.get("container_id")
        timeout = kwargs.get("timeout", 30)
        tail = kwargs.get("tail", 50)

        if not action or not container_id:
            return {
                "success": False,
                "error": "Missing required arguments: action and container_id",
                "timestamp": datetime.now().isoformat()
            }

        # Mock Docker operations
        operations = {
            "start": self._mock_start_container,
            "stop": self._mock_stop_container,
            "restart": self._mock_restart_container,
            "status": self._mock_container_status,
            "logs": self._mock_get_logs,
            "remove": self._mock_remove_container
        }

        if action not in operations:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "timestamp": datetime.now().isoformat()
            }

        try:
            result = await operations[action](container_id, timeout, tail)
            
            # Log the operation
            operation_record = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "container_id": container_id,
                "success": result.get("success", False),
                "duration_seconds": result.get("duration", 0)
            }
            self.last_operations.append(operation_record)

            return result

        except Exception as e:
            return {
                "success": False,
                "action": action,
                "container_id": container_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _mock_start_container(self, container_id: str, timeout: int, **_) -> Dict[str, Any]:
        """Mock starting a container."""
        return {
            "success": True,
            "action": "start",
            "container_id": container_id,
            "result": f"Container {container_id} started",
            "duration": 2.5,
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_stop_container(self, container_id: str, timeout: int, **_) -> Dict[str, Any]:
        """Mock stopping a container."""
        return {
            "success": True,
            "action": "stop",
            "container_id": container_id,
            "result": f"Container {container_id} stopped",
            "duration": 5.0,
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_restart_container(self, container_id: str, timeout: int, **_) -> Dict[str, Any]:
        """Mock restarting a container."""
        return {
            "success": True,
            "action": "restart",
            "container_id": container_id,
            "result": f"Container {container_id} restarted",
            "duration": 3.2,
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_container_status(self, container_id: str, **_) -> Dict[str, Any]:
        """Mock getting container status."""
        return {
            "success": True,
            "action": "status",
            "container_id": container_id,
            "result": {
                "status": "running",
                "uptime_seconds": 3600,
                "cpu_usage_percent": 15.5,
                "memory_usage_mb": 256
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_get_logs(self, container_id: str, timeout: int, tail: int, **_) -> Dict[str, Any]:
        """Mock retrieving container logs."""
        return {
            "success": True,
            "action": "logs",
            "container_id": container_id,
            "result": {
                "lines": tail,
                "recent_logs": [
                    "2026-01-01T12:00:00Z INFO Application started",
                    "2026-01-01T12:00:05Z INFO Server listening on port 8000",
                    "2026-01-01T12:05:00Z WARN High memory usage detected"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _mock_remove_container(self, container_id: str, **_) -> Dict[str, Any]:
        """Mock removing a container."""
        return {
            "success": True,
            "action": "remove",
            "container_id": container_id,
            "result": f"Container {container_id} removed",
            "timestamp": datetime.now().isoformat()
        }


class DockerImageTool(BaseTool):
    """
    Tool for Docker image operations.

    Agents use this tool to:
    - Pull images
    - Build images
    - List images
    - Remove images
    - Inspect image metadata
    """

    def __init__(self):
        """Initialize Docker image tool."""
        super().__init__(
            name="docker_image",
            description="Manages Docker images (pull, build, list, remove, inspect)"
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a Docker image operation.

        Args:
            action (str): Operation to perform
                - "pull": Download image from registry
                - "build": Build image from Dockerfile
                - "list": List available images
                - "inspect": Get image details
                - "remove": Delete image
            image (str): Image name (for pull/remove/inspect)
            dockerfile (str, optional): Path to Dockerfile (for build)
            tag (str, optional): Image tag

        Returns:
            Dict with operation result and metadata
        """

        action = kwargs.get("action")
        image = kwargs.get("image")

        if not action:
            return {
                "success": False,
                "error": "Missing required argument: action",
                "timestamp": datetime.now().isoformat()
            }

        # Mock image operations
        if action == "pull":
            return {
                "success": True,
                "action": "pull",
                "image": image,
                "result": f"Image {image} pulled successfully",
                "timestamp": datetime.now().isoformat()
            }
        elif action == "list":
            return {
                "success": True,
                "action": "list",
                "result": {
                    "images": ["nginx:latest", "postgres:14", "redis:7"]
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "timestamp": datetime.now().isoformat()
            }


class DockerHealthTool(BaseTool):
    """
    Tool for Docker health monitoring and diagnosis.

    Agents use this tool to:
    - Check Docker daemon status
    - Get resource usage statistics
    - Diagnose container health
    - Monitor system capacity
    """

    def __init__(self):
        """Initialize Docker health tool."""
        super().__init__(
            name="docker_health",
            description="Monitors Docker daemon and container health"
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Check Docker health status.

        Args:
            check_type (str): Type of health check
                - "daemon": Check Docker daemon status
                - "system": Get system resource usage
                - "container_health": Check specific container health

        Returns:
            Dict with health status and metrics
        """

        check_type = kwargs.get("check_type", "daemon")

        if check_type == "daemon":
            return {
                "success": True,
                "check_type": "daemon",
                "result": {
                    "status": "healthy",
                    "version": "24.0.0",
                    "containers": 12,
                    "images": 5
                },
                "timestamp": datetime.now().isoformat()
            }
        elif check_type == "system":
            return {
                "success": True,
                "check_type": "system",
                "result": {
                    "cpu_available": 8,
                    "cpu_used": 2.5,
                    "memory_available_gb": 16,
                    "memory_used_gb": 6.2,
                    "disk_available_gb": 100,
                    "disk_used_gb": 45
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Unknown check type: {check_type}",
                "timestamp": datetime.now().isoformat()
            }


def get_docker_tools() -> List[BaseTool]:
    """
    Get all available Docker tools as a collection.

    This is useful for agents that need to work with multiple Docker tools.

    Returns:
        List of initialized Docker tool instances
    """
    return [
        DockerContainerTool(),
        DockerImageTool(),
        DockerHealthTool()
    ]
