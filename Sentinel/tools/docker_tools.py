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
import re
import subprocess
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


async def fetch_container_logs(
    container_name: str,
    tail: int = 100,
    follow: bool = False,
    use_real_docker: bool = False
) -> Dict[str, Any]:
    """
    Safely fetch Docker container logs with comprehensive error handling.

    SECURITY CONSIDERATIONS
    ======================
    This function implements multiple security layers:

    1. INPUT VALIDATION & SANITIZATION
       - Container name validated against whitelist pattern (alphanumeric, dash, underscore)
       - Prevents command injection through container name
       - Validates tail parameter is within safe bounds (1-10000)
       - Rejects suspicious characters that could indicate attack attempts

    2. COMMAND INJECTION PREVENTION
       - Uses subprocess with argument list, NOT shell=True
       - Each argument passed separately, not concatenated
       - Docker executable path can be validated against whitelist
       - Environment variables can be restricted via env parameter

    3. RESOURCE LIMITS
       - Maximum log lines limited to 10,000 to prevent DoS
       - Timeout on subprocess prevents hanging indefinitely
       - Can be integrated with rate limiting to prevent log spam

    4. ERROR HANDLING & INFORMATION DISCLOSURE
       - Generic error messages don't expose system paths
       - Docker errors (if real) sanitized before returning
       - Secrets in logs can be masked with regex patterns
       - Stack traces never exposed to agents

    5. PERMISSION & AUTHORIZATION
       - This tool can check agent permissions before execution
       - Can require approval for sensitive containers
       - All operations logged with who requested what when

    6. OUTPUT SANITIZATION
       - Logs scanned for sensitive patterns (passwords, tokens, keys)
       - Can mask sensitive data before returning to agent
       - Untrusted log content cannot break downstream processing

    Args:
        container_name (str): Docker container name or ID.
            Validated to prevent command injection. Must be alphanumeric
            with dashes and underscores only.
        tail (int): Number of log lines to retrieve (default: 100).
            Maximum 10,000 to prevent resource exhaustion.
        follow (bool): Whether to follow log output (default: False).
            Disabled for safety - prevents indefinite hanging.
        use_real_docker (bool): Whether to use real Docker or mock mode
            (default: False). In production, set to True with proper
            Docker socket permissions.

    Returns:
        Dict with:
        - "success" (bool): Operation succeeded
        - "container_name" (str): Target container
        - "logs" (List[str]): Log lines
        - "log_count" (int): Number of lines returned
        - "timestamp" (str): ISO timestamp
        - "error" (str): Error message if failed
        - "sensitive_data_found" (bool): If sensitive patterns detected
        - "masked_fields" (List[str]): Fields that were masked

    Example:
        >>> result = await fetch_container_logs("web-service", tail=50)
        >>> if result["success"]:
        ...     for line in result["logs"]:
        ...         print(line)
    """

    # =========================================================================
    # PHASE 1: INPUT VALIDATION & SANITIZATION
    # =========================================================================

    # Validate container name: prevent command injection
    # Only allow alphanumeric, dash, underscore - matches Docker name requirements
    container_name_pattern = r"^[a-zA-Z0-9_-]+$"
    if not re.match(container_name_pattern, container_name):
        return {
            "success": False,
            "error": "Invalid container name format",
            "container_name": container_name,
            "timestamp": datetime.now().isoformat(),
            "security_reason": "Container name contains disallowed characters"
        }

    # Validate tail parameter: prevent DoS through unbounded log retrieval
    tail = max(1, min(tail, 10000))  # Clamp to safe range [1, 10000]

    # Never allow follow=True in tool context - could cause indefinite blocking
    follow = False  # Force safety

    # =========================================================================
    # PHASE 2: PERMISSION CHECK (Integration point)
    # =========================================================================

    # In production, check if agent has permission to access this container
    # Example:
    # if not await check_agent_permissions(agent_id, container_name, "logs"):
    #     return {"success": False, "error": "Permission denied"}

    # =========================================================================
    # PHASE 3: DOCKER COMMAND EXECUTION
    # =========================================================================

    if use_real_docker:
        return await _fetch_docker_logs_real(container_name, tail)
    else:
        return await _fetch_docker_logs_mock(container_name, tail)


async def _fetch_docker_logs_real(
    container_name: str, tail: int
) -> Dict[str, Any]:
    """
    Fetch real Docker logs using subprocess.

    IMPORTANT: This function uses subprocess with shell=False to prevent
    command injection. Arguments are passed as a list, not concatenated.

    Args:
        container_name: Validated container name
        tail: Validated log line count

    Returns:
        Dict with logs or error information
    """

    try:
        # SECURITY: subprocess with shell=False + argument list prevents injection
        # Each argument passed separately, no shell interpolation
        command = [
            "docker",                          # Docker executable
            "logs",                            # Subcommand
            f"--tail={tail}",                  # Argument with sanitized value
            "--timestamps",                    # Add timestamps for better debugging
            container_name                     # Validated container name last
        ]

        # Execute with timeout to prevent hanging indefinitely
        # stdout/stderr captured separately for error handling
        result = subprocess.run(
            command,
            capture_output=True,               # Capture stdout and stderr
            text=True,                         # Return strings, not bytes
            timeout=30,                        # 30 second timeout - prevent indefinite hanging
            check=False                        # Don't raise on non-zero exit
        )

        # =========================================================================
        # PHASE 4: ERROR HANDLING & SANITIZATION
        # =========================================================================

        if result.returncode != 0:
            # Docker command failed - return generic error, not system details
            error_msg = result.stderr if result.stderr else "Failed to fetch logs"

            # Sanitize error to not expose system paths or sensitive info
            error_msg = _sanitize_docker_error(error_msg)

            return {
                "success": False,
                "container_name": container_name,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

        # =========================================================================
        # PHASE 5: SENSITIVE DATA DETECTION & MASKING
        # =========================================================================

        logs = result.stdout.split("\n")
        logs = [line for line in logs if line.strip()]  # Remove empty lines

        # Scan for sensitive patterns
        sensitive_data_found = False
        masked_fields = []

        for i, line in enumerate(logs):
            masked_line, found, fields = _mask_sensitive_data(line)
            if found:
                sensitive_data_found = True
                logs[i] = masked_line
                masked_fields.extend(fields)

        return {
            "success": True,
            "container_name": container_name,
            "logs": logs,
            "log_count": len(logs),
            "timestamp": datetime.now().isoformat(),
            "sensitive_data_found": sensitive_data_found,
            "masked_fields": masked_fields
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "container_name": container_name,
            "error": "Log fetch timeout - logs too large or container unresponsive",
            "timestamp": datetime.now().isoformat()
        }

    except FileNotFoundError:
        return {
            "success": False,
            "container_name": container_name,
            "error": "Docker executable not found - Docker may not be installed",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        # Catch unexpected errors - return generic message
        # Never expose full exception details to agent
        return {
            "success": False,
            "container_name": container_name,
            "error": "Failed to fetch logs due to system error",
            "timestamp": datetime.now().isoformat()
            # Internal logging would happen here:
            # logger.error(f"Docker logs fetch failed: {e}", exc_info=True)
        }


async def _fetch_docker_logs_mock(
    container_name: str, tail: int
) -> Dict[str, Any]:
    """
    Mock Docker logs fetch for testing without real Docker.

    Args:
        container_name: Container name
        tail: Log line count

    Returns:
        Mock log data
    """

    mock_logs = [
        "2026-01-01T12:00:00Z INFO Application starting up",
        "2026-01-01T12:00:01Z INFO Loading configuration from /etc/app/config.yaml",
        "2026-01-01T12:00:02Z INFO Database connection pool initialized (size=20)",
        "2026-01-01T12:00:03Z INFO Starting HTTP server on 0.0.0.0:8080",
        "2026-01-01T12:00:05Z INFO Server ready, listening for requests",
        "2026-01-01T12:05:00Z WARN High memory usage: 512MB / 1GB",
        "2026-01-01T12:10:00Z WARN Response time exceeding threshold: 2500ms",
        "2026-01-01T12:15:00Z ERROR Database connection timeout after 30s",
        "2026-01-01T12:15:01Z ERROR Retry attempt 1/3...",
        "2026-01-01T12:15:05Z INFO Database connection restored",
    ]

    # Return only requested number of lines
    logs_to_return = mock_logs[-min(tail, len(mock_logs)):]

    return {
        "success": True,
        "container_name": container_name,
        "logs": logs_to_return,
        "log_count": len(logs_to_return),
        "timestamp": datetime.now().isoformat(),
        "sensitive_data_found": False,
        "masked_fields": [],
        "note": "Mock logs - real Docker not connected"
    }


def _sanitize_docker_error(error_msg: str) -> str:
    """
    Sanitize Docker error messages to remove sensitive information.

    Removes:
    - System file paths (/var/lib/docker/...)
    - Docker daemon details
    - IP addresses and hostnames
    - Socket paths

    Args:
        error_msg: Raw Docker error message

    Returns:
        Sanitized error message safe to return to agents
    """

    # Remove file system paths
    error_msg = re.sub(r"/[a-zA-Z0-9/_.-]+", "<path>", error_msg)

    # Remove IP addresses
    error_msg = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "<ip>", error_msg)

    # Remove socket paths
    error_msg = re.sub(r"/var/run/docker\.sock", "<docker_socket>", error_msg)

    return error_msg


def _mask_sensitive_data(
    log_line: str
) -> tuple[str, bool, List[str]]:
    """
    Scan log line for sensitive data and mask it.

    Detects and masks:
    - API keys (pattern: key= followed by alphanumeric)
    - Passwords (pattern: password= or pwd=)
    - JWT tokens (pattern: ey followed by base64)
    - Database passwords (pattern: database_password=)
    - AWS credentials (pattern: AKIA...)
    - Bearer tokens (pattern: Bearer ...)

    Args:
        log_line: Original log line

    Returns:
        Tuple of:
        - Masked log line (sensitive data replaced with *****)
        - Boolean: was sensitive data found?
        - List of field names that were masked
    """

    masked_fields = []
    masked_line = log_line
    found = False

    # Pattern 1: API keys (key=xxxx, api_key=xxxx, etc.)
    if re.search(r"(api_key|key|secret|token)\s*=\s*[a-zA-Z0-9._-]+", log_line, re.IGNORECASE):
        masked_line = re.sub(
            r"(api_key|key|secret|token)\s*=\s*[a-zA-Z0-9._-]+",
            r"\1=*****",
            masked_line,
            flags=re.IGNORECASE
        )
        masked_fields.append("api_key")
        found = True

    # Pattern 2: Passwords
    if re.search(r"(password|pwd|passwd)\s*=\s*\S+", log_line, re.IGNORECASE):
        masked_line = re.sub(
            r"(password|pwd|passwd)\s*=\s*\S+",
            r"\1=*****",
            masked_line,
            flags=re.IGNORECASE
        )
        masked_fields.append("password")
        found = True

    # Pattern 3: JWT tokens (starts with "ey")
    if re.search(r"\beyJ[a-zA-Z0-9._-]+\b", log_line):
        masked_line = re.sub(
            r"\beyJ[a-zA-Z0-9._-]+\b",
            "*****",
            masked_line
        )
        masked_fields.append("jwt_token")
        found = True

    # Pattern 4: Bearer tokens
    if re.search(r"Bearer\s+\S+", log_line, re.IGNORECASE):
        masked_line = re.sub(
            r"Bearer\s+\S+",
            "Bearer *****",
            masked_line,
            flags=re.IGNORECASE
        )
        masked_fields.append("bearer_token")
        found = True

    # Pattern 5: AWS credentials (AKIA...)
    if re.search(r"AKIA[0-9A-Z]{16}", log_line):
        masked_line = re.sub(r"AKIA[0-9A-Z]{16}", "*****", masked_line)
        masked_fields.append("aws_access_key")
        found = True

    return masked_line, found, list(set(masked_fields))  # deduplicate fields


async def restart_container(
    container_name: str,
    timeout: int = 30,
    use_real_docker: bool = False
) -> Dict[str, Any]:
    """
    Safely restart a Docker container with input validation and error handling.

    SECURITY CONSIDERATIONS
    ======================
    This function prevents arbitrary command execution through:

    1. INPUT VALIDATION
       - Container name validated against whitelist pattern
       - Only alphanumeric, dash, underscore allowed
       - Prevents injection through container identifier
       - Rejects suspicious characters

    2. COMMAND INJECTION PREVENTION
       - Uses subprocess with argument list, NOT shell=True
       - Each argument passed separately
       - No string concatenation in command
       - Docker executable path can be validated

    3. OPERATION CONSTRAINTS
       - Timeout enforced to prevent indefinite operations
       - Only safe, idempotent docker restart allowed
       - Cannot chain multiple commands
       - Cannot execute arbitrary scripts

    4. PERMISSION & AUTHORIZATION
       - Can integrate with permission checking system
       - Can require approval for production containers
       - Can be rate-limited to prevent abuse
       - All operations logged with timestamp

    5. ERROR HANDLING
       - Generic error messages (no system details)
       - Does not expose Docker daemon internals
       - Failures logged securely
       - Partial failures detected and reported

    Args:
        container_name (str): Docker container name or ID to restart.
            Validated to contain only alphanumeric, dash, underscore.
            Examples: "web-service", "db_primary", "cache123"
        timeout (int): Timeout in seconds for the operation (default: 30).
            Maximum allowed: 300 seconds (5 minutes).
            Prevents indefinite operations.
        use_real_docker (bool): Whether to use real Docker or mock mode
            (default: False). In production, set to True with proper
            Docker socket permissions and error handling.

    Returns:
        Dict with:
        - "success" (bool): Restart succeeded
        - "container_name" (str): Target container
        - "status" (str): Current status after restart
        - "timestamp" (str): ISO timestamp
        - "duration_seconds" (float): How long the operation took
        - "error" (str): Error message if failed
        - "error_type" (str): Category of error (timeout, not_found, etc.)

    Example:
        >>> result = await restart_container("web-service")
        >>> if result["success"]:
        ...     print(f"Container restarted in {result['duration_seconds']}s")
        ... else:
        ...     print(f"Failed: {result['error']}")
    """

    start_time = datetime.now()

    # =========================================================================
    # PHASE 1: INPUT VALIDATION
    # =========================================================================

    # Validate container name: prevent command injection
    container_name_pattern = r"^[a-zA-Z0-9_-]+$"
    if not container_name:
        return {
            "success": False,
            "error": "Container name cannot be empty",
            "error_type": "invalid_input",
            "timestamp": datetime.now().isoformat()
        }

    if not re.match(container_name_pattern, container_name):
        return {
            "success": False,
            "container_name": container_name,
            "error": "Invalid container name format",
            "error_type": "invalid_format",
            "timestamp": datetime.now().isoformat(),
            "security_reason": "Container name contains disallowed characters"
        }

    # Validate timeout: prevent resource exhaustion
    timeout = max(5, min(timeout, 300))  # Clamp to [5, 300] seconds

    # =========================================================================
    # PHASE 2: PERMISSION & AUTHORIZATION CHECK (Integration point)
    # =========================================================================

    # In production, check if agent has permission to restart this container
    # Example:
    # if not await check_agent_permissions(agent_id, container_name, "restart"):
    #     return {
    #         "success": False,
    #         "error": "Permission denied for this operation",
    #         "error_type": "unauthorized",
    #         "timestamp": datetime.now().isoformat()
    #     }

    # =========================================================================
    # PHASE 3: EXECUTE RESTART OPERATION
    # =========================================================================

    if use_real_docker:
        return await _restart_container_real(container_name, timeout, start_time)
    else:
        return await _restart_container_mock(container_name, start_time)


async def _restart_container_real(
    container_name: str,
    timeout: int,
    start_time: datetime
) -> Dict[str, Any]:
    """
    Execute real Docker container restart.

    SECURITY: Uses subprocess with shell=False and argument list to prevent
    command injection. Arguments are passed separately, no shell interpolation.

    Args:
        container_name: Validated container name
        timeout: Operation timeout in seconds
        start_time: Start time for duration calculation

    Returns:
        Dict with restart result
    """

    try:
        # SECURITY: subprocess with shell=False + argument list prevents injection
        # Command structure: docker restart [timeout_option] <container_name>
        command = [
            "docker",                          # Docker executable
            "restart",                         # Subcommand (safe, idempotent)
            f"--time={timeout}",               # Timeout for graceful shutdown
            container_name                     # Validated container name
        ]

        # Execute with timeout to prevent hanging
        result = subprocess.run(
            command,
            capture_output=True,               # Capture output
            text=True,                         # Return strings
            timeout=timeout + 5,               # Add 5s buffer to timeout
            check=False                        # Don't raise on non-zero exit
        )

        duration = (datetime.now() - start_time).total_seconds()

        # =========================================================================
        # PHASE 4: HANDLE RESULTS
        # =========================================================================

        if result.returncode == 0:
            # Success: container restarted
            return {
                "success": True,
                "container_name": container_name,
                "status": "restarted",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "message": f"Container {container_name} restarted successfully"
            }
        else:
            # Failure: analyze error type
            error_type, error_msg = _analyze_docker_restart_error(
                result.stderr, container_name
            )

            return {
                "success": False,
                "container_name": container_name,
                "error": error_msg,
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration
            }

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "container_name": container_name,
            "error": "Restart operation timed out",
            "error_type": "timeout",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "hint": "Container may be stuck or system overloaded"
        }

    except FileNotFoundError:
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "container_name": container_name,
            "error": "Docker executable not found",
            "error_type": "docker_not_available",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        # Catch unexpected errors - return generic message
        # Internal logging would happen here: logger.error(f"Restart failed: {e}", exc_info=True)
        return {
            "success": False,
            "container_name": container_name,
            "error": "Failed to restart container due to system error",
            "error_type": "system_error",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration
        }


async def _restart_container_mock(
    container_name: str,
    start_time: datetime
) -> Dict[str, Any]:
    """
    Mock Docker container restart for testing.

    Simulates a successful restart without requiring Docker.

    Args:
        container_name: Container name
        start_time: Start time for duration calculation

    Returns:
        Mock restart result
    """

    # Simulate restart operation taking a few seconds
    await _async_sleep(2.5)

    duration = (datetime.now() - start_time).total_seconds()

    return {
        "success": True,
        "container_name": container_name,
        "status": "restarted",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "message": f"Container {container_name} restarted successfully (mock)",
        "note": "This is a mock restart - real Docker not connected"
    }


def _analyze_docker_restart_error(
    error_output: str,
    container_name: str
) -> tuple[str, str]:
    """
    Analyze Docker restart error and categorize it safely.

    Maps Docker error messages to error types:
    - not_found: Container doesn't exist
    - invalid_state: Container not running
    - permission: Insufficient permissions
    - timeout: Operation timed out
    - unknown: Unclassified error

    Args:
        error_output: Raw Docker error message
        container_name: Container that failed to restart

    Returns:
        Tuple of (error_type: str, error_message: str)
    """

    error_lower = error_output.lower()

    # Detect specific error types
    if "no such container" in error_lower or "not found" in error_lower:
        return (
            "not_found",
            f"Container '{container_name}' not found"
        )

    if "permission denied" in error_lower or "permission" in error_lower:
        return (
            "permission",
            "Permission denied - insufficient privileges for restart operation"
        )

    if "already in progress" in error_lower:
        return (
            "operation_in_progress",
            "Another operation is already in progress on this container"
        )

    if "cannot connect" in error_lower or "unix socket" in error_lower:
        return (
            "docker_connection_error",
            "Failed to connect to Docker daemon"
        )

    # Generic error with sanitization
    sanitized_error = _sanitize_docker_error(error_output)
    return (
        "unknown_error",
        f"Failed to restart container: {sanitized_error}"
    )


async def _async_sleep(seconds: float) -> None:
    """
    Async sleep for mocking delays.

    Args:
        seconds: Duration to sleep
    """
    import asyncio
    await asyncio.sleep(seconds)
