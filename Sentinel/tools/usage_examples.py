"""
Example usage of registered tools by Sentinel agents.

This module demonstrates how agents use the tool registry to:
1. Discover available tools
2. Get tool documentation for decision-making
3. Call tools with parameters
4. Handle tool results

AGENT WORKFLOW WITH TOOLS
==========================

1. Agent receives state from LangGraph
2. Agent calls get_registry() to access tools
3. Agent lists available tools: registry.list_tools_for_agent(categories=[...])
4. Agent reads tool descriptions and schemas
5. Agent calls tool with parameters: await registry.call_tool(tool_name, **kwargs)
6. Agent processes result and updates state
7. Agent returns updated state to LangGraph

LANGCHAIN INTEGRATION
====================

For LangChain/LangGraph integration, agents can:

1. With LangChain StructuredTool:
   ```python
   from langchain.tools import tool
   from sentinel.tools import get_registry
   
   @tool
   async def docker_logs_tool(container_name: str, tail: int = 100):
       registry = get_registry()
       return await registry.call_tool(
           "docker_container_logs",
           container_name=container_name,
           tail=tail
       )
   ```

2. With bind_tools() in LangGraph:
   ```python
   from langgraph.prebuilt import create_react_agent
   from sentinel.tools import get_registry
   
   registry = get_registry()
   tools = registry.list_tools_for_agent()
   
   agent = create_react_agent(
       llm,
       tools=tools
   )
   ```

3. Direct tool access in agents:
   ```python
   class DiagnosisAgent(BaseAgent):
       async def run(self, state):
           registry = get_registry()
           
           # Get available Docker tools
           docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
           
           # Call a tool
           result = await registry.call_tool(
               "docker_container_logs",
               container_name=state.get("target_container"),
               tail=200
           )
           
           # Process result
           if result["success"]:
               state["logs"] = result["logs"]
           else:
               state["error"] = result["error"]
           
           return state
   ```
"""

from typing import Any, Dict, List
from sentinel.tools import (
    get_registry,
    ToolCategory,
    ToolRegistry
)


async def example_agent_with_docker_tools() -> Dict[str, Any]:
    """
    Example: Agent querying Docker tools.

    Shows how an agent discovers and uses Docker tools.
    """
    registry = get_registry()

    # 1. List all Docker tools available
    docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
    print(f"Available Docker tools: {len(docker_tools)}")
    for tool in docker_tools:
        print(f"  - {tool.name}: {tool.description}")

    # 2. Get specific tool documentation
    tool_info = registry.get_tool("docker_container_logs")
    print(f"\nTool details:")
    print(f"  Name: {tool_info.name}")
    print(f"  Category: {tool_info.category.value}")
    print(f"  Description: {tool_info.description}")
    print(f"  Restrictions:")
    for restriction in tool_info.restrictions:
        print(f"    - {restriction}")

    # 3. Call the tool
    result = await registry.call_tool(
        "docker_container_logs",
        container_name="web-service",
        tail=50
    )

    print(f"\nTool result:")
    print(f"  Success: {result.get('success')}")
    if result.get('success'):
        print(f"  Container: {result.get('container_name')}")
        print(f"  Lines returned: {result.get('lines_returned')}")
    else:
        print(f"  Error: {result.get('error')}")

    return result


async def example_agent_with_kubernetes_tools() -> Dict[str, Any]:
    """
    Example: Agent querying Kubernetes tools.

    Shows how an agent discovers and uses Kubernetes tools.
    """
    registry = get_registry()

    # 1. List all Kubernetes tools
    k8s_tools = registry.get_tools_by_category(ToolCategory.KUBERNETES)
    print(f"Available Kubernetes tools: {len(k8s_tools)}")
    for tool in k8s_tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # 2. Get tool for pod status
    pod_status_tool = registry.get_tool("k8s_pod_status")
    print(f"\nPod status tool input schema:")
    for prop_name, prop_schema in pod_status_tool.input_schema["properties"].items():
        print(f"  - {prop_name}: {prop_schema.get('description', 'No description')}")

    # 3. Query pod status
    result = await registry.call_tool(
        "k8s_pod_status",
        namespace="production",
        pod_name="web-service-7d9f4c"
    )

    print(f"\nPod status result:")
    print(f"  Success: {result.get('success')}")
    if result.get('success'):
        status = result.get('status', {})
        print(f"  Pod: {result.get('pod_name')}")
        print(f"  Phase: {status.get('phase')}")
        print(f"  Ready: {status.get('ready')}")
    else:
        print(f"  Error: {result.get('error')} ({result.get('error_type')})")

    return result


async def example_agent_tool_discovery() -> None:
    """
    Example: Agent discovering all available tools.

    Shows how agents can list and understand all tools they can use.
    """
    registry = get_registry()

    # Get all tools
    all_tools = registry.list_tools()
    print(f"Total tools registered: {len(all_tools)}\n")

    # Group by category
    tools_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for tool in all_tools:
        category = tool["category"]
        if category not in tools_by_category:
            tools_by_category[category] = []
        tools_by_category[category].append(tool)

    # Print organized view
    for category, tools in sorted(tools_by_category.items()):
        print(f"{category.upper()} TOOLS ({len(tools)}):")
        for tool in tools:
            print(f"  {tool['name']}")
            print(f"    - {tool['description'][:80]}...")
            print(f"    - Input: {list(tool['input_schema'].get('properties', {}).keys())}")
            if tool.get('restrictions'):
                print(f"    - Restrictions: {len(tool['restrictions'])} rules")
        print()


async def example_agent_tool_description() -> None:
    """
    Example: Agent getting detailed tool documentation.

    Shows how agents can understand tool capabilities and restrictions.
    """
    registry = get_registry()

    # Get detailed description of a tool
    description = registry.describe_tool("k8s_pod_restart")
    print(description)

    # Get another tool
    description = registry.describe_tool("docker_container_restart")
    print(description)


async def example_agent_conditional_tool_use() -> Dict[str, Any]:
    """
    Example: Agent deciding which tool to use based on state.

    Shows how agents might choose between tools based on what they're trying
    to diagnose or fix.
    """
    registry = get_registry()

    # Simulate agent state
    incident_state = {
        "target": "web-service",
        "infrastructure": "docker",  # or "kubernetes"
        "action": "get_logs"
    }

    # Decide which tool to use
    if incident_state["infrastructure"] == "docker":
        if incident_state["action"] == "get_logs":
            tool_name = "docker_container_logs"
        elif incident_state["action"] == "restart":
            tool_name = "docker_container_restart"
    elif incident_state["infrastructure"] == "kubernetes":
        if incident_state["action"] == "get_status":
            tool_name = "k8s_pod_status"
        elif incident_state["action"] == "restart":
            tool_name = "k8s_pod_restart"

    # Verify tool exists
    tool = registry.get_tool(tool_name)
    if not tool:
        return {"error": f"Tool {tool_name} not found"}

    # Log tool choice
    print(f"Selected tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Input requirements: {tool.input_schema['required']}")

    # Call appropriate tool
    if incident_state["infrastructure"] == "docker":
        result = await registry.call_tool(
            tool_name,
            container_name=incident_state["target"]
        )
    else:
        result = await registry.call_tool(
            tool_name,
            namespace="production",
            pod_name=incident_state["target"]
        )

    return result


# ============================================================================
# LANGCHAIN TOOL INTEGRATION EXAMPLE
# ============================================================================

def get_langchain_tools() -> List[Dict[str, Any]]:
    """
    Get tools in LangChain format for use with language models.

    Returns:
        List of tool definitions compatible with LangChain
    """
    registry = get_registry()
    return registry.list_tools()


def get_langchain_tool_by_name(name: str) -> Dict[str, Any]:
    """
    Get a specific tool in LangChain format.

    Args:
        name: Tool name

    Returns:
        Tool definition for LangChain
    """
    registry = get_registry()
    tool = registry.get_tool(name)
    if not tool:
        raise ValueError(f"Tool '{name}' not found")
    return tool.to_dict()


if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 70)
        print("SENTINEL TOOL REGISTRY - EXAMPLES")
        print("=" * 70)
        print()

        print("1. TOOL DISCOVERY")
        print("-" * 70)
        await example_agent_tool_discovery()

        print("\n2. DOCKER TOOLS USAGE")
        print("-" * 70)
        await example_agent_with_docker_tools()

        print("\n3. KUBERNETES TOOLS USAGE")
        print("-" * 70)
        await example_agent_with_kubernetes_tools()

        print("\n4. CONDITIONAL TOOL SELECTION")
        print("-" * 70)
        result = await example_agent_conditional_tool_use()
        print(f"Result: {result}")

    asyncio.run(main())
