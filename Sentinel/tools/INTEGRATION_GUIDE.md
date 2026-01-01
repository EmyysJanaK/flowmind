"""
SENTINEL TOOL REGISTRATION & AGENT INTEGRATION GUIDE

This document explains how tools are registered in Sentinel and how agents
use them through the tool registry system.

============================================================================
1. TOOL REGISTRATION ARCHITECTURE
============================================================================

Tools flow through three layers:

LAYER 1: TOOL IMPLEMENTATIONS (docker_tools.py, k8s_tools.py)
----------
- Async functions that perform actual work
- Input validation and safety checks
- Error handling and categorization
- Security constraints (RBAC hooks, command injection prevention)
- Examples:
  * fetch_container_logs(): Fetch Docker logs with data masking
  * restart_pod(): Delete Kubernetes pod (triggers restart)

LAYER 2: TOOL REGISTRY (registry.py)
----------
- ToolRegistration: Stores metadata for a single tool
  * name: LLM-friendly name ("docker_container_logs")
  * description: Clear English description for LLMs
  * category: ToolCategory (DOCKER, KUBERNETES, MONITORING, etc.)
  * async_func: The actual function to execute
  * input_schema: JSON Schema for parameters (for LLM understanding)
  * output_schema: JSON Schema for return values
  * restrictions: Safety restrictions (what tool CANNOT do)
  * examples: Usage examples for LLMs

- ToolRegistry: Central registry managing all tools
  * register(): Add tool to registry
  * get_tool(): Retrieve by name
  * get_tools_by_category(): Filter by category
  * list_tools(): All tools for LLM consumption
  * call_tool(): Execute tool with validation
  * describe_tool(): Human-readable documentation

LAYER 3: AGENT ACCESS (agents use registry)
----------
- Agents call get_registry() to access tools
- Agents list available tools and their schemas
- Agents call tools with parameters
- Agents handle results and update state

============================================================================
2. TOOL REGISTRATION PROCESS
============================================================================

When adding a new tool:

1. IMPLEMENT THE TOOL (docker_tools.py or k8s_tools.py)
   ```python
   async def my_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
       '''Tool implementation'''
       # ... actual work ...
       return {"success": True, "result": ...}
   ```

2. CREATE TOOL REGISTRATION (registry.py)
   ```python
   registry.register(
       ToolRegistration(
           name="my_tool",  # LLM-friendly name
           description="Clear description of what tool does",
           category=ToolCategory.DOCKER,
           async_func=my_tool,
           input_schema={
               "type": "object",
               "properties": {
                   "param1": {
                       "type": "string",
                       "description": "What is param1?"
                   },
                   "param2": {
                       "type": "integer",
                       "description": "What is param2?"
                   }
               },
               "required": ["param1"]
           },
           output_schema={...},  # What tool returns
           restrictions=[
               "Cannot do X",
               "Cannot do Y"
           ],
           examples=[
               "Example 1: ...",
               "Example 2: ..."
           ]
       )
   )
   ```

3. VERIFY TOOL WORKS
   ```python
   registry = get_registry()
   result = await registry.call_tool("my_tool", param1="value", param2=20)
   ```

============================================================================
3. AGENT USAGE PATTERNS
============================================================================

PATTERN 1: SIMPLE TOOL CALL
---------
```python
class MyAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        
        result = await registry.call_tool(
            "docker_container_logs",
            container_name=state["target"],
            tail=100
        )
        
        state["logs"] = result.get("logs")
        state["error"] = result.get("error")
        
        return state
```

PATTERN 2: DISCOVERING AVAILABLE TOOLS
---------
```python
class IntelligentAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        
        # What tools are available?
        all_tools = registry.list_tools()
        
        # Filter by category
        docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
        
        # Get tool for decision-making
        tool = registry.get_tool("docker_container_logs")
        print(f"Can I use this? Restrictions: {tool.restrictions}")
        
        return state
```

PATTERN 3: CONDITIONAL TOOL SELECTION
---------
```python
class DecisionAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        
        # Choose tool based on state
        if state["infrastructure"] == "docker":
            tool_name = "docker_container_logs"
        else:
            tool_name = "k8s_pod_status"
        
        # Verify tool exists
        if not registry.get_tool(tool_name):
            state["error"] = f"Tool {tool_name} not available"
            return state
        
        # Call tool
        result = await registry.call_tool(tool_name, ...)
        
        return state
```

PATTERN 4: TOOL PIPELINE
---------
```python
class PipelineAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        
        # 1. Get logs
        logs_result = await registry.call_tool(
            "docker_container_logs",
            container_name=state["container"]
        )
        
        # 2. If error in logs, restart
        if "ERROR" in logs_result["logs"]:
            restart_result = await registry.call_tool(
                "docker_container_restart",
                container_name=state["container"]
            )
            state["action"] = "restarted"
        
        state["logs"] = logs_result
        state["restart"] = restart_result
        
        return state
```

PATTERN 5: ERROR HANDLING
---------
```python
class RobustAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        
        try:
            result = await registry.call_tool(
                "k8s_pod_restart",
                namespace="production",
                pod_name=state["pod"]
            )
            
            if not result["success"]:
                error_type = result.get("error_type")
                
                if error_type == "permission_error":
                    state["escalate_to_human"] = True
                elif error_type == "not_found":
                    state["pod_already_deleted"] = True
                elif error_type == "timeout":
                    state["retry_later"] = True
            
        except ValueError as e:
            state["error"] = f"Tool not found: {e}"
        
        return state
```

============================================================================
4. LANGCHAIN/LANGGRAPH INTEGRATION
============================================================================

APPROACH 1: Direct Tool Access in Agents
----------
```python
from sentinel.tools import get_registry, ToolCategory

class MyAgent(BaseAgent):
    async def run(self, state):
        registry = get_registry()
        result = await registry.call_tool("docker_container_logs", ...)
        return state
```

APPROACH 2: LangChain StructuredTool Wrapper
----------
```python
from langchain.tools import tool
from sentinel.tools import get_registry

@tool
async def docker_logs(container_name: str, tail: int = 100) -> Dict:
    '''Get Docker container logs with sensitive data masked.'''
    registry = get_registry()
    return await registry.call_tool(
        "docker_container_logs",
        container_name=container_name,
        tail=tail
    )

@tool
async def k8s_pod_status(namespace: str, pod_name: str) -> Dict:
    '''Get Kubernetes pod status.'''
    registry = get_registry()
    return await registry.call_tool(
        "k8s_pod_status",
        namespace=namespace,
        pod_name=pod_name
    )

# Use with LangGraph
from langgraph.prebuilt import create_react_agent

tools = [docker_logs, k8s_pod_status]
agent = create_react_agent(llm, tools)
```

APPROACH 3: Dynamic Tool Binding
----------
```python
from sentinel.tools import get_registry, get_langchain_tools
from langgraph.prebuilt import create_react_agent

registry = get_registry()

# Get all registered tools in LangChain format
langchain_tools = get_langchain_tools()

# Or filter by category
docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)

# Create agent with tools
agent = create_react_agent(llm, tools=langchain_tools)
```

APPROACH 4: Tool Binding in Node Function
----------
```python
from langgraph.graph import StateGraph
from sentinel.tools import get_registry

def my_agent_node(state):
    registry = get_registry()
    tools = registry.list_tools()
    
    # Bind tools to agent
    agent_with_tools = my_llm.bind_tools(tools)
    
    # Use agent
    return agent_with_tools.invoke(state)

graph = StateGraph(...)
graph.add_node("agent", my_agent_node)
```

============================================================================
5. TOOL SCHEMAS FOR LLM CONSUMPTION
============================================================================

Tools are registered with JSON schemas that help LLMs understand:
- What parameters to provide
- What values are required/optional
- What types (string, int, float, bool, array, object)
- What the tool returns
- Restrictions and safety rules

Example input schema:
```json
{
    "type": "object",
    "properties": {
        "container_name": {
            "type": "string",
            "description": "Name of Docker container"
        },
        "tail": {
            "type": "integer",
            "description": "Number of log lines",
            "minimum": 1,
            "maximum": 10000,
            "default": 100
        }
    },
    "required": ["container_name"]
}
```

Output schema describes what the tool returns:
```json
{
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "logs": {"type": "string"},
        "error": {"type": "string"},
        "error_type": {"type": "string"}
    }
}
```

============================================================================
6. TOOL DISCOVERY FOR AGENTS
============================================================================

Agents can discover tools in multiple ways:

```python
registry = get_registry()

# Get all tools
all_tools = registry.list_tools()

# Get all tools for agent usage (same)
agent_tools = registry.list_tools_for_agent()

# Get tools by category
docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
k8s_tools = registry.get_tools_by_category(ToolCategory.KUBERNETES)

# Get specific tool
tool = registry.get_tool("docker_container_logs")
print(tool.name)              # "docker_container_logs"
print(tool.description)       # "Fetch Docker container logs..."
print(tool.input_schema)      # JSON schema for inputs
print(tool.restrictions)      # Safety restrictions
print(tool.examples)          # Usage examples for LLMs

# Get human-readable description
description = registry.describe_tool("docker_container_logs")
```

============================================================================
7. TOOL EXECUTION & ERROR HANDLING
============================================================================

Tools return consistent error format:

```python
{
    "success": bool,
    "result": {...},                    # If successful
    "error": str,                       # If failed
    "error_type": str,                  # Category: not_found, permission, timeout, etc
    "timestamp": str                    # ISO timestamp
}
```

Agents should check:
```python
result = await registry.call_tool(...)

if result["success"]:
    # Process result
    process(result)
else:
    # Handle error
    error_type = result.get("error_type")
    
    if error_type == "not_found":
        # Resource doesn't exist
    elif error_type == "permission_error":
        # Need escalation
    elif error_type == "timeout":
        # Try again later
    elif error_type == "invalid_input":
        # Bad parameters
    else:
        # Unknown error
```

============================================================================
8. TOOL SECURITY & RESTRICTIONS
============================================================================

Each tool documents restrictions:

```python
tool = registry.get_tool("k8s_pod_restart")

for restriction in tool.restrictions:
    print(restriction)
    # "RESTRICTED: Only pod deletion allowed"
    # "PROTECTED: Cannot delete pods in system namespaces"
    # "RBAC ENFORCED: Service account must have 'delete pods' permission"
```

Agents should:
1. Read restrictions before calling tool
2. Verify they have proper permissions
3. Understand what the tool CANNOT do
4. Handle permission errors gracefully
5. Escalate to humans when needed

============================================================================
9. REGISTERING CUSTOM TOOLS
============================================================================

To add new tools:

1. Implement async function in appropriate module (docker_tools.py, etc.)
2. Register in registry.py:

```python
from .registry import ToolRegistration, ToolCategory, get_registry

# At bottom of registry.py:
registry = get_registry()
registry.register(
    ToolRegistration(
        name="my_custom_tool",
        description="Clear description",
        category=ToolCategory.MONITORING,
        async_func=my_custom_tool,
        input_schema={...},
        output_schema={...},
        restrictions=[...],
        examples=[...]
    )
)
```

3. Export from __init__.py if needed
4. Test with usage_examples.py
5. Update agent code to use new tool

============================================================================
10. TESTING TOOLS
============================================================================

Run example usage:
```bash
python -m sentinel.tools.usage_examples
```

Test specific tool in Python:
```python
import asyncio
from sentinel.tools import get_registry

async def test():
    registry = get_registry()
    result = await registry.call_tool(
        "docker_container_logs",
        container_name="web-service",
        tail=50
    )
    print(result)

asyncio.run(test())
```

Test tool discovery:
```python
from sentinel.tools import get_registry, ToolCategory

registry = get_registry()
docker_tools = registry.get_tools_by_category(ToolCategory.DOCKER)
print(f"Available Docker tools: {[t.name for t in docker_tools]}")
```

============================================================================
11. LANGCHAIN COMPATIBILITY
============================================================================

Sentinel tools are compatible with LangChain's tool system:

```python
from langchain.tools import Tool
from sentinel.tools import get_registry

registry = get_registry()

# Convert Sentinel tool to LangChain Tool
tool_reg = registry.get_tool("docker_container_logs")
langchain_tool = Tool(
    name=tool_reg.name,
    description=tool_reg.description,
    func=tool_reg.async_func,
    # ... additional LangChain properties
)
```

Or use directly with bind_tools():
```python
tools_metadata = registry.list_tools()
agent = llm.bind_tools(tools_metadata)
```

============================================================================
"""

# This module serves as comprehensive documentation
# It is not meant to be executed directly
