# Sentinel

A LangGraph-based multi-agent system for intelligent workflow orchestration and autonomous task execution.

## Project Goal

Sentinel is designed to create a flexible, scalable framework for building AI-powered multi-agent systems using LangGraph. It enables agents to collaborate, share context, and execute complex workflows with memory management and real-time API interfaces.

## Key Features

- **Multi-Agent Architecture**: Modular agent system with specialized roles
- **Graph-Based Workflows**: LangGraph-powered state management and agent coordination
- **Tool System**: Extensible tool framework for agents to interact with external systems
- **Memory Management**: Persistent conversation history and shared state storage
- **REST API**: FastAPI-based HTTP interface for workflow interactions
- **Async Support**: Full async/await support for non-blocking operations

## Project Structure

```
Sentinel/
├── agents/              # Individual agent implementations
│   └── base.py         # BaseAgent abstract class
├── graph/              # LangGraph workflow definitions
│   ├── state.py        # GraphState dataclass
│   └── workflow.py     # Workflow graph creation
├── tools/              # Tool definitions for agents
│   └── base.py         # BaseTool abstract class
├── memory/             # Memory and state management
│   └── store.py        # MemoryStore implementation
├── api/                # REST API and server setup
│   ├── server.py       # FastAPI application factory
│   └── routes/         # API endpoint definitions
└── README.md           # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip or uv

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from sentinel.graph import create_workflow
from sentinel.memory import MemoryStore

# Create workflow
workflow = create_workflow()

# Initialize memory
memory = MemoryStore()

# Run workflow (to be implemented)
# result = await workflow.invoke(...)
```

## Development

### Adding a New Agent

1. Create a new file in `agents/` extending `BaseAgent`
2. Implement the `execute()` method
3. Register the agent in the workflow graph

### Adding a New Tool

1. Create a new file in `tools/` extending `BaseTool`
2. Implement the `execute()` method
3. Make tools available to agents

## Architecture

- **Agents**: Specialized AI entities that handle specific tasks
- **Graph**: LangGraph-based coordination layer managing agent interactions
- **Tools**: External capabilities agents can leverage
- **Memory**: Shared context and conversation history
- **API**: HTTP interface for external system integration

## License

MIT

## Contributing

Contributions are welcome! Please ensure code follows the project structure and patterns.
