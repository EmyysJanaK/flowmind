# Sentinel

A **LangGraph-based multi-agent system** for intelligent incident investigation, analysis, and autonomous remediation. Sentinel implements a self-correcting workflow where specialized agents collaborate to resolve production incidents through iterative investigation and validation cycles.

## Project Goal

Sentinel demonstrates how to build **resilient, self-improving multi-agent systems** using LangGraph. Instead of linear automation chains, Sentinel uses a cyclic graph architecture that enables agents to:
- Investigate incidents from multiple angles
- Adapt strategies based on feedback
- Loop back for re-investigation if initial remediation fails
- Escalate to humans when issues exceed automation capabilities
- Maintain complete audit trails of all decisions

## Key Features

- **Multi-Agent Architecture**: Four specialized agents with distinct responsibilities
- **Cyclic Graph Workflow**: Self-correcting loops that retry failed remediations
- **State-Based Coordination**: Shared state object flowing through all agents
- **Self-Reflection**: Post-execution validation with automatic retry strategies
- **Safety by Design**: Separation of planning from execution, human approval gates
- **Complete Auditability**: Conversation history + metadata tracking all decisions
- **Async Support**: Full async/await for non-blocking operation
- **Memory Management**: Persistent context across investigation cycles

## Architecture Overview

### Agent Responsibilities

#### 🔍 **Detective Agent**
- **Role**: Incident analyzer and evidence gatherer
- **Responsibilities**:
  - Extract key details from incident description
  - Identify affected services and error patterns
  - Generate multiple root cause hypotheses
  - Rank hypotheses by confidence score
  - Collect and structure evidence
- **Output**: Hypotheses with supporting evidence and confidence scores
- **Re-runs**: If initial remediation fails, re-investigates with new context

#### 🔬 **Researcher Agent**
- **Role**: Solution discoverer and knowledge synthesizer
- **Responsibilities**:
  - Query knowledge base for solutions to each hypothesis
  - Retrieve relevant runbooks and best practices
  - Generate fixes ranked by effectiveness + feasibility
  - Create detailed implementation plans
  - Suggest alternative approaches
- **Output**: Prioritized fixes with step-by-step procedures
- **Future**: Integrates with RAG for LLM-generated recommendations

#### 👨‍💼 **Operator Agent**
- **Role**: Remediation strategist and planner
- **Responsibilities**:
  - Evaluate researcher recommendations against current context
  - Select optimal fix based on risk/effectiveness trade-off
  - Create detailed action plan with safety checkpoints
  - Design rollback procedures
  - Assess execution risks and success criteria
- **Output**: Actionable plan ready for execution
- **Key Principle**: Plans but does NOT execute (execution is separate)

#### ✅ **Reflection Agent** (via reflection module)
- **Role**: Validator and decision maker
- **Responsibilities**:
  - Validate whether remediation succeeded
  - Compare actual results against success criteria
  - Identify failure reasons if applicable
  - Decide: Success / Retry / Try Alternative / Escalate
  - Prevent infinite loops with retry limits
- **Output**: Reflection outcome + retry strategy
- **Enables**: Self-correcting loops and adaptive agent behavior

### Why LangGraph Instead of Linear Chains?

Traditional linear automation chains execute sequentially and cannot adapt:
```
Input → Agent1 → Agent2 → Agent3 → Output
```

**Problems with linear chains:**
- ❌ No feedback loops - can't react to failures
- ❌ All-or-nothing execution - no partial recovery
- ❌ One-shot planning - can't try alternatives
- ❌ Limited visibility - can't validate mid-workflow
- ❌ No human intervention points - can't escalate gracefully

**LangGraph's advantages:**
- ✅ **Conditional routing**: Route based on reflection outcomes
- ✅ **Cyclic flows**: Loop back to re-investigate with new strategies
- ✅ **Complex state management**: Rich state object with full history
- ✅ **Explicit decision points**: Route nodes that implement business logic
- ✅ **Flexible execution**: Can retry, skip, or escalate at any point
- ✅ **Visualization support**: Graph structure can be visualized

**Sentinel's cyclic flow enables:**
1. **Resilience**: Failed fixes trigger automatic retry attempts
2. **Adaptation**: System learns from failures and tries different approaches
3. **Safety**: Maximum retry count prevents infinite loops
4. **Transparency**: Every decision recorded in conversation history
5. **Human-in-loop**: Escalates when automation reaches limits

## Workflow: The Agent Loop

```
┌────────────────────────────────────────────────────────────────┐
│                      SENTINEL WORKFLOW                        │
└────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────┐
     │         INCIDENT DESCRIPTION            │
     │  "DB timeout every morning at 9 AM"    │
     └──────────────────┬──────────────────────┘
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │      🔍 DETECTIVE AGENT                │
     │  • Extract incident details             │
     │  • Gather evidence                      │
     │  • Generate hypotheses (3)              │
     │  • Rank by confidence                   │
     └──────────────────┬──────────────────────┘
                        │
        detective_findings: {
          hypotheses: [...],
          evidence: {...}
        }
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │     🔬 RESEARCHER AGENT                │
     │  • Query knowledge base                 │
     │  • Generate fixes (6 per hypothesis)    │
     │  • Rank by effectiveness               │
     │  • Create implementation plans          │
     └──────────────────┬──────────────────────┘
                        │
        researcher_recommendations: {
          fixes: [...],
          primary_recommendation: {...},
          alternatives: [...]
        }
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │     👨‍💼 OPERATOR AGENT                   │
     │  • Select best fix                      │
     │  • Create action plan                   │
     │  • Design rollback procedure            │
     │  • Assess risks                         │
     │  • Add safety checkpoints               │
     └──────────────────┬──────────────────────┘
                        │
        operator_action_plan: {
          action_plan: {phases: [...]},
          risk_assessment: {...},
          rollback_plan: {...}
        }
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │     ⚙️  EXECUTE REMEDIATION            │
     │  • Run action plan steps                │
     │  • Track progress & errors              │
     │  • Generate execution log               │
     └──────────────────┬──────────────────────┘
                        │
        execution_result: {
          status: "success",
          errors: [],
          duration_seconds: 45.5
        }
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │     ✅ REFLECTION & VALIDATION          │
     │  • Check success criteria               │
     │  • Identify issues (if any)             │
     │  • Decide: Success/Retry/Escalate      │
     └──────────────────┬──────────────────────┘
                        │
                        │ Decision Router
                        │
        ┌───────────────┼───────────────┬──────────────┐
        │               │               │              │
        ▼               ▼               ▼              ▼
    ┌────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐
    │SUCCESS │    │ RETRY  │    │ESCALATE  │    │  ABORT  │
    └────────┘    └────┬───┘    └──────────┘    └─────────┘
      ✓ Done           │              │              │
                       └──────┬───────┴──────────────┘
                              │
                    [Loop back to Detective]
                       Max 3 retries
                              │
                        (with adjustments)
                              │
                              ▼
                    ┌──────────────────┐
                    │   COMPLETION     │
                    │  Status: success │
                    │  or escalated    │
                    └──────────────────┘

═══════════════════════════════════════════════════════════════════

Key Characteristics:
• Cyclic: Loops back for re-investigation if needed
• Self-correcting: Tries adjustments before escalating
• Bounded: Max 3 retries prevents infinite loops
• Auditable: All decisions in conversation_history
• Safe: Execution separated from planning
• Human-integrated: Escalates when needed
```

## State Flow Through Agents

Each agent reads and writes to a shared `SentinelState` object:

```python
SentinelState {
  # User input
  user_input: "Database timeout at 9 AM"
  
  # Agent outputs (populated sequentially)
  detective_findings: {...}           # ← Detective writes
  researcher_recommendations: {...}    # ← Researcher writes
  operator_action_plan: {...}          # ← Operator writes
  execution_result: {...}              # ← Executor writes (future)
  
  # Metadata
  conversation_history: [...]  # All agent messages + decisions
  metadata: {
    workflow_id: "...",
    retry_count: 2,
    reflection_history: [...],
    needs_reinvestigation: False
  }
  
  # Status tracking
  status: "planning" | "executing" | "completed" | "failed"
  error: "..."
}
```

## Project Structure

```
Sentinel/
├── agents/                      # Individual agent implementations
│   ├── __init__.py
│   ├── base.py                  # BaseAgent abstract class
│   ├── detective_agent.py        # Analyzes incidents
│   ├── researcher_agent.py       # Researches solutions
│   └── operator_agent.py         # Plans remediation
│
├── graph/                       # LangGraph workflow
│   ├── __init__.py
│   ├── state.py                 # SentinelState dataclass
│   ├── workflow.py              # LangGraph workflow definition
│   └── reflection.py            # Self-correction logic
│
├── tools/                       # Tool framework (extensible)
│   ├── __init__.py
│   └── base.py                  # BaseTool abstract class
│
├── memory/                      # State & history management
│   ├── __init__.py
│   └── store.py                 # MemoryStore implementation
│
├── api/                         # REST API layer (future)
│   ├── __init__.py
│   └── server.py                # FastAPI application
│
├── config.py                    # Configuration management
├── main.py                      # Entry point & examples
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip or Poetry

### Installation

```bash
# Clone or navigate to the project
cd Sentinel

# Install dependencies
pip install -r requirements.txt
```

### Running the Workflow

```bash
# Run the sample incident workflow
python main.py
```

This will:
1. Initialize the LangGraph workflow
2. Run through all agents with a sample incident
3. Print state transitions and findings
4. Show final resolution status

### Custom Incidents

```python
from sentinel.graph import create_workflow, SentinelState
import asyncio

async def test_custom_incident():
    workflow = create_workflow()
    state = SentinelState(user_input="Your incident description")
    result = await workflow.ainvoke(state)
    print(f"Status: {result.status}")
    return result

# Run it
asyncio.run(test_custom_incident())
```

## Agent Development Guide

### Creating a New Agent

```python
from sentinel.agents import BaseAgent

class MyAgent(BaseAgent):
    """Custom agent for specialized task."""
    
    def __init__(self):
        super().__init__(
            name="my_agent",
            description="Does something specific"
        )
    
    async def run(self, state: SentinelState) -> SentinelState:
        # Read from state
        data = state.detective_findings
        
        # Process
        result = self.process(data)
        
        # Write to state
        state.my_results = result
        state.add_message(
            role="agent",
            agent_name=self.name,
            content="What I did"
        )
        
        return state
```

### Integrating into Workflow

1. Add node to `workflow.py`:
   ```python
   workflow.add_node("my_agent", my_agent_node)
   ```

2. Connect with edges:
   ```python
   workflow.add_edge("researcher", "my_agent")
   workflow.add_edge("my_agent", "reflection")
   ```

## Self-Correction & Resilience

Sentinel's key innovation is the **reflection loop**:

1. **Execution**: Action plan is executed
2. **Reflection**: Validates against success criteria
3. **Decision**: 
   - ✅ Success → Close incident
   - 🔄 Retry → Adjust and try again (max 3x)
   - 🔀 Alternative → Try different fix from researcher
   - 👤 Escalate → Send to humans with full context
4. **Loop**: Retries go back to Detective for re-analysis

This enables:
- **Autonomy**: System tries multiple strategies automatically
- **Learning**: Each failure provides diagnostic data
- **Safety**: Bounded retries prevent infinite loops
- **Transparency**: All attempts recorded in history

## Configuration

See `config.py` for settings:

```python
class Config:
    DEBUG = False
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    LLM_MODEL = "gpt-4"
    LLM_TEMPERATURE = 0.7
    LOG_LEVEL = "INFO"
```

## License

MIT

## Contributing

Contributions welcome! Areas for enhancement:
- [ ] Integrate with real LLMs (OpenAI, Claude)
- [ ] Connect to actual knowledge bases (Pinecone, Weaviate)
- [ ] Implement ExecutorAgent for real command execution
- [ ] Add REST API endpoints
- [ ] Build web dashboard for workflow visualization
- [ ] Create more specialized agents
- [ ] Add observability (Datadog, New Relic integration)
