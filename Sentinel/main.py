"""
Main entry point for Sentinel multi-agent system.

This script demonstrates the Sentinel workflow by:
1. Initializing the LangGraph workflow
2. Creating a sample incident
3. Running the workflow through all agents
4. Printing state transitions for debugging and visibility
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

from sentinel.graph import SentinelState, create_workflow


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_state_summary(state: SentinelState, stage: str) -> None:
    """Print a formatted summary of current state."""
    summary = state.get_summary()
    
    print(f"[{stage.upper()}] State Snapshot:")
    print(f"  Status: {summary['status']}")
    print(f"  Incident: {summary['incident']}")
    print(f"  Hypotheses: {summary['hypotheses_count']}")
    print(f"  Fixes Available: {summary['fixes_available']}")
    print(f"  Action Planned: {summary['action_planned']}")
    if summary['error']:
        print(f"  Error: {summary['error']}")
    if summary['execution_status']:
        print(f"  Execution Status: {summary['execution_status']}")


def print_detective_findings(findings: Dict[str, Any]) -> None:
    """Print detective agent findings."""
    if not findings:
        return
    
    print("\n  Detective Findings:")
    print(f"    Incident Summary: {findings.get('incident_summary', 'N/A')}")
    
    hypotheses = findings.get('hypotheses', [])
    print(f"    Hypotheses ({len(hypotheses)}):")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"      {i}. {hyp.get('hypothesis')} (confidence: {hyp.get('confidence', 0):.0%})")
        print(f"         {hyp.get('explanation')}")
    
    evidence = findings.get('evidence', {})
    if evidence:
        print(f"    Evidence:")
        print(f"      Services: {', '.join(evidence.get('affected_services', []))}")
        print(f"      Patterns: {', '.join(evidence.get('error_patterns', []))}")
        print(f"      Frequency: {evidence.get('frequency', 'unknown')}")


def print_researcher_recommendations(recommendations: Dict[str, Any]) -> None:
    """Print researcher agent recommendations."""
    if not recommendations:
        return
    
    print("\n  Researcher Recommendations:")
    
    fixes = recommendations.get('fixes', [])
    print(f"    Fixes ({len(fixes)}):")
    for i, fix in enumerate(fixes[:3], 1):  # Show top 3
        print(f"      {i}. {fix.get('title')}")
        print(f"         Effectiveness: {fix.get('effectiveness_score', 0):.0%}, "
              f"Feasibility: {fix.get('feasibility_score', 0):.0%}")
        print(f"         Time: ~{fix.get('time_to_implement_minutes', 0)} min")
    
    primary = recommendations.get('primary_recommendation', {})
    if primary:
        print(f"\n    Primary Recommendation: {primary.get('title')}")
        print(f"      {primary.get('description')}")


def print_operator_action_plan(action_plan: Dict[str, Any]) -> None:
    """Print operator agent action plan."""
    if not action_plan:
        return
    
    print("\n  Operator Action Plan:")
    print(f"    Selected Action: {action_plan.get('selected_action', {}).get('title')}")
    print(f"    Risk Level: {action_plan.get('risk_assessment', {}).get('overall_risk', 'unknown')}")
    print(f"    Approval Status: {action_plan.get('approval_status')}")
    
    plan = action_plan.get('action_plan', {})
    phases = plan.get('phases', [])
    if phases:
        print(f"    Phases ({len(phases)}):")
        for phase in phases:
            phase_name = phase.get('name', 'Unknown')
            step_count = len(phase.get('steps', []))
            print(f"      - {phase_name} ({step_count} steps)")
    
    print(f"\n    Rationale: {action_plan.get('decision_rationale', '')}")


def print_reflection_results(reflection: Dict[str, Any]) -> None:
    """Print reflection results."""
    if not reflection:
        return
    
    print("\n  Reflection Results:")
    print(f"    Outcome: {reflection.get('outcome')}")
    print(f"    Analysis: {reflection.get('analysis')}")
    
    next_steps = reflection.get('next_steps', [])
    if next_steps:
        print(f"    Next Steps:")
        for step in next_steps:
            print(f"      - {step}")


def print_conversation_history(history: list) -> None:
    """Print conversation history."""
    if not history:
        return
    
    print("\n  Conversation History:")
    for msg in history[-5:]:  # Show last 5 messages
        role = msg.get('role', 'unknown').upper()
        agent = msg.get('agent_name', '')
        content = msg.get('content', '')
        
        if agent:
            print(f"    [{role}:{agent}] {content}")
        else:
            print(f"    [{role}] {content}")


async def run_sentinel_workflow() -> None:
    """
    Run the Sentinel workflow with a sample incident.
    
    This demonstrates the complete flow:
    1. Initialize graph
    2. Create sample incident state
    3. Run workflow
    4. Print results at each stage
    """

    print_section("SENTINEL: Multi-Agent Incident Resolution System")
    
    # Initialize the workflow
    print("Initializing Sentinel workflow graph...")
    try:
        workflow = create_workflow()
        print("✓ Workflow graph created successfully")
    except Exception as e:
        print(f"✗ Failed to create workflow: {e}")
        return

    # Create sample incident state
    print("\nCreating sample incident...")
    sample_incident = (
        "Database connection timeout errors occurring every morning at 9 AM. "
        "The system responds with 'Connection pool exhausted' errors for approximately "
        "5-10 minutes before recovery. This is affecting production services. "
        "Monitoring shows high CPU usage on the database server during these periods."
    )
    
    initial_state = SentinelState(
        user_input=sample_incident,
        status="initialized"
    )
    
    initial_state.set_metadata("workflow_id", f"incident-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    initial_state.set_metadata("priority", "high")
    initial_state.set_metadata("environment", "production")
    initial_state.add_message(role="user", content=f"Incident: {sample_incident}")
    
    print(f"✓ Sample incident created")
    print(f"  Incident: {sample_incident[:100]}...")
    
    # Run the workflow
    print_section("Running Workflow")
    print("Executing Sentinel agents in sequence...\n")
    
    try:
        # Invoke the workflow
        result_state = await workflow.ainvoke(initial_state)
        
        # Print final results
        print_section("Workflow Execution Complete")
        
        # Final state summary
        final_summary = result_state.get_summary()
        print("Final State Summary:")
        print(json.dumps(final_summary, indent=2))
        
        # Detective findings
        if result_state.detective_findings:
            print_section("Detective Findings", "-")
            print_detective_findings(result_state.detective_findings)
        
        # Researcher recommendations
        if result_state.researcher_recommendations:
            print_section("Researcher Recommendations", "-")
            print_researcher_recommendations(result_state.researcher_recommendations)
        
        # Operator action plan
        if result_state.operator_action_plan:
            print_section("Operator Action Plan", "-")
            print_operator_action_plan(result_state.operator_action_plan)
        
        # Reflection results
        if result_state.metadata.get('latest_reflection'):
            print_section("Reflection & Validation", "-")
            print_reflection_results(result_state.metadata['latest_reflection'])
        
        # Conversation history
        if result_state.conversation_history:
            print_section("Complete Conversation History", "-")
            print_conversation_history(result_state.conversation_history)
        
        # Workflow metadata
        print_section("Workflow Metadata", "-")
        print("Metadata:")
        for key, value in result_state.metadata.items():
            if key not in ['reflection_history']:  # Skip verbose history
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    print(f"  {key}: <complex object>")
                else:
                    print(f"  {key}: {value}")
        
        # Status and error information
        print_section("Final Status", "-")
        print(f"Status: {result_state.status}")
        if result_state.error:
            print(f"Error: {result_state.error}")
        else:
            print("✓ No errors")
        
        print_section("Workflow Summary")
        print(f"Total Messages: {len(result_state.conversation_history)}")
        print(f"Hypotheses Generated: {len(result_state.detective_findings.get('hypotheses', []))}")
        print(f"Fixes Researched: {len(result_state.researcher_recommendations.get('fixes', []))}")
        print(f"Final Status: {result_state.status}")
        print("\n✓ Sentinel workflow completed successfully\n")

    except Exception as e:
        print(f"\n✗ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()


async def run_with_custom_incident(incident_description: str) -> None:
    """
    Run Sentinel with a custom incident description.
    
    Args:
        incident_description: Custom incident to investigate
    """
    
    workflow = create_workflow()
    
    initial_state = SentinelState(user_input=incident_description)
    initial_state.set_metadata("workflow_id", f"incident-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    initial_state.add_message(role="user", content=f"Incident: {incident_description}")
    
    print(f"\nRunning Sentinel with incident: {incident_description}\n")
    
    try:
        result_state = await workflow.ainvoke(initial_state)
        print(f"✓ Workflow completed with status: {result_state.status}")
        return result_state
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


async def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 70)
    print(" SENTINEL: Multi-Agent Incident Resolution System")
    print("=" * 70)
    print("\nStarting Sentinel workflow...\n")
    
    # Run with sample incident
    await run_sentinel_workflow()
    
    # Optionally run with custom incident
    # custom_incident = "API service experiencing 50% error rate in production"
    # result = await run_with_custom_incident(custom_incident)


if __name__ == "__main__":
    # Run the async workflow
    asyncio.run(main())
