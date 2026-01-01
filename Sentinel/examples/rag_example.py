"""
RAG Retrieval Example for Sentinel

This script demonstrates how to use the RAG retrieval system
to find relevant runbooks and incidents for incident resolution.

Run this script to see the RAG system in action:
    python examples/rag_example.py
"""

import asyncio
from datetime import datetime

# Import Sentinel components
from memory import (
    # RAG Retriever
    retrieve_for_incident,
    get_grounding_context,
    IncidentContext,
    
    # Data loading
    store_resolved_incident,
    load_runbooks,
    
    # Stores
    get_rag_retriever,
)


async def populate_knowledge_base():
    """
    Populate the knowledge base with sample runbooks and incidents.
    
    In production, this would be done through:
    - Loading runbooks from markdown files
    - Storing incidents as they are resolved
    - Importing historical data
    """
    print("📚 Populating knowledge base with sample data...")
    
    # Load runbooks from the runbooks directory
    try:
        runbook_result = await load_runbooks("runbooks/")
        print(f"✅ Loaded {runbook_result.loaded} runbooks ({runbook_result.chunks} chunks)")
    except Exception as e:
        print(f"⚠️  Could not load runbooks: {e}")
        print("   (This is okay - we'll use sample incidents only)")
    
    # Add some sample resolved incidents
    sample_incidents = [
        {
            "incident_id": "INC-2024-001",
            "title": "Kubernetes pods crashing with OOMKilled",
            "description": "Multiple pods in the web-api service are crashing with exit code 137 (OOMKilled). Users reporting 500 errors. Memory usage spikes visible in monitoring before crashes.",
            "service": "web-api",
            "severity": "critical",
            "error_type": "OOMKilled",
            "root_cause": "Memory limit set too low for the workload. Application was loading large datasets into memory during peak traffic.",
            "resolution_summary": "Increased memory limits from 512Mi to 1Gi and added memory usage monitoring alerts",
            "resolution_steps": [
                {"action": "Checked pod logs and found OOMKilled exit code", "outcome": "Confirmed memory exhaustion"},
                {"action": "Analyzed memory usage patterns", "outcome": "Found spikes during data processing"},
                {"action": "Increased memory limit to 1Gi", "outcome": "Pods stopped crashing"},
                {"action": "Added memory usage alerts at 80%", "outcome": "Proactive monitoring enabled"},
            ],
            "prevented_by": "Memory usage monitoring and auto-scaling based on memory metrics",
            "tags": ["kubernetes", "memory", "oom", "monitoring"],
        },
        {
            "incident_id": "INC-2024-002",
            "title": "PostgreSQL connection timeouts",
            "description": "Applications unable to connect to PostgreSQL database. Connection timeout errors in logs. Database appears to be running but not accepting connections.",
            "service": "postgresql",
            "severity": "high",
            "error_type": "connection_timeout",
            "root_cause": "Connection pool exhausted due to long-running queries that weren't properly closed",
            "resolution_summary": "Killed long-running queries and increased connection pool size",
            "resolution_steps": [
                {"action": "Checked database connectivity", "outcome": "Database responsive but connections maxed"},
                {"action": "Identified long-running queries", "outcome": "Found 15 queries running for >1 hour"},
                {"action": "Terminated blocking queries", "outcome": "Connections freed up"},
                {"action": "Increased max_connections to 200", "outcome": "More capacity for peak load"},
            ],
            "prevented_by": "Query timeout settings and connection pool monitoring",
            "tags": ["database", "postgresql", "connections", "performance"],
        },
        {
            "incident_id": "INC-2024-003",
            "title": "Container restart loop in Docker",
            "description": "Container keeps restarting immediately after startup. Exit code 1. Application logs show configuration errors.",
            "service": "user-service",
            "severity": "medium",
            "error_type": "restart_loop",
            "root_cause": "Missing environment variable for database connection string",
            "resolution_summary": "Added missing DATABASE_URL environment variable to container configuration",
            "resolution_steps": [
                {"action": "Checked container logs", "outcome": "Found configuration error on startup"},
                {"action": "Verified environment variables", "outcome": "DATABASE_URL was missing"},
                {"action": "Updated container config with DATABASE_URL", "outcome": "Container started successfully"},
                {"action": "Added config validation to startup script", "outcome": "Early detection of missing vars"},
            ],
            "prevented_by": "Configuration validation and deployment checklists",
            "tags": ["docker", "configuration", "environment", "startup"],
        }
    ]
    
    # Store incidents
    for incident_data in sample_incidents:
        try:
            await store_resolved_incident(**incident_data)
            print(f"✅ Stored incident: {incident_data['title'][:50]}...")
        except Exception as e:
            print(f"❌ Failed to store incident {incident_data['incident_id']}: {e}")
    
    print("📚 Knowledge base populated!\n")


async def demonstrate_rag_retrieval():
    """
    Demonstrate RAG retrieval with various incident scenarios.
    """
    print("🔍 Demonstrating RAG Retrieval System")
    print("=" * 50)
    
    # Scenario 1: Memory issue
    print("\n📋 SCENARIO 1: Memory-related incident")
    print("-" * 30)
    
    result1 = await retrieve_for_incident(
        description="Our pods are crashing with exit code 137, memory usage looks high",
        service="web-api",
        severity="high",
        error_type="OOMKilled",
        top_k=5
    )
    
    print(f"📊 Retrieved {result1.total_retrieved} documents (confidence: {result1.confidence:.2f})")
    print(f"⏱️  Search time: {result1.search_time_ms}ms")
    
    for i, doc in enumerate(result1.get_top_k(3), 1):
        print(f"\n{i}. [{doc.content_type.upper()}] {doc.title}")
        print(f"   Score: {doc.score:.3f} | Semantic: {doc.semantic_score:.3f}")
        print(f"   Reasoning: {doc.explanation.reasoning}")
        if doc.relevant_sections:
            print(f"   Key section: {doc.relevant_sections[0][:100]}...")
    
    # Scenario 2: Database connectivity
    print("\n\n📋 SCENARIO 2: Database connectivity issue")
    print("-" * 35)
    
    result2 = await retrieve_for_incident(
        description="Can't connect to database, getting timeout errors",
        service="postgresql", 
        severity="critical",
        additional_context="Applications reporting 'connection refused' errors",
        top_k=5
    )
    
    print(f"📊 Retrieved {result2.total_retrieved} documents (confidence: {result2.confidence:.2f})")
    
    for i, doc in enumerate(result2.get_top_k(3), 1):
        print(f"\n{i}. [{doc.content_type.upper()}] {doc.title}")
        print(f"   Score: {doc.score:.3f}")
        print(f"   Reasoning: {doc.explanation.reasoning}")
    
    # Scenario 3: General container issue
    print("\n\n📋 SCENARIO 3: Container startup problem")
    print("-" * 32)
    
    result3 = await retrieve_for_incident(
        description="Container won't start, keeps restarting immediately",
        service="user-service",
        severity="medium",
        top_k=5
    )
    
    print(f"📊 Retrieved {result3.total_retrieved} documents (confidence: {result3.confidence:.2f})")
    
    for i, doc in enumerate(result3.get_top_k(2), 1):
        print(f"\n{i}. [{doc.content_type.upper()}] {doc.title}")
        print(f"   Score: {doc.score:.3f}")
        print(f"   Reasoning: {doc.explanation.reasoning}")


async def demonstrate_grounding_context():
    """
    Demonstrate how to get grounding context for LLM prompts.
    """
    print("\n\n🤖 LLM GROUNDING CONTEXT EXAMPLE")
    print("=" * 40)
    
    # Get grounding context for a specific incident
    context = await get_grounding_context(
        description="Pods are getting OOMKilled and users can't access the service",
        service="web-api",
        severity="critical",
        max_length=2000
    )
    
    print("📄 Generated grounding context:")
    print("-" * 30)
    print(context)
    
    print("\n💡 This context can be used in an LLM prompt like:")
    print("-" * 45)
    
    example_prompt = f"""
You are an expert SRE helping resolve a critical incident.

{context}

CURRENT INCIDENT:
Pods are getting OOMKilled and users can't access the service.
Service: web-api
Severity: critical

Based on the knowledge base above, provide:
1. Most likely root causes
2. Immediate steps to resolve
3. Steps to prevent recurrence

Be specific and reference the relevant knowledge base entries.
"""
    
    print(example_prompt)


async def analyze_retrieval_performance():
    """
    Analyze the performance and accuracy of the RAG system.
    """
    print("\n\n📈 RAG SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Test various queries
    test_queries = [
        ("memory issues kubernetes", "web-api", "high"),
        ("database connection problems", "postgresql", "critical"),
        ("container startup failures", "user-service", "medium"),
        ("pod crash exit code 137", "kubernetes", "high"),
        ("timeout connecting to database", "postgresql", "high"),
    ]
    
    total_time = 0
    results_summary = []
    
    for description, service, severity in test_queries:
        start_time = datetime.now()
        
        result = await retrieve_for_incident(
            description=description,
            service=service,
            severity=severity,
            top_k=5
        )
        
        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds() * 1000
        total_time += query_time
        
        results_summary.append({
            "query": description,
            "confidence": result.confidence,
            "retrieved": result.total_retrieved,
            "time_ms": query_time,
            "top_score": result.all_documents[0].score if result.all_documents else 0
        })
    
    print(f"📊 Performance Summary ({len(test_queries)} queries):")
    print(f"   Average query time: {total_time / len(test_queries):.1f}ms")
    print(f"   Average confidence: {sum(r['confidence'] for r in results_summary) / len(results_summary):.3f}")
    print(f"   Average documents retrieved: {sum(r['retrieved'] for r in results_summary) / len(results_summary):.1f}")
    
    print("\n📋 Individual Query Results:")
    for r in results_summary:
        print(f"   '{r['query'][:30]}...' - Confidence: {r['confidence']:.3f}, Time: {r['time_ms']:.1f}ms")


async def main():
    """
    Main demonstration function.
    """
    print("🚀 Sentinel RAG Retrieval System Demo")
    print("=" * 50)
    print("This demo shows how the RAG system retrieves relevant")
    print("runbooks and incidents for incident resolution.\n")
    
    # Initialize the system
    print("🔧 Initializing RAG retriever...")
    retriever = await get_rag_retriever()
    print("✅ RAG retriever initialized\n")
    
    # Populate with sample data
    await populate_knowledge_base()
    
    # Run demonstrations
    await demonstrate_rag_retrieval()
    await demonstrate_grounding_context()
    await analyze_retrieval_performance()
    
    print("\n\n🎉 Demo completed!")
    print("\nKey features demonstrated:")
    print("✅ Multi-source retrieval (runbooks + incidents)")
    print("✅ Multi-factor ranking (semantic + metadata + quality)")
    print("✅ Structured results with explanations")
    print("✅ LLM grounding context generation")
    print("✅ Performance analytics")
    
    print("\nNext steps:")
    print("• Load real runbooks from your documentation")
    print("• Store resolved incidents as they happen")
    print("• Integrate with your LLM for automated responses")
    print("• Customize ranking weights for your use case")


if __name__ == "__main__":
    asyncio.run(main())