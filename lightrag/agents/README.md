# RIF Agent LightRAG Integration

This directory contains the LightRAG integration modules for RIF agents, enabling intelligent knowledge capture and retrieval for each specialized agent type.

## Overview

The agent integration system provides:

- **Specialized RAG capabilities** for each RIF agent type
- **Knowledge capture hooks** for storing learnings during agent execution  
- **Intelligent querying** based on agent context and needs
- **Cross-agent knowledge sharing** and pattern recognition
- **Session management** for tracking knowledge operations

## Architecture

```
lightrag/agents/
├── agent_integration.py    # Core agent RAG classes
├── utils.py               # Utility functions and session management
├── examples.py            # Usage examples for each agent type
├── test_integration.py    # Integration tests
└── README.md             # This documentation
```

## Agent Types and Capabilities

### RIF-Analyst (`RIFAnalystRAG`)
- **Find similar issues** from past resolutions
- **Discover relevant patterns** for requirements analysis
- **Analyze with historical context** using knowledge base
- **Store analysis results** for future reference

```python
from agent_integration import get_analyst_rag

analyst = get_analyst_rag()
similar_issues = analyst.find_similar_issues("user authentication system")
analysis = analyst.analyze_with_history(issue_data)
doc_id = analyst.store_analysis_results("issue_123", analysis)
```

### RIF-Architect (`RIFArchitectRAG`)
- **Find architectural decisions** from past projects
- **Discover design patterns** applicable to requirements
- **Design with knowledge** using proven approaches
- **Store design decisions** for future reference

```python
from agent_integration import get_architect_rag

architect = get_architect_rag()
decisions = architect.find_architectural_decisions("microservices architecture")
design = architect.design_with_knowledge(requirements)
doc_id = architect.store_design_decision(decision_data)
```

### RIF-Implementer (`RIFImplementerRAG`)
- **Find code examples** and reusable components
- **Discover implementation patterns** for specific tasks
- **Plan implementation** using knowledge base guidance
- **Store code snippets** for future reuse

```python
from agent_integration import get_implementer_rag

implementer = get_implementer_rag()
code_examples = implementer.find_code_examples("JWT validation", "Python")
implementation = implementer.implement_with_knowledge(task_data)
doc_id = implementer.store_code_snippet(code_data)
```

### RIF-Validator (`RIFValidatorRAG`)
- **Find test patterns** and testing strategies
- **Discover quality gates** for different project types
- **Plan validation** using proven test approaches
- **Store test patterns** for future reference

```python
from agent_integration import get_validator_rag

validator = get_validator_rag()
test_patterns = validator.find_test_patterns("authentication middleware")
validation = validator.validate_with_knowledge(validation_data)
doc_id = validator.store_test_pattern(pattern_data)
```

### RIF-Planner (`RIFPlannerRAG`)
- **Find planning templates** for different complexity levels
- **Discover workflow patterns** for project types
- **Plan with knowledge** using proven project approaches
- **Store planning templates** for future use

```python
from agent_integration import get_planner_rag

planner = get_planner_rag()
templates = planner.find_planning_templates("high", "microservices")
plan = planner.plan_with_knowledge(planning_data)
doc_id = planner.store_planning_template(template_data)
```

## Session Management

Use session management for organized knowledge operations:

```python
from utils import agent_knowledge_session

with agent_knowledge_session('analyst', 'issue_analysis_001') as session:
    # Query knowledge
    results = session.query_knowledge("similar authentication issues")
    
    # Capture new knowledge
    session.capture_knowledge(
        content="Authentication pattern discovered",
        knowledge_type="pattern",
        metadata={"domain": "security", "complexity": "medium"}
    )
    
    # Get session summary
    summary = session.get_session_summary()
    print(f"Performed {summary['queries_performed']} queries")
```

## Advanced Pattern Analysis

Use the pattern matcher for sophisticated knowledge discovery:

```python
from utils import KnowledgePatternMatcher
from agent_integration import get_analyst_rag

analyst = get_analyst_rag()
matcher = KnowledgePatternMatcher(analyst)

# Find how knowledge evolved over time
evolution = matcher.find_evolution_patterns("authentication", 90)

# Find successful patterns
success_patterns = matcher.find_success_patterns("microservices", 0.8)

# Find anti-patterns to avoid
anti_patterns = matcher.find_anti_patterns("authentication")
```

## Cross-Agent Knowledge Search

Search knowledge across multiple agent types:

```python
from utils import cross_agent_knowledge_search

results = cross_agent_knowledge_search(
    "real-time notifications",
    ["analyst", "architect", "implementer"]
)

for agent_type, agent_results in results.items():
    print(f"{agent_type}: {len(agent_results)} results")
```

## Knowledge Migration

Bulk migrate existing knowledge into the system:

```python
from utils import bulk_migrate_knowledge

doc_ids = bulk_migrate_knowledge(
    source_files=["pattern1.json", "pattern2.json"],
    agent_type="architect",
    knowledge_type="pattern"
)
```

## Knowledge Export

Export agent knowledge for backup or analysis:

```python
from utils import export_agent_knowledge

export_paths = export_agent_knowledge("analyst", "./exports/")
# Returns: {"patterns": "./exports/analyst_patterns.json", ...}
```

## Factory Pattern

Create agent instances using the factory:

```python
from agent_integration import create_agent_rag

# Create any agent type
analyst = create_agent_rag("analyst")
architect = create_agent_rag("architect")
implementer = create_agent_rag("implementer")
validator = create_agent_rag("validator")
planner = create_agent_rag("planner")
```

## Testing

Run the integration tests to verify functionality:

```bash
cd lightrag/agents
python3 test_integration.py
```

Expected output:
```
RIF Agent LightRAG Integration Tests
==================================================
✓ Import Test PASSED
✓ Agent Creation Test PASSED
✓ LightRAG Connection Test PASSED
✓ Knowledge Operations Test PASSED
✓ Agent-Specific Methods Test PASSED
✓ Utils Functionality Test PASSED
==================================================
Test Results: 6/6 tests passed
🎉 All tests passed! Agent integration is working correctly.
```

## Examples

Run the example workflows to see agent integration in action:

```bash
cd lightrag/agents
python3 examples.py
```

This will demonstrate:
- Analyst workflow for issue analysis
- Architect workflow for system design
- Implementer workflow for code planning
- Validator workflow for test planning
- Planner workflow for project planning
- Advanced pattern analysis
- Multi-agent coordination

## Knowledge Collections

The system uses four main knowledge collections:

| Collection | Purpose | Used By |
|------------|---------|---------|
| `patterns` | Successful patterns and templates | All agents |
| `decisions` | Architectural decisions and rationale | Architect, Analyst |
| `code_snippets` | Reusable code examples | Implementer |
| `issue_resolutions` | Resolved issues and solutions | Analyst, Validator |

## Error Handling

The integration includes robust error handling:

- **Connection failures**: Graceful degradation when LightRAG unavailable
- **Query errors**: Fallback to basic search if advanced queries fail
- **Storage failures**: Logging and continuation without blocking agent execution
- **Missing data**: Empty result sets instead of exceptions

## Performance Considerations

- **Connection pooling**: Single LightRAG instance shared across agents
- **Query optimization**: Agent-specific query patterns and filters
- **Batch operations**: Utilities for bulk knowledge operations
- **Session tracking**: Minimal overhead for knowledge operation tracking

## Integration with RIF Workflow

The agent integration automatically:

1. **Captures knowledge** during agent execution
2. **Queries relevant context** before starting tasks
3. **Shares insights** across agent handoffs
4. **Learns from outcomes** to improve future performance
5. **Provides recommendations** based on historical success

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure ChromaDB dependencies are installed
2. **Collection conflicts**: Reset collections if needed
3. **Permission errors**: Check file system permissions for knowledge directory
4. **Query syntax errors**: Use simple strings for queries, complex filters separately

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Reset Collections

Reset collections if corrupted:

```python
from lightrag.core.lightrag_core import LightRAGCore
rag = LightRAGCore()
rag.reset_collection("patterns")  # Reset specific collection
```

## Contributing

When adding new agent integration features:

1. **Add methods** to appropriate agent RAG class
2. **Update tests** in `test_integration.py`
3. **Add examples** in `examples.py`
4. **Update documentation** in this README

## Next Steps

Phase 4 will add:
- **Real-time learning** during agent execution
- **Advanced analytics** on knowledge usage
- **Performance optimization** based on usage patterns
- **Integration testing** with full RIF workflow