# LightRAG Migration Compatibility Guide

## Overview

The enhanced LightRAG Knowledge Adapter (Issue #36) provides comprehensive migration support for gradually transitioning from legacy knowledge systems to LightRAG while maintaining performance and compatibility.

## Features

### 🔄 Query/Response Translation
- **File-based systems**: Convert file paths to content searches
- **JSON-based systems**: Transform structured queries to natural language
- **Extensible**: Register custom translators for any legacy system

### 🎯 Context Optimization (Issue #34 Integration)
- Automatic integration with RIF Context Optimizer
- Agent-specific result optimization
- Token-aware pruning for context windows
- Relevance scoring and ranking

### 📊 Performance Monitoring
- Real-time migration performance tracking
- Response time analysis (optimized vs unoptimized)
- Translation success/error rates
- Historical performance data

### 🏭 Enhanced Factory Support
- Migration-aware adapter creation
- Backward compatibility maintained
- Easy feature toggling

## Usage Examples

### Basic Migration-Compatible Adapter

```python
from knowledge.lightrag_adapter import get_migration_compatible_adapter

# Create adapter with full migration support
adapter = get_migration_compatible_adapter()

# Check capabilities
print(f"Migration features: {adapter.migration_features_enabled}")
print(f"Context optimizer: {adapter.context_optimizer is not None}")
```

### Agent-Optimized Retrieval

```python
# Retrieve knowledge optimized for specific agent type
result = adapter.retrieve_knowledge_for_agent(
    query="authentication patterns",
    agent_type="rif-implementer",
    collection="patterns",
    n_results=5,
    legacy_system="file_based"  # Optional: translate from legacy system
)

print(f"Results: {len(result['results'])}")
print(f"Optimization applied: {result['optimization_applied']}")
print(f"Translation applied: {result['translation_applied']}")
print(f"Duration: {result['performance_info']['duration']:.3f}s")
```

### Legacy System Translation

```python
# Register custom translator for your legacy system
def my_legacy_query_translator(query: str) -> str:
    # Custom translation logic
    if query.startswith("LEGACY:"):
        return query.replace("LEGACY:", "").strip()
    return query

def my_legacy_response_translator(results):
    # Custom response format conversion
    return [
        {
            "legacy_id": result["id"],
            "legacy_content": result["content"],
            "legacy_score": 1.0 - result.get("distance", 0.0)
        }
        for result in results
    ]

# Register translators
adapter.register_query_translator("my_system", my_legacy_query_translator)
adapter.register_response_translator("my_system", my_legacy_response_translator)

# Use with translation
results = adapter.retrieve_knowledge(
    query="LEGACY: find user management code",
    legacy_system="my_system",
    agent_type="rif-implementer"
)
```

### Performance Monitoring

```python
# Get migration metrics
metrics = adapter.get_migration_metrics()

print(f"Queries translated: {metrics['queries_translated']}")
print(f"Average response time: {metrics.get('avg_response_time', 'N/A')}")
print(f"Context optimizations: {metrics['context_optimizations']}")

# Performance breakdown
if 'optimized_avg_time' in metrics:
    print(f"Optimized queries avg: {metrics['optimized_avg_time']:.3f}s")
if 'unoptimized_avg_time' in metrics:
    print(f"Unoptimized queries avg: {metrics['unoptimized_avg_time']:.3f}s")
```

## Supported Legacy Systems

### File-Based Systems
- **Query Translation**: File paths → content-based searches
- **Response Translation**: Adds file-like metadata structure
- **Use Case**: Migrating from file-based knowledge storage

```python
# File-based query examples
adapter.retrieve_knowledge(
    query="/docs/authentication/oauth.md",  # Translates to "content related to oauth.md"
    legacy_system="file_based"
)
```

### JSON-Based Systems
- **Query Translation**: Structured JSON → natural language
- **Response Translation**: Standard format → structured JSON response
- **Use Case**: Migrating from JSON-based APIs

```python
# JSON-based query examples
adapter.retrieve_knowledge(
    query='{"type": "pattern", "complexity": "high"}',  # Translates to "type pattern complexity high"
    legacy_system="json_based"
)
```

## Context Optimization Integration

The adapter seamlessly integrates with the Context Optimizer from Issue #34:

```python
# Context optimization is automatic when agent_type is specified
results = adapter.retrieve_knowledge(
    query="database connection patterns",
    agent_type="rif-implementer",  # Triggers context optimization
    collection="patterns",
    optimize_for_agent=True  # Default: True
)

# Results are automatically:
# 1. Relevance scored
# 2. Token-optimized for agent consumption
# 3. Pruned to fit context windows
# 4. Diversity-balanced
```

## Migration Strategy

### Phase 1: Parallel Operation
```python
# Run both systems in parallel
legacy_results = legacy_system.search("user auth")
lightrag_results = adapter.retrieve_knowledge(
    query="user auth",
    legacy_system="your_legacy_type"
)

# Compare and validate results
```

### Phase 2: Gradual Migration
```python
# Use LightRAG with legacy fallback
try:
    results = adapter.retrieve_knowledge_for_agent(
        query=query,
        agent_type=agent_type,
        legacy_system="your_legacy_type"
    )
except Exception:
    # Fallback to legacy system
    results = legacy_system.search(query)
```

### Phase 3: Full Migration
```python
# Pure LightRAG usage
adapter = get_migration_compatible_adapter()
results = adapter.retrieve_knowledge_for_agent(
    query=query,
    agent_type=agent_type
    # No legacy_system parameter needed
)
```

## Performance Considerations

### Optimization Benefits
- **Context optimization**: ~30-50% token reduction
- **Relevance scoring**: Better result quality
- **Agent-specific tuning**: Improved agent performance

### Migration Overhead
- **Query translation**: ~5-10ms overhead
- **Response translation**: ~2-5ms overhead
- **Performance monitoring**: ~1ms overhead

### Recommendations
1. **Enable context optimization** for all agent interactions
2. **Monitor migration metrics** to track performance
3. **Use custom translators** for complex legacy systems
4. **Gradually reduce legacy system usage** based on metrics

## Troubleshooting

### Common Issues

**Context Optimizer Not Available**
```
Warning: Context optimizer not available - some migration features may be limited
```
- Ensure Issue #34 context optimizer is properly installed
- Check import paths and dependencies

**Translation Errors**
```
Warning: Query translation failed for legacy_system: <error>
```
- Verify translator function signatures
- Check for exception handling in custom translators
- Review legacy system format compatibility

**Performance Degradation**
- Check migration metrics for bottlenecks
- Consider disabling unnecessary translations
- Monitor context optimization impact

### Debug Mode

```python
import logging
logging.getLogger('knowledge.lightrag_adapter').setLevel(logging.DEBUG)

# Detailed logging will show:
# - Translation attempts
# - Context optimization decisions
# - Performance timings
# - Error details
```

## Integration with RIF Agents

All RIF agents can use the migration-compatible adapter:

```python
# In RIF agent code
from knowledge.lightrag_adapter import get_migration_compatible_adapter

class RIFImplementer:
    def __init__(self):
        self.knowledge = get_migration_compatible_adapter()
    
    def find_patterns(self, requirement):
        return self.knowledge.retrieve_knowledge_for_agent(
            query=requirement,
            agent_type="rif-implementer",
            collection="patterns",
            n_results=3
        )
```

## Next Steps

1. **Test in your environment** with actual legacy data
2. **Register custom translators** for your specific legacy systems
3. **Monitor performance metrics** during migration
4. **Gradually increase LightRAG usage** while reducing legacy system dependency
5. **Provide feedback** on translation accuracy and performance

For more information, see:
- Issue #36: Create LightRAG compatibility interface
- Issue #34: Optimize context for agent consumption
- `knowledge/context/optimizer.py` - Context optimization implementation
- `test_lightrag_compatibility.py` - Comprehensive test examples