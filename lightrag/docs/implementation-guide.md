# LightRAG Implementation Guide for RIF Framework

## Overview

This document describes the LightRAG implementation for the RIF (Reactive Intelligence Framework) system. LightRAG provides intelligent knowledge storage, retrieval, and learning capabilities using vector embeddings and ChromaDB.

## Architecture

### Core Components

```
/Users/cal/DEV/RIF/lightrag/
├── core/
│   └── lightrag_core.py          # Main RAG functionality
├── embeddings/
│   └── embedding_manager.py      # Embedding generation and caching
├── collections/                  # (Future) Collection-specific logic
├── tests/
│   └── test_basic.py            # Basic functionality tests
├── docs/
│   └── implementation-guide.md  # This document
├── requirements.txt             # Python dependencies
└── init_lightrag.py            # Initialization and migration script
```

### Collections Structure

LightRAG uses 4 specialized ChromaDB collections:

1. **patterns** - Successful code patterns and templates
2. **decisions** - Architectural decisions and rationale  
3. **code_snippets** - Reusable code examples and functions
4. **issue_resolutions** - Resolved issues and their solutions

## Installation

### Phase 1: Foundation Setup (Current)

The foundation has been implemented with:
- Directory structure created
- Core modules implemented
- Dependencies defined
- Basic testing framework

### Phase 2: Dependency Installation (Next)

```bash
# Install Python dependencies
pip install -r /Users/cal/DEV/RIF/lightrag/requirements.txt

# Initialize LightRAG system
python /Users/cal/DEV/RIF/lightrag/init_lightrag.py

# Run tests
python /Users/cal/DEV/RIF/lightrag/init_lightrag.py --test
```

### Phase 3: Knowledge Migration

The initialization script will automatically migrate existing knowledge from:
- `/Users/cal/DEV/RIF/knowledge/patterns/` → patterns collection
- `/Users/cal/DEV/RIF/knowledge/decisions/` → decisions collection  
- `/Users/cal/DEV/RIF/knowledge/issues/` → issue_resolutions collection

## Usage

### Basic Usage

```python
from lightrag.core.lightrag_core import get_lightrag_instance

# Get LightRAG instance
rag = get_lightrag_instance()

# Store knowledge
doc_id = rag.store_knowledge(
    collection_name="patterns",
    content="This is a successful pattern for...",
    metadata={
        "type": "pattern",
        "complexity": "medium",
        "tags": ["api", "authentication"]
    }
)

# Retrieve similar knowledge
results = rag.retrieve_knowledge(
    query="authentication patterns",
    collection_name="patterns",
    n_results=3
)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Similarity: {1 - result['distance']}")
```

### Convenience Functions for RIF Agents

```python
from lightrag.core.lightrag_core import store_pattern, find_similar_patterns

# Store a pattern
pattern_data = {
    "name": "JWT Authentication",
    "code": "...",
    "complexity": "medium",
    "tags": ["auth", "jwt"]
}
doc_id = store_pattern(pattern_data)

# Find similar patterns
similar = find_similar_patterns("JWT authentication")
```

## Embedding System

### Backends

1. **OpenAI (Primary)**: Uses text-embedding-ada-002 model
   - Requires OPENAI_API_KEY environment variable
   - High quality embeddings
   - API cost considerations

2. **Local (Fallback)**: Uses sentence-transformers
   - Model: all-MiniLM-L6-v2 (384 dimensions)
   - No API costs
   - Runs locally

### Caching

- Embeddings are cached to reduce API costs
- Cache located at `/Users/cal/DEV/RIF/lightrag/embeddings_cache/`
- Automatic cache management with cleanup utilities

## Integration with RIF Agents

### Agent Integration Points

Each RIF agent can integrate with LightRAG:

```python
# In agent code
from lightrag.core.lightrag_core import get_lightrag_instance, store_pattern

def rif_analyst_process_issue(issue_data):
    rag = get_lightrag_instance()
    
    # Find similar past issues
    similar_issues = rag.retrieve_knowledge(
        query=issue_data["title"],
        collection_name="issue_resolutions",
        n_results=5
    )
    
    # Store new analysis pattern
    if analysis_successful:
        pattern_data = {
            "issue_type": issue_data["type"],
            "analysis_approach": "...",
            "success_factors": "..."
        }
        store_pattern(pattern_data)
```

### Workflow Integration

LightRAG enhances the RIF workflow by:
1. **Learning from Past Solutions**: Agents query similar past issues
2. **Pattern Recognition**: Successful approaches are automatically stored
3. **Knowledge Accumulation**: Each resolution improves future performance
4. **Context Awareness**: Agents have access to historical context

## Configuration

### Environment Variables

```bash
# Optional: OpenAI API key for embeddings
export OPENAI_API_KEY="your-api-key"

# Optional: Custom knowledge path
export RIF_KNOWLEDGE_PATH="/custom/path/to/knowledge"
```

### Customization

#### Custom Collections

```python
# Add custom collection to lightrag_core.py
self.collections = {
    "patterns": "Successful code patterns and templates",
    "decisions": "Architectural decisions and rationale", 
    "code_snippets": "Reusable code examples and functions",
    "issue_resolutions": "Resolved issues and their solutions",
    "custom_collection": "Your custom collection description"
}
```

#### Custom Embedding Models

```python
# Modify embedding_manager.py
embedding_manager = EmbeddingManager(
    embedding_model="text-embedding-ada-003",  # Different OpenAI model
    local_model="all-mpnet-base-v2"            # Different local model
)
```

## Monitoring and Maintenance

### Collection Statistics

```python
rag = get_lightrag_instance()
stats = rag.get_collection_stats()
print(stats)
# Output:
# {
#   "patterns": {"count": 42, "description": "..."},
#   "decisions": {"count": 15, "description": "..."}
# }
```

### Cache Management

```python
from lightrag.embeddings.embedding_manager import get_embedding_manager

manager = get_embedding_manager()
cache_stats = manager.get_cache_stats()
print(f"Cache: {cache_stats['file_count']} files, {cache_stats['total_size_mb']} MB")

# Clear cache if needed
cleared_count = manager.clear_cache()
```

### Export and Backup

```python
# Export collection to JSON
rag.export_collection("patterns", "/backup/patterns.json")

# Export all collections
for collection in ["patterns", "decisions", "code_snippets", "issue_resolutions"]:
    rag.export_collection(collection, f"/backup/{collection}.json")
```

## Performance Considerations

### Embedding Optimization

1. **Batch Processing**: Use `get_embeddings_batch()` for multiple texts
2. **Caching**: Enable caching to reduce API calls
3. **Local Fallback**: Configure local models for offline usage

### Query Optimization

1. **Collection-Specific Searches**: Query specific collections when possible
2. **Result Limits**: Use appropriate `n_results` limits
3. **Metadata Filters**: Use filters to narrow search scope

### Storage Optimization

1. **ChromaDB Maintenance**: Regular database maintenance
2. **Collection Management**: Archive old collections periodically
3. **Cache Cleanup**: Monitor and clean embedding cache

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r /Users/cal/DEV/RIF/lightrag/requirements.txt
   ```

2. **OpenAI API Errors**: Check API key and quota
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. **ChromaDB Errors**: Check write permissions and disk space
   ```bash
   ls -la /Users/cal/DEV/RIF/knowledge/chromadb/
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now LightRAG operations will show detailed logs
```

## Future Enhancements

### Phase 2: Knowledge Migration
- Migrate all existing JSON knowledge to ChromaDB
- Validate data integrity
- Performance optimization

### Phase 3: Agent Integration  
- Integrate with all 5 RIF agents
- Workflow automation
- Real-time learning

### Phase 4: Advanced Features
- Feedback loops for continuous improvement
- Advanced query capabilities
- Multi-modal embeddings (code + documentation)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test results in `/Users/cal/DEV/RIF/knowledge/checkpoints/`
3. Run diagnostic tests: `python /Users/cal/DEV/RIF/lightrag/init_lightrag.py --status`

---

**Implementation Status**: Phase 1 Complete (Foundation)  
**Next Phase**: Dependency installation and knowledge migration  
**Agent**: RIF-Implementer  
**Date**: August 18, 2025