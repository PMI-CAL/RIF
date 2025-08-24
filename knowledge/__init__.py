"""
RIF Knowledge Management System

This package provides a unified interface for knowledge management operations
in the RIF framework, supporting multiple backend implementations while
maintaining a consistent API for agents.

Key components:
- KnowledgeInterface: Abstract interface for all knowledge operations
- LightRAGKnowledgeAdapter: Implementation using LightRAG with ChromaDB
- MockKnowledgeAdapter: In-memory implementation for testing
- KnowledgeSystemFactory: Factory for creating configured instances

Usage for RIF agents:
```python
# Get the default configured knowledge system
from knowledge import get_knowledge_system

knowledge = get_knowledge_system()

# Store a pattern
pattern_id = knowledge.store_pattern({
    "title": "Successful implementation pattern",
    "description": "Pattern for implementing feature X",
    "complexity": "medium",
    "source": "issue-25"
})

# Search for similar patterns
patterns = knowledge.search_patterns("implementation approach", limit=3)

# Store an architectural decision
decision_id = knowledge.store_decision({
    "title": "Choice of abstraction layer design",
    "status": "accepted",
    "context": "Need to decouple agents from LightRAG",
    "decision": "Use abstract interface with factory pattern",
    "impact": "high"
})
```

The interface automatically handles:
- Content serialization/deserialization
- Metadata standardization
- Error handling and logging
- Semantic search across collections
- Collection statistics and system info
"""

from .interface import (
    KnowledgeInterface,
    KnowledgeSystemError,
    KnowledgeStorageError,
    KnowledgeRetrievalError,
    KnowledgeSystemFactory,
    get_knowledge_system
)

from .lightrag_adapter import (
    LightRAGKnowledgeAdapter,
    MockKnowledgeAdapter,
    get_lightrag_adapter,
    get_mock_adapter,
    LIGHTRAG_AVAILABLE
)

# Make key functions and classes available at package level
__all__ = [
    # Core interface
    'KnowledgeInterface',
    'get_knowledge_system',
    
    # Factory and implementations
    'KnowledgeSystemFactory',
    'LightRAGKnowledgeAdapter',
    'MockKnowledgeAdapter',
    
    # Convenience functions
    'get_lightrag_adapter',
    'get_mock_adapter',
    
    # Exceptions
    'KnowledgeSystemError',
    'KnowledgeStorageError',
    'KnowledgeRetrievalError',
    
    # Status
    'LIGHTRAG_AVAILABLE'
]

# Version info
__version__ = '1.0.0'

# Configuration defaults
DEFAULT_COLLECTIONS = {
    'patterns': 'Successful code patterns and templates',
    'decisions': 'Architectural decisions and rationale',
    'code_snippets': 'Reusable code examples and functions',
    'issue_resolutions': 'Resolved issues and their solutions',
    'learnings': 'Learning and experience data from completed tasks',
    'checkpoints': 'Progress checkpoints for recovery',
    'metrics': 'Performance and quality metrics'
}


def configure_default_system(implementation: str = "lightrag", **kwargs):
    """
    Configure the default knowledge system implementation.
    
    This function allows easy configuration of the knowledge system
    that will be returned by get_knowledge_system().
    
    Args:
        implementation: Implementation name ('lightrag' or 'mock')
        **kwargs: Additional configuration arguments
        
    Example:
        configure_default_system("lightrag", knowledge_path="/path/to/knowledge")
        configure_default_system("mock")  # For testing
    """
    KnowledgeSystemFactory.set_default_implementation(implementation)


def get_available_implementations():
    """
    Get list of available knowledge system implementations.
    
    Returns:
        List of implementation names that can be used
    """
    return KnowledgeSystemFactory.get_available_implementations()


def create_knowledge_system(implementation: str, **kwargs) -> KnowledgeInterface:
    """
    Create a specific knowledge system implementation.
    
    Args:
        implementation: Implementation name
        **kwargs: Implementation-specific arguments
        
    Returns:
        KnowledgeInterface implementation
        
    Example:
        # Create LightRAG instance with custom path
        lightrag = create_knowledge_system("lightrag", knowledge_path="/custom/path")
        
        # Create mock instance for testing
        mock = create_knowledge_system("mock")
    """
    return KnowledgeSystemFactory.create(implementation, **kwargs)


# Module-level convenience functions for backward compatibility
def store_pattern(pattern_data: dict, pattern_id: str = None) -> str:
    """Store pattern using default knowledge system."""
    return get_knowledge_system().store_pattern(pattern_data, pattern_id)


def store_decision(decision_data: dict, decision_id: str = None) -> str:
    """Store decision using default knowledge system."""
    return get_knowledge_system().store_decision(decision_data, decision_id)


def search_patterns(query: str, complexity: str = None, limit: int = 5) -> list:
    """Search patterns using default knowledge system."""
    return get_knowledge_system().search_patterns(query, complexity, limit)


def search_decisions(query: str, status: str = None, limit: int = 5) -> list:
    """Search decisions using default knowledge system."""
    return get_knowledge_system().search_decisions(query, status, limit)


def find_similar_issues(description: str, limit: int = 5) -> list:
    """Find similar issues using default knowledge system."""
    return get_knowledge_system().find_similar_issues(description, limit)