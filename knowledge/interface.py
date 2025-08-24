"""
RIF Knowledge Management Interface

This module provides an abstract interface for knowledge management operations
in the RIF framework, enabling decoupling of agents from specific knowledge
system implementations (LightRAG, ChromaDB, etc.).

The interface follows the Repository pattern and supports dependency injection
to allow different knowledge backends to be plugged in without changing agent code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging


class KnowledgeInterface(ABC):
    """
    Abstract base class defining the knowledge management interface for RIF agents.
    
    This interface provides a consistent API for knowledge storage, retrieval,
    and management operations, allowing agents to work with different knowledge
    systems without code changes.
    
    Implementations must support:
    - Vector-based semantic search
    - Metadata filtering
    - Document lifecycle management (CRUD operations)
    - Collection-based organization
    - Pattern and decision storage convenience methods
    """
    
    @abstractmethod
    def store_knowledge(self, 
                       collection: str, 
                       content: Union[str, Dict[str, Any]], 
                       metadata: Optional[Dict[str, Any]] = None,
                       doc_id: Optional[str] = None) -> Optional[str]:
        """
        Store knowledge item in the specified collection.
        
        Args:
            collection: Collection name (e.g., 'patterns', 'decisions', 'learnings')
            content: Content to store (string or dictionary)
            metadata: Optional metadata dictionary for filtering and categorization
            doc_id: Optional document ID (auto-generated if None)
            
        Returns:
            Document ID if successful, None on failure
            
        Raises:
            ValueError: If collection name is invalid
            RuntimeError: If storage operation fails
        """
        pass
    
    @abstractmethod
    def retrieve_knowledge(self, 
                          query: str, 
                          collection: Optional[str] = None, 
                          n_results: int = 5,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge items using semantic search.
        
        Args:
            query: Search query string (semantic search)
            collection: Specific collection to search (None for all collections)
            n_results: Maximum number of results to return
            filters: Optional metadata filters (e.g., {'complexity': 'high'})
            
        Returns:
            List of matching documents with structure:
            [
                {
                    "id": "doc_id",
                    "content": "document content",
                    "metadata": {"key": "value"},
                    "collection": "collection_name",
                    "distance": 0.2,  # Similarity score (lower = more similar)
                }
            ]
        """
        pass
    
    @abstractmethod
    def update_knowledge(self,
                        collection: str,
                        doc_id: str,
                        content: Optional[Union[str, Dict[str, Any]]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing knowledge item.
        
        Args:
            collection: Collection name
            doc_id: Document ID to update
            content: New content (None to keep existing)
            metadata: New/additional metadata (None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_knowledge(self, collection: str, doc_id: str) -> bool:
        """
        Delete knowledge item from collection.
        
        Args:
            collection: Collection name
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collections.
        
        Returns:
            Dictionary with collection statistics:
            {
                "patterns": {"count": 150, "description": "..."},
                "decisions": {"count": 75, "description": "..."},
                ...
            }
        """
        pass
    
    # Convenience methods for common RIF agent operations
    
    def store_pattern(self, pattern_data: Dict[str, Any], pattern_id: Optional[str] = None) -> Optional[str]:
        """
        Store a successful code pattern.
        
        This is a convenience method for RIF agents to store patterns with
        standardized metadata structure.
        
        Args:
            pattern_data: Pattern information dictionary with keys:
                - title: Pattern name/title
                - description: Detailed description
                - code: Code example (optional)
                - complexity: 'low', 'medium', 'high', 'very-high'
                - source: Source identifier (e.g., 'issue-25')
                - tags: List of tags for categorization
                - context: Usage context and conditions
            pattern_id: Optional pattern ID
            
        Returns:
            Pattern ID if successful, None otherwise
        """
        # Standardize metadata for patterns
        metadata = {
            "type": "pattern",
            "complexity": pattern_data.get("complexity", "medium"),
            "source": pattern_data.get("source", "unknown"),
            "tags": pattern_data.get("tags", []),
            "title": pattern_data.get("title", "Untitled Pattern")
        }
        
        # Convert tags list to string if needed
        if isinstance(metadata["tags"], list):
            metadata["tags"] = ",".join(metadata["tags"])
        
        return self.store_knowledge("patterns", pattern_data, metadata, pattern_id)
    
    def store_decision(self, decision_data: Dict[str, Any], decision_id: Optional[str] = None) -> Optional[str]:
        """
        Store an architectural decision.
        
        This is a convenience method for RIF agents to store decisions with
        standardized metadata structure following ADR (Architecture Decision Record) format.
        
        Args:
            decision_data: Decision information dictionary with keys:
                - title: Decision title
                - status: 'proposed', 'accepted', 'deprecated', 'superseded'
                - context: Problem/context that led to the decision
                - decision: The actual decision made
                - consequences: Positive/negative consequences
                - alternatives: Alternatives considered
                - impact: 'low', 'medium', 'high'
                - tags: List of tags for categorization
            decision_id: Optional decision ID
            
        Returns:
            Decision ID if successful, None otherwise
        """
        # Standardize metadata for decisions
        metadata = {
            "type": "decision",
            "status": decision_data.get("status", "accepted"),
            "impact": decision_data.get("impact", "medium"),
            "tags": decision_data.get("tags", []),
            "title": decision_data.get("title", "Untitled Decision")
        }
        
        # Convert tags list to string if needed
        if isinstance(metadata["tags"], list):
            metadata["tags"] = ",".join(metadata["tags"])
        
        return self.store_knowledge("decisions", decision_data, metadata, decision_id)
    
    def store_learning(self, learning_data: Dict[str, Any], learning_id: Optional[str] = None) -> Optional[str]:
        """
        Store learning/experience data from completed tasks.
        
        Args:
            learning_data: Learning information dictionary with keys:
                - title: Learning title
                - description: What was learned
                - issue_id: Related GitHub issue ID
                - complexity: Task complexity that generated this learning
                - success_factors: What led to success
                - challenges: Challenges encountered
                - recommendations: Recommendations for future similar tasks
                - source: Source identifier
            learning_id: Optional learning ID
            
        Returns:
            Learning ID if successful, None otherwise
        """
        metadata = {
            "type": "learning",
            "source": learning_data.get("source", "unknown"),
            "issue_id": learning_data.get("issue_id"),
            "complexity": learning_data.get("complexity", "medium"),
            "title": learning_data.get("title", "Untitled Learning")
        }
        
        return self.store_knowledge("learnings", learning_data, metadata, learning_id)
    
    def search_patterns(self, query: str, complexity: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for patterns with optional complexity filtering.
        
        Args:
            query: Search query
            complexity: Optional complexity filter ('low', 'medium', 'high', 'very-high')
            limit: Maximum results
            
        Returns:
            List of matching patterns
        """
        filters = {"complexity": complexity} if complexity else None
        return self.retrieve_knowledge(query, "patterns", limit, filters)
    
    def search_decisions(self, query: str, status: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for architectural decisions with optional status filtering.
        
        Args:
            query: Search query
            status: Optional status filter ('proposed', 'accepted', 'deprecated', 'superseded')
            limit: Maximum results
            
        Returns:
            List of matching decisions
        """
        filters = {"status": status} if status else None
        return self.retrieve_knowledge(query, "decisions", limit, filters)
    
    def find_similar_issues(self, description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar resolved issues based on description.
        
        Args:
            description: Issue description to match against
            limit: Maximum results
            
        Returns:
            List of similar resolved issues
        """
        return self.retrieve_knowledge(description, "issue_resolutions", limit)
    
    def get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently stored patterns.
        
        This is useful for agents to see what patterns have been learned recently.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of recent patterns sorted by creation time
        """
        # Most implementations should be able to sort by timestamp metadata
        return self.retrieve_knowledge("*", "patterns", limit)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the knowledge system implementation.
        
        Returns:
            Dictionary with system information:
            {
                "implementation": "LightRAGAdapter",
                "version": "1.0.0",
                "backend": "ChromaDB",
                "collections": ["patterns", "decisions", ...],
                "features": ["semantic_search", "metadata_filtering", ...]
            }
        """
        # Default implementation - concrete classes should override
        return {
            "implementation": self.__class__.__name__,
            "version": "unknown",
            "backend": "unknown",
            "collections": [],
            "features": ["basic_storage", "basic_retrieval"]
        }


class KnowledgeSystemError(Exception):
    """Base exception for knowledge system operations."""
    pass


class KnowledgeStorageError(KnowledgeSystemError):
    """Exception raised for storage operation failures."""
    pass


class KnowledgeRetrievalError(KnowledgeSystemError):
    """Exception raised for retrieval operation failures."""
    pass


class KnowledgeSystemFactory:
    """
    Factory for creating knowledge system implementations.
    
    This factory supports dependency injection by allowing the knowledge
    system implementation to be configured externally.
    """
    
    _default_implementation = None
    _implementations = {}
    
    @classmethod
    def register_implementation(cls, name: str, implementation_class: type):
        """
        Register a knowledge system implementation.
        
        Args:
            name: Implementation name (e.g., 'lightrag', 'mock', 'file_based')
            implementation_class: Class implementing KnowledgeInterface
        """
        if not issubclass(implementation_class, KnowledgeInterface):
            raise ValueError(f"Implementation must inherit from KnowledgeInterface")
        
        cls._implementations[name] = implementation_class
        
    @classmethod
    def set_default_implementation(cls, name: str):
        """
        Set the default implementation to use.
        
        Args:
            name: Name of registered implementation
        """
        if name not in cls._implementations:
            raise ValueError(f"Implementation '{name}' not registered")
        
        cls._default_implementation = name
    
    @classmethod
    def create(cls, implementation: Optional[str] = None, **kwargs) -> KnowledgeInterface:
        """
        Create a knowledge system instance.
        
        Args:
            implementation: Implementation name (uses default if None)
            **kwargs: Additional arguments passed to implementation constructor
            
        Returns:
            KnowledgeInterface implementation instance
        """
        impl_name = implementation or cls._default_implementation
        
        if impl_name is None:
            raise ValueError("No implementation specified and no default set")
        
        if impl_name not in cls._implementations:
            raise ValueError(f"Unknown implementation: {impl_name}")
        
        impl_class = cls._implementations[impl_name]
        return impl_class(**kwargs)
    
    @classmethod
    def get_available_implementations(cls) -> List[str]:
        """Get list of registered implementation names."""
        return list(cls._implementations.keys())


# Convenience function for getting the default knowledge system
def get_knowledge_system() -> KnowledgeInterface:
    """
    Get the default knowledge system instance.
    
    This function provides a simple way for agents to get a knowledge system
    without dealing with the factory directly.
    
    Returns:
        Configured KnowledgeInterface implementation
        
    Raises:
        ValueError: If no implementation is configured
    """
    return KnowledgeSystemFactory.create()