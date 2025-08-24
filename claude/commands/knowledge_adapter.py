#!/usr/bin/env python3
"""
Knowledge System Adapter
Transparent adapter layer that provides a unified interface to both legacy and LightRAG
knowledge systems, with shadow mode integration. This ensures existing agent operations
remain unaffected while enabling parallel testing.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Import shadow mode components
from shadow_mode import get_shadow_processor, shadow_store_knowledge, shadow_retrieve_knowledge

# Import the new knowledge interface system
knowledge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'knowledge')
if knowledge_path not in sys.path:
    sys.path.insert(0, knowledge_path)

try:
    from knowledge import get_knowledge_system, LIGHTRAG_AVAILABLE
    KNOWLEDGE_SYSTEM_AVAILABLE = True
except ImportError:
    KNOWLEDGE_SYSTEM_AVAILABLE = False
    LIGHTRAG_AVAILABLE = False


class KnowledgeAdapter:
    """
    Unified knowledge system adapter that provides transparent access to both
    legacy and LightRAG systems with shadow mode support.
    
    This adapter ensures that existing agent code continues to work unchanged
    while enabling shadow mode testing of the new system.
    """
    
    def __init__(self, enable_shadow_mode: bool = None):
        """Initialize the knowledge adapter."""
        self.logger = logging.getLogger(f"{__name__}.KnowledgeAdapter")
        
        # Initialize the new knowledge system interface
        self.knowledge_system = None
        if KNOWLEDGE_SYSTEM_AVAILABLE:
            try:
                self.knowledge_system = get_knowledge_system()
                self.logger.info(f"Knowledge system initialized: {self.knowledge_system.get_system_info()['implementation']}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize knowledge system: {e}")
        
        # Determine shadow mode setting
        if enable_shadow_mode is None:
            # Check environment variable
            enable_shadow_mode = os.getenv('RIF_SHADOW_MODE_ENABLED', 'true').lower() == 'true'
        
        self.shadow_mode_enabled = enable_shadow_mode
        
        # Initialize shadow processor if enabled
        self.shadow_processor = None
        if self.shadow_mode_enabled:
            self.shadow_processor = get_shadow_processor()
            if self.shadow_processor and self.shadow_processor.is_enabled():
                self.logger.info("Shadow mode enabled for knowledge operations")
            else:
                self.logger.info("Shadow mode disabled or not configured")
                self.shadow_mode_enabled = False
        
        # Initialize legacy system paths
        self.knowledge_path = Path(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'knowledge'
        ))
        
        # Map collections to legacy folders
        self.collection_map = {
            'patterns': 'patterns',
            'decisions': 'decisions',
            'code_snippets': 'learning',
            'issue_resolutions': 'issues',
            'learnings': 'learning',
            'checkpoints': 'checkpoints'
        }
    
    def store_knowledge(self, 
                       collection: str, 
                       content: Union[str, Dict[str, Any]], 
                       metadata: Dict[str, Any] = None,
                       doc_id: str = None) -> Optional[str]:
        """
        Store knowledge in the appropriate system(s).
        
        This method provides a unified interface that works with both legacy
        JSON files and the new LightRAG system, with optional shadow mode testing.
        
        Args:
            collection: Collection name (patterns, decisions, etc.)
            content: Content to store (string or dict)
            metadata: Optional metadata dictionary
            doc_id: Optional document ID
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Normalize inputs
            metadata = metadata or {}
            
            # Use new knowledge interface if available
            if self.knowledge_system:
                return self.knowledge_system.store_knowledge(collection, content, metadata, doc_id)
            
            # Use shadow mode if enabled
            if self.shadow_mode_enabled and self.shadow_processor:
                content_str = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)
                return shadow_store_knowledge(collection, content_str, metadata, doc_id)
            
            # Fall back to legacy system
            content_str = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)
            return self._store_legacy(collection, content_str, metadata, doc_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store knowledge in {collection}: {e}")
            return None
    
    def retrieve_knowledge(self, 
                          query: str, 
                          collection: str = None, 
                          n_results: int = 5,
                          filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge from the appropriate system(s).
        
        This method provides a unified interface for knowledge retrieval with
        shadow mode support for testing new systems.
        
        Args:
            query: Search query string
            collection: Optional collection to search
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching documents
        """
        try:
            # Use new knowledge interface if available
            if self.knowledge_system:
                return self.knowledge_system.retrieve_knowledge(query, collection, n_results, filters)
            
            # Use shadow mode if enabled
            if self.shadow_mode_enabled and self.shadow_processor:
                return shadow_retrieve_knowledge(query, collection, n_results)
            
            # Fall back to legacy system
            return self._retrieve_legacy(query, collection, n_results, filters)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve knowledge for query '{query}': {e}")
            return []
    
    def update_knowledge(self,
                        collection: str,
                        doc_id: str,
                        content: Union[str, Dict[str, Any]] = None,
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Update existing knowledge item.
        
        Args:
            collection: Collection name
            doc_id: Document ID to update
            content: New content (None to keep existing)
            metadata: New metadata (None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize content if provided
            if content is not None and isinstance(content, dict):
                content = json.dumps(content, indent=2)
            
            # For now, only support legacy system updates
            # Shadow mode doesn't support updates yet
            return self._update_legacy(collection, doc_id, content, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge {doc_id} in {collection}: {e}")
            return False
    
    def search_patterns(self, query: str, complexity: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for patterns with specific criteria.
        
        This is a convenience method for agents that frequently search patterns.
        
        Args:
            query: Search query
            complexity: Optional complexity filter
            limit: Maximum results
            
        Returns:
            List of matching patterns
        """
        filters = {"complexity": complexity} if complexity else None
        return self.retrieve_knowledge(query, "patterns", limit, filters)
    
    def search_decisions(self, query: str, status: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for architectural decisions.
        
        Args:
            query: Search query
            status: Optional status filter
            limit: Maximum results
            
        Returns:
            List of matching decisions
        """
        filters = {"status": status} if status else None
        return self.retrieve_knowledge(query, "decisions", limit, filters)
    
    def store_pattern(self, pattern_data: Dict[str, Any], pattern_id: str = None) -> Optional[str]:
        """
        Store a successful pattern.
        
        This is a convenience method that maintains compatibility with existing
        RIF agent code while enabling shadow mode testing.
        
        Args:
            pattern_data: Pattern data dictionary
            pattern_id: Optional pattern ID
            
        Returns:
            Pattern ID if successful
        """
        # Use new interface convenience method if available
        if self.knowledge_system:
            return self.knowledge_system.store_pattern(pattern_data, pattern_id)
        
        # Fall back to legacy method
        metadata = {
            "type": "pattern",
            "source": pattern_data.get("source", "unknown"),
            "complexity": pattern_data.get("complexity", "medium"),
            "tags": pattern_data.get("tags", [])
        }
        
        if isinstance(metadata["tags"], list):
            metadata["tags"] = ",".join(metadata["tags"])
        
        return self.store_knowledge("patterns", pattern_data, metadata, pattern_id)
    
    def store_decision(self, decision_data: Dict[str, Any], decision_id: str = None) -> Optional[str]:
        """
        Store an architectural decision.
        
        Args:
            decision_data: Decision data dictionary
            decision_id: Optional decision ID
            
        Returns:
            Decision ID if successful
        """
        # Use new interface convenience method if available
        if self.knowledge_system:
            return self.knowledge_system.store_decision(decision_data, decision_id)
        
        # Fall back to legacy method
        metadata = {
            "type": "decision",
            "status": decision_data.get("status", "active"),
            "impact": decision_data.get("impact", "medium"),
            "tags": decision_data.get("tags", [])
        }
        
        if isinstance(metadata["tags"], list):
            metadata["tags"] = ",".join(metadata["tags"])
        
        return self.store_knowledge("decisions", decision_data, metadata, decision_id)
    
    def store_learning(self, learning_data: Dict[str, Any], learning_id: str = None) -> Optional[str]:
        """
        Store learning/experience data.
        
        Args:
            learning_data: Learning data dictionary
            learning_id: Optional learning ID
            
        Returns:
            Learning ID if successful
        """
        # Use new interface convenience method if available
        if self.knowledge_system:
            return self.knowledge_system.store_learning(learning_data, learning_id)
        
        # Fall back to legacy method
        metadata = {
            "type": "learning",
            "source": learning_data.get("source", "unknown"),
            "issue_id": learning_data.get("issue_id"),
            "complexity": learning_data.get("complexity", "medium")
        }
        
        return self.store_knowledge("learnings", learning_data, metadata, learning_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of knowledge systems."""
        status = {
            "adapter_initialized": True,
            "knowledge_interface_available": KNOWLEDGE_SYSTEM_AVAILABLE,
            "shadow_mode_enabled": self.shadow_mode_enabled,
            "shadow_mode_active": False,
            "lightrag_available": LIGHTRAG_AVAILABLE,
            "primary_system": "interface" if self.knowledge_system else "legacy",
            "collections_available": list(self.collection_map.keys())
        }
        
        # Add interface system info if available
        if self.knowledge_system:
            try:
                interface_info = self.knowledge_system.get_system_info()
                status["interface_system"] = interface_info
            except Exception as e:
                status["interface_system_error"] = str(e)
        
        if self.shadow_processor:
            shadow_status = self.shadow_processor.get_status()
            status.update({
                "shadow_mode_active": shadow_status["enabled"],
                "primary_system": shadow_status.get("primary_system", "legacy"),
                "shadow_system": shadow_status.get("shadow_system", "lightrag"),
                "shadow_metrics": shadow_status.get("metrics", {})
            })
        
        return status
    
    # Legacy system implementations
    
    def _store_legacy(self, collection: str, content: str, metadata: Dict[str, Any], doc_id: str = None) -> Optional[str]:
        """Store in legacy file-based system."""
        try:
            # Map collection to folder
            folder = self.collection_map.get(collection, collection)
            folder_path = self.knowledge_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Generate ID if not provided
            if not doc_id:
                import hashlib
                from datetime import datetime
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                doc_id = f"{collection}_{timestamp}_{content_hash}"
            
            # Prepare data
            data = {
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat(),
                "collection": collection
            }
            
            # Write to file
            file_path = folder_path / f"{doc_id}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Stored knowledge in legacy system: {file_path}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Legacy store failed: {e}")
            return None
    
    def _retrieve_legacy(self, query: str, collection: str = None, n_results: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve from legacy file-based system."""
        try:
            results = []
            
            # Determine collections to search
            if collection:
                collections = [collection]
            else:
                collections = list(self.collection_map.keys())
            
            # Search each collection
            for coll in collections:
                folder = self.collection_map.get(coll, coll)
                folder_path = self.knowledge_path / folder
                
                if not folder_path.exists():
                    continue
                
                # Search JSON files
                for file_path in folder_path.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Apply filters if provided
                        if filters and not self._matches_filters(data.get('metadata', {}), filters):
                            continue
                        
                        # Simple text search
                        content = data.get('content', '')
                        if isinstance(content, dict):
                            content = json.dumps(content)
                        
                        if query.lower() in content.lower():
                            # Calculate simple relevance score
                            relevance = content.lower().count(query.lower()) / len(content.split()) if content else 0
                            
                            result = {
                                "collection": coll,
                                "id": data.get('id', file_path.stem),
                                "content": data.get('content', ''),
                                "metadata": data.get('metadata', {}),
                                "distance": 1.0 - min(relevance, 1.0)  # Convert to distance
                            }
                            results.append(result)
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to read {file_path}: {e}")
            
            # Sort by relevance and limit results
            results.sort(key=lambda x: x.get('distance', 1.0))
            return results[:n_results]
            
        except Exception as e:
            self.logger.error(f"Legacy retrieve failed: {e}")
            return []
    
    def _update_legacy(self, collection: str, doc_id: str, content: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update in legacy file-based system."""
        try:
            folder = self.collection_map.get(collection, collection)
            folder_path = self.knowledge_path / folder
            file_path = folder_path / f"{doc_id}.json"
            
            if not file_path.exists():
                self.logger.error(f"Document {doc_id} not found in {collection}")
                return False
            
            # Load existing data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Update content if provided
            if content is not None:
                data['content'] = content
            
            # Update metadata if provided
            if metadata is not None:
                existing_metadata = data.get('metadata', {})
                existing_metadata.update(metadata)
                data['metadata'] = existing_metadata
            
            # Update timestamp
            from datetime import datetime
            data['updated_timestamp'] = datetime.utcnow().isoformat()
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Updated knowledge in legacy system: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Legacy update failed: {e}")
            return False
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # Handle different value types
            if isinstance(value, str) and isinstance(metadata_value, str):
                if value.lower() not in metadata_value.lower():
                    return False
            elif metadata_value != value:
                return False
        
        return True


# Global adapter instance
_knowledge_adapter = None


def get_knowledge_adapter() -> KnowledgeAdapter:
    """Get global knowledge adapter instance."""
    global _knowledge_adapter
    
    if _knowledge_adapter is None:
        _knowledge_adapter = KnowledgeAdapter()
    
    return _knowledge_adapter


# Convenience functions that maintain backward compatibility
def store_knowledge(collection: str, content: Union[str, Dict], metadata: Dict[str, Any] = None, doc_id: str = None) -> Optional[str]:
    """Store knowledge using the adapter."""
    adapter = get_knowledge_adapter()
    return adapter.store_knowledge(collection, content, metadata, doc_id)


def retrieve_knowledge(query: str, collection: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
    """Retrieve knowledge using the adapter."""
    adapter = get_knowledge_adapter()
    return adapter.retrieve_knowledge(query, collection, n_results)


def store_pattern(pattern_data: Dict[str, Any], pattern_id: str = None) -> Optional[str]:
    """Store pattern using the adapter."""
    adapter = get_knowledge_adapter()
    return adapter.store_pattern(pattern_data, pattern_id)


def store_decision(decision_data: Dict[str, Any], decision_id: str = None) -> Optional[str]:
    """Store decision using the adapter."""
    adapter = get_knowledge_adapter()
    return adapter.store_decision(decision_data, decision_id)


def search_patterns(query: str, complexity: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Search patterns using the adapter."""
    adapter = get_knowledge_adapter()
    return adapter.search_patterns(query, complexity, limit)


def search_decisions(query: str, status: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Search decisions using the adapter."""
    adapter = get_knowledge_adapter()
    return adapter.search_decisions(query, status, limit)


if __name__ == "__main__":
    # Test the adapter
    adapter = KnowledgeAdapter()
    
    print("Knowledge Adapter Status:")
    status = adapter.get_system_status()
    print(json.dumps(status, indent=2))
    
    # Test store operation
    print("\nTesting store operation...")
    test_pattern = {
        "title": "Test Pattern",
        "description": "This is a test pattern",
        "complexity": "low",
        "source": "adapter_test"
    }
    
    pattern_id = adapter.store_pattern(test_pattern)
    print(f"Stored pattern: {pattern_id}")
    
    # Test retrieve operation
    print("\nTesting retrieve operation...")
    results = adapter.search_patterns("test pattern")
    print(f"Found {len(results)} results")
    
    if results:
        print(f"First result: {results[0].get('id')}")
    
    print("\nAdapter test completed.")