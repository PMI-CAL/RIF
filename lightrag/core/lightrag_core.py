"""
LightRAG Core Module for RIF Framework
Handles knowledge storage, retrieval, and management using ChromaDB vector database.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
import hashlib


class LightRAGCore:
    """
    Core LightRAG implementation for RIF framework.
    Manages knowledge storage and retrieval with ChromaDB backend.
    """

    def __init__(self, knowledge_path: str = None):
        """
        Initialize LightRAG with ChromaDB backend.
        
        Args:
            knowledge_path: Path to knowledge directory (defaults to ../knowledge)
        """
        if knowledge_path is None:
            # Default to RIF knowledge directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            knowledge_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "knowledge")
        
        self.knowledge_path = knowledge_path
        self.db_path = os.path.join(knowledge_path, "chromadb")
        
        # Ensure directories exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Collection definitions based on RIF architecture
        self.collections = {
            "patterns": "Successful code patterns and templates",
            "decisions": "Architectural decisions and rationale", 
            "code_snippets": "Reusable code examples and functions",
            "issue_resolutions": "Resolved issues and their solutions"
        }
        
        # Initialize collections
        self._init_collections()
        
    def _init_collections(self):
        """Initialize ChromaDB collections for each knowledge type."""
        self.collection_objects = {}
        
        for name, description in self.collections.items():
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name=name)
                self.logger.info(f"Loaded existing collection: {name}")
            except Exception:
                # Create new collection if it doesn't exist
                collection = self.client.create_collection(
                    name=name,
                    metadata={"description": description}
                )
                self.logger.info(f"Created new collection: {name}")
            
            self.collection_objects[name] = collection
    
    def store_knowledge(self, 
                       collection_name: str,
                       content: str,
                       metadata: Dict[str, Any],
                       doc_id: str = None) -> str:
        """
        Store knowledge item in specified collection.
        
        Args:
            collection_name: Name of collection to store in
            content: Text content to store
            metadata: Metadata dictionary
            doc_id: Optional document ID (auto-generated if None)
            
        Returns:
            Document ID of stored item
        """
        if collection_name not in self.collection_objects:
            raise ValueError(f"Collection {collection_name} not found")
        
        # Generate ID if not provided
        if doc_id is None:
            doc_id = self._generate_doc_id(content, metadata)
        
        # Add timestamp to metadata and ensure ChromaDB compatibility
        metadata = metadata.copy()
        metadata["timestamp"] = datetime.utcnow().isoformat()
        metadata["content_hash"] = hashlib.md5(content.encode()).hexdigest()
        
        # Convert lists to comma-separated strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ",".join(str(v) for v in value)
        
        # Store in collection
        collection = self.collection_objects[collection_name]
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        self.logger.info(f"Stored document {doc_id} in collection {collection_name}")
        return doc_id
    
    def retrieve_knowledge(self,
                          query: str,
                          collection_name: str = None,
                          n_results: int = 5,
                          filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge items based on query.
        
        Args:
            query: Search query string
            collection_name: Specific collection to search (None for all)
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching documents with metadata
        """
        results = []
        
        # Determine which collections to search
        collections_to_search = [collection_name] if collection_name else self.collections.keys()
        
        for coll_name in collections_to_search:
            if coll_name not in self.collection_objects:
                continue
                
            collection = self.collection_objects[coll_name]
            
            try:
                # Build query parameters
                query_params = {
                    "query_texts": [query],
                    "n_results": n_results
                }
                
                if filters:
                    query_params["where"] = filters
                
                # Execute search
                search_results = collection.query(**query_params)
                
                # Format results
                for i in range(len(search_results["ids"][0])):
                    result = {
                        "collection": coll_name,
                        "id": search_results["ids"][0][i],
                        "content": search_results["documents"][0][i],
                        "metadata": search_results["metadatas"][0][i],
                        "distance": search_results["distances"][0][i] if "distances" in search_results else None
                    }
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error searching collection {coll_name}: {e}")
        
        # Sort by relevance (lower distance = more relevant)
        results.sort(key=lambda x: x["distance"] or float('inf'))
        
        return results[:n_results]
    
    def update_knowledge(self,
                        collection_name: str,
                        doc_id: str,
                        content: str = None,
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Update existing knowledge item.
        
        Args:
            collection_name: Name of collection
            doc_id: Document ID to update
            content: New content (None to keep existing)
            metadata: New metadata (None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        if collection_name not in self.collection_objects:
            return False
        
        collection = self.collection_objects[collection_name]
        
        try:
            # Get existing document
            existing = collection.get(ids=[doc_id])
            if not existing["ids"]:
                return False
            
            # Prepare update data
            update_data = {"ids": [doc_id]}
            
            if content is not None:
                update_data["documents"] = [content]
                
            if metadata is not None:
                # Merge with existing metadata
                existing_meta = existing["metadatas"][0] if existing["metadatas"] else {}
                updated_meta = existing_meta.copy()
                updated_meta.update(metadata)
                updated_meta["updated_timestamp"] = datetime.utcnow().isoformat()
                update_data["metadatas"] = [updated_meta]
            
            # Update document
            collection.update(**update_data)
            
            self.logger.info(f"Updated document {doc_id} in collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_knowledge(self, collection_name: str, doc_id: str) -> bool:
        """
        Delete knowledge item from collection.
        
        Args:
            collection_name: Name of collection
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if collection_name not in self.collection_objects:
            return False
        
        collection = self.collection_objects[collection_name]
        
        try:
            collection.delete(ids=[doc_id])
            self.logger.info(f"Deleted document {doc_id} from collection {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections."""
        stats = {}
        
        for name, collection in self.collection_objects.items():
            try:
                count = collection.count()
                stats[name] = {
                    "count": count,
                    "description": self.collections[name]
                }
            except Exception as e:
                stats[name] = {
                    "error": str(e),
                    "description": self.collections[name]
                }
        
        return stats
    
    def _generate_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate unique document ID based on content and metadata."""
        # Create hash from content and key metadata
        content_for_hash = content + json.dumps(metadata, sort_keys=True)
        hash_value = hashlib.sha256(content_for_hash.encode()).hexdigest()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"doc_{timestamp}_{hash_value[:8]}"
    
    def export_collection(self, collection_name: str, output_path: str) -> bool:
        """Export collection to JSON file."""
        if collection_name not in self.collection_objects:
            return False
        
        collection = self.collection_objects[collection_name]
        
        try:
            # Get all documents
            all_docs = collection.get()
            
            # Format for export
            export_data = {
                "collection": collection_name,
                "exported_at": datetime.utcnow().isoformat(),
                "count": len(all_docs["ids"]),
                "documents": []
            }
            
            for i in range(len(all_docs["ids"])):
                doc = {
                    "id": all_docs["ids"][i],
                    "content": all_docs["documents"][i] if all_docs["documents"] else None,
                    "metadata": all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                }
                export_data["documents"].append(doc)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(all_docs['ids'])} documents to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting collection {collection_name}: {e}")
            return False
    
    def reset_collection(self, collection_name: str) -> bool:
        """Reset (clear) a collection."""
        if collection_name not in self.collection_objects:
            return False
        
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=collection_name)
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": self.collections[collection_name]}
            )
            self.collection_objects[collection_name] = collection
            
            self.logger.info(f"Reset collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting collection {collection_name}: {e}")
            return False


# Convenience functions for RIF agents
def get_lightrag_instance() -> LightRAGCore:
    """Get a shared LightRAG instance."""
    if not hasattr(get_lightrag_instance, "_instance"):
        get_lightrag_instance._instance = LightRAGCore()
    return get_lightrag_instance._instance


def store_pattern(pattern_data: Dict[str, Any], pattern_id: str = None) -> str:
    """Store a successful pattern."""
    rag = get_lightrag_instance()
    content = json.dumps(pattern_data, indent=2)
    tags = pattern_data.get("tags", [])
    if isinstance(tags, list):
        tags = ",".join(tags)
    metadata = {
        "type": "pattern",
        "source": pattern_data.get("source", "unknown"),
        "complexity": pattern_data.get("complexity", "medium"),
        "tags": tags
    }
    return rag.store_knowledge("patterns", content, metadata, pattern_id)


def store_decision(decision_data: Dict[str, Any], decision_id: str = None) -> str:
    """Store an architectural decision."""
    rag = get_lightrag_instance()
    content = json.dumps(decision_data, indent=2)
    tags = decision_data.get("tags", [])
    if isinstance(tags, list):
        tags = ",".join(tags)
    metadata = {
        "type": "decision",
        "status": decision_data.get("status", "active"),
        "impact": decision_data.get("impact", "medium"),
        "tags": tags
    }
    return rag.store_knowledge("decisions", content, metadata, decision_id)


def find_similar_patterns(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Find similar patterns for a given query."""
    rag = get_lightrag_instance()
    return rag.retrieve_knowledge(query, "patterns", limit)


def find_relevant_decisions(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Find relevant architectural decisions for a query."""
    rag = get_lightrag_instance()
    return rag.retrieve_knowledge(query, "decisions", limit)