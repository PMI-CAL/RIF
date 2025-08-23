"""
LightRAG Visualization Data Access Layer
Provides interface to ChromaDB collections and LightRAG core functionality.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml

# Add parent directory to path to import LightRAG core
current_dir = Path(__file__).parent
lightrag_dir = current_dir.parent
sys.path.insert(0, str(lightrag_dir))

try:
    from core.lightrag_core import LightRAGCore, get_lightrag_instance
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    logging.error(f"Failed to import LightRAG dependencies: {e}")
    raise


class LightRAGDataAccess:
    """
    Data access layer for LightRAG visualization tool.
    Provides high-level interface to ChromaDB collections and metadata.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize data access layer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LightRAG core
        try:
            knowledge_path = self._resolve_knowledge_path()
            self.lightrag = LightRAGCore(knowledge_path)
            self.logger.info(f"Connected to LightRAG at {knowledge_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LightRAG: {e}")
            raise
            
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_timeout = self.config.get('performance', {}).get('cache_timeout', 3600)
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = current_dir / "config.yaml"
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _resolve_knowledge_path(self) -> str:
        """Resolve the knowledge directory path."""
        # Try config first
        db_path = self.config.get('database', {}).get('chromadb_path')
        if db_path:
            resolved_path = (current_dir / db_path).resolve()
            if resolved_path.exists():
                return str(resolved_path.parent)
        
        # Try default RIF knowledge directory
        rif_knowledge = current_dir.parent.parent / "knowledge"
        if rif_knowledge.exists():
            return str(rif_knowledge)
            
        # Fall back to lightrag knowledge directory
        lightrag_knowledge = current_dir.parent / "knowledge"
        if lightrag_knowledge.exists():
            return str(lightrag_knowledge)
            
        raise FileNotFoundError("Could not locate ChromaDB knowledge directory")
    
    def get_collection_overview(self) -> Dict[str, Any]:
        """
        Get overview of all collections with statistics.
        
        Returns:
            Dictionary with collection statistics and metadata
        """
        try:
            stats = self.lightrag.get_collection_stats()
            
            # Enhance with additional metadata
            overview = {
                "collections": stats,
                "total_documents": sum(coll.get("count", 0) for coll in stats.values()),
                "total_collections": len(stats),
                "database_path": self.lightrag.db_path,
                "last_updated": self._get_database_last_modified()
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Failed to get collection overview: {e}")
            return {"error": str(e)}
    
    def get_collection_details(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection details and sample documents
        """
        try:
            if collection_name not in self.lightrag.collection_objects:
                return {"error": f"Collection '{collection_name}' not found"}
            
            collection = self.lightrag.collection_objects[collection_name]
            
            # Get basic stats
            count = collection.count()
            
            # Get sample documents
            sample_size = min(10, count)
            if sample_size > 0:
                sample_docs = collection.get(limit=sample_size)
            else:
                sample_docs = {"ids": [], "documents": [], "metadatas": []}
            
            # Analyze metadata fields
            metadata_fields = set()
            for metadata in sample_docs.get("metadatas", []):
                if metadata:
                    metadata_fields.update(metadata.keys())
            
            details = {
                "name": collection_name,
                "count": count,
                "description": self.lightrag.collections.get(collection_name, "Unknown"),
                "metadata_fields": list(metadata_fields),
                "sample_documents": self._format_sample_docs(sample_docs),
                "last_updated": self._get_collection_last_updated(collection_name)
            }
            
            return details
            
        except Exception as e:
            self.logger.error(f"Failed to get collection details for {collection_name}: {e}")
            return {"error": str(e)}
    
    def search_documents(self, 
                        query: str,
                        collection_name: str = None,
                        filters: Dict[str, Any] = None,
                        limit: int = None) -> List[Dict[str, Any]]:
        """
        Search documents across collections.
        
        Args:
            query: Search query string
            collection_name: Specific collection to search (None for all)
            filters: Metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching documents with metadata
        """
        try:
            if limit is None:
                limit = self.config.get('performance', {}).get('search_limit', 100)
            
            results = self.lightrag.retrieve_knowledge(
                query=query,
                collection_name=collection_name,
                n_results=limit,
                filters=filters
            )
            
            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                enhanced_result = result.copy()
                enhanced_result['relevance_score'] = 1 - (result.get('distance', 1))
                enhanced_result['content_preview'] = self._create_content_preview(result.get('content', ''))
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return [{"error": str(e)}]
    
    def get_documents_paginated(self,
                               collection_name: str,
                               page: int = 0,
                               page_size: int = None) -> Dict[str, Any]:
        """
        Get documents from a collection with pagination.
        
        Args:
            collection_name: Name of the collection
            page: Page number (0-based)
            page_size: Number of documents per page
            
        Returns:
            Dictionary with documents and pagination info
        """
        try:
            if page_size is None:
                page_size = self.config.get('performance', {}).get('pagination_size', 50)
            
            if collection_name not in self.lightrag.collection_objects:
                return {"error": f"Collection '{collection_name}' not found"}
            
            collection = self.lightrag.collection_objects[collection_name]
            total_count = collection.count()
            
            # Calculate pagination
            offset = page * page_size
            if offset >= total_count:
                return {
                    "documents": [],
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_count": total_count,
                        "total_pages": (total_count + page_size - 1) // page_size,
                        "has_next": False,
                        "has_prev": page > 0
                    }
                }
            
            # Get documents for this page
            limit = min(page_size, total_count - offset)
            docs = collection.get(
                limit=limit,
                offset=offset
            )
            
            # Format documents
            formatted_docs = []
            for i in range(len(docs.get("ids", []))):
                doc = {
                    "id": docs["ids"][i],
                    "content": docs["documents"][i] if docs.get("documents") else None,
                    "metadata": docs["metadatas"][i] if docs.get("metadatas") else {},
                    "content_preview": self._create_content_preview(
                        docs["documents"][i] if docs.get("documents") else ""
                    )
                }
                formatted_docs.append(doc)
            
            return {
                "documents": formatted_docs,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size,
                    "has_next": offset + limit < total_count,
                    "has_prev": page > 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pagination failed for {collection_name}: {e}")
            return {"error": str(e)}
    
    def get_document_relationships(self, 
                                 doc_id: str,
                                 collection_name: str,
                                 k: int = None) -> List[Dict[str, Any]]:
        """
        Get related documents based on similarity.
        
        Args:
            doc_id: Document ID to find relationships for
            collection_name: Collection containing the document
            k: Number of similar documents to return
            
        Returns:
            List of related documents with similarity scores
        """
        try:
            if k is None:
                k = self.config.get('visualization', {}).get('default_k_neighbors', 5)
            
            if collection_name not in self.lightrag.collection_objects:
                return []
            
            collection = self.lightrag.collection_objects[collection_name]
            
            # Get the source document
            source_doc = collection.get(ids=[doc_id])
            if not source_doc.get("ids"):
                return []
            
            # Use document content as query to find similar documents
            content = source_doc["documents"][0] if source_doc.get("documents") else ""
            if not content:
                return []
            
            # Search for similar documents
            similar_docs = collection.query(
                query_texts=[content],
                n_results=k + 1  # +1 to exclude self
            )
            
            # Format relationships, excluding the source document
            relationships = []
            for i, doc_id_result in enumerate(similar_docs["ids"][0]):
                if doc_id_result != doc_id:  # Exclude self
                    relationship = {
                        "id": doc_id_result,
                        "content": similar_docs["documents"][0][i],
                        "metadata": similar_docs["metadatas"][0][i],
                        "similarity": 1 - similar_docs["distances"][0][i],
                        "content_preview": self._create_content_preview(
                            similar_docs["documents"][0][i]
                        )
                    }
                    relationships.append(relationship)
            
            return relationships[:k]  # Limit to requested number
            
        except Exception as e:
            self.logger.error(f"Failed to get relationships for {doc_id}: {e}")
            return []
    
    def export_collection(self, 
                         collection_name: str,
                         format: str = "json",
                         filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Export collection data.
        
        Args:
            collection_name: Name of collection to export
            format: Export format (json, csv)
            filters: Optional metadata filters
            
        Returns:
            Dictionary with export data or error information
        """
        try:
            if collection_name not in self.lightrag.collection_objects:
                return {"error": f"Collection '{collection_name}' not found"}
            
            collection = self.lightrag.collection_objects[collection_name]
            
            # Get all documents (with potential filtering)
            if filters:
                # If filters are provided, we need to get all docs and filter manually
                all_docs = collection.get()
                filtered_docs = self._apply_filters(all_docs, filters)
            else:
                filtered_docs = collection.get()
            
            # Check export size limit
            max_size = self.config.get('export', {}).get('max_export_size', 10000)
            if len(filtered_docs.get("ids", [])) > max_size:
                return {"error": f"Export size ({len(filtered_docs['ids'])}) exceeds limit ({max_size})"}
            
            # Format based on requested format
            if format.lower() == "json":
                return self._format_json_export(collection_name, filtered_docs)
            elif format.lower() == "csv":
                return self._format_csv_export(collection_name, filtered_docs)
            else:
                return {"error": f"Unsupported export format: {format}"}
                
        except Exception as e:
            self.logger.error(f"Export failed for {collection_name}: {e}")
            return {"error": str(e)}
    
    def _format_sample_docs(self, docs: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Format sample documents for display."""
        samples = []
        for i in range(min(limit, len(docs.get("ids", [])))):
            sample = {
                "id": docs["ids"][i],
                "content_preview": self._create_content_preview(
                    docs["documents"][i] if docs.get("documents") else ""
                ),
                "metadata": docs["metadatas"][i] if docs.get("metadatas") else {}
            }
            samples.append(sample)
        return samples
    
    def _create_content_preview(self, content: str, max_length: int = 200) -> str:
        """Create a preview of document content."""
        if not content:
            return ""
        
        # Clean and truncate content
        preview = content.strip()
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        
        return preview
    
    def _get_database_last_modified(self) -> str:
        """Get the last modification time of the database."""
        try:
            db_file = Path(self.lightrag.db_path) / "chroma.sqlite3"
            if db_file.exists():
                import datetime
                mtime = db_file.stat().st_mtime
                return datetime.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            pass
        return "Unknown"
    
    def _get_collection_last_updated(self, collection_name: str) -> str:
        """Get the last update time for a collection."""
        # This is a placeholder - ChromaDB doesn't provide this directly
        # We could implement this by tracking timestamps in metadata
        return "Unknown"
    
    def _apply_filters(self, docs: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply metadata filters to documents."""
        if not filters or not docs.get("metadatas"):
            return docs
        
        filtered_indices = []
        for i, metadata in enumerate(docs["metadatas"]):
            if self._matches_filters(metadata, filters):
                filtered_indices.append(i)
        
        # Create filtered document set
        filtered_docs = {
            "ids": [docs["ids"][i] for i in filtered_indices],
            "documents": [docs["documents"][i] for i in filtered_indices] if docs.get("documents") else None,
            "metadatas": [docs["metadatas"][i] for i in filtered_indices] if docs.get("metadatas") else None
        }
        
        return filtered_docs
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def _format_json_export(self, collection_name: str, docs: Dict[str, Any]) -> Dict[str, Any]:
        """Format documents for JSON export."""
        export_data = {
            "collection": collection_name,
            "exported_at": self._get_current_timestamp(),
            "count": len(docs.get("ids", [])),
            "documents": []
        }
        
        for i in range(len(docs.get("ids", []))):
            doc = {
                "id": docs["ids"][i],
                "content": docs["documents"][i] if docs.get("documents") else None,
                "metadata": docs["metadatas"][i] if docs.get("metadatas") else {}
            }
            export_data["documents"].append(doc)
        
        return {"data": export_data, "format": "json"}
    
    def _format_csv_export(self, collection_name: str, docs: Dict[str, Any]) -> Dict[str, Any]:
        """Format documents for CSV export."""
        # This would need pandas for proper CSV formatting
        # For now, return structured data that can be converted to CSV
        rows = []
        
        for i in range(len(docs.get("ids", []))):
            row = {
                "id": docs["ids"][i],
                "content": docs["documents"][i] if docs.get("documents") else "",
                "metadata": json.dumps(docs["metadatas"][i] if docs.get("metadatas") else {})
            }
            rows.append(row)
        
        return {"data": rows, "format": "csv"}
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        import datetime
        return datetime.datetime.utcnow().isoformat()


# Convenience function for getting data access instance
_data_access_instance = None

def get_data_access(config_path: str = None) -> LightRAGDataAccess:
    """Get shared data access instance."""
    global _data_access_instance
    if _data_access_instance is None:
        _data_access_instance = LightRAGDataAccess(config_path)
    return _data_access_instance