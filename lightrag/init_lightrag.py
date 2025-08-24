#!/usr/bin/env python3
"""
LightRAG Initialization Script for RIF Framework
Sets up the LightRAG system and migrates existing knowledge.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Add the RIF project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from lightrag.core.lightrag_core import LightRAGCore
    from lightrag.embeddings.embedding_manager import EmbeddingManager
except ImportError as e:
    print(f"Error importing LightRAG modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class LightRAGInitializer:
    """Handles LightRAG initialization and knowledge migration."""
    
    def __init__(self, knowledge_path: str = None):
        """
        Initialize the LightRAG system.
        
        Args:
            knowledge_path: Path to existing knowledge directory
        """
        if knowledge_path is None:
            knowledge_path = os.path.join(project_root, "knowledge")
        
        self.knowledge_path = knowledge_path
        self.logger = self._setup_logging()
        
        # Initialize components
        self.rag = None
        self.embedding_manager = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """
        Initialize LightRAG system.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Starting LightRAG initialization...")
        
        try:
            # Step 1: Initialize embedding manager
            self.logger.info("Initializing embedding manager...")
            self.embedding_manager = EmbeddingManager()
            
            # Step 2: Initialize RAG core
            self.logger.info("Initializing LightRAG core...")
            self.rag = LightRAGCore(self.knowledge_path)
            
            # Step 3: Verify collections
            self.logger.info("Verifying collections...")
            stats = self.rag.get_collection_stats()
            for name, info in stats.items():
                if "error" in info:
                    self.logger.error(f"Collection {name} error: {info['error']}")
                    return False
                self.logger.info(f"Collection {name}: {info['count']} documents")
            
            # Step 4: Migrate existing knowledge if present
            self.logger.info("Checking for existing knowledge to migrate...")
            migrated_count = self.migrate_existing_knowledge()
            if migrated_count > 0:
                self.logger.info(f"Migrated {migrated_count} knowledge items")
            
            # Step 5: Create checkpoint
            self._create_initialization_checkpoint()
            
            self.logger.info("LightRAG initialization completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"LightRAG initialization failed: {e}")
            return False
    
    def migrate_existing_knowledge(self) -> int:
        """
        Migrate existing knowledge from JSON files to LightRAG collections.
        
        Returns:
            Number of items migrated
        """
        migrated_count = 0
        
        # Migration mappings: directory -> collection
        migration_map = {
            "patterns": "patterns",
            "decisions": "decisions", 
            "issues": "issue_resolutions"
        }
        
        for source_dir, collection_name in migration_map.items():
            source_path = os.path.join(self.knowledge_path, source_dir)
            
            if not os.path.exists(source_path):
                continue
            
            # Process JSON files in source directory
            for filename in os.listdir(source_path):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(source_path, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Create content and metadata for LightRAG
                    if isinstance(data, dict):
                        content = json.dumps(data, indent=2)
                        metadata = {
                            "source_file": filename,
                            "migrated_from": source_dir,
                            "original_timestamp": data.get("timestamp", "unknown")
                        }
                        
                        # Add specific metadata based on collection
                        if collection_name == "patterns":
                            metadata.update({
                                "type": "pattern",
                                "complexity": data.get("complexity", "medium"),
                                "tags": data.get("tags", [])
                            })
                        elif collection_name == "decisions":
                            metadata.update({
                                "type": "decision", 
                                "status": data.get("status", "active"),
                                "impact": data.get("impact", "medium")
                            })
                        elif collection_name == "issue_resolutions":
                            metadata.update({
                                "type": "issue_resolution",
                                "issue_number": data.get("issue_number", "unknown"),
                                "status": data.get("status", "resolved")
                            })
                        
                        # Store in LightRAG
                        doc_id = f"migrated_{filename.replace('.json', '')}"
                        self.rag.store_knowledge(collection_name, content, metadata, doc_id)
                        migrated_count += 1
                        
                        self.logger.info(f"Migrated {filename} to {collection_name}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to migrate {filename}: {e}")
        
        return migrated_count
    
    def _create_initialization_checkpoint(self):
        """Create initialization checkpoint."""
        checkpoint_dir = os.path.join(self.knowledge_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            "type": "lightrag_initialization",
            "timestamp": datetime.utcnow().isoformat(),
            "collections": self.rag.get_collection_stats(),
            "embedding_backend": self.embedding_manager.backend,
            "cache_stats": self.embedding_manager.get_cache_stats()
        }
        
        checkpoint_file = os.path.join(checkpoint_dir, "lightrag-init.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Created initialization checkpoint: {checkpoint_file}")
    
    def test_functionality(self) -> bool:
        """
        Test basic LightRAG functionality.
        
        Returns:
            True if all tests pass
        """
        self.logger.info("Testing LightRAG functionality...")
        
        try:
            # Test 1: Store and retrieve a test document
            test_content = "This is a test pattern for RIF framework initialization."
            test_metadata = {
                "type": "test",
                "category": "initialization",
                "tags": ["test", "init"]
            }
            
            doc_id = self.rag.store_knowledge("patterns", test_content, test_metadata, "test_doc")
            self.logger.info("✓ Test document stored successfully")
            
            # Test 2: Search for the test document
            results = self.rag.retrieve_knowledge("test pattern initialization", "patterns", 1)
            if not results:
                self.logger.error("✗ Failed to retrieve test document")
                return False
            
            self.logger.info("✓ Test document retrieved successfully")
            
            # Test 3: Test embeddings
            embedding = self.embedding_manager.get_embedding("test embedding functionality")
            if embedding is None:
                self.logger.warning("⚠ Embedding test failed - may need OpenAI API key or local model")
            else:
                self.logger.info("✓ Embedding generation successful")
            
            # Test 4: Clean up test document
            self.rag.delete_knowledge("patterns", "test_doc")
            self.logger.info("✓ Test document cleaned up")
            
            self.logger.info("All functionality tests passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Functionality test failed: {e}")
            return False
    
    def print_status(self):
        """Print current LightRAG status."""
        if not self.rag:
            print("LightRAG not initialized")
            return
        
        print("\n" + "="*50)
        print("LIGHTRAG STATUS")
        print("="*50)
        
        # Collection stats
        stats = self.rag.get_collection_stats()
        print(f"Collections ({len(stats)}):")
        for name, info in stats.items():
            if "error" in info:
                print(f"  ✗ {name}: ERROR - {info['error']}")
            else:
                print(f"  ✓ {name}: {info['count']} documents - {info['description']}")
        
        # Embedding stats
        if self.embedding_manager:
            cache_stats = self.embedding_manager.get_cache_stats()
            print(f"\nEmbedding Manager:")
            print(f"  Backend: {cache_stats['backend']}")
            print(f"  Cache: {cache_stats['file_count']} files ({cache_stats['total_size_mb']} MB)")
        
        print("="*50)


def main():
    """Main initialization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize LightRAG for RIF framework")
    parser.add_argument("--knowledge-path", help="Path to knowledge directory")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    parser.add_argument("--status", action="store_true", help="Show LightRAG status")
    parser.add_argument("--migrate-only", action="store_true", help="Only migrate existing knowledge")
    
    args = parser.parse_args()
    
    # Initialize LightRAG
    initializer = LightRAGInitializer(args.knowledge_path)
    
    if args.status:
        # Just show status
        try:
            initializer.rag = LightRAGCore(initializer.knowledge_path)
            initializer.embedding_manager = EmbeddingManager()
            initializer.print_status()
        except Exception as e:
            print(f"Error getting status: {e}")
        return
    
    if args.migrate_only:
        # Just migrate knowledge
        try:
            initializer.rag = LightRAGCore(initializer.knowledge_path)
            count = initializer.migrate_existing_knowledge()
            print(f"Migrated {count} knowledge items")
        except Exception as e:
            print(f"Migration failed: {e}")
        return
    
    # Full initialization
    success = initializer.initialize()
    
    if success:
        if args.test:
            test_success = initializer.test_functionality()
            if not test_success:
                sys.exit(1)
        
        initializer.print_status()
        print("\nLightRAG is ready for use!")
        
        # Print usage examples
        print("\nUsage Examples:")
        print("  from lightrag.core.lightrag_core import get_lightrag_instance")
        print("  rag = get_lightrag_instance()")
        print("  rag.store_knowledge('patterns', 'content', {'type': 'pattern'})")
        print("  results = rag.retrieve_knowledge('search query', 'patterns')")
        
    else:
        print("LightRAG initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()