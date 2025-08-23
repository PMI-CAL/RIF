"""
Tests for RIF Knowledge Management Interface

This test suite validates the KnowledgeInterface abstraction and its implementations
to ensure proper decoupling between agents and knowledge systems.
"""

import pytest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Add knowledge module to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge'))

from knowledge.interface import (
    KnowledgeInterface,
    KnowledgeSystemFactory,
    KnowledgeSystemError,
    KnowledgeStorageError,
    KnowledgeRetrievalError,
    get_knowledge_system
)
from knowledge.lightrag_adapter import (
    LightRAGKnowledgeAdapter,
    MockKnowledgeAdapter,
    LIGHTRAG_AVAILABLE
)


class TestKnowledgeInterface:
    """Test the abstract KnowledgeInterface base class."""
    
    def test_interface_cannot_be_instantiated(self):
        """Test that the abstract interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            KnowledgeInterface()
    
    def test_interface_methods_are_abstract(self):
        """Test that interface methods are properly abstract."""
        # Create a minimal implementation that doesn't implement all methods
        class IncompleteImplementation(KnowledgeInterface):
            def store_knowledge(self, collection, content, metadata=None, doc_id=None):
                return "test_id"
        
        # Should fail because not all abstract methods are implemented
        with pytest.raises(TypeError):
            IncompleteImplementation()


class TestMockKnowledgeAdapter:
    """Test the mock knowledge adapter implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = MockKnowledgeAdapter()
    
    def test_initialization(self):
        """Test mock adapter initialization."""
        assert isinstance(self.adapter, KnowledgeInterface)
        assert self.adapter.storage == {}
        assert self.adapter.next_id == 1
    
    def test_store_knowledge_basic(self):
        """Test basic knowledge storage."""
        content = "Test content"
        metadata = {"type": "test", "complexity": "low"}
        
        doc_id = self.adapter.store_knowledge("test_collection", content, metadata)
        
        assert doc_id is not None
        assert doc_id.startswith("mock_")
        assert "test_collection" in self.adapter.storage
        assert doc_id in self.adapter.storage["test_collection"]
        
        stored_doc = self.adapter.storage["test_collection"][doc_id]
        assert stored_doc["content"] == content
        assert stored_doc["metadata"] == metadata
    
    def test_store_knowledge_dict_content(self):
        """Test storing dictionary content."""
        content_dict = {"title": "Test", "description": "Test description"}
        
        doc_id = self.adapter.store_knowledge("patterns", content_dict)
        
        stored_content = self.adapter.storage["patterns"][doc_id]["content"]
        # Should be JSON serialized
        assert isinstance(stored_content, str)
        parsed_content = json.loads(stored_content)
        assert parsed_content == content_dict
    
    def test_retrieve_knowledge_simple(self):
        """Test basic knowledge retrieval."""
        # Store some test data
        self.adapter.store_knowledge("patterns", "Test pattern content", {"type": "pattern"})
        self.adapter.store_knowledge("patterns", "Another pattern", {"type": "pattern"})
        self.adapter.store_knowledge("decisions", "Test decision", {"type": "decision"})
        
        # Search for patterns
        results = self.adapter.retrieve_knowledge("pattern", "patterns", n_results=5)
        
        assert len(results) == 2
        for result in results:
            assert "pattern" in result["content"].lower()
            assert result["collection"] == "patterns"
            assert "distance" in result
    
    def test_retrieve_knowledge_with_filters(self):
        """Test knowledge retrieval with metadata filters."""
        self.adapter.store_knowledge("patterns", "High complexity pattern", 
                                    {"complexity": "high", "type": "pattern"})
        self.adapter.store_knowledge("patterns", "Low complexity pattern", 
                                    {"complexity": "low", "type": "pattern"})
        
        # Filter by complexity
        results = self.adapter.retrieve_knowledge("pattern", "patterns", 
                                                filters={"complexity": "high"})
        
        assert len(results) == 1
        assert results[0]["metadata"]["complexity"] == "high"
    
    def test_update_knowledge(self):
        """Test knowledge updates."""
        # Store initial content
        doc_id = self.adapter.store_knowledge("test", "Initial content", {"version": 1})
        
        # Update content only
        success = self.adapter.update_knowledge("test", doc_id, "Updated content")
        assert success
        assert self.adapter.storage["test"][doc_id]["content"] == "Updated content"
        assert self.adapter.storage["test"][doc_id]["metadata"]["version"] == 1
        
        # Update metadata only
        success = self.adapter.update_knowledge("test", doc_id, metadata={"version": 2, "updated": True})
        assert success
        assert self.adapter.storage["test"][doc_id]["content"] == "Updated content"
        assert self.adapter.storage["test"][doc_id]["metadata"]["version"] == 2
        assert self.adapter.storage["test"][doc_id]["metadata"]["updated"] is True
    
    def test_delete_knowledge(self):
        """Test knowledge deletion."""
        doc_id = self.adapter.store_knowledge("test", "Content to delete")
        
        # Verify it exists
        assert doc_id in self.adapter.storage["test"]
        
        # Delete it
        success = self.adapter.delete_knowledge("test", doc_id)
        assert success
        assert doc_id not in self.adapter.storage["test"]
        
        # Try to delete non-existent document
        success = self.adapter.delete_knowledge("test", "nonexistent")
        assert not success
    
    def test_convenience_methods(self):
        """Test convenience methods for patterns and decisions."""
        # Test store_pattern
        pattern_data = {
            "title": "Test Pattern",
            "description": "A test pattern",
            "complexity": "medium",
            "tags": ["test", "pattern"]
        }
        pattern_id = self.adapter.store_pattern(pattern_data)
        assert pattern_id is not None
        
        # Test store_decision
        decision_data = {
            "title": "Test Decision",
            "status": "accepted",
            "impact": "high",
            "tags": ["test", "decision"]
        }
        decision_id = self.adapter.store_decision(decision_data)
        assert decision_id is not None
        
        # Test search_patterns
        patterns = self.adapter.search_patterns("test", complexity="medium")
        assert len(patterns) >= 1
        
        # Test search_decisions  
        decisions = self.adapter.search_decisions("test", status="accepted")
        assert len(decisions) >= 1
    
    def test_get_collection_stats(self):
        """Test collection statistics."""
        # Add some data
        self.adapter.store_knowledge("patterns", "Pattern 1")
        self.adapter.store_knowledge("patterns", "Pattern 2") 
        self.adapter.store_knowledge("decisions", "Decision 1")
        
        stats = self.adapter.get_collection_stats()
        
        assert "patterns" in stats
        assert stats["patterns"]["count"] == 2
        assert "decisions" in stats
        assert stats["decisions"]["count"] == 1
    
    def test_get_system_info(self):
        """Test system information."""
        info = self.adapter.get_system_info()
        
        assert info["implementation"] == "MockKnowledgeAdapter"
        assert info["backend"] == "in-memory"
        assert "features" in info
        assert "basic_storage" in info["features"]


@pytest.mark.skipif(not LIGHTRAG_AVAILABLE, reason="LightRAG not available")
class TestLightRAGKnowledgeAdapter:
    """Test the LightRAG knowledge adapter implementation."""
    
    def setup_method(self):
        """Set up test fixtures with temporary knowledge directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = LightRAGKnowledgeAdapter(knowledge_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LightRAG adapter initialization."""
        assert isinstance(self.adapter, KnowledgeInterface)
        assert hasattr(self.adapter, 'lightrag')
        
        # Check system info
        info = self.adapter.get_system_info()
        assert info["implementation"] == "LightRAGKnowledgeAdapter"
        assert info["backend"] == "ChromaDB"
        assert "semantic_search" in info["features"]
    
    def test_store_and_retrieve_knowledge(self):
        """Test storing and retrieving knowledge with LightRAG."""
        content = "This is a test pattern for semantic search capabilities"
        metadata = {"type": "pattern", "complexity": "medium", "tags": "test,pattern"}
        
        # Store knowledge
        doc_id = self.adapter.store_knowledge("patterns", content, metadata)
        assert doc_id is not None
        
        # Retrieve knowledge
        results = self.adapter.retrieve_knowledge("semantic search", "patterns", n_results=5)
        assert len(results) > 0
        
        # Check result structure
        result = results[0]
        assert "id" in result
        assert "content" in result
        assert "metadata" in result
        assert "collection" in result
        assert "distance" in result
    
    def test_convenience_methods_integration(self):
        """Test convenience methods with LightRAG backend."""
        # Store pattern
        pattern_data = {
            "title": "Integration Test Pattern",
            "description": "Pattern for testing LightRAG integration",
            "complexity": "medium",
            "source": "test-25",
            "tags": ["integration", "test"]
        }
        pattern_id = self.adapter.store_pattern(pattern_data)
        assert pattern_id is not None
        
        # Store decision
        decision_data = {
            "title": "Integration Test Decision", 
            "context": "Testing LightRAG decision storage",
            "decision": "Use LightRAG for knowledge management",
            "status": "accepted",
            "impact": "high",
            "tags": ["integration", "architecture"]
        }
        decision_id = self.adapter.store_decision(decision_data)
        assert decision_id is not None
        
        # Search patterns
        patterns = self.adapter.search_patterns("integration", limit=5)
        assert len(patterns) > 0
        
        # Search decisions
        decisions = self.adapter.search_decisions("LightRAG", status="accepted", limit=5)
        assert len(decisions) > 0
    
    def test_collection_stats(self):
        """Test collection statistics with LightRAG."""
        # Add some data
        self.adapter.store_pattern({"title": "Test Pattern 1", "complexity": "low"})
        self.adapter.store_pattern({"title": "Test Pattern 2", "complexity": "high"})
        self.adapter.store_decision({"title": "Test Decision", "status": "accepted"})
        
        stats = self.adapter.get_collection_stats()
        
        assert isinstance(stats, dict)
        # Check that collections exist
        expected_collections = ["patterns", "decisions", "code_snippets", "issue_resolutions"]
        for collection in expected_collections:
            if collection in stats:
                assert "count" in stats[collection]


class TestKnowledgeSystemFactory:
    """Test the knowledge system factory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing implementations for clean testing
        KnowledgeSystemFactory._implementations = {}
        KnowledgeSystemFactory._default_implementation = None
    
    def test_register_implementation(self):
        """Test registering implementations."""
        KnowledgeSystemFactory.register_implementation("mock", MockKnowledgeAdapter)
        
        assert "mock" in KnowledgeSystemFactory.get_available_implementations()
        
        # Test invalid registration
        with pytest.raises(ValueError):
            KnowledgeSystemFactory.register_implementation("invalid", dict)
    
    def test_create_implementation(self):
        """Test creating implementations through factory."""
        KnowledgeSystemFactory.register_implementation("mock", MockKnowledgeAdapter)
        
        adapter = KnowledgeSystemFactory.create("mock")
        assert isinstance(adapter, MockKnowledgeAdapter)
        
        # Test unknown implementation
        with pytest.raises(ValueError):
            KnowledgeSystemFactory.create("unknown")
    
    def test_default_implementation(self):
        """Test default implementation setting."""
        KnowledgeSystemFactory.register_implementation("mock", MockKnowledgeAdapter)
        KnowledgeSystemFactory.set_default_implementation("mock")
        
        # Should create default implementation
        adapter = KnowledgeSystemFactory.create()
        assert isinstance(adapter, MockKnowledgeAdapter)
        
        # Test setting invalid default
        with pytest.raises(ValueError):
            KnowledgeSystemFactory.set_default_implementation("unknown")


class TestKnowledgeIntegration:
    """Test integration between interface and implementations."""
    
    def test_interface_compliance(self):
        """Test that all implementations comply with interface."""
        # Test mock adapter
        mock_adapter = MockKnowledgeAdapter()
        assert isinstance(mock_adapter, KnowledgeInterface)
        
        # Test all required methods exist
        required_methods = [
            'store_knowledge', 'retrieve_knowledge', 'update_knowledge', 
            'delete_knowledge', 'get_collection_stats', 'store_pattern',
            'store_decision', 'search_patterns', 'search_decisions'
        ]
        
        for method in required_methods:
            assert hasattr(mock_adapter, method)
            assert callable(getattr(mock_adapter, method))
        
        # Test LightRAG adapter if available
        if LIGHTRAG_AVAILABLE:
            with tempfile.TemporaryDirectory() as temp_dir:
                lightrag_adapter = LightRAGKnowledgeAdapter(knowledge_path=temp_dir)
                assert isinstance(lightrag_adapter, KnowledgeInterface)
                
                for method in required_methods:
                    assert hasattr(lightrag_adapter, method)
                    assert callable(getattr(lightrag_adapter, method))
    
    def test_cross_implementation_compatibility(self):
        """Test that different implementations handle same data consistently."""
        # Test data
        pattern_data = {
            "title": "Cross-implementation Test",
            "description": "Testing compatibility between implementations",
            "complexity": "medium",
            "tags": ["test", "compatibility"]
        }
        
        # Test with mock adapter
        mock_adapter = MockKnowledgeAdapter()
        mock_pattern_id = mock_adapter.store_pattern(pattern_data)
        assert mock_pattern_id is not None
        
        mock_results = mock_adapter.search_patterns("compatibility")
        assert len(mock_results) > 0
        
        # Test with LightRAG adapter if available
        if LIGHTRAG_AVAILABLE:
            with tempfile.TemporaryDirectory() as temp_dir:
                lightrag_adapter = LightRAGKnowledgeAdapter(knowledge_path=temp_dir)
                lightrag_pattern_id = lightrag_adapter.store_pattern(pattern_data)
                assert lightrag_pattern_id is not None
                
                lightrag_results = lightrag_adapter.search_patterns("compatibility")
                assert len(lightrag_results) > 0
                
                # Both should return results in same format
                mock_result = mock_results[0]
                lightrag_result = lightrag_results[0]
                
                for key in ["id", "content", "metadata", "collection"]:
                    assert key in mock_result
                    assert key in lightrag_result


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = MockKnowledgeAdapter()
    
    def test_invalid_collection_handling(self):
        """Test handling of invalid collection names."""
        # Empty collection name should still work with mock adapter
        doc_id = self.adapter.store_knowledge("", "test content")
        assert doc_id is not None
    
    def test_none_values_handling(self):
        """Test handling of None values."""
        # None content should be handled gracefully
        doc_id = self.adapter.store_knowledge("test", None)
        assert doc_id is not None
        
        # None metadata should be handled
        doc_id = self.adapter.store_knowledge("test", "content", None)
        assert doc_id is not None
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        results = self.adapter.retrieve_knowledge("")
        assert isinstance(results, list)
        
        results = self.adapter.search_patterns("")
        assert isinstance(results, list)
    
    def test_update_nonexistent_document(self):
        """Test updating non-existent documents."""
        success = self.adapter.update_knowledge("test", "nonexistent_id", "new content")
        assert not success
    
    def test_large_content_handling(self):
        """Test handling of large content."""
        large_content = "x" * 10000  # 10KB of text
        doc_id = self.adapter.store_knowledge("test", large_content)
        assert doc_id is not None
        
        results = self.adapter.retrieve_knowledge("x", "test")
        assert len(results) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])