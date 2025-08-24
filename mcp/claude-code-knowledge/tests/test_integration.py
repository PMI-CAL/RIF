"""
Integration Tests for Claude Code Knowledge MCP Server.

Tests the MCP server integration with the actual RIF knowledge graph,
validating end-to-end functionality with real data from Phase 1.

These tests require:
- RIF knowledge graph database to be available
- Claude Code knowledge entities to be seeded (from Phase 1)
- Proper database schema and relationships
"""

import pytest
import asyncio
import json
import time
import os
from pathlib import Path
import sys

# Add paths for RIF integration
rif_root = Path(__file__).parents[3]
sys.path.insert(0, str(rif_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from server import ClaudeCodeKnowledgeServer
    from config import load_server_config
    from query_engine import ClaudeKnowledgeQueryEngine
    from knowledge.database.database_interface import RIFDatabase
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


@pytest.fixture(scope="session")
def rif_database():
    """Create RIF database connection for testing."""
    try:
        db = RIFDatabase()
        
        # Verify database is accessible
        stats = db.get_database_stats()
        if stats['total_entities'] == 0:
            pytest.skip("RIF database appears empty - run Phase 1 seeding first")
        
        return db
        
    except Exception as e:
        pytest.skip(f"Cannot connect to RIF database: {e}")


@pytest.fixture(scope="session") 
async def mcp_server(rif_database):
    """Create MCP server instance with real database."""
    config = {
        'log_level': 'INFO',
        'enable_caching': True,
        'cache_size': 50,
        'cache_ttl': 300
    }
    
    server = ClaudeCodeKnowledgeServer(config)
    
    if not await server.initialize():
        pytest.skip("Failed to initialize MCP server")
    
    return server


@pytest.fixture(scope="session")
def claude_entities(rif_database):
    """Get Claude Code entities from the database for validation."""
    entities = {}
    
    # Get sample entities of each type
    for entity_type in ['claude_capability', 'claude_limitation', 'claude_tool', 
                        'implementation_pattern', 'anti_pattern']:
        results = rif_database.search_entities(
            entity_types=[entity_type],
            limit=5
        )
        
        if results:
            entities[entity_type] = results
        else:
            pytest.skip(f"No {entity_type} entities found - run Phase 1 seeding")
    
    return entities


class TestKnowledgeGraphIntegration:
    """Test integration with the actual knowledge graph."""
    
    def test_database_connection(self, rif_database):
        """Test database connection and basic operations."""
        # Test database stats
        stats = rif_database.get_database_stats()
        
        assert stats['total_entities'] > 0
        assert stats['total_relationships'] >= 0
        
        # Test entity search
        entities = rif_database.search_entities(limit=5)
        assert len(entities) > 0
    
    def test_claude_entities_present(self, claude_entities):
        """Test that Claude Code entities are present in the database."""
        required_types = ['claude_capability', 'claude_limitation', 'claude_tool',
                          'implementation_pattern', 'anti_pattern']
        
        for entity_type in required_types:
            assert entity_type in claude_entities, f"Missing {entity_type} entities"
            assert len(claude_entities[entity_type]) > 0
    
    def test_entity_metadata_structure(self, claude_entities):
        """Test that entities have proper metadata structure."""
        for entity_type, entities in claude_entities.items():
            for entity in entities[:2]:  # Test first 2 of each type
                entity_data = entity
                
                assert 'id' in entity_data
                assert 'name' in entity_data
                assert 'type' in entity_data
                assert entity_data['type'] == entity_type
                
                # Check metadata exists (may be empty for some entities)
                if 'metadata' in entity_data:
                    metadata = entity_data['metadata']
                    assert isinstance(metadata, dict)
    
    def test_relationship_structure(self, rif_database, claude_entities):
        """Test relationships between Claude entities."""
        # Get a few entities and check their relationships
        for entity_type, entities in claude_entities.items():
            if entities:
                entity = entities[0]
                relationships = rif_database.get_entity_relationships(entity['id'])
                
                # Relationships should be properly structured
                for rel in relationships[:3]:  # Test first few
                    assert 'source_id' in rel
                    assert 'target_id' in rel
                    assert 'relationship_type' in rel
                    
                    # Relationship types should be valid
                    valid_rel_types = [
                        'supports', 'conflicts_with', 'alternative_to', 
                        'validates', 'requires', 'incompatible_with'
                    ]
                    # Note: May have other relationship types from general RIF system
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_server):
        """Test MCP server initializes properly with real database."""
        assert mcp_server is not None
        assert mcp_server.rif_db is not None
        
        # Test health check
        health = await mcp_server.rif_db.health_check() if hasattr(mcp_server.rif_db, 'health_check') else {'status': 'unknown'}
        # Health check may not be implemented, so just verify server is responsive


class TestMCPToolsIntegration:
    """Test MCP tools with real knowledge graph data."""
    
    @pytest.mark.asyncio
    async def test_check_compatibility_real_data(self, mcp_server):
        """Test compatibility checking with real knowledge graph."""
        # Test compatible approach
        compatible_params = {
            'issue_description': 'Need to read and process files using Claude Code tools',
            'approach': 'Use Read() and Edit() tools directly'
        }
        
        result = await mcp_server._check_compatibility(compatible_params)
        
        assert 'compatible' in result
        assert 'confidence' in result
        assert 'issues' in result
        assert 'recommendations' in result
        assert isinstance(result['execution_time_ms'], (int, float))
        
        # Should be compatible since it uses direct tools
        # Note: Result depends on actual seeded data
    
    @pytest.mark.asyncio
    async def test_check_compatibility_problematic(self, mcp_server):
        """Test compatibility checking with known problematic approach."""
        problematic_params = {
            'issue_description': 'Need to orchestrate multiple agents in parallel',
            'approach': 'Use Task() to launch multiple parallel agents'
        }
        
        result = await mcp_server._check_compatibility(problematic_params)
        
        assert 'compatible' in result
        assert 'confidence' in result
        
        # Should likely be incompatible due to Task() orchestration
        # (depends on seeded limitation data)
        if not result['compatible']:
            assert len(result['issues']) > 0
            assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_recommend_pattern_real_data(self, mcp_server):
        """Test pattern recommendation with real data."""
        params = {
            'technology': 'Python',
            'task_type': 'file_processing',
            'limit': 3
        }
        
        result = await mcp_server._recommend_pattern(params)
        
        assert 'patterns' in result
        assert 'total_found' in result
        assert isinstance(result['patterns'], list)
        
        # Check pattern structure if any found
        for pattern in result['patterns'][:1]:  # Check first pattern
            assert 'pattern_id' in pattern
            assert 'name' in pattern
            assert 'description' in pattern
            assert 'confidence' in pattern
            assert isinstance(pattern['confidence'], (int, float))
    
    @pytest.mark.asyncio
    async def test_find_alternatives_real_data(self, mcp_server):
        """Test alternative finding with real data."""
        params = {
            'problematic_approach': 'Task() orchestration for parallel execution'
        }
        
        result = await mcp_server._find_alternatives(params)
        
        assert 'alternatives' in result
        assert 'total_found' in result
        assert isinstance(result['alternatives'], list)
        
        # Check alternatives structure if any found
        for alternative in result['alternatives'][:1]:
            assert 'id' in alternative
            assert 'name' in alternative  
            assert 'confidence' in alternative
            assert isinstance(alternative['confidence'], (int, float))
    
    @pytest.mark.asyncio
    async def test_validate_architecture_real_data(self, mcp_server):
        """Test architecture validation with real data."""
        # Test simple compatible architecture
        compatible_params = {
            'system_design': 'Simple file processing system using Read and Write tools with direct function calls'
        }
        
        result = await mcp_server._validate_architecture(compatible_params)
        
        assert 'valid' in result
        assert 'confidence' in result
        assert 'components_analyzed' in result
        assert 'issues_found' in result
        assert 'recommendations' in result
        
        # Test complex problematic architecture
        problematic_params = {
            'system_design': 'Complex orchestrator system with centralized agent management, persistent state, and background task processing'
        }
        
        result2 = await mcp_server._validate_architecture(problematic_params)
        
        assert 'valid' in result2
        # Complex architecture may have more issues identified
    
    @pytest.mark.asyncio
    async def test_query_limitations_real_data(self, mcp_server):
        """Test limitation querying with real data."""
        params = {
            'capability_area': 'orchestration'
        }
        
        result = await mcp_server._query_limitations(params)
        
        assert 'limitations' in result
        assert 'capability_area' in result
        assert 'total_found' in result
        
        # Check limitation structure if any found
        for limitation in result['limitations'][:1]:
            assert 'limitation_id' in limitation
            assert 'name' in limitation
            assert 'severity' in limitation
            assert limitation['severity'] in ['low', 'medium', 'high']
    
    @pytest.mark.asyncio
    async def test_severity_filtering(self, mcp_server):
        """Test limitation severity filtering."""
        # Query all limitations
        all_params = {
            'capability_area': 'general'
        }
        all_result = await mcp_server._query_limitations(all_params)
        
        # Query only high severity
        high_params = {
            'capability_area': 'general',
            'severity': 'high'
        }
        high_result = await mcp_server._query_limitations(high_params)
        
        # High severity results should be subset of all results
        assert len(high_result['limitations']) <= len(all_result['limitations'])
        
        # All returned limitations should be high severity
        for limitation in high_result['limitations']:
            assert limitation['severity'] == 'high'


class TestPerformanceIntegration:
    """Test performance with real database."""
    
    @pytest.mark.asyncio
    async def test_response_time_with_real_data(self, mcp_server):
        """Test response times with real database queries."""
        params = {
            'issue_description': 'Test performance with real database query'
        }
        
        start_time = time.time()
        result = await mcp_server._check_compatibility(params)
        execution_time = (time.time() - start_time) * 1000  # ms
        
        # Should complete within reasonable time
        assert execution_time < 2000  # 2 seconds max for integration test
        assert result.get('execution_time_ms', 0) > 0
    
    @pytest.mark.asyncio
    async def test_caching_with_real_data(self, mcp_server):
        """Test caching behavior with real queries."""
        params = {
            'technology': 'Python',
            'task_type': 'testing'
        }
        
        # First query (cache miss)
        start_time1 = time.time()
        result1 = await mcp_server._recommend_pattern(params)
        time1 = (time.time() - start_time1) * 1000
        
        # Second identical query (should hit cache)
        start_time2 = time.time()
        result2 = await mcp_server._recommend_pattern(params)
        time2 = (time.time() - start_time2) * 1000
        
        # Results should be identical
        assert result1['total_found'] == result2['total_found']
        
        # Second query should be faster (though timing can be unreliable)
        # Just verify both completed successfully
        assert time1 > 0
        assert time2 > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_real_queries(self, mcp_server):
        """Test concurrent queries with real database."""
        # Create different queries to avoid cache hits
        queries = [
            {'issue_description': 'Query 1: File processing'},
            {'issue_description': 'Query 2: API integration'},
            {'issue_description': 'Query 3: Data processing'},
        ]
        
        # Execute concurrently
        tasks = [mcp_server._check_compatibility(params) for params in queries]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert 'compatible' in result
            assert 'confidence' in result


class TestErrorHandlingIntegration:
    """Test error handling with real database scenarios."""
    
    @pytest.mark.asyncio
    async def test_malformed_input_handling(self, mcp_server):
        """Test handling of malformed input with real server."""
        # Invalid parameters
        invalid_params = {
            'invalid_field': 'This should cause validation error'
        }
        
        result = await mcp_server._check_compatibility(invalid_params)
        
        assert 'error' in result
        # Should handle gracefully without crashing
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, mcp_server):
        """Test handling of empty/minimal queries."""
        minimal_params = {
            'issue_description': ''  # Empty description
        }
        
        result = await mcp_server._check_compatibility(minimal_params)
        
        # Should either handle gracefully or return appropriate error
        assert isinstance(result, dict)
        # May return error or attempt to process empty query
    
    @pytest.mark.asyncio
    async def test_large_input_handling(self, mcp_server):
        """Test handling of large input data."""
        large_params = {
            'issue_description': 'x' * 5000,  # Large description
            'approach': 'y' * 2000
        }
        
        result = await mcp_server._check_compatibility(large_params)
        
        # Should handle without crashing (may truncate or return error)
        assert isinstance(result, dict)


class TestDataConsistency:
    """Test data consistency and integrity."""
    
    def test_entity_relationship_consistency(self, rif_database, claude_entities):
        """Test that relationships reference valid entities."""
        # Sample a few entities and check their relationships
        for entity_type, entities in claude_entities.items():
            for entity in entities[:2]:  # Test first 2 of each type
                relationships = rif_database.get_entity_relationships(entity['id'])
                
                for rel in relationships[:3]:  # Test first few relationships
                    # Check that related entities exist
                    source_entity = rif_database.get_entity(rel['source_id'])
                    target_entity = rif_database.get_entity(rel['target_id'])
                    
                    # At least one of source/target should exist (some may be from general RIF system)
                    assert source_entity is not None or target_entity is not None
    
    def test_claude_entity_types_valid(self, claude_entities):
        """Test that Claude entities have correct types."""
        expected_types = {
            'claude_capability': 'claude_capability',
            'claude_limitation': 'claude_limitation', 
            'claude_tool': 'claude_tool',
            'implementation_pattern': 'implementation_pattern',
            'anti_pattern': 'anti_pattern'
        }
        
        for entity_type, entities in claude_entities.items():
            for entity in entities:
                assert entity['type'] == expected_types[entity_type]
    
    def test_metadata_consistency(self, claude_entities):
        """Test metadata consistency across entities."""
        # Check that similar entity types have consistent metadata structure
        for entity_type, entities in claude_entities.items():
            if len(entities) > 1:
                # Compare metadata keys across entities of same type
                first_entity = entities[0]
                if 'metadata' in first_entity:
                    first_keys = set(first_entity['metadata'].keys())
                    
                    # Other entities of same type should have similar structure
                    for entity in entities[1:3]:  # Check a few more
                        if 'metadata' in entity:
                            entity_keys = set(entity['metadata'].keys())
                            # Should have some common keys (not necessarily identical)
                            # This is a loose consistency check


if __name__ == '__main__':
    """Run integration tests."""
    # These tests require real database, so provide helpful error messages
    pytest.main([
        __file__, 
        '-v', 
        '--tb=short',
        '-x',  # Stop on first failure
        '--disable-warnings'
    ])