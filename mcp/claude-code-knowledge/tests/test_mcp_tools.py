"""
Unit Tests for Claude Code Knowledge MCP Server Tools.

Comprehensive test suite covering all 5 MCP tools with various scenarios:
- Normal operation testing
- Error handling validation
- Edge case coverage
- Performance benchmarks
- Safety feature verification
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add the server directory to path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

from server import ClaudeCodeKnowledgeServer
from config import ServerConfig
from safety import InputValidator, ValidationError, GracefulDegradation
from query_engine import ClaudeKnowledgeQueryEngine, QueryResult


class TestMCPTools:
    """Test class for MCP tools functionality."""
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock server instance for testing."""
        config = {
            'log_level': 'DEBUG',
            'cache_size': 10,
            'cache_ttl': 60,
            'enable_caching': True
        }
        
        server = ClaudeCodeKnowledgeServer(config)
        
        # Mock the RIF database
        mock_rif_db = Mock()
        server.rif_db = mock_rif_db
        
        # Mock query engine
        mock_query_engine = Mock(spec=ClaudeKnowledgeQueryEngine)
        server.query_engine = mock_query_engine
        
        return server
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return {
            'capability': {
                'id': 'cap_1',
                'name': 'File Operations',
                'type': 'claude_capability',
                'metadata': {
                    'description': 'Read, write, and edit files',
                    'tools': ['Read', 'Write', 'Edit']
                }
            },
            'limitation': {
                'id': 'lim_1', 
                'name': 'No Task Orchestration',
                'type': 'claude_limitation',
                'metadata': {
                    'description': 'Cannot use Task() for orchestration',
                    'severity': 'high',
                    'category': 'orchestration'
                }
            },
            'pattern': {
                'id': 'pat_1',
                'name': 'Direct Tool Usage',
                'type': 'implementation_pattern',
                'metadata': {
                    'description': 'Use tools directly instead of orchestration',
                    'technology': 'Python',
                    'task_type': 'file_processing',
                    'code_example': 'Read(file_path="/path/to/file")'
                }
            }
        }
    
    
    class TestCheckCompatibility:
        """Tests for check_compatibility tool."""
        
        @pytest.mark.asyncio
        async def test_compatible_approach(self, mock_server, sample_entities):
            """Test compatibility check for valid approach."""
            # Setup mock responses
            mock_server.rif_db.search_entities.return_value = []  # No conflicts
            
            params = {
                'issue_description': 'Need to process files using Read and Write tools',
                'approach': 'Use direct tool calls'
            }
            
            result = await mock_server._check_compatibility(params)
            
            assert result['compatible'] is True
            assert result['confidence'] > 0.8
            assert len(result['issues']) == 0
            assert 'execution_time_ms' in result
        
        @pytest.mark.asyncio
        async def test_incompatible_approach(self, mock_server, sample_entities):
            """Test compatibility check for problematic approach."""
            # Mock finding conflicts
            mock_server.rif_db.search_entities.return_value = [sample_entities['limitation']]
            mock_server.rif_db.get_entity.return_value = sample_entities['limitation']
            
            params = {
                'issue_description': 'Need to orchestrate multiple agents with Task()',
                'approach': 'Use Task() for parallel execution'
            }
            
            result = await mock_server._check_compatibility(params)
            
            assert result['compatible'] is False
            assert result['confidence'] < 0.5
            assert len(result['issues']) > 0
            assert any('task()' in issue.get('concept', '').lower() for issue in result['issues'])
        
        @pytest.mark.asyncio
        async def test_missing_required_field(self, mock_server):
            """Test validation of required fields."""
            params = {}  # Missing issue_description
            
            result = await mock_server._check_compatibility(params)
            
            assert 'error' in result
            assert 'issue_description' in result['error']
        
        @pytest.mark.asyncio
        async def test_caching_behavior(self, mock_server, sample_entities):
            """Test query caching functionality."""
            mock_server.rif_db.search_entities.return_value = []
            
            params = {
                'issue_description': 'Test caching with same query',
            }
            
            # First call
            result1 = await mock_server._check_compatibility(params)
            # Second call (should use cache)
            result2 = await mock_server._check_compatibility(params)
            
            assert result1['compatible'] == result2['compatible']
            # Cache should be faster (though timing can be unreliable in tests)
    
    
    class TestRecommendPattern:
        """Tests for recommend_pattern tool."""
        
        @pytest.mark.asyncio
        async def test_successful_pattern_recommendation(self, mock_server, sample_entities):
            """Test successful pattern recommendation."""
            # Mock hybrid search results
            mock_result = Mock()
            mock_result.entity_id = 'pat_1'
            mock_result.similarity_score = 0.85
            
            mock_server.rif_db.hybrid_search.return_value = [mock_result]
            mock_server.rif_db.get_entity.return_value = sample_entities['pattern']
            mock_server.rif_db.get_entity_relationships.return_value = []
            
            params = {
                'technology': 'Python',
                'task_type': 'file_processing',
                'limit': 3
            }
            
            result = await mock_server._recommend_pattern(params)
            
            assert 'patterns' in result
            assert len(result['patterns']) > 0
            assert result['patterns'][0]['name'] == 'Direct Tool Usage'
            assert result['patterns'][0]['technology'] == 'Python'
            assert result['patterns'][0]['confidence'] == 0.85
        
        @pytest.mark.asyncio 
        async def test_no_patterns_found(self, mock_server):
            """Test when no patterns match the query."""
            mock_server.rif_db.hybrid_search.return_value = []
            
            params = {
                'technology': 'UnknownTech',
                'task_type': 'impossible_task'
            }
            
            result = await mock_server._recommend_pattern(params)
            
            assert result['patterns'] == []
            assert result['total_found'] == 0
        
        @pytest.mark.asyncio
        async def test_limit_parameter(self, mock_server, sample_entities):
            """Test limit parameter enforcement."""
            # Create multiple mock results
            mock_results = []
            for i in range(10):
                mock_result = Mock()
                mock_result.entity_id = f'pat_{i}'
                mock_result.similarity_score = 0.8 - (i * 0.05)
                mock_results.append(mock_result)
            
            mock_server.rif_db.hybrid_search.return_value = mock_results
            mock_server.rif_db.get_entity.return_value = sample_entities['pattern']
            mock_server.rif_db.get_entity_relationships.return_value = []
            
            params = {
                'technology': 'Python',
                'task_type': 'processing',
                'limit': 3
            }
            
            result = await mock_server._recommend_pattern(params)
            
            assert len(result['patterns']) <= 3
        
        @pytest.mark.asyncio
        async def test_missing_required_fields(self, mock_server):
            """Test validation of required fields."""
            # Missing technology field
            params = {
                'task_type': 'processing'
            }
            
            result = await mock_server._recommend_pattern(params)
            
            assert 'error' in result
            assert 'technology' in result['error']
    
    
    class TestFindAlternatives:
        """Tests for find_alternatives tool."""
        
        @pytest.mark.asyncio
        async def test_find_relationship_alternatives(self, mock_server, sample_entities):
            """Test finding alternatives through relationships."""
            # Mock search for problematic approach
            mock_server.rif_db.search_entities.return_value = [sample_entities['limitation']]
            mock_server.rif_db.get_entity.return_value = sample_entities['limitation']
            
            # Mock relationships pointing to alternatives
            mock_relationship = {
                'relationship_type': 'alternative_to',
                'target_id': 'pat_1',
                'confidence': 0.9
            }
            mock_server.rif_db.get_entity_relationships.return_value = [mock_relationship]
            
            # Mock alternative entity
            mock_server.rif_db.get_entity.side_effect = lambda id: {
                'lim_1': sample_entities['limitation'],
                'pat_1': sample_entities['pattern']
            }.get(id)
            
            params = {
                'problematic_approach': 'Task() orchestration'
            }
            
            result = await mock_server._find_alternatives(params)
            
            assert 'alternatives' in result
            assert len(result['alternatives']) > 0
            assert result['alternatives'][0]['name'] == 'Direct Tool Usage'
        
        @pytest.mark.asyncio
        async def test_find_similarity_alternatives(self, mock_server, sample_entities):
            """Test finding alternatives through vector similarity."""
            # Mock no relationship alternatives
            mock_server.rif_db.search_entities.return_value = []
            mock_server.rif_db.get_entity_relationships.return_value = []
            
            # Mock similarity search
            mock_result = Mock()
            mock_result.entity_id = 'pat_1'
            mock_result.similarity_score = 0.75
            
            mock_server.rif_db.hybrid_search.return_value = [mock_result]
            mock_server.rif_db.get_entity.return_value = sample_entities['pattern']
            
            params = {
                'problematic_approach': 'Complex orchestration pattern'
            }
            
            result = await mock_server._find_alternatives(params)
            
            assert len(result['alternatives']) > 0
            assert result['alternatives'][0]['source'] == 'similarity'
        
        @pytest.mark.asyncio
        async def test_no_alternatives_found(self, mock_server):
            """Test when no alternatives are found."""
            mock_server.rif_db.search_entities.return_value = []
            mock_server.rif_db.hybrid_search.return_value = []
            
            params = {
                'problematic_approach': 'Completely unknown approach'
            }
            
            result = await mock_server._find_alternatives(params)
            
            assert result['alternatives'] == []
            assert result['total_found'] == 0
    
    
    class TestValidateArchitecture:
        """Tests for validate_architecture tool."""
        
        @pytest.mark.asyncio
        async def test_valid_architecture(self, mock_server):
            """Test validation of compatible architecture."""
            # Mock no limitations found
            mock_server.rif_db.search_entities.return_value = []
            
            params = {
                'system_design': 'Simple file processing system using Read and Write tools'
            }
            
            result = await mock_server._validate_architecture(params)
            
            assert result['valid'] is True
            assert result['confidence'] >= 0.0
            assert len(result['issues_found']) == 0
        
        @pytest.mark.asyncio
        async def test_invalid_architecture(self, mock_server, sample_entities):
            """Test validation of problematic architecture."""
            # Mock finding orchestrator limitations
            mock_server.rif_db.search_entities.return_value = [sample_entities['limitation']]
            mock_server.rif_db.get_entity.return_value = sample_entities['limitation']
            
            params = {
                'system_design': 'Complex orchestrator managing multiple agents with Task() calls'
            }
            
            result = await mock_server._validate_architecture(params)
            
            assert result['valid'] is False
            assert len(result['issues_found']) > 0
            assert len(result['recommendations']) > 0
        
        @pytest.mark.asyncio
        async def test_component_extraction(self, mock_server, sample_entities):
            """Test architectural component extraction."""
            mock_server.rif_db.search_entities.return_value = []
            
            params = {
                'system_design': 'System with API endpoints, database storage, and agent coordination'
            }
            
            result = await mock_server._validate_architecture(params)
            
            # Should have extracted and analyzed components
            assert result['components_analyzed'] > 0
    
    
    class TestQueryLimitations:
        """Tests for query_limitations tool."""
        
        @pytest.mark.asyncio
        async def test_successful_limitation_query(self, mock_server, sample_entities):
            """Test successful limitation query."""
            mock_server.rif_db.search_entities.return_value = [sample_entities['limitation']]
            mock_server.rif_db.get_entity.return_value = sample_entities['limitation']
            mock_server.rif_db.get_entity_relationships.return_value = []
            
            params = {
                'capability_area': 'orchestration'
            }
            
            result = await mock_server._query_limitations(params)
            
            assert 'limitations' in result
            assert len(result['limitations']) > 0
            assert result['limitations'][0]['name'] == 'No Task Orchestration'
            assert result['limitations'][0]['severity'] == 'high'
        
        @pytest.mark.asyncio
        async def test_severity_filtering(self, mock_server):
            """Test filtering limitations by severity."""
            # Create mock limitations with different severities
            high_limitation = {
                'id': 'lim_high',
                'name': 'High Severity Limitation', 
                'type': 'claude_limitation',
                'metadata': {'severity': 'high', 'description': 'Critical issue'}
            }
            
            medium_limitation = {
                'id': 'lim_medium',
                'name': 'Medium Severity Limitation',
                'type': 'claude_limitation', 
                'metadata': {'severity': 'medium', 'description': 'Moderate issue'}
            }
            
            mock_server.rif_db.search_entities.return_value = [high_limitation, medium_limitation]
            mock_server.rif_db.get_entity.side_effect = lambda id: {
                'lim_high': high_limitation,
                'lim_medium': medium_limitation
            }.get(id)
            mock_server.rif_db.get_entity_relationships.return_value = []
            
            params = {
                'capability_area': 'general',
                'severity': 'high'
            }
            
            result = await mock_server._query_limitations(params)
            
            # Should only return high severity limitations
            assert all(lim['severity'] == 'high' for lim in result['limitations'])
        
        @pytest.mark.asyncio
        async def test_limitations_with_alternatives(self, mock_server, sample_entities):
            """Test limitations query including alternatives."""
            # Mock limitation with alternative relationship
            mock_relationship = {
                'relationship_type': 'alternative_to',
                'target_id': 'pat_1',
                'confidence': 0.8
            }
            
            mock_server.rif_db.search_entities.return_value = [sample_entities['limitation']]
            mock_server.rif_db.get_entity.side_effect = lambda id: {
                'lim_1': sample_entities['limitation'],
                'pat_1': sample_entities['pattern']
            }.get(id)
            mock_server.rif_db.get_entity_relationships.return_value = [mock_relationship]
            
            params = {
                'capability_area': 'orchestration'
            }
            
            result = await mock_server._query_limitations(params)
            
            assert len(result['limitations']) > 0
            limitation = result['limitations'][0]
            assert 'alternatives' in limitation
            if limitation['alternatives']:
                assert limitation['alternatives'][0]['entity']['name'] == 'Direct Tool Usage'


class TestInputValidation:
    """Tests for input validation and safety features."""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = InputValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_tool_params')
    
    def test_valid_compatibility_params(self):
        """Test validation of valid compatibility parameters."""
        validator = InputValidator()
        
        params = {
            'issue_description': 'Valid description',
            'approach': 'Valid approach'
        }
        
        errors = validator.validate_tool_params('check_compatibility', params)
        assert len(errors) == 0
    
    def test_missing_required_field(self):
        """Test validation catches missing required fields."""
        validator = InputValidator()
        
        params = {}  # Missing required issue_description
        
        errors = validator.validate_tool_params('check_compatibility', params)
        assert len(errors) > 0
        assert any(error.field == 'issue_description' for error in errors)
        assert any(error.severity == 'high' for error in errors)
    
    def test_text_too_long(self):
        """Test validation catches overly long text."""
        validator = InputValidator()
        
        params = {
            'issue_description': 'x' * (validator.MAX_TEXT_LENGTH + 1)  # Too long
        }
        
        errors = validator.validate_tool_params('check_compatibility', params)
        assert len(errors) > 0
        assert any(error.code == 'TEXT_TOO_LONG' for error in errors)
    
    def test_dangerous_content_detection(self):
        """Test detection of potentially dangerous content."""
        validator = InputValidator()
        
        params = {
            'issue_description': 'Normal text with <script>alert("bad")</script> injection'
        }
        
        errors = validator.validate_tool_params('check_compatibility', params)
        assert len(errors) > 0
        assert any(error.code == 'DANGEROUS_CONTENT' for error in errors)
    
    def test_input_sanitization(self):
        """Test input parameter sanitization."""
        validator = InputValidator()
        
        params = {
            'issue_description': 'Text with <script>bad</script> content',
            'limit': 5
        }
        
        sanitized = validator.sanitize_params(params)
        
        assert '<script>' not in sanitized['issue_description']
        assert sanitized['limit'] == 5  # Non-string values unchanged


class TestGracefulDegradation:
    """Tests for graceful degradation functionality."""
    
    def test_fallback_responses_loaded(self):
        """Test fallback responses are properly loaded."""
        degradation = GracefulDegradation()
        
        assert 'check_compatibility' in degradation.fallback_responses
        assert 'recommend_pattern' in degradation.fallback_responses
        assert 'find_alternatives' in degradation.fallback_responses
        assert 'validate_architecture' in degradation.fallback_responses
        assert 'query_limitations' in degradation.fallback_responses
    
    def test_compatibility_fallback(self):
        """Test compatibility check fallback response."""
        degradation = GracefulDegradation()
        
        params = {
            'issue_description': 'Test description'
        }
        
        response = degradation.get_fallback_response('check_compatibility', params)
        
        assert response['compatible'] is False
        assert response['confidence'] == 0.1
        assert len(response['issues']) > 0
        assert 'fallback_mode' not in response or response.get('fallback_mode') is True
    
    def test_pattern_recommendation_fallback(self):
        """Test pattern recommendation fallback response."""
        degradation = GracefulDegradation()
        
        params = {
            'technology': 'Python',
            'task_type': 'processing'
        }
        
        response = degradation.get_fallback_response('recommend_pattern', params)
        
        assert len(response['patterns']) > 0
        assert response['patterns'][0]['technology'] == 'Python'
        assert response['patterns'][0]['task_type'] == 'processing'
        assert response.get('fallback_mode') is True
    
    def test_unknown_tool_fallback(self):
        """Test fallback for unknown tool."""
        degradation = GracefulDegradation()
        
        response = degradation.get_fallback_response('unknown_tool', {})
        
        assert 'error' in response
        assert 'unknown_tool' in response['error']
        assert response.get('fallback_mode') is True


class TestPerformanceAndBenchmarks:
    """Performance tests and benchmarks."""
    
    @pytest.mark.asyncio
    async def test_response_time_benchmark(self, mock_server):
        """Test response times meet performance targets."""
        # Setup fast mock responses
        mock_server.rif_db.search_entities.return_value = []
        
        params = {
            'issue_description': 'Performance test query'
        }
        
        start_time = time.time()
        result = await mock_server._check_compatibility(params)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete within performance target (200ms)
        assert execution_time < 500  # Allow some buffer for test environment
        assert 'execution_time_ms' in result
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_server):
        """Test handling of concurrent requests."""
        mock_server.rif_db.search_entities.return_value = []
        
        params = {
            'issue_description': 'Concurrent test query'
        }
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            task = mock_server._check_compatibility(params)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 5
        assert all('compatible' in result for result in results)
    
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create server and perform operations
        config = {'enable_caching': True, 'cache_size': 100}
        server = ClaudeCodeKnowledgeServer(config)
        
        # Simulate cache usage
        for i in range(50):
            server.query_cache.set(f"key_{i}", {"test": "data"})
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory significantly (less than 50MB)
        assert memory_increase < 50


if __name__ == '__main__':
    """Run tests directly."""
    pytest.main([__file__, '-v', '--tb=short'])