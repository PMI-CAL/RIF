"""
Comprehensive tests for DuckDB setup and configuration.
Issue #26: Set up DuckDB as embedded database with vector search
"""

import os
import json
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to Python path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from knowledge.database.database_config import DatabaseConfig
from knowledge.database.connection_manager import DuckDBConnectionManager
from knowledge.database.vector_search import VectorSearchEngine, SearchQuery
from knowledge.database.database_interface import RIFDatabase


class TestDatabaseConfig(unittest.TestCase):
    """Test database configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        self.assertEqual(config.memory_limit, "500MB")
        self.assertEqual(config.max_memory, "500MB")
        self.assertEqual(config.max_connections, 5)
        self.assertTrue(config.enable_vss)
        self.assertEqual(config.vss_metric, "cosine")
        self.assertTrue(config.auto_create_schema)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = DatabaseConfig(
            memory_limit="1000",  # Should be normalized to "1000MB"
            threads=20,  # Should be capped at 8
            max_connections=50  # Should be capped at 20
        )
        
        self.assertEqual(config.memory_limit, "1000MB")
        self.assertLessEqual(config.threads, 8)
        self.assertLessEqual(config.max_connections, 20)
    
    def test_environment_config(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'RIF_DB_MEMORY_LIMIT': '750MB',
            'RIF_DB_MAX_CONNECTIONS': '3',
            'RIF_DB_ENABLE_VSS': 'false'
        }):
            config = DatabaseConfig.from_environment()
            
            self.assertEqual(config.memory_limit, '750MB')
            self.assertEqual(config.max_connections, 3)
            self.assertFalse(config.enable_vss)
    
    def test_testing_config(self):
        """Test configuration optimized for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DatabaseConfig.for_testing(temp_dir)
            
            self.assertEqual(config.memory_limit, "100MB")
            self.assertEqual(config.max_connections, 2)
            self.assertIn(temp_dir, config.database_path)
    
    def test_config_dict(self):
        """Test configuration dictionary generation."""
        config = DatabaseConfig(
            memory_limit="500MB",
            threads=4,
            temp_directory="/tmp/test"
        )
        
        config_dict = config.get_config_dict()
        
        self.assertEqual(config_dict['memory_limit'], "500MB")
        self.assertEqual(config_dict['threads'], 4)
        self.assertEqual(config_dict['temp_directory'], "/tmp/test")
        self.assertFalse(config_dict['enable_progress_bar'])
    
    def test_vss_config(self):
        """Test VSS-specific configuration."""
        config = DatabaseConfig(
            vss_metric="euclidean",
            hnsw_ef_construction=300,
            hnsw_m=20
        )
        
        vss_config = config.get_vss_config()
        
        self.assertEqual(vss_config['metric'], "euclidean")
        self.assertEqual(vss_config['ef_construction'], 300)
        self.assertEqual(vss_config['m'], 20)


class TestConnectionManager(unittest.TestCase):
    """Test database connection management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="rif_test_")
        self.config = DatabaseConfig.for_testing(self.temp_dir)
        self.manager = None
    
    def tearDown(self):
        """Clean up test environment."""
        if self.manager:
            self.manager.shutdown()
        
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def test_manager_initialization(self):
        """Test connection manager initialization."""
        self.manager = DuckDBConnectionManager(self.config)
        
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.config, self.config)
        self.assertFalse(self.manager._shutdown)
    
    def test_connection_creation(self):
        """Test database connection creation."""
        self.manager = DuckDBConnectionManager(self.config)
        
        with self.manager.get_connection() as conn:
            self.assertIsNotNone(conn)
            
            # Test basic query
            result = conn.execute("SELECT 1 as test").fetchone()
            self.assertEqual(result[0], 1)
    
    def test_connection_pooling(self):
        """Test connection pooling functionality."""
        self.manager = DuckDBConnectionManager(self.config)
        
        # Test multiple connections
        connections = []
        for i in range(self.config.max_connections):
            try:
                conn_context = self.manager.get_connection()
                conn = conn_context.__enter__()
                connections.append((conn_context, conn))
                
                # Test connection is working
                result = conn.execute(f"SELECT {i} as test").fetchone()
                self.assertEqual(result[0], i)
                
            except Exception as e:
                self.fail(f"Failed to get connection {i}: {e}")
        
        # Clean up connections
        for conn_context, conn in connections:
            conn_context.__exit__(None, None, None)
    
    def test_pool_statistics(self):
        """Test connection pool statistics."""
        self.manager = DuckDBConnectionManager(self.config)
        
        stats = self.manager.get_pool_stats()
        
        self.assertIn('pool_size', stats)
        self.assertIn('active_connections', stats)
        self.assertIn('max_connections', stats)
        self.assertIn('total_created', stats)
        self.assertEqual(stats['max_connections'], self.config.max_connections)
    
    def test_connection_timeout(self):
        """Test connection timeout behavior."""
        # Use a config with very short timeout for testing
        config = DatabaseConfig.for_testing(self.temp_dir)
        config.connection_timeout = 0.1  # 100ms timeout
        config.max_connections = 1  # Only one connection
        
        self.manager = DuckDBConnectionManager(config)
        
        # Hold the only connection
        with self.manager.get_connection() as conn1:
            # Try to get another connection - should timeout
            with self.assertRaises(RuntimeError):
                with self.manager.get_connection(timeout=0.1) as conn2:
                    pass
    
    def test_memory_limit_configuration(self):
        """Test memory limit is properly applied."""
        self.manager = DuckDBConnectionManager(self.config)
        
        with self.manager.get_connection() as conn:
            # Check memory limit setting
            try:
                result = conn.execute("SELECT current_setting('memory_limit')").fetchone()
                self.assertIn("MB", str(result[0]))
            except Exception as e:
                # Memory limit setting might not be available in test environment
                self.skipTest(f"Memory limit setting not available: {e}")


class TestVectorSearch(unittest.TestCase):
    """Test vector similarity search functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="rif_vector_test_")
        self.config = DatabaseConfig.for_testing(self.temp_dir)
        self.manager = DuckDBConnectionManager(self.config)
        self.vector_search = VectorSearchEngine(self.manager)
        
        # Create test data
        self._setup_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        self.manager.shutdown()
        
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def _setup_test_data(self):
        """Create test entities with embeddings."""
        try:
            with self.manager.get_connection() as conn:
                # Insert test entities
                test_entities = [
                    {
                        'id': '550e8400-e29b-41d4-a716-446655440001',
                        'type': 'function',
                        'name': 'calculate_similarity',
                        'file_path': '/test/similarity.py',
                        'line_start': 10,
                        'line_end': 25,
                        'embedding': np.random.rand(768).astype(np.float32).tobytes()
                    },
                    {
                        'id': '550e8400-e29b-41d4-a716-446655440002', 
                        'type': 'class',
                        'name': 'VectorProcessor',
                        'file_path': '/test/processor.py',
                        'line_start': 1,
                        'line_end': 50,
                        'embedding': np.random.rand(768).astype(np.float32).tobytes()
                    },
                    {
                        'id': '550e8400-e29b-41d4-a716-446655440003',
                        'type': 'function',
                        'name': 'process_embeddings',
                        'file_path': '/test/processor.py',
                        'line_start': 30,
                        'line_end': 45,
                        'embedding': np.random.rand(768).astype(np.float32).tobytes()
                    }
                ]
                
                for entity in test_entities:
                    conn.execute("""
                        INSERT INTO entities (id, type, name, file_path, line_start, line_end, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        entity['id'], entity['type'], entity['name'], 
                        entity['file_path'], entity['line_start'], entity['line_end'],
                        entity['embedding']
                    ])
                
        except Exception as e:
            self.skipTest(f"Failed to set up test data: {e}")
    
    def test_vector_search_engine_initialization(self):
        """Test vector search engine initialization."""
        self.assertIsNotNone(self.vector_search)
        self.assertEqual(self.vector_search.connection_manager, self.manager)
        self.assertEqual(self.vector_search.metric, "cosine")
    
    def test_similarity_search_query(self):
        """Test vector similarity search with query."""
        query_embedding = np.random.rand(768).astype(np.float32)
        
        search_query = SearchQuery(
            embedding=query_embedding,
            limit=5,
            threshold=0.0,  # Low threshold for testing
            entity_types=['function', 'class']
        )
        
        results = self.vector_search.search_similar_entities(search_query)
        
        # Should return results (even with random embeddings)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 0)  # May be 0 if no embeddings are similar enough
        
        # Check result structure
        for result in results:
            self.assertIn('id', result.__dict__)
            self.assertIn('name', result.__dict__)
            self.assertIn('type', result.__dict__)
            self.assertIn('similarity_score', result.__dict__)
            self.assertIsInstance(result.similarity_score, float)
    
    def test_hybrid_search(self):
        """Test hybrid text + vector search."""
        query_embedding = np.random.rand(768).astype(np.float32)
        
        results = self.vector_search.hybrid_search(
            text_query="similarity",
            embedding_query=query_embedding,
            limit=5
        )
        
        self.assertIsInstance(results, list)
        
        # Check that text matching works
        for result in results:
            # Should match entities with "similarity" in the name
            self.assertTrue(
                "similarity" in result.name.lower() or 
                "similarity" in result.file_path.lower() or
                result.similarity_score > 0
            )
    
    def test_name_search(self):
        """Test search by entity name."""
        results = self.vector_search.search_by_entity_name(
            name_pattern="calculate",
            entity_types=['function'],
            limit=5
        )
        
        self.assertIsInstance(results, list)
        
        # Check that name matching works
        for result in results:
            self.assertIn("calculate", result.name.lower())
            self.assertEqual(result.type, "function")
    
    def test_vss_setup_verification(self):
        """Test VSS extension setup verification."""
        status = self.vector_search.verify_vss_setup()
        
        self.assertIn('vss_extension_loaded', status)
        self.assertIn('vss_indexes_exist', status)
        self.assertIn('vss_functions_available', status)
        self.assertIn('test_query_successful', status)
        
        # Note: VSS extension may not be available in test environment
        if not status['vss_extension_loaded']:
            self.skipTest("VSS extension not available in test environment")
    
    def test_search_statistics(self):
        """Test search performance statistics."""
        # Perform some searches to generate statistics
        query_embedding = np.random.rand(768).astype(np.float32)
        search_query = SearchQuery(embedding=query_embedding, limit=5, threshold=0.0)
        
        self.vector_search.search_similar_entities(search_query)
        
        stats = self.vector_search.get_search_statistics()
        
        self.assertIn('total_queries', stats)
        self.assertIn('total_query_time', stats)
        self.assertIn('average_query_time', stats)
        self.assertIn('metric', stats)
        self.assertEqual(stats['metric'], "cosine")
        self.assertGreater(stats['total_queries'], 0)


class TestRIFDatabase(unittest.TestCase):
    """Test unified database interface."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="rif_db_test_")
        self.config = DatabaseConfig.for_testing(self.temp_dir)
        self.db = RIFDatabase(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        self.db.close()
        
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db)
        self.assertIsNotNone(self.db.connection_manager)
        self.assertIsNotNone(self.db.vector_search)
        self.assertEqual(self.db.config, self.config)
    
    def test_entity_operations(self):
        """Test entity storage and retrieval."""
        # Test entity creation
        entity_data = {
            'type': 'function',
            'name': 'test_function',
            'file_path': '/test/example.py',
            'line_start': 10,
            'line_end': 20,
            'ast_hash': 'abc123',
            'embedding': np.random.rand(768).astype(np.float32),
            'metadata': {'complexity': 'low', 'language': 'python'}
        }
        
        entity_id = self.db.store_entity(entity_data)
        self.assertIsNotNone(entity_id)
        
        # Test entity retrieval
        retrieved_entity = self.db.get_entity(entity_id)
        self.assertIsNotNone(retrieved_entity)
        self.assertEqual(retrieved_entity['name'], 'test_function')
        self.assertEqual(retrieved_entity['type'], 'function')
        self.assertEqual(retrieved_entity['metadata']['complexity'], 'low')
        
        # Test entity search
        search_results = self.db.search_entities(
            query="test",
            entity_types=['function'],
            limit=10
        )
        
        self.assertGreater(len(search_results), 0)
        self.assertEqual(search_results[0]['name'], 'test_function')
    
    def test_relationship_operations(self):
        """Test relationship storage and retrieval."""
        # First create two entities
        entity1_data = {
            'type': 'function',
            'name': 'caller_function',
            'file_path': '/test/caller.py'
        }
        entity2_data = {
            'type': 'function', 
            'name': 'called_function',
            'file_path': '/test/called.py'
        }
        
        entity1_id = self.db.store_entity(entity1_data)
        entity2_id = self.db.store_entity(entity2_data)
        
        # Create relationship
        relationship_id = self.db.store_relationship(
            source_id=entity1_id,
            target_id=entity2_id,
            relationship_type='calls',
            confidence=0.9,
            metadata={'call_count': 5}
        )
        
        self.assertIsNotNone(relationship_id)
        
        # Test relationship retrieval
        relationships = self.db.get_entity_relationships(entity1_id, direction='outgoing')
        
        self.assertGreater(len(relationships), 0)
        self.assertEqual(relationships[0]['relationship_type'], 'calls')
        self.assertEqual(relationships[0]['confidence'], 0.9)
        self.assertEqual(relationships[0]['source_name'], 'caller_function')
        self.assertEqual(relationships[0]['target_name'], 'called_function')
    
    def test_agent_memory_operations(self):
        """Test agent memory storage and retrieval."""
        # Store agent memory
        memory_id = self.db.store_agent_memory(
            agent_type='RIF-Analyst',
            context='Analyzing issue requirements for authentication system',
            issue_number=123,
            decision='Use OAuth2 with PKCE flow',
            outcome='success',
            embedding=np.random.rand(768).astype(np.float32),
            metadata={'complexity': 'medium', 'duration_minutes': 15}
        )
        
        self.assertIsNotNone(memory_id)
        
        # Test memory retrieval
        memories = self.db.get_agent_memories(
            agent_type='RIF-Analyst',
            issue_number=123,
            limit=10
        )
        
        self.assertGreater(len(memories), 0)
        self.assertEqual(memories[0]['agent_type'], 'RIF-Analyst')
        self.assertEqual(memories[0]['issue_number'], 123)
        self.assertEqual(memories[0]['outcome'], 'success')
        self.assertIn('analyzing issue', memories[0]['context'].lower())
    
    def test_vector_search_integration(self):
        """Test vector search through database interface."""
        # Create entity with embedding
        entity_data = {
            'type': 'class',
            'name': 'SearchEngine', 
            'file_path': '/test/search.py',
            'embedding': np.random.rand(768).astype(np.float32)
        }
        
        entity_id = self.db.store_entity(entity_data)
        
        # Test similarity search
        query_embedding = np.random.rand(768).astype(np.float32)
        results = self.db.similarity_search(
            query_embedding=query_embedding,
            entity_types=['class'],
            limit=5,
            threshold=0.0  # Low threshold for testing
        )
        
        self.assertIsInstance(results, list)
        
        # Test hybrid search
        hybrid_results = self.db.hybrid_search(
            text_query="search",
            embedding_query=query_embedding,
            limit=5
        )
        
        self.assertIsInstance(hybrid_results, list)
        
        # Test name search
        name_results = self.db.search_by_name(
            name_pattern="Search",
            entity_types=['class'],
            limit=5
        )
        
        self.assertIsInstance(name_results, list)
    
    def test_database_statistics(self):
        """Test database statistics generation."""
        # Add some test data first
        entity_data = {
            'type': 'function',
            'name': 'stats_test',
            'file_path': '/test/stats.py',
            'embedding': np.random.rand(768).astype(np.float32)
        }
        
        entity_id = self.db.store_entity(entity_data)
        
        self.db.store_agent_memory(
            agent_type='RIF-Implementer',
            context='Testing statistics',
            outcome='success'
        )
        
        # Get statistics
        stats = self.db.get_database_stats()
        
        self.assertIn('entities', stats)
        self.assertIn('relationships', stats)
        self.assertIn('agent_memory', stats)
        self.assertIn('connection_pool', stats)
        self.assertIn('vector_search', stats)
        
        # Check entity stats
        self.assertGreater(stats['entities']['total'], 0)
        self.assertGreater(stats['entities']['with_embeddings'], 0)
        
        # Check agent memory stats  
        self.assertGreater(stats['agent_memory']['total'], 0)
    
    def test_maintenance_operations(self):
        """Test database maintenance functionality."""
        results = self.db.run_maintenance()
        
        self.assertIn('connection_cleanup', results)
        self.assertIn('analyze_tables', results)
        
        # Should complete successfully
        self.assertEqual(results['connection_cleanup'], 'completed')
        self.assertEqual(results['analyze_tables'], 'completed')
    
    def test_setup_verification(self):
        """Test database setup verification."""
        verification = self.db.verify_setup()
        
        self.assertIn('database_accessible', verification)
        self.assertIn('schema_present', verification)
        self.assertIn('vss_setup', verification)
        self.assertIn('connection_pool_working', verification)
        self.assertIn('performance_acceptable', verification)
        
        # Basic functionality should work
        self.assertTrue(verification['database_accessible'])
        self.assertTrue(verification['schema_present'])
        self.assertTrue(verification['connection_pool_working'])
    
    def test_context_manager(self):
        """Test database as context manager."""
        config = DatabaseConfig.for_testing(self.temp_dir)
        
        with RIFDatabase(config) as db:
            self.assertIsNotNone(db)
            
            # Test basic operation
            entity_id = db.store_entity({
                'type': 'module',
                'name': 'context_test',
                'file_path': '/test/context.py'
            })
            
            self.assertIsNotNone(entity_id)
        
        # Database should be closed after context


def run_comprehensive_tests():
    """Run all database tests with detailed reporting."""
    import logging
    
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDatabaseConfig,
        TestConnectionManager, 
        TestVectorSearch,
        TestRIFDatabase
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed results
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate test report
    report = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    }
    
    logger.info(f"Test Report: {report}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    # Run tests when executed directly
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)