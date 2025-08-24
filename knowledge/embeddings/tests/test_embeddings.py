"""
Tests for the embedding generation system.
"""

import unittest
import tempfile
import os
import numpy as np
from pathlib import Path

from ...extraction.entity_types import EntityType, CodeEntity, SourceLocation
from ..text_processor import EntityTextExtractor, ProcessedText
from ..embedding_generator import EmbeddingGenerator, LocalEmbeddingModel
from ..embedding_storage import EmbeddingStorage
from ..embedding_pipeline import EmbeddingPipeline


class TestTextProcessor(unittest.TestCase):
    """Test text processing for embeddings."""
    
    def setUp(self):
        self.extractor = EntityTextExtractor()
    
    def test_function_text_extraction(self):
        """Test text extraction from function entities."""
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="calculate_sum",
            file_path="/test.py",
            location=SourceLocation(line_start=1, line_end=5),
            metadata={
                'parameters': ['a', 'b'],
                'docstring': 'Calculate the sum of two numbers',
                'return_type': 'int',
                'language': 'python'
            }
        )
        
        processed = self.extractor.extract_text(entity)
        
        self.assertIsInstance(processed, ProcessedText)
        self.assertIn('function calculate_sum', processed.text)
        self.assertIn('parameters a b', processed.text)
        self.assertIn('purpose calculate sum two numbers', processed.text)
        self.assertIn('returns int', processed.text)
        self.assertIn('language python', processed.text)
        self.assertTrue(processed.content_hash)
        self.assertEqual(processed.metadata['entity_type'], 'function')
    
    def test_class_text_extraction(self):
        """Test text extraction from class entities."""
        entity = CodeEntity(
            type=EntityType.CLASS,
            name="Calculator",
            file_path="/test.py",
            location=SourceLocation(line_start=10, line_end=20),
            metadata={
                'methods': ['add', 'subtract', 'multiply'],
                'extends': 'BaseCalculator',
                'language': 'python'
            }
        )
        
        processed = self.extractor.extract_text(entity)
        
        self.assertIn('class Calculator', processed.text)
        self.assertIn('methods add subtract multiply', processed.text)
        self.assertIn('extends BaseCalculator', processed.text)
        self.assertEqual(processed.metadata['entity_type'], 'class')
    
    def test_module_text_extraction(self):
        """Test text extraction from module entities."""
        entity = CodeEntity(
            type=EntityType.MODULE,
            name="math_utils",
            file_path="/test.py",
            metadata={
                'imports': ['numpy', 'scipy', 'pandas'],
                'exports': ['calculate', 'process'],
                'language': 'python'
            }
        )
        
        processed = self.extractor.extract_text(entity)
        
        self.assertIn('module math_utils', processed.text)
        self.assertIn('imports numpy scipy pandas', processed.text)
        self.assertIn('exports calculate process', processed.text)
        self.assertEqual(processed.metadata['entity_type'], 'module')
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        raw_text = "This is a TEST with 123 numbers and special@chars!"
        cleaned = self.extractor.clean_text(raw_text)
        
        # Should be lowercase, normalized, and cleaned
        self.assertNotIn('@', cleaned)
        self.assertNotIn('!', cleaned)
        self.assertIn('NUM', cleaned)  # Numbers replaced with NUM
        self.assertTrue(cleaned.islower())
    
    def test_context_enhancement(self):
        """Test context enhancement with related entities."""
        base_entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="main_function",
            metadata={'language': 'python'}
        )
        
        context_entities = [
            CodeEntity(type=EntityType.CLASS, name="Helper"),
            CodeEntity(type=EntityType.FUNCTION, name="utility_func"),
            CodeEntity(type=EntityType.VARIABLE, name="config")
        ]
        
        processed = self.extractor.extract_text(base_entity)
        enhanced = self.extractor.enhance_with_context(processed, context_entities)
        
        self.assertIn('context', enhanced.text)
        self.assertIn('related', enhanced.text)
        self.assertTrue(enhanced.metadata.get('context_enhanced', False))
        self.assertEqual(enhanced.metadata.get('context_entity_count'), 3)


class TestLocalEmbeddingModel(unittest.TestCase):
    """Test the local embedding model."""
    
    def setUp(self):
        self.model = LocalEmbeddingModel(embedding_dim=64, max_features=100)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.embedding_dim, 64)
        self.assertEqual(self.model.max_features, 100)
        self.assertFalse(self.model.is_fitted)
    
    def test_hash_embedding_unfitted(self):
        """Test hash-based embedding when model is not fitted."""
        embedding = self.model.encode("test function add numbers")
        
        self.assertEqual(len(embedding), 64)
        self.assertEqual(embedding.dtype, np.float32)
        
        # Should be deterministic
        embedding2 = self.model.encode("test function add numbers")
        np.testing.assert_array_equal(embedding, embedding2)
        
        # Different text should give different embedding
        embedding3 = self.model.encode("different text content")
        self.assertFalse(np.array_equal(embedding, embedding3))
    
    def test_model_fitting(self):
        """Test model fitting with sample texts."""
        texts = [
            "function add numbers sum",
            "class calculator operations",
            "module utilities helper functions",
            "variable store data value",
            "function multiply numbers product"
        ]
        
        self.model.fit(texts)
        
        self.assertTrue(self.model.is_fitted)
        self.assertGreater(len(self.model.vocabulary), 0)
        self.assertGreater(len(self.model.idf_scores), 0)
        self.assertEqual(self.model.document_count, len(texts))
    
    def test_embedding_generation_fitted(self):
        """Test embedding generation with fitted model."""
        texts = [
            "function add numbers sum",
            "class calculator operations", 
            "module utilities helper functions"
        ]
        
        self.model.fit(texts)
        
        # Generate embeddings
        embedding1 = self.model.encode("function add numbers", {'entity_type': 'function'})
        embedding2 = self.model.encode("class calculator", {'entity_type': 'class'})
        
        self.assertEqual(len(embedding1), 64)
        self.assertEqual(len(embedding2), 64)
        
        # Should be unit vectors
        self.assertAlmostEqual(np.linalg.norm(embedding1), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(embedding2), 1.0, places=5)
        
        # Similar content should have high similarity
        embedding3 = self.model.encode("function sum numbers", {'entity_type': 'function'})
        similarity = np.dot(embedding1, embedding3)
        self.assertGreater(similarity, 0.5)  # Should be somewhat similar
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        self.assertEqual(info['model_type'], 'tfidf_local')
        self.assertEqual(info['embedding_dim'], 64)
        self.assertEqual(info['max_features'], 100)
        self.assertIn('feature_weights', info)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test the embedding generator."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = EmbeddingGenerator(embedding_dim=64)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_single_embedding_generation(self):
        """Test generating embedding for a single entity."""
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="test_function",
            metadata={
                'parameters': ['x', 'y'],
                'language': 'python'
            }
        )
        
        result = self.generator.generate_embedding(entity)
        
        self.assertEqual(result.entity_id, str(entity.id))
        self.assertEqual(len(result.embedding), 64)
        self.assertTrue(result.content_hash)
        self.assertIn('generation_time', result.metadata)
        self.assertGreater(result.generation_time, 0)
    
    def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        entities = [
            CodeEntity(type=EntityType.FUNCTION, name=f"func_{i}")
            for i in range(5)
        ]
        
        results = self.generator.generate_embeddings_batch(entities, batch_size=2)
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertEqual(len(result.embedding), 64)
            self.assertTrue(result.content_hash)
    
    def test_caching_behavior(self):
        """Test embedding caching."""
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="cached_function"
        )
        
        # Generate first time (cache miss)
        result1 = self.generator.generate_embedding(entity, use_cache=True)
        
        # Generate second time (should be cache hit)
        result2 = self.generator.generate_embedding(entity, use_cache=True)
        
        # Should be identical
        np.testing.assert_array_equal(result1.embedding, result2.embedding)
        self.assertEqual(result1.content_hash, result2.content_hash)
        
        # Check metrics
        metrics = self.generator.get_metrics()
        self.assertGreater(metrics['cache_hits'], 0)
    
    def test_model_fitting(self):
        """Test model fitting with entities."""
        entities = [
            CodeEntity(type=EntityType.FUNCTION, name="add_numbers",
                      metadata={'parameters': ['a', 'b'], 'language': 'python'}),
            CodeEntity(type=EntityType.CLASS, name="Calculator",
                      metadata={'methods': ['add', 'subtract'], 'language': 'python'}),
            CodeEntity(type=EntityType.MODULE, name="math_utils",
                      metadata={'imports': ['numpy'], 'language': 'python'}),
        ]
        
        # Fit model
        self.generator.fit_model(entities)
        
        # Model should now be fitted
        self.assertTrue(self.generator.model.is_fitted)
        
        # Generate embedding with fitted model
        result = self.generator.generate_embedding(entities[0])
        self.assertEqual(len(result.embedding), 64)
    
    def test_metrics_tracking(self):
        """Test metrics tracking."""
        initial_metrics = self.generator.get_metrics()
        self.assertEqual(initial_metrics['embeddings_generated'], 0)
        
        entity = CodeEntity(type=EntityType.FUNCTION, name="test")
        self.generator.generate_embedding(entity)
        
        updated_metrics = self.generator.get_metrics()
        self.assertEqual(updated_metrics['embeddings_generated'], 1)
        self.assertGreater(updated_metrics['total_generation_time'], 0)


class TestEmbeddingStorage(unittest.TestCase):
    """Test embedding storage functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_embeddings.duckdb")
        
        try:
            self.storage = EmbeddingStorage(self.db_path)
            self.storage_available = True
        except Exception:
            self.storage_available = False
    
    def tearDown(self):
        if hasattr(self, 'storage'):
            self.storage.close()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_embedding_schema_setup(self):
        """Test database schema setup for embeddings."""
        if not self.storage_available:
            self.skipTest("DuckDB not available")
        
        # Storage should initialize without errors
        self.assertIsNotNone(self.storage._get_connection())
    
    def test_embedding_storage_and_retrieval(self):
        """Test storing and retrieving embeddings."""
        if not self.storage_available:
            self.skipTest("DuckDB not available")
        
        # First, we need to create some entities in the database
        # This test assumes the entities table exists from the extraction system
        conn = self.storage._get_connection()
        
        # Create a test entity
        entity_id = "test-entity-123"
        conn.execute("""
            INSERT INTO entities (id, type, name, file_path, line_start, line_end)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [entity_id, "function", "test_func", "/test.py", 1, 5])
        
        # Create embedding result
        from ..embedding_generator import EmbeddingResult
        
        embedding = np.random.rand(64).astype(np.float32)
        embedding_result = EmbeddingResult(
            entity_id=entity_id,
            embedding=embedding,
            content_hash="test_hash_123",
            metadata={'model': 'test', 'generation_time': 0.1},
            generation_time=0.1
        )
        
        # Store embedding
        store_result = self.storage.store_embeddings([embedding_result])
        
        self.assertGreater(store_result['inserted'] + store_result['updated'], 0)
        
        # Retrieve embedding
        retrieved_embedding = self.storage.get_entity_embedding(entity_id)
        
        self.assertIsNotNone(retrieved_embedding)
        self.assertEqual(len(retrieved_embedding), 64)


class TestEmbeddingPipeline(unittest.TestCase):
    """Test the complete embedding pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_pipeline.duckdb")
        
        try:
            self.pipeline = EmbeddingPipeline(self.db_path, embedding_dim=64)
            self.pipeline_available = True
        except Exception:
            self.pipeline_available = False
    
    def tearDown(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.close()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        if not self.pipeline_available:
            self.skipTest("Pipeline dependencies not available")
        
        self.assertIsNotNone(self.pipeline.entity_storage)
        self.assertIsNotNone(self.pipeline.embedding_generator)
        self.assertIsNotNone(self.pipeline.embedding_storage)
        
        # Test metrics
        metrics = self.pipeline.get_pipeline_metrics()
        self.assertIn('pipeline', metrics)
        self.assertIn('embedding_generation', metrics)
    
    def test_model_fitting(self):
        """Test model fitting with sample entities."""
        if not self.pipeline_available:
            self.skipTest("Pipeline dependencies not available")
        
        # Create sample entities for fitting
        sample_entities = [
            CodeEntity(type=EntityType.FUNCTION, name="add_func"),
            CodeEntity(type=EntityType.CLASS, name="Calculator"),
            CodeEntity(type=EntityType.MODULE, name="utils")
        ]
        
        # Mock the training sample method
        original_method = self.pipeline._get_training_sample
        self.pipeline._get_training_sample = lambda size: sample_entities[:size]
        
        try:
            result = self.pipeline.fit_embedding_model(sample_size=3, save_model=False)
            
            self.assertTrue(result['success'])
            self.assertEqual(result['entities_used_for_fitting'], 3)
            self.assertIn('model_info', result)
        
        finally:
            # Restore original method
            self.pipeline._get_training_sample = original_method


if __name__ == '__main__':
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)