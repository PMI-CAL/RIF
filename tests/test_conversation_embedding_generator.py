#!/usr/bin/env python3
"""
Test suite for Conversation Embedding Generator.

Tests TF-IDF based embedding generation including:
- Text preprocessing and tokenization
- TF-IDF model training
- Embedding generation (single and batch)
- Cache management
- Integration with conversation storage
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from knowledge.conversations.embedding_generator import (
    ConversationEmbeddingGenerator,
    create_embedding_generator,
    setup_embeddings_for_capture_engine
)


class TestConversationEmbeddingGenerator(unittest.TestCase):
    """Test the main embedding generator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConversationEmbeddingGenerator(
            vector_dim=128,  # Smaller for testing
            batch_size=10,
            cache_dir=self.temp_dir,
            max_features=100
        )
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of embedding generator"""
        self.assertEqual(self.generator.vector_dim, 128)
        self.assertEqual(self.generator.batch_size, 10)
        self.assertFalse(self.generator.is_trained)
        self.assertEqual(len(self.generator.training_texts), 0)
        self.assertEqual(len(self.generator.embedding_cache), 0)
    
    def test_tokenize_text(self):
        """Test text tokenization with cleaning and stemming"""
        text = "This is a sample text for testing tokenization!"
        tokens = self.generator._tokenize_text(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        # Should contain stemmed versions of words
        self.assertTrue(all(isinstance(token, str) for token in tokens))
        # Should not contain punctuation or very short words
        self.assertTrue(all(len(token) > 2 for token in tokens))
    
    def test_add_training_texts(self):
        """Test adding training texts to corpus"""
        texts = [
            "This is the first training text",
            "Here is another text for training",
            "A third text to build the corpus"
        ]
        
        initial_count = len(self.generator.training_texts)
        self.generator.add_training_texts(texts)
        
        self.assertEqual(len(self.generator.training_texts), initial_count + 3)
        
        # Test filtering of short/empty texts
        invalid_texts = ["", "hi", "   ", None]
        self.generator.add_training_texts(invalid_texts)
        
        # Should not add invalid texts
        self.assertEqual(len(self.generator.training_texts), initial_count + 3)
    
    def test_train_embeddings_model(self):
        """Test training the embedding model"""
        # Add sufficient training texts
        training_texts = [
            "Machine learning algorithms for natural language processing",
            "Deep learning models require large amounts of training data",
            "Neural networks can learn complex patterns from text",
            "Natural language understanding involves semantic analysis",
            "Text classification uses supervised learning techniques",
            "Information retrieval systems use vector space models",
            "Semantic similarity measures text relatedness",
            "Word embeddings capture semantic relationships",
            "Document classification requires feature extraction",
            "Text preprocessing includes tokenization and stemming"
        ]
        
        self.generator.add_training_texts(training_texts)
        
        # Train the model
        success = self.generator.train_embeddings_model()
        
        self.assertTrue(success)
        self.assertTrue(self.generator.is_trained)
    
    def test_generate_embedding_single(self):
        """Test generating embedding for single text"""
        # First train the model
        self._train_test_model()
        
        test_text = "This is a test text for embedding generation"
        embedding = self.generator.generate_embedding(test_text)
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 128)  # Target dimension
        self.assertTrue(all(isinstance(x, float) for x in embedding))
    
    def test_generate_embedding_batch(self):
        """Test batch embedding generation"""
        # Train the model first
        self._train_test_model()
        
        test_texts = [
            "First test text for batch processing",
            "Second text in the batch",
            "Third text for testing",
            ""  # Empty text should return None
        ]
        
        embeddings = self.generator.generate_embeddings_batch(test_texts)
        
        self.assertEqual(len(embeddings), 4)
        self.assertIsNotNone(embeddings[0])
        self.assertIsNotNone(embeddings[1])
        self.assertIsNotNone(embeddings[2])
        self.assertIsNone(embeddings[3])  # Empty text
        
        # Check dimensions
        for embedding in embeddings[:3]:
            self.assertEqual(len(embedding), 128)
    
    def test_embedding_caching(self):
        """Test embedding caching functionality"""
        self._train_test_model()
        
        test_text = "This text will be cached"
        
        # Generate embedding first time
        embedding1 = self.generator.generate_embedding(test_text)
        initial_cache_misses = self.generator.generation_stats['cache_misses']
        
        # Generate same embedding second time (should be cached)
        embedding2 = self.generator.generate_embedding(test_text)
        cache_hits = self.generator.generation_stats['cache_hits']
        
        self.assertEqual(embedding1, embedding2)
        self.assertGreater(cache_hits, 0)
    
    def test_adjust_embedding_dimension(self):
        """Test embedding dimension adjustment"""
        # Test truncation
        long_embedding = [0.1] * 200
        adjusted = self.generator._adjust_embedding_dimension(long_embedding)
        self.assertEqual(len(adjusted), 128)
        
        # Test padding
        short_embedding = [0.1] * 50
        adjusted = self.generator._adjust_embedding_dimension(short_embedding)
        self.assertEqual(len(adjusted), 128)
        # Padded with zeros
        self.assertEqual(adjusted[50:], [0.0] * 78)
        
        # Test exact match
        exact_embedding = [0.1] * 128
        adjusted = self.generator._adjust_embedding_dimension(exact_embedding)
        self.assertEqual(adjusted, exact_embedding)
    
    def test_create_searchable_text(self):
        """Test searchable text creation from event data"""
        event_type = "tool_use"
        event_data = {
            'tool_name': 'Read',
            'description': 'Reading file contents',
            'result': 'Successfully read file',
            'status': 'completed',
            'agent_type': 'rif-implementer'
        }
        
        searchable_text = self.generator._create_searchable_text(event_type, event_data)
        
        self.assertIn('Event: tool_use', searchable_text)
        self.assertIn('Tool: Read', searchable_text)
        self.assertIn('description: Reading file contents', searchable_text)
        self.assertIn('Agent: rif-implementer', searchable_text)
    
    def test_get_generation_stats(self):
        """Test generation statistics"""
        stats = self.generator.get_generation_stats()
        
        expected_keys = [
            'embeddings_generated', 'cache_hits', 'cache_misses',
            'avg_generation_time_ms', 'batch_operations',
            'is_trained', 'training_text_count', 'cache_size',
            'cache_hit_rate', 'vector_dimension'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['vector_dimension'], 128)
        self.assertFalse(stats['is_trained'])  # Not trained in this test
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add something to cache
        self.generator.embedding_cache['test'] = [0.1] * 128
        self.assertEqual(len(self.generator.embedding_cache), 1)
        
        self.generator.clear_cache()
        self.assertEqual(len(self.generator.embedding_cache), 0)
    
    def test_model_persistence(self):
        """Test saving and loading trained models"""
        # Train model
        self._train_test_model()
        
        # Save model
        self.generator._save_trained_model()
        
        # Create new generator and load model
        new_generator = ConversationEmbeddingGenerator(
            vector_dim=128,
            cache_dir=self.temp_dir
        )
        
        success = new_generator._load_trained_model()
        self.assertTrue(success)
        self.assertTrue(new_generator.is_trained)
    
    def _train_test_model(self):
        """Helper method to train a test model"""
        training_texts = [
            "Artificial intelligence and machine learning algorithms",
            "Natural language processing for text analysis",
            "Deep learning neural networks and training",
            "Computer vision and image recognition systems",
            "Data science and statistical analysis methods",
            "Software engineering and development practices",
            "Database systems and query optimization",
            "Web development and user interface design",
            "Mobile application development frameworks",
            "Cloud computing and distributed systems"
        ]
        
        self.generator.add_training_texts(training_texts)
        return self.generator.train_embeddings_model()


class TestStorageIntegration(unittest.TestCase):
    """Test integration with conversation storage"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConversationEmbeddingGenerator(
            vector_dim=64,  # Small for testing
            cache_dir=self.temp_dir,
            max_features=50
        )
        
        # Mock storage backend
        self.mock_storage = Mock()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_auto_train_from_storage(self):
        """Test auto-training from storage backend"""
        # Mock storage response
        mock_rows = [
            ('tool_use', json.dumps({'description': 'Reading file contents', 'tool_name': 'Read'})),
            ('decision', json.dumps({'decision_point': 'Choose implementation approach', 'rationale': 'Best practice'})),
            ('completion', json.dumps({'summary': 'Task completed successfully', 'result': 'All tests passed'}))
        ] * 40  # Repeat to get enough training data
        
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        self.mock_storage.connection.execute.return_value = mock_result
        
        # Test auto-training
        success = self.generator.auto_train_from_storage(self.mock_storage, min_texts=100)
        
        self.assertTrue(success)
        self.assertTrue(self.generator.is_trained)
        self.assertGreater(len(self.generator.training_texts), 100)
    
    def test_update_conversation_embeddings(self):
        """Test updating embeddings for conversation events"""
        # Train model first
        self._train_simple_model()
        
        # Mock conversation events
        mock_events = [
            {
                'event_id': 'event1',
                'event_type': 'tool_use',
                'event_data': json.dumps({'description': 'Reading file', 'tool_name': 'Read'})
            },
            {
                'event_id': 'event2', 
                'event_type': 'decision',
                'event_data': json.dumps({'decision_point': 'Implementation choice'})
            }
        ]
        
        self.mock_storage.get_conversation_events.return_value = mock_events
        self.mock_storage.connection.execute = Mock()  # Mock database updates
        
        # Test embedding update
        result = self.generator.update_conversation_embeddings('conv123', self.mock_storage)
        
        self.assertIn('updated', result)
        self.assertGreater(result['updated'], 0)
        # Should have called database update for each event
        self.assertEqual(self.mock_storage.connection.execute.call_count, 2)
    
    def _train_simple_model(self):
        """Helper to train a simple model for testing"""
        texts = [f"Sample training text number {i}" for i in range(20)]
        self.generator.add_training_texts(texts)
        return self.generator.train_embeddings_model()


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for easy integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_embedding_generator(self):
        """Test factory function"""
        generator = create_embedding_generator(
            vector_dim=256,
            cache_dir=self.temp_dir
        )
        
        self.assertIsInstance(generator, ConversationEmbeddingGenerator)
        self.assertEqual(generator.vector_dim, 256)
    
    def test_setup_embeddings_for_capture_engine(self):
        """Test setting up embeddings for capture engine"""
        # Mock capture engine and storage
        mock_capture_engine = Mock()
        mock_storage = Mock()
        
        # Mock auto-training response
        mock_rows = [('test', json.dumps({'description': 'test'}))] * 150
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_storage.connection.execute.return_value = mock_result
        
        # Test setup
        success = setup_embeddings_for_capture_engine(
            mock_capture_engine, 
            mock_storage, 
            auto_train=True
        )
        
        # Should call set_embedding_generator on capture engine
        self.assertTrue(mock_capture_engine.set_embedding_generator.called)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConversationEmbeddingGenerator(
            vector_dim=384,
            batch_size=50,
            cache_dir=self.temp_dir
        )
        
        # Train with substantial corpus
        training_texts = []
        topics = [
            "machine learning and artificial intelligence",
            "software engineering and development practices", 
            "data science and statistical analysis",
            "web development and user interfaces",
            "database systems and optimization",
            "cloud computing and distributed systems",
            "mobile application development",
            "cybersecurity and information protection",
            "computer networks and protocols",
            "algorithm design and analysis"
        ]
        
        for i in range(50):  # Generate 500 training texts
            for topic in topics:
                text = f"Advanced research in {topic} with practical applications and theoretical foundations number {i}"
                training_texts.append(text)
        
        self.generator.add_training_texts(training_texts)
        self.generator.train_embeddings_model()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_single_embedding_performance(self):
        """Test single embedding generation performance"""
        import time
        
        test_text = "Performance testing for single embedding generation with substantial text content"
        
        start_time = time.time()
        embedding = self.generator.generate_embedding(test_text)
        generation_time = (time.time() - start_time) * 1000
        
        self.assertIsNotNone(embedding)
        self.assertLess(generation_time, 100)  # Should be under 100ms
    
    def test_batch_embedding_performance(self):
        """Test batch embedding generation performance"""
        import time
        
        # Generate test texts
        test_texts = [
            f"Batch performance testing text number {i} with substantial content for realistic testing"
            for i in range(100)
        ]
        
        start_time = time.time()
        embeddings = self.generator.generate_embeddings_batch(test_texts)
        total_time = time.time() - start_time
        
        valid_embeddings = [e for e in embeddings if e is not None]
        self.assertEqual(len(valid_embeddings), 100)
        
        # Should process at least 50 embeddings per second
        embeddings_per_second = len(valid_embeddings) / total_time
        self.assertGreater(embeddings_per_second, 50)
    
    def test_cache_effectiveness(self):
        """Test embedding cache effectiveness"""
        test_texts = [
            "First test text for caching",
            "Second test text for caching",
            "First test text for caching",  # Repeat
            "Third test text for caching",
            "Second test text for caching",  # Repeat
            "First test text for caching"   # Repeat again
        ]
        
        embeddings = self.generator.generate_embeddings_batch(test_texts)
        stats = self.generator.get_generation_stats()
        
        # Should have some cache hits
        self.assertGreater(stats['cache_hits'], 0)
        self.assertGreater(stats['cache_hit_rate'], 0.3)  # At least 30% hit rate


if __name__ == '__main__':
    # Set up test environment
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversation Embedding Generator Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ EXCELLENT: Conversation embedding system is working correctly!")
    elif success_rate >= 80:
        print("✅ GOOD: Conversation embedding system is mostly functional")
    elif success_rate >= 70:
        print("⚠️  WARNING: Some issues detected in conversation embedding system")
    else:
        print("❌ CRITICAL: Major issues in conversation embedding system")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)