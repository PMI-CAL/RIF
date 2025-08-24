#!/usr/bin/env python3
"""
Simple test suite for Conversation Embedding Generator core functionality.
Tests the basic structure and interfaces without heavy ML dependencies.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class MockTfidfVectorizer:
    """Mock TF-IDF vectorizer for testing"""
    def __init__(self, **kwargs):
        self.max_features = kwargs.get('max_features', 1000)
        self.is_fitted = False
        
    def fit(self, texts):
        self.is_fitted = True
        return self
        
    def transform(self, texts):
        # Return mock vectors
        import numpy as np
        return np.random.rand(len(texts), min(100, self.max_features))
        
    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

class MockTruncatedSVD:
    """Mock SVD reducer for testing"""
    def __init__(self, **kwargs):
        self.n_components = kwargs.get('n_components', 100)
        self.n_components_ = self.n_components
        
    def fit(self, X):
        return self
        
    def transform(self, X):
        import numpy as np
        return np.random.rand(X.shape[0], self.n_components)
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MockStandardScaler:
    """Mock standard scaler for testing"""
    def fit(self, X):
        return self
        
    def transform(self, X):
        return X
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MockPipeline:
    """Mock pipeline for testing"""
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}
        
    def fit(self, X):
        result = X
        for name, step in self.steps:
            if hasattr(step, 'fit'):
                step.fit(result)
            if hasattr(step, 'transform'):
                result = step.transform(result)
        return self
        
    def transform(self, X):
        result = X
        for name, step in self.steps:
            if hasattr(step, 'transform'):
                result = step.transform(result)
        return result
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)


# Patch the imports to use mock classes
import sys
from unittest.mock import patch, MagicMock

# Mock the sklearn imports
sklearn_mock = MagicMock()
sklearn_mock.feature_extraction.text.TfidfVectorizer = MockTfidfVectorizer
sklearn_mock.decomposition.TruncatedSVD = MockTruncatedSVD
sklearn_mock.preprocessing.StandardScaler = MockStandardScaler
sklearn_mock.pipeline.Pipeline = MockPipeline

sys.modules['sklearn'] = sklearn_mock
sys.modules['sklearn.feature_extraction'] = sklearn_mock.feature_extraction
sys.modules['sklearn.feature_extraction.text'] = sklearn_mock.feature_extraction.text
sys.modules['sklearn.decomposition'] = sklearn_mock.decomposition
sys.modules['sklearn.preprocessing'] = sklearn_mock.preprocessing
sys.modules['sklearn.pipeline'] = sklearn_mock.pipeline

# Mock NLTK
nltk_mock = MagicMock()
nltk_mock.download = MagicMock()
nltk_mock.corpus.stopwords = MagicMock()
nltk_mock.tokenize.word_tokenize = lambda text: text.lower().split()
nltk_mock.stem.PorterStemmer.return_value.stem = lambda word: word

sys.modules['nltk'] = nltk_mock
sys.modules['nltk.corpus'] = nltk_mock.corpus
sys.modules['nltk.tokenize'] = nltk_mock.tokenize
sys.modules['nltk.stem'] = nltk_mock.stem

# Mock numpy
numpy_mock = MagicMock()
numpy_mock.random.rand = lambda *args: [[0.1] * args[1] for _ in range(args[0])]
numpy_mock.mean = lambda x: sum(x) / len(x) if x else 0

sys.modules['numpy'] = numpy_mock
sys.modules['np'] = numpy_mock

# Now import the actual module
from knowledge.conversations.embedding_generator import (
    ConversationEmbeddingGenerator,
    create_embedding_generator
)


class TestEmbeddingGeneratorStructure(unittest.TestCase):
    """Test the structure and basic functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConversationEmbeddingGenerator(
            vector_dim=128,
            batch_size=10,
            cache_dir=self.temp_dir,
            max_features=100
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.generator.vector_dim, 128)
        self.assertEqual(self.generator.batch_size, 10)
        self.assertFalse(self.generator.is_trained)
        self.assertEqual(len(self.generator.training_texts), 0)
    
    def test_add_training_texts(self):
        """Test adding training texts"""
        texts = ["First text", "Second text", "Third text"]
        self.generator.add_training_texts(texts)
        
        self.assertEqual(len(self.generator.training_texts), 3)
    
    def test_tokenize_text(self):
        """Test text tokenization"""
        text = "This is a sample text for testing!"
        tokens = self.generator._tokenize_text(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_text_hash(self):
        """Test text hashing for caching"""
        text = "Sample text for hashing"
        hash1 = self.generator._get_text_hash(text)
        hash2 = self.generator._get_text_hash(text)
        
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertGreater(len(hash1), 0)
    
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
        
        # Test exact match
        exact_embedding = [0.1] * 128
        adjusted = self.generator._adjust_embedding_dimension(exact_embedding)
        self.assertEqual(adjusted, exact_embedding)
    
    def test_create_searchable_text(self):
        """Test searchable text creation"""
        event_type = "tool_use"
        event_data = {
            'tool_name': 'Read',
            'description': 'Reading file contents',
            'result': 'Successfully read file',
            'status': 'completed'
        }
        
        searchable_text = self.generator._create_searchable_text(event_type, event_data)
        
        self.assertIn('Event: tool_use', searchable_text)
        self.assertIn('Tool: Read', searchable_text)
        self.assertIn('description: Reading file contents', searchable_text)
    
    def test_train_embeddings_model(self):
        """Test model training process"""
        # Add training texts
        texts = [f"Training text number {i}" for i in range(20)]
        self.generator.add_training_texts(texts)
        
        # Attempt to train
        success = self.generator.train_embeddings_model()
        
        # Should succeed with mocked components
        self.assertTrue(success)
        self.assertTrue(self.generator.is_trained)
    
    def test_generate_embedding(self):
        """Test embedding generation"""
        # Train first
        texts = [f"Training text {i}" for i in range(10)]
        self.generator.add_training_texts(texts)
        self.generator.train_embeddings_model()
        
        # Generate embedding
        test_text = "Test text for embedding"
        embedding = self.generator.generate_embedding(test_text)
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 128)
    
    def test_generate_embeddings_batch(self):
        """Test batch embedding generation"""
        # Train first
        texts = [f"Training text {i}" for i in range(10)]
        self.generator.add_training_texts(texts)
        self.generator.train_embeddings_model()
        
        # Generate batch
        test_texts = ["First test", "Second test", "Third test"]
        embeddings = self.generator.generate_embeddings_batch(test_texts)
        
        self.assertEqual(len(embeddings), 3)
        for embedding in embeddings:
            self.assertIsNotNone(embedding)
            self.assertEqual(len(embedding), 128)
    
    def test_caching_functionality(self):
        """Test embedding caching"""
        # Train model
        texts = [f"Training text {i}" for i in range(10)]
        self.generator.add_training_texts(texts)
        self.generator.train_embeddings_model()
        
        test_text = "Cache test text"
        
        # First generation
        embedding1 = self.generator.generate_embedding(test_text)
        
        # Second generation (should use cache)
        embedding2 = self.generator.generate_embedding(test_text)
        
        self.assertEqual(embedding1, embedding2)
        self.assertGreater(len(self.generator.embedding_cache), 0)
    
    def test_get_generation_stats(self):
        """Test statistics retrieval"""
        stats = self.generator.get_generation_stats()
        
        required_keys = [
            'embeddings_generated', 'cache_hits', 'cache_misses',
            'is_trained', 'vector_dimension'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['vector_dimension'], 128)
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add to cache
        self.generator.embedding_cache['test'] = [0.1] * 128
        self.assertEqual(len(self.generator.embedding_cache), 1)
        
        # Clear cache
        self.generator.clear_cache()
        self.assertEqual(len(self.generator.embedding_cache), 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
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


class TestIntegrationPoints(unittest.TestCase):
    """Test integration with other components"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConversationEmbeddingGenerator(cache_dir=self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_storage_integration_interface(self):
        """Test storage integration interface"""
        # Mock storage backend
        mock_storage = MagicMock()
        mock_storage.get_conversation_events.return_value = [
            {
                'event_id': 'test1',
                'event_type': 'tool_use',
                'event_data': json.dumps({'description': 'Test event'})
            }
        ]
        mock_storage.connection.execute = MagicMock()
        
        # Train model first
        self.generator.add_training_texts(['Test training text'])
        self.generator.train_embeddings_model()
        
        # Test update conversation embeddings
        result = self.generator.update_conversation_embeddings('conv123', mock_storage)
        
        self.assertIn('updated', result)
        self.assertIsInstance(result['updated'], int)
    
    def test_auto_train_interface(self):
        """Test auto-training interface"""
        mock_storage = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ('tool_use', json.dumps({'description': f'Training text {i}'}))
            for i in range(150)  # Enough for training
        ]
        mock_storage.connection.execute.return_value = mock_result
        
        # Test auto-training
        success = self.generator.auto_train_from_storage(mock_storage, min_texts=100)
        
        # Should succeed with enough mock data
        self.assertTrue(success)
        self.assertTrue(self.generator.is_trained)


if __name__ == '__main__':
    # Run tests
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Embedding Generator Structure Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ EXCELLENT: Embedding generator structure is correct!")
    else:
        print("❌ Issues detected in embedding generator structure")
    
    sys.exit(0 if result.wasSuccessful() else 1)