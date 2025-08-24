#!/usr/bin/env python3
"""
Conversation Embedding Generator for RIF Agents

Generates vector embeddings for conversation events to enable semantic search
using TF-IDF vectorization with batch processing for efficiency.

This module provides:
- TF-IDF-based embedding generation for text content
- Batch processing for performance optimization
- Integration with DuckDB vector storage
- Embedding cache management
- Support for different embedding strategies
"""

import json
import time
import pickle
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationEmbeddingGenerator:
    """
    TF-IDF based embedding generator for conversation events.
    
    Provides semantic embeddings for text content using TF-IDF vectorization
    with dimensionality reduction and batch processing for efficiency.
    """
    
    def __init__(self, 
                 vector_dim: int = 768,
                 batch_size: int = 100,
                 cache_dir: str = "knowledge/embeddings_cache",
                 max_features: int = 5000,
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Initialize the embedding generator.
        
        Args:
            vector_dim: Dimension of output embeddings (default 768 for compatibility)
            batch_size: Batch size for processing multiple texts
            cache_dir: Directory for caching trained models and embeddings
            max_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency for features
            max_df: Maximum document frequency for features
        """
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TF-IDF pipeline
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words='english',
            tokenizer=self._tokenize_text,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        # Dimensionality reduction to get to target vector size
        self.svd_reducer = TruncatedSVD(
            n_components=min(vector_dim, max_features),
            random_state=42
        )
        
        # Standardization for better similarity calculations
        self.scaler = StandardScaler()
        
        # Combined pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.tfidf_vectorizer),
            ('svd', self.svd_reducer),
            ('scaler', self.scaler)
        ])
        
        # Stemmer for text preprocessing
        self.stemmer = PorterStemmer()
        
        # Training state
        self.is_trained = False
        self.training_texts = []
        self.embedding_cache = {}
        
        # Performance tracking
        self.generation_stats = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_generation_time_ms': 0.0,
            'batch_operations': 0
        }
        
        # Try to download required NLTK data
        self._setup_nltk_data()
        
        logger.info(f"ConversationEmbeddingGenerator initialized with {vector_dim}D vectors, batch size {batch_size}")
    
    def _setup_nltk_data(self):
        """Download required NLTK data if not present"""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}. Using fallback tokenization.")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Custom tokenization with stemming and cleanup"""
        if not text:
            return []
        
        try:
            # Basic tokenization if NLTK not available
            tokens = word_tokenize(text.lower()) if 'word_tokenize' in globals() else text.lower().split()
        except:
            tokens = text.lower().split()
        
        # Clean and stem tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove non-alphabetic tokens and very short tokens
            if token.isalpha() and len(token) > 2:
                stemmed = self.stemmer.stem(token)
                cleaned_tokens.append(stemmed)
        
        return cleaned_tokens
    
    def add_training_texts(self, texts: List[str]) -> None:
        """
        Add texts to the training corpus for TF-IDF learning.
        
        Args:
            texts: List of text strings to add to training corpus
        """
        if not texts:
            return
        
        # Clean and preprocess texts
        processed_texts = []
        for text in texts:
            if text and isinstance(text, str):
                # Clean text: remove excessive whitespace, normalize
                cleaned = re.sub(r'\s+', ' ', text.strip())
                if len(cleaned) > 10:  # Only include substantial texts
                    processed_texts.append(cleaned)
        
        self.training_texts.extend(processed_texts)
        logger.info(f"Added {len(processed_texts)} texts to training corpus (total: {len(self.training_texts)})")
    
    def train_embeddings_model(self, force_retrain: bool = False) -> bool:
        """
        Train the TF-IDF model on accumulated training texts.
        
        Args:
            force_retrain: Force retraining even if model already trained
            
        Returns:
            True if training successful, False otherwise
        """
        if self.is_trained and not force_retrain:
            logger.info("Embedding model already trained. Use force_retrain=True to retrain.")
            return True
        
        if not self.training_texts:
            logger.warning("No training texts available. Cannot train embedding model.")
            return False
        
        try:
            logger.info(f"Training embedding model on {len(self.training_texts)} texts...")
            start_time = time.time()
            
            # Train the pipeline
            self.pipeline.fit(self.training_texts)
            
            # Adjust dimensions if needed
            actual_components = self.pipeline.named_steps['svd'].n_components_
            if actual_components < self.vector_dim:
                logger.info(f"Adjusting vector dimension from {self.vector_dim} to {actual_components}")
                self.vector_dim = actual_components
            
            self.is_trained = True
            training_time = time.time() - start_time
            
            logger.info(f"Embedding model trained successfully in {training_time:.2f}s")
            
            # Save trained model
            self._save_trained_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train embedding model: {e}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not text or not isinstance(text, str):
            return None
        
        # Check cache first
        text_hash = self._get_text_hash(text)
        if text_hash in self.embedding_cache:
            self.generation_stats['cache_hits'] += 1
            return self.embedding_cache[text_hash]
        
        if not self.is_trained:
            logger.warning("Embedding model not trained. Cannot generate embeddings.")
            return None
        
        start_time = time.time()
        
        try:
            # Generate embedding using trained pipeline
            embedding_matrix = self.pipeline.transform([text])
            embedding = embedding_matrix[0].tolist()
            
            # Pad or truncate to target dimension
            embedding = self._adjust_embedding_dimension(embedding)
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            
            # Update stats
            generation_time = (time.time() - start_time) * 1000
            self._update_generation_stats(generation_time)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (same order as input texts)
        """
        if not texts:
            return []
        
        if not self.is_trained:
            logger.warning("Embedding model not trained. Cannot generate embeddings.")
            return [None] * len(texts)
        
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {self.batch_size}")
        start_time = time.time()
        
        embeddings = []
        
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._generate_batch_embeddings(batch)
                embeddings.extend(batch_embeddings)
            
            # Update batch operation stats
            self.generation_stats['batch_operations'] += 1
            
            batch_time = (time.time() - start_time) * 1000
            logger.info(f"Generated {len(embeddings)} embeddings in {batch_time:.1f}ms ({len(embeddings) / (batch_time / 1000):.1f} embeddings/sec)")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)
    
    def _generate_batch_embeddings(self, batch_texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a single batch"""
        embeddings = []
        
        # Check cache for each text and collect uncached texts
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(batch_texts):
            if not text or not isinstance(text, str):
                embeddings.append(None)
                continue
                
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
                self.generation_stats['cache_hits'] += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.generation_stats['cache_misses'] += 1
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                embedding_matrix = self.pipeline.transform(uncached_texts)
                
                for j, (embedding_vec, original_index) in enumerate(zip(embedding_matrix, uncached_indices)):
                    embedding = embedding_vec.tolist()
                    embedding = self._adjust_embedding_dimension(embedding)
                    
                    # Update cache and results
                    text = batch_texts[original_index]
                    text_hash = self._get_text_hash(text)
                    self.embedding_cache[text_hash] = embedding
                    embeddings[original_index] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Fill remaining with None
                for i in uncached_indices:
                    embeddings[i] = None
        
        return embeddings
    
    def update_conversation_embeddings(self, 
                                     conversation_id: str, 
                                     storage_backend) -> Dict[str, Any]:
        """
        Update embeddings for all events in a conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            storage_backend: ConversationStorageBackend instance
            
        Returns:
            Dictionary with update statistics
        """
        logger.info(f"Updating embeddings for conversation {conversation_id}")
        start_time = time.time()
        
        try:
            # Get all events for conversation
            events = storage_backend.get_conversation_events(conversation_id, limit=1000)
            
            if not events:
                return {'updated': 0, 'error': 'No events found'}
            
            # Extract texts that need embeddings
            texts_to_embed = []
            event_indices = []
            
            for i, event in enumerate(events):
                event_data = event.get('event_data', {})
                if isinstance(event_data, str):
                    event_data = json.loads(event_data)
                
                # Create searchable text from event
                searchable_text = self._create_searchable_text(event['event_type'], event_data)
                if searchable_text and len(searchable_text.strip()) > 10:
                    texts_to_embed.append(searchable_text)
                    event_indices.append(i)
            
            if not texts_to_embed:
                return {'updated': 0, 'error': 'No suitable texts found for embedding'}
            
            # Generate embeddings in batch
            embeddings = self.generate_embeddings_batch(texts_to_embed)
            
            # Update events with embeddings
            updated_count = 0
            for embedding, event_index in zip(embeddings, event_indices):
                if embedding:
                    event = events[event_index]
                    # Update embedding in database
                    try:
                        storage_backend.connection.execute("""
                            UPDATE conversation_events 
                            SET embedding = ?
                            WHERE event_id = ?
                        """, [embedding, event['event_id']])
                        updated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to update embedding for event {event['event_id']}: {e}")
            
            total_time = time.time() - start_time
            logger.info(f"Updated embeddings for {updated_count} events in {total_time:.2f}s")
            
            return {
                'updated': updated_count,
                'total_events': len(events),
                'processing_time_s': total_time,
                'embeddings_per_second': updated_count / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to update conversation embeddings: {e}")
            return {'updated': 0, 'error': str(e)}
    
    def _create_searchable_text(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Create searchable text from event data"""
        text_parts = [f"Event: {event_type}"]
        
        # Extract key text fields from event data
        text_fields = ['description', 'summary', 'content', 'message', 'result', 'output', 'rationale', 'decision_point']
        
        for field in text_fields:
            if field in event_data:
                value = event_data[field]
                if isinstance(value, str) and len(value.strip()) > 0:
                    text_parts.append(f"{field}: {value[:500]}")  # Limit length
        
        # Add other relevant fields
        if 'tool_name' in event_data:
            text_parts.append(f"Tool: {event_data['tool_name']}")
        
        if 'status' in event_data:
            text_parts.append(f"Status: {event_data['status']}")
        
        if 'agent_type' in event_data:
            text_parts.append(f"Agent: {event_data['agent_type']}")
        
        return ". ".join(text_parts)
    
    def _adjust_embedding_dimension(self, embedding: List[float]) -> List[float]:
        """Adjust embedding to target dimension by padding or truncating"""
        if len(embedding) == self.vector_dim:
            return embedding
        elif len(embedding) > self.vector_dim:
            return embedding[:self.vector_dim]
        else:
            # Pad with zeros
            padding = [0.0] * (self.vector_dim - len(embedding))
            return embedding + padding
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _update_generation_stats(self, generation_time_ms: float):
        """Update generation performance statistics"""
        self.generation_stats['embeddings_generated'] += 1
        
        current_avg = self.generation_stats['avg_generation_time_ms']
        total_generated = self.generation_stats['embeddings_generated']
        
        self.generation_stats['avg_generation_time_ms'] = (
            (current_avg * (total_generated - 1) + generation_time_ms) / total_generated
        )
    
    def _save_trained_model(self):
        """Save trained model to cache for reuse"""
        try:
            model_cache_path = self.cache_dir / "trained_model.pkl"
            
            model_data = {
                'pipeline': self.pipeline,
                'vector_dim': self.vector_dim,
                'is_trained': self.is_trained,
                'training_timestamp': datetime.now().isoformat(),
                'training_text_count': len(self.training_texts)
            }
            
            with open(model_cache_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved trained model to {model_cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save trained model: {e}")
    
    def _load_trained_model(self) -> bool:
        """Load trained model from cache"""
        try:
            model_cache_path = self.cache_dir / "trained_model.pkl"
            
            if not model_cache_path.exists():
                return False
            
            with open(model_cache_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.vector_dim = model_data['vector_dim']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Loaded trained model from {model_cache_path}")
            logger.info(f"Model trained on {model_data['training_text_count']} texts at {model_data['training_timestamp']}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load trained model: {e}")
            return False
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        total_requests = self.generation_stats['cache_hits'] + self.generation_stats['cache_misses']
        cache_hit_rate = self.generation_stats['cache_hits'] / max(total_requests, 1)
        
        stats = self.generation_stats.copy()
        stats.update({
            'is_trained': self.is_trained,
            'training_text_count': len(self.training_texts),
            'cache_size': len(self.embedding_cache),
            'cache_hit_rate': cache_hit_rate,
            'vector_dimension': self.vector_dim
        })
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache to free memory"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"Cleared embedding cache ({cache_size} entries)")
    
    def auto_train_from_storage(self, storage_backend, min_texts: int = 100) -> bool:
        """
        Automatically train embedding model from existing conversation data.
        
        Args:
            storage_backend: ConversationStorageBackend instance
            min_texts: Minimum number of texts needed for training
            
        Returns:
            True if training successful, False otherwise
        """
        logger.info("Auto-training embedding model from existing conversation data...")
        
        try:
            # Get recent conversation events
            query = """
                SELECT event_type, event_data 
                FROM conversation_events 
                WHERE event_data IS NOT NULL 
                AND LENGTH(event_data) > 50
                ORDER BY timestamp DESC 
                LIMIT 1000
            """
            
            result = storage_backend.connection.execute(query)
            rows = result.fetchall()
            
            if len(rows) < min_texts:
                logger.warning(f"Insufficient conversation data for training ({len(rows)} < {min_texts})")
                return False
            
            # Extract texts for training
            training_texts = []
            for row in rows:
                event_type, event_data_str = row
                try:
                    event_data = json.loads(event_data_str) if isinstance(event_data_str, str) else event_data_str
                    searchable_text = self._create_searchable_text(event_type, event_data)
                    if searchable_text and len(searchable_text.strip()) > 20:
                        training_texts.append(searchable_text)
                except Exception as e:
                    continue
            
            if len(training_texts) < min_texts:
                logger.warning(f"Insufficient valid texts extracted ({len(training_texts)} < {min_texts})")
                return False
            
            # Add to training corpus and train
            self.add_training_texts(training_texts)
            return self.train_embeddings_model()
            
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")
            return False


# Convenience functions for easy integration

def create_embedding_generator(**kwargs) -> ConversationEmbeddingGenerator:
    """Factory function to create embedding generator with default configuration"""
    generator = ConversationEmbeddingGenerator(**kwargs)
    
    # Try to load existing trained model
    if generator._load_trained_model():
        logger.info("Loaded existing trained embedding model")
    
    return generator


def setup_embeddings_for_capture_engine(capture_engine, storage_backend, auto_train: bool = True) -> bool:
    """
    Set up embedding generation for a conversation capture engine.
    
    Args:
        capture_engine: ConversationCaptureEngine instance
        storage_backend: ConversationStorageBackend instance  
        auto_train: Whether to automatically train from existing data
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Create embedding generator
        generator = create_embedding_generator()
        
        # Auto-train if requested and no trained model exists
        if auto_train and not generator.is_trained:
            generator.auto_train_from_storage(storage_backend)
        
        # Set up generator function for capture engine
        def embedding_function(text: str) -> Optional[List[float]]:
            return generator.generate_embedding(text)
        
        capture_engine.set_embedding_generator(embedding_function)
        
        logger.info("Successfully set up embeddings for conversation capture engine")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up embeddings: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    print("RIF Conversation Embedding Generator")
    print("=" * 40)
    
    # Create generator
    generator = create_embedding_generator(vector_dim=384, batch_size=50)
    
    # Example training texts
    example_texts = [
        "Analyzing requirements for user authentication system",
        "Implementing secure login with encryption and validation", 
        "Running comprehensive test suite for quality assurance",
        "Validating security measures and compliance standards",
        "Optimizing performance and scalability of microservices",
        "Designing system architecture for distributed components"
    ]
    
    # Add training texts and train
    generator.add_training_texts(example_texts)
    
    if generator.train_embeddings_model():
        print("✅ Model trained successfully")
        
        # Generate test embeddings
        test_text = "Testing the embedding generation system"
        embedding = generator.generate_embedding(test_text)
        
        if embedding:
            print(f"✅ Generated {len(embedding)}-dimensional embedding")
            print(f"   Sample values: {embedding[:5]}...")
        
        # Test batch generation
        batch_texts = ["First test text", "Second test text", "Third test text"]
        batch_embeddings = generator.generate_embeddings_batch(batch_texts)
        
        valid_embeddings = [e for e in batch_embeddings if e is not None]
        print(f"✅ Generated {len(valid_embeddings)} embeddings in batch")
        
        # Show stats
        stats = generator.get_generation_stats()
        print(f"\nGeneration Statistics:")
        print(f"  - Embeddings generated: {stats['embeddings_generated']}")
        print(f"  - Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  - Average generation time: {stats['avg_generation_time_ms']:.1f}ms")
        
    else:
        print("❌ Model training failed")