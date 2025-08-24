"""
Local embedding generation system for code entities.

This implementation uses TF-IDF and text features to create semantic embeddings
without requiring heavy transformer models.
"""

import hashlib
import time
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

from ..extraction.entity_types import CodeEntity
from .text_processor import EntityTextExtractor, ProcessedText


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    entity_id: str
    embedding: np.ndarray
    content_hash: str
    metadata: Dict[str, Any]
    generation_time: float


class LocalEmbeddingModel:
    """
    Local embedding model using TF-IDF and text features.
    
    This provides a lightweight alternative to transformer models while
    still generating meaningful vector representations for code similarity.
    """
    
    def __init__(self, embedding_dim: int = 384, max_features: int = 5000):
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.logger = logging.getLogger(__name__)
        
        # TF-IDF components
        self.vocabulary = {}  # word -> index mapping
        self.idf_scores = {}  # word -> IDF score
        self.feature_names = []
        
        # Model state
        self.is_fitted = False
        self.document_count = 0
        
        # Feature weights for different components
        self.feature_weights = {
            'tfidf': 0.6,      # TF-IDF features
            'structural': 0.2,  # Structural features (entity type, etc.)
            'semantic': 0.2     # Semantic features (keywords, etc.)
        }
    
    def fit(self, texts: List[str]):
        """Fit the model on a collection of texts."""
        self.logger.info(f"Fitting embedding model on {len(texts)} documents")
        
        # Build vocabulary and calculate IDF scores
        self._build_vocabulary(texts)
        self._calculate_idf_scores(texts)
        
        self.is_fitted = True
        self.document_count = len(texts)
        
        self.logger.info(f"Model fitted with {len(self.vocabulary)} features")
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_counts = Counter()
        
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Select most common words up to max_features
        most_common = word_counts.most_common(self.max_features)
        
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(most_common)}
        self.feature_names = [word for word, idx in self.vocabulary.items()]
    
    def _calculate_idf_scores(self, texts: List[str]):
        """Calculate IDF scores for vocabulary words."""
        document_frequencies = defaultdict(int)
        
        for text in texts:
            words_in_doc = set(text.split())
            for word in words_in_doc:
                if word in self.vocabulary:
                    document_frequencies[word] += 1
        
        # Calculate IDF scores: log(N / df)
        num_docs = len(texts)
        self.idf_scores = {}
        for word in self.vocabulary:
            df = document_frequencies.get(word, 1)
            idf = np.log(num_docs / df)
            self.idf_scores[word] = idf
    
    def encode(self, text: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.is_fitted:
            # If not fitted, create a simple hash-based embedding
            return self._hash_embedding(text)
        
        # Generate TF-IDF features
        tfidf_features = self._compute_tfidf_features(text)
        
        # Generate structural features
        structural_features = self._compute_structural_features(metadata or {})
        
        # Generate semantic features
        semantic_features = self._compute_semantic_features(text, metadata or {})
        
        # Combine features with weights
        embedding = (
            self.feature_weights['tfidf'] * tfidf_features +
            self.feature_weights['structural'] * structural_features +
            self.feature_weights['semantic'] * semantic_features
        )
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _compute_tfidf_features(self, text: str) -> np.ndarray:
        """Compute TF-IDF features for text."""
        words = text.split()
        word_counts = Counter(words)
        
        # Create sparse TF-IDF vector
        tfidf_vector = np.zeros(len(self.vocabulary))
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                word_idx = self.vocabulary[word]
                tf = count / len(words) if words else 0
                idf = self.idf_scores.get(word, 0)
                tfidf_vector[word_idx] = tf * idf
        
        # Pad or truncate to embedding dimension
        if len(tfidf_vector) < self.embedding_dim:
            padded = np.zeros(self.embedding_dim)
            padded[:len(tfidf_vector)] = tfidf_vector
            return padded
        else:
            return tfidf_vector[:self.embedding_dim]
    
    def _compute_structural_features(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Compute structural features based on entity metadata."""
        features = np.zeros(self.embedding_dim)
        
        # Entity type features
        entity_type = metadata.get('entity_type', 'unknown')
        type_mapping = {
            'function': [1.0, 0.0, 0.0, 0.0, 0.0],
            'class': [0.0, 1.0, 0.0, 0.0, 0.0],
            'module': [0.0, 0.0, 1.0, 0.0, 0.0],
            'variable': [0.0, 0.0, 0.0, 1.0, 0.0],
            'constant': [0.0, 0.0, 0.0, 1.0, 0.0],
            'interface': [0.0, 0.0, 0.0, 0.0, 1.0],
            'enum': [0.0, 0.0, 0.0, 0.0, 1.0],
        }
        type_features = type_mapping.get(entity_type, [0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add type features to beginning of vector
        features[:len(type_features)] = type_features
        
        # Complexity features (for functions)
        complexity = metadata.get('complexity', 1)
        if 'complexity' in metadata:
            complexity_idx = min(5 + int(complexity) % 10, self.embedding_dim - 1)
            features[complexity_idx] = min(complexity / 10.0, 1.0)
        
        # Count features
        method_count = metadata.get('method_count', 0)
        if method_count > 0:
            count_idx = min(15 + method_count % 20, self.embedding_dim - 1)
            features[count_idx] = min(method_count / 20.0, 1.0)
        
        return features
    
    def _compute_semantic_features(self, text: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Compute semantic features based on text content."""
        features = np.zeros(self.embedding_dim)
        
        words = text.split()
        if not words:
            return features
        
        # Keyword density features
        keyword_categories = {
            'data': ['data', 'information', 'record', 'store', 'save'],
            'algorithm': ['sort', 'search', 'find', 'process', 'compute'],
            'interface': ['api', 'interface', 'endpoint', 'service', 'client'],
            'utility': ['util', 'helper', 'tool', 'common', 'shared'],
            'test': ['test', 'spec', 'mock', 'stub', 'assert'],
        }
        
        for idx, (category, keywords) in enumerate(keyword_categories.items()):
            if idx >= self.embedding_dim:
                break
                
            matches = sum(1 for word in words if word in keywords)
            density = matches / len(words)
            features[idx] = min(density * 10, 1.0)  # Scale and cap at 1.0
        
        # Name-based features
        name = metadata.get('name', '')
        if name:
            # CamelCase/snake_case patterns
            has_camel_case = bool(re.search(r'[a-z][A-Z]', name))
            has_snake_case = '_' in name
            
            if has_camel_case and len(features) > 10:
                features[10] = 1.0
            if has_snake_case and len(features) > 11:
                features[11] = 1.0
        
        return features
    
    def _hash_embedding(self, text: str) -> np.ndarray:
        """Generate a simple hash-based embedding when model is not fitted."""
        # Create a deterministic embedding based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float array
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        
        # Resize to desired dimension
        if len(embedding) < self.embedding_dim:
            # Repeat pattern if needed
            repeats = (self.embedding_dim // len(embedding)) + 1
            embedding = np.tile(embedding, repeats)[:self.embedding_dim]
        else:
            embedding = embedding[:self.embedding_dim]
        
        # Normalize
        embedding = (embedding - 128) / 128.0  # Convert to [-1, 1] range
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': 'tfidf_local',
            'embedding_dim': self.embedding_dim,
            'max_features': self.max_features,
            'vocabulary_size': len(self.vocabulary),
            'is_fitted': self.is_fitted,
            'document_count': self.document_count,
            'feature_weights': self.feature_weights
        }


class EmbeddingGenerator:
    """
    Main embedding generator that coordinates text processing and model inference.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 cache_size: int = 10000,
                 model_path: Optional[str] = None):
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.text_extractor = EntityTextExtractor()
        self.model = LocalEmbeddingModel(embedding_dim=embedding_dim)
        
        # Caching
        self.embedding_cache = {}  # content_hash -> embedding
        self.cache_access_order = []  # For LRU eviction
        
        # Performance metrics
        self.metrics = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generation_time': 0.0
        }
        
        # Load pre-fitted model if available
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def fit_model(self, entities: List[CodeEntity], save_path: Optional[str] = None):
        """Fit the embedding model on a collection of entities."""
        self.logger.info(f"Fitting embedding model on {len(entities)} entities")
        
        # Extract texts from all entities
        texts = []
        for entity in entities:
            processed_text = self.text_extractor.extract_text(entity)
            texts.append(processed_text.text)
        
        # Fit the model
        self.model.fit(texts)
        
        # Save model if path provided
        if save_path:
            self._save_model(save_path)
            
        self.logger.info("Model fitting complete")
    
    def generate_embedding(self, entity: CodeEntity, 
                          use_cache: bool = True) -> EmbeddingResult:
        """Generate embedding for a single entity."""
        start_time = time.time()
        
        # Extract and process text
        processed_text = self.text_extractor.extract_text(entity)
        
        # Check cache first
        if use_cache and processed_text.content_hash in self.embedding_cache:
            embedding = self.embedding_cache[processed_text.content_hash]
            self.metrics['cache_hits'] += 1
            self._update_cache_access(processed_text.content_hash)
        else:
            # Generate new embedding
            embedding = self.model.encode(processed_text.text, processed_text.metadata)
            
            # Cache the result
            if use_cache:
                self._cache_embedding(processed_text.content_hash, embedding)
                self.metrics['cache_misses'] += 1
        
        generation_time = time.time() - start_time
        self.metrics['embeddings_generated'] += 1
        self.metrics['total_generation_time'] += generation_time
        
        return EmbeddingResult(
            entity_id=str(entity.id),
            embedding=embedding,
            content_hash=processed_text.content_hash,
            metadata={
                'processed_text': processed_text.text,
                'original_metadata': processed_text.metadata,
                'model_info': self.model.get_model_info(),
                'generation_time': generation_time
            },
            generation_time=generation_time
        )
    
    def generate_embeddings_batch(self, entities: List[CodeEntity],
                                 batch_size: int = 100,
                                 use_cache: bool = True) -> List[EmbeddingResult]:
        """Generate embeddings for multiple entities in batches."""
        results = []
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            self.logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} entities")
            
            batch_results = []
            for entity in batch:
                result = self.generate_embedding(entity, use_cache)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Cleanup cache periodically
            if len(self.embedding_cache) > self.cache_size:
                self._cleanup_cache()
        
        self.logger.info(f"Generated {len(results)} embeddings")
        return results
    
    def _cache_embedding(self, content_hash: str, embedding: np.ndarray):
        """Cache an embedding with LRU eviction."""
        self.embedding_cache[content_hash] = embedding
        
        if content_hash in self.cache_access_order:
            self.cache_access_order.remove(content_hash)
        self.cache_access_order.append(content_hash)
        
        # Evict if cache is too large
        if len(self.embedding_cache) > self.cache_size:
            self._cleanup_cache()
    
    def _update_cache_access(self, content_hash: str):
        """Update access order for cached item."""
        if content_hash in self.cache_access_order:
            self.cache_access_order.remove(content_hash)
            self.cache_access_order.append(content_hash)
    
    def _cleanup_cache(self):
        """Remove least recently used cache entries."""
        target_size = int(self.cache_size * 0.8)  # Remove 20% when cleaning up
        
        while len(self.embedding_cache) > target_size and self.cache_access_order:
            lru_hash = self.cache_access_order.pop(0)
            self.embedding_cache.pop(lru_hash, None)
    
    def _save_model(self, save_path: str):
        """Save the fitted model to disk."""
        model_data = {
            'vocabulary': self.model.vocabulary,
            'idf_scores': self.model.idf_scores,
            'feature_names': self.model.feature_names,
            'embedding_dim': self.model.embedding_dim,
            'max_features': self.model.max_features,
            'is_fitted': self.model.is_fitted,
            'document_count': self.model.document_count,
            'feature_weights': self.model.feature_weights
        }
        
        with open(save_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def _load_model(self, model_path: str):
        """Load a pre-fitted model from disk."""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        self.model.vocabulary = model_data['vocabulary']
        self.model.idf_scores = model_data['idf_scores']
        self.model.feature_names = model_data['feature_names']
        self.model.embedding_dim = model_data['embedding_dim']
        self.model.max_features = model_data['max_features']
        self.model.is_fitted = model_data['is_fitted']
        self.model.document_count = model_data['document_count']
        self.model.feature_weights = model_data['feature_weights']
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = dict(self.metrics)
        
        # Calculate derived metrics
        if metrics['embeddings_generated'] > 0:
            metrics['avg_generation_time'] = (
                metrics['total_generation_time'] / metrics['embeddings_generated']
            )
            
            total_requests = metrics['cache_hits'] + metrics['cache_misses']
            if total_requests > 0:
                metrics['cache_hit_rate'] = metrics['cache_hits'] / total_requests
        else:
            metrics['avg_generation_time'] = 0.0
            metrics['cache_hit_rate'] = 0.0
        
        # Add cache info
        metrics['cache_size'] = len(self.embedding_cache)
        metrics['cache_capacity'] = self.cache_size
        
        # Add model info
        metrics['model'] = self.model.get_model_info()
        
        return metrics
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.cache_access_order.clear()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generation_time': 0.0
        }