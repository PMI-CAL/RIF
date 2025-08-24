"""
Embedding Manager for LightRAG
Handles text embeddings with fallback strategies for RIF framework.
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import pickle
import hashlib
from datetime import datetime

try:
    # Try OpenAI first
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    # Fallback to sentence-transformers
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingManager:
    """
    Manages text embeddings with multiple backends and caching.
    Supports OpenAI embeddings with local sentence-transformers fallback.
    """
    
    def __init__(self, 
                 cache_dir: str = None,
                 embedding_model: str = "text-embedding-ada-002",
                 local_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager.
        
        Args:
            cache_dir: Directory for embedding cache
            embedding_model: OpenAI embedding model name
            local_model: Local sentence-transformers model name
        """
        # Setup cache directory
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(os.path.dirname(current_dir), "embeddings_cache")
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Model configuration
        self.embedding_model = embedding_model
        self.local_model_name = local_model
        
        # Initialize backends
        self.openai_client = None
        self.local_model = None
        
        # Setup OpenAI if available
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = openai.OpenAI(api_key=api_key)
                    # Test the connection
                    self._test_openai_connection()
                except Exception as e:
                    logging.warning(f"OpenAI setup failed: {e}")
                    self.openai_client = None
        
        # Setup local model as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.local_model = SentenceTransformer(local_model)
                logging.info(f"Loaded local embedding model: {local_model}")
            except Exception as e:
                logging.error(f"Failed to load local model: {e}")
                self.local_model = None
        
        # Determine active backend
        self.backend = self._determine_backend()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EmbeddingManager initialized with backend: {self.backend}")
    
    def _determine_backend(self) -> str:
        """Determine which embedding backend to use."""
        if self.openai_client:
            return "openai"
        elif self.local_model:
            return "local"
        else:
            return "none"
    
    def _test_openai_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=["test"]
            )
            return True
        except Exception:
            return False
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        content = f"{text}_{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Get embedding for text using available backend.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array, or None if failed
        """
        if not text or not text.strip():
            return None
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text, self.backend)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding based on backend
        embedding = None
        
        if self.backend == "openai":
            embedding = self._get_openai_embedding(text)
        elif self.backend == "local":
            embedding = self._get_local_embedding(text)
        
        # Cache the result
        if embedding is not None and use_cache:
            cache_key = self._get_cache_key(text, self.backend)
            self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def _get_openai_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
        except Exception as e:
            self.logger.error(f"OpenAI embedding failed: {e}")
            # Fallback to local model if OpenAI fails
            if self.local_model:
                self.logger.info("Falling back to local model")
                return self._get_local_embedding(text)
            return None
    
    def _get_local_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from local sentence-transformers model."""
        try:
            embedding = self.local_model.encode(text)
            return np.array(embedding)
        except Exception as e:
            self.logger.error(f"Local embedding failed: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors (same order as input)
        """
        embeddings = []
        
        # Check which texts need embedding (not in cache)
        texts_to_embed = []
        cached_embeddings = {}
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, self.backend)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    cached_embeddings[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            if self.backend == "openai":
                self._get_openai_embeddings_batch(texts_to_embed, embeddings, use_cache)
            elif self.backend == "local":
                self._get_local_embeddings_batch(texts_to_embed, embeddings, use_cache)
        
        # Combine cached and new embeddings in correct order
        result = [None] * len(texts)
        
        # Add cached embeddings
        for i, embedding in cached_embeddings.items():
            result[i] = embedding
        
        # Add new embeddings
        for (original_index, _), embedding in zip(texts_to_embed, embeddings):
            result[original_index] = embedding
        
        return result
    
    def _get_openai_embeddings_batch(self, texts_to_embed: List[tuple], embeddings: List, use_cache: bool):
        """Get batch embeddings from OpenAI."""
        try:
            texts_only = [text for _, text in texts_to_embed]
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts_only
            )
            
            for i, (original_index, text) in enumerate(texts_to_embed):
                embedding = np.array(response.data[i].embedding)
                embeddings.append(embedding)
                
                # Cache the result
                if use_cache:
                    cache_key = self._get_cache_key(text, self.backend)
                    self._save_to_cache(cache_key, embedding)
                    
        except Exception as e:
            self.logger.error(f"OpenAI batch embedding failed: {e}")
            # Fallback to individual local embeddings
            if self.local_model:
                for _, text in texts_to_embed:
                    embedding = self._get_local_embedding(text)
                    embeddings.append(embedding)
    
    def _get_local_embeddings_batch(self, texts_to_embed: List[tuple], embeddings: List, use_cache: bool):
        """Get batch embeddings from local model."""
        try:
            texts_only = [text for _, text in texts_to_embed]
            batch_embeddings = self.local_model.encode(texts_only)
            
            for i, (original_index, text) in enumerate(texts_to_embed):
                embedding = np.array(batch_embeddings[i])
                embeddings.append(embedding)
                
                # Cache the result
                if use_cache:
                    cache_key = self._get_cache_key(text, self.backend)
                    self._save_to_cache(cache_key, embedding)
                    
        except Exception as e:
            self.logger.error(f"Local batch embedding failed: {e}")
            # Fallback to individual embeddings
            for _, text in texts_to_embed:
                embedding = self._get_local_embedding(text)
                embeddings.append(embedding)
    
    def get_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1), or None if failed
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return None
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def clear_cache(self) -> int:
        """
        Clear embedding cache.
        
        Returns:
            Number of files cleared
        """
        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
                    count += 1
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        stats = {
            "cache_dir": self.cache_dir,
            "file_count": 0,
            "total_size_mb": 0,
            "backend": self.backend
        }
        
        try:
            total_size = 0
            file_count = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            stats["file_count"] = file_count
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats


# Global embedding manager instance
_embedding_manager = None


def get_embedding_manager() -> EmbeddingManager:
    """Get global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def embed_text(text: str) -> Optional[np.ndarray]:
    """Convenience function to embed a single text."""
    manager = get_embedding_manager()
    return manager.get_embedding(text)


def embed_texts(texts: List[str]) -> List[Optional[np.ndarray]]:
    """Convenience function to embed multiple texts."""
    manager = get_embedding_manager()
    return manager.get_embeddings_batch(texts)


def text_similarity(text1: str, text2: str) -> Optional[float]:
    """Convenience function to calculate text similarity."""
    manager = get_embedding_manager()
    return manager.get_similarity(text1, text2)