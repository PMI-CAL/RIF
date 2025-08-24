"""
Vector embedding generation package for the RIF hybrid knowledge system.

This package provides semantic vector embeddings for code entities to enable
similarity search and intelligent code queries.
"""

from .embedding_generator import EmbeddingGenerator, LocalEmbeddingModel
from .text_processor import TextProcessor, EntityTextExtractor
from .embedding_storage import EmbeddingStorage
from .embedding_pipeline import EmbeddingPipeline

__all__ = [
    'EmbeddingGenerator',
    'LocalEmbeddingModel',
    'TextProcessor',
    'EntityTextExtractor',
    'EmbeddingStorage', 
    'EmbeddingPipeline'
]