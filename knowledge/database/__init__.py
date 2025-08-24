"""
DuckDB database interface for RIF knowledge system.
Provides connection management, pooling, and vector search capabilities.
Issue #26: Set up DuckDB as embedded database with vector search
"""

from .connection_manager import DuckDBConnectionManager
from .vector_search import VectorSearchEngine
from .database_config import DatabaseConfig

__all__ = [
    'DuckDBConnectionManager',
    'VectorSearchEngine', 
    'DatabaseConfig'
]