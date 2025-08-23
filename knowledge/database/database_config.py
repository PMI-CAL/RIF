"""
DuckDB database configuration for RIF system.
Issue #26: Set up DuckDB as embedded database with vector search
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Configuration for DuckDB database connections."""
    
    # Database file path
    database_path: str = "knowledge/chromadb/entities.duckdb"
    
    # Memory settings (Issue #26 requirement: 500MB limit)
    memory_limit: str = "500MB"
    max_memory: str = "500MB"
    
    # Connection pool settings
    max_connections: int = 5
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    
    # Performance settings
    threads: int = field(default_factory=lambda: min(4, os.cpu_count() or 2))
    temp_directory: Optional[str] = None
    
    # VSS extension settings
    enable_vss: bool = True
    vss_metric: str = "cosine"
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16
    
    # Schema settings
    auto_create_schema: bool = True
    schema_files: list = field(default_factory=lambda: [
        "knowledge/schema/duckdb_simple_schema.sql"
    ])
    
    # Backup and maintenance settings
    auto_checkpoint: bool = True
    checkpoint_interval: int = 1000  # transactions
    wal_mode: bool = True
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Ensure database directory exists
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize memory settings
        if isinstance(self.memory_limit, str) and not self.memory_limit.upper().endswith(('MB', 'GB')):
            if self.memory_limit.isdigit():
                self.memory_limit = f"{self.memory_limit}MB"
        
        if isinstance(self.max_memory, str) and not self.max_memory.upper().endswith(('MB', 'GB')):
            if self.max_memory.isdigit():
                self.max_memory = f"{self.max_memory}MB"
        
        # Set temp directory if not specified
        if self.temp_directory is None:
            self.temp_directory = str(db_path.parent / "temp")
            Path(self.temp_directory).mkdir(parents=True, exist_ok=True)
        
        # Validate thread count
        self.threads = max(1, min(self.threads, 8))  # Between 1-8 threads
        
        # Validate connection pool settings
        self.max_connections = max(1, min(self.max_connections, 20))  # Between 1-20 connections
    
    def get_connection_string(self) -> str:
        """Get the DuckDB connection string with configuration parameters."""
        return self.database_path
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for DuckDB settings."""
        return {
            'memory_limit': self.memory_limit,
            'max_memory': self.max_memory,
            'threads': self.threads,
            'temp_directory': self.temp_directory,
            'enable_progress_bar': False,  # Disable for production
            'autoinstall_known_extensions': True,
            'autoload_known_extensions': True,
            'wal_autocheckpoint': self.checkpoint_interval if self.auto_checkpoint else 0,
        }
    
    def get_vss_config(self) -> Dict[str, Any]:
        """Get VSS-specific configuration."""
        return {
            'metric': self.vss_metric,
            'ef_construction': self.hnsw_ef_construction,
            'm': self.hnsw_m,
        }
    
    @classmethod
    def from_environment(cls, prefix: str = "RIF_DB_") -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        config_kwargs = {}
        
        # Map environment variables to config fields
        env_mapping = {
            f'{prefix}PATH': 'database_path',
            f'{prefix}MEMORY_LIMIT': 'memory_limit', 
            f'{prefix}MAX_MEMORY': 'max_memory',
            f'{prefix}MAX_CONNECTIONS': 'max_connections',
            f'{prefix}CONNECTION_TIMEOUT': 'connection_timeout',
            f'{prefix}IDLE_TIMEOUT': 'idle_timeout',
            f'{prefix}THREADS': 'threads',
            f'{prefix}TEMP_DIR': 'temp_directory',
            f'{prefix}ENABLE_VSS': 'enable_vss',
            f'{prefix}VSS_METRIC': 'vss_metric',
            f'{prefix}AUTO_CREATE_SCHEMA': 'auto_create_schema',
            f'{prefix}AUTO_CHECKPOINT': 'auto_checkpoint',
            f'{prefix}CHECKPOINT_INTERVAL': 'checkpoint_interval',
        }
        
        for env_var, config_field in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Type conversion based on field
                if config_field in ['max_connections', 'threads', 'checkpoint_interval', 'hnsw_ef_construction', 'hnsw_m']:
                    config_kwargs[config_field] = int(env_value)
                elif config_field in ['connection_timeout', 'idle_timeout']:
                    config_kwargs[config_field] = float(env_value)
                elif config_field in ['enable_vss', 'auto_create_schema', 'auto_checkpoint', 'wal_mode']:
                    config_kwargs[config_field] = env_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    config_kwargs[config_field] = env_value
        
        return cls(**config_kwargs)
    
    @classmethod
    def for_testing(cls, temp_dir: Optional[str] = None) -> 'DatabaseConfig':
        """Create configuration optimized for testing."""
        if temp_dir is None:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="rif_test_db_")
        
        test_db_path = os.path.join(temp_dir, "test_entities.duckdb")
        
        return cls(
            database_path=test_db_path,
            memory_limit="100MB",  # Smaller for tests
            max_memory="100MB",
            max_connections=2,     # Fewer connections for tests
            connection_timeout=5.0,
            idle_timeout=30.0,     # Shorter timeout for tests
            threads=2,
            temp_directory=os.path.join(temp_dir, "temp"),
            auto_checkpoint=False,  # Manual checkpoints in tests
            checkpoint_interval=100,
        )
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"DatabaseConfig(path={self.database_path}, "
                f"memory={self.memory_limit}, "
                f"connections={self.max_connections}, "
                f"vss_enabled={self.enable_vss})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"DatabaseConfig("
                f"database_path='{self.database_path}', "
                f"memory_limit='{self.memory_limit}', "
                f"max_connections={self.max_connections}, "
                f"enable_vss={self.enable_vss}, "
                f"threads={self.threads})")