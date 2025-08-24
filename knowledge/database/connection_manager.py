"""
DuckDB connection manager with pooling and memory management.
Issue #26: Set up DuckDB as embedded database with vector search
"""

import logging
import threading
import time
import weakref
from contextlib import contextmanager
from queue import Queue, Empty, Full
from typing import Dict, Any, Optional, Generator, List
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .database_config import DatabaseConfig


@dataclass
class ConnectionInfo:
    """Information about a database connection."""
    connection: duckdb.DuckDBPyConnection
    created_at: float
    last_used: float
    in_use: bool = False
    use_count: int = 0
    thread_id: Optional[int] = None


class DuckDBConnectionManager:
    """
    Thread-safe connection manager for DuckDB with pooling and memory limits.
    
    Features:
    - Connection pooling with configurable limits
    - Memory management (500MB limit as per Issue #26)
    - Automatic VSS extension loading
    - Schema initialization
    - Connection health monitoring
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        
        # Connection pool management
        self._pool: Queue[ConnectionInfo] = Queue(maxsize=self.config.max_connections)
        self._active_connections: Dict[int, ConnectionInfo] = {}
        self._connection_counter = 0
        self._pool_lock = threading.RLock()
        self._shutdown = False
        
        # Schema initialization state
        self._schema_initialized = False
        self._schema_lock = threading.Lock()
        
        # Health monitoring
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # 1 minute
        
        # Initialize the connection manager
        self._initialize()
        
        # Register cleanup on shutdown
        weakref.finalize(self, self._cleanup_all_connections)
    
    def _initialize(self):
        """Initialize the connection manager."""
        self.logger.info(f"Initializing DuckDB connection manager: {self.config}")
        
        # Ensure database directory exists
        db_path = Path(self.config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create initial connections to fill the pool
        initial_connections = min(2, self.config.max_connections)
        for _ in range(initial_connections):
            try:
                conn_info = self._create_connection()
                if not self._pool.full():
                    self._pool.put(conn_info, block=False)
            except Exception as e:
                self.logger.warning(f"Failed to create initial connection: {e}")
        
        self.logger.info(f"Connection manager initialized with {self._pool.qsize()} connections")
    
    def _create_connection(self) -> ConnectionInfo:
        """Create a new database connection with configuration applied."""
        try:
            # Create the connection
            conn = duckdb.connect(
                database=self.config.get_connection_string(),
                read_only=False
            )
            
            # Apply configuration settings
            self._apply_connection_settings(conn)
            
            # Load VSS extension if enabled
            if self.config.enable_vss:
                self._load_vss_extension(conn)
            
            # Initialize schema if needed
            self._ensure_schema_initialized(conn)
            
            # Create connection info
            current_time = time.time()
            conn_info = ConnectionInfo(
                connection=conn,
                created_at=current_time,
                last_used=current_time,
                thread_id=threading.get_ident()
            )
            
            self._connection_counter += 1
            self.logger.debug(f"Created new connection #{self._connection_counter}")
            
            return conn_info
            
        except Exception as e:
            self.logger.error(f"Failed to create database connection: {e}")
            raise
    
    def _apply_connection_settings(self, conn: duckdb.DuckDBPyConnection):
        """Apply configuration settings to a connection."""
        config_dict = self.config.get_config_dict()
        
        for setting, value in config_dict.items():
            try:
                if setting == 'wal_autocheckpoint' and value > 0:
                    conn.execute(f"PRAGMA wal_autocheckpoint={value}")
                elif setting in ['memory_limit', 'max_memory'] and value:
                    conn.execute(f"SET {setting}='{value}'")
                elif setting in ['threads'] and value:
                    conn.execute(f"SET {setting}={value}")
                elif setting in ['temp_directory'] and value:
                    conn.execute(f"SET temp_directory='{value}'")
                elif setting in ['enable_progress_bar']:
                    conn.execute(f"SET {setting}={str(value).lower()}")
                elif setting in ['autoinstall_known_extensions', 'autoload_known_extensions']:
                    conn.execute(f"SET {setting}={str(value).lower()}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to set {setting}={value}: {e}")
        
        # Note: HNSW persistence setting is handled after VSS extension is loaded
        
        self.logger.debug(f"Applied configuration settings to connection")
    
    def _load_vss_extension(self, conn: duckdb.DuckDBPyConnection):
        """Load and configure the VSS extension for vector similarity search."""
        try:
            # Install VSS extension if not already installed
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
            
            # Enable experimental HNSW persistence after VSS is loaded
            try:
                conn.execute("SET hnsw_enable_experimental_persistence=true")
                self.logger.debug("HNSW experimental persistence enabled")
            except Exception as e:
                self.logger.warning(f"Failed to enable HNSW persistence: {e}")
            
            # Verify VSS extension is loaded
            result = conn.execute("""
                SELECT extension_name, loaded, installed 
                FROM duckdb_extensions() 
                WHERE extension_name = 'vss'
            """).fetchone()
            
            if result and result[1] and result[2]:  # loaded and installed
                self.logger.debug("VSS extension loaded successfully")
            else:
                self.logger.warning("VSS extension may not be properly loaded")
                
        except Exception as e:
            self.logger.error(f"Failed to load VSS extension: {e}")
            # Continue without VSS - the system should still work for basic operations
            
    def _ensure_schema_initialized(self, conn: duckdb.DuckDBPyConnection):
        """Ensure the database schema is initialized (thread-safe)."""
        if self._schema_initialized:
            return
            
        with self._schema_lock:
            if self._schema_initialized:  # Double-check pattern
                return
                
            if not self.config.auto_create_schema:
                self._schema_initialized = True
                return
            
            self.logger.info("Initializing database schema...")
            
            try:
                # Execute schema files in order
                for schema_file in self.config.schema_files:
                    schema_path = Path(schema_file)
                    if schema_path.exists():
                        with open(schema_path, 'r') as f:
                            schema_sql = f.read()
                        
                        # Execute schema SQL (split by semicolon for multiple statements)
                        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                        for statement in statements:
                            try:
                                conn.execute(statement)
                            except Exception as e:
                                # Log but continue with other statements
                                self.logger.warning(f"Schema statement failed: {e}")
                        
                        self.logger.debug(f"Executed schema file: {schema_file}")
                    else:
                        self.logger.warning(f"Schema file not found: {schema_file}")
                
                # Verify basic tables exist
                tables_query = """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'main' 
                    AND table_name IN ('entities', 'relationships', 'agent_memory')
                """
                tables = conn.execute(tables_query).fetchall()
                
                if len(tables) >= 3:  # All core tables exist
                    self.logger.info("Database schema initialized successfully")
                    self._schema_initialized = True
                else:
                    self.logger.warning(f"Schema initialization incomplete. Found tables: {[t[0] for t in tables]}")
                    # Mark as initialized anyway to prevent infinite retries
                    self._schema_initialized = True
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize schema: {e}")
                # Mark as initialized to prevent infinite retries
                self._schema_initialized = True
                raise
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Get a connection from the pool (context manager).
        
        Args:
            timeout: Maximum time to wait for a connection (uses config default if None)
            
        Yields:
            DuckDB connection
            
        Example:
            with manager.get_connection() as conn:
                result = conn.execute("SELECT * FROM entities LIMIT 10").fetchall()
        """
        if self._shutdown:
            raise RuntimeError("Connection manager is shutdown")
            
        timeout = timeout or self.config.connection_timeout
        conn_info = None
        
        try:
            # Try to get connection from pool
            try:
                conn_info = self._pool.get(timeout=timeout)
                self.logger.debug("Got connection from pool")
            except Empty:
                # Pool is empty, try to create a new connection
                if len(self._active_connections) < self.config.max_connections:
                    conn_info = self._create_connection()
                    self.logger.debug("Created new connection (pool empty)")
                else:
                    raise RuntimeError(f"Maximum connections ({self.config.max_connections}) reached")
            
            # Mark connection as in use
            with self._pool_lock:
                conn_info.in_use = True
                conn_info.last_used = time.time()
                conn_info.use_count += 1
                conn_info.thread_id = threading.get_ident()
                conn_id = id(conn_info.connection)
                self._active_connections[conn_id] = conn_info
            
            # Yield the connection
            yield conn_info.connection
            
        except Exception as e:
            self.logger.error(f"Error getting database connection: {e}")
            raise
            
        finally:
            # Return connection to pool or clean up
            if conn_info:
                self._return_connection(conn_info)
    
    def _return_connection(self, conn_info: ConnectionInfo):
        """Return a connection to the pool or close it if pool is full."""
        with self._pool_lock:
            conn_info.in_use = False
            conn_info.last_used = time.time()
            conn_id = id(conn_info.connection)
            self._active_connections.pop(conn_id, None)
            
            # Check if connection is still healthy
            if self._is_connection_healthy(conn_info):
                try:
                    # Try to return to pool
                    self._pool.put(conn_info, block=False)
                    self.logger.debug("Returned connection to pool")
                except Full:
                    # Pool is full, close the connection
                    self._close_connection(conn_info)
                    self.logger.debug("Pool full, closed connection")
            else:
                # Connection is unhealthy, close it
                self._close_connection(conn_info)
                self.logger.debug("Closed unhealthy connection")
    
    def _is_connection_healthy(self, conn_info: ConnectionInfo) -> bool:
        """Check if a connection is still healthy and usable."""
        try:
            # Simple health check query
            conn_info.connection.execute("SELECT 1").fetchone()
            
            # Check if connection is too old
            max_age = 3600  # 1 hour
            if time.time() - conn_info.created_at > max_age:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Connection health check failed: {e}")
            return False
    
    def _close_connection(self, conn_info: ConnectionInfo):
        """Close a database connection safely."""
        try:
            if conn_info.connection:
                conn_info.connection.close()
                self.logger.debug("Connection closed")
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool."""
        with self._pool_lock:
            return {
                'pool_size': self._pool.qsize(),
                'active_connections': len(self._active_connections),
                'max_connections': self.config.max_connections,
                'total_created': self._connection_counter,
                'schema_initialized': self._schema_initialized,
                'config': str(self.config)
            }
    
    def cleanup_idle_connections(self):
        """Clean up idle connections that have exceeded the idle timeout."""
        if time.time() - self._last_cleanup < self._cleanup_interval:
            return  # Don't cleanup too frequently
        
        self._last_cleanup = time.time()
        idle_threshold = time.time() - self.config.idle_timeout
        
        # Get all connections from pool and check if they're idle
        connections_to_close = []
        connections_to_keep = []
        
        while not self._pool.empty():
            try:
                conn_info = self._pool.get_nowait()
                if conn_info.last_used < idle_threshold:
                    connections_to_close.append(conn_info)
                else:
                    connections_to_keep.append(conn_info)
            except Empty:
                break
        
        # Close idle connections
        for conn_info in connections_to_close:
            self._close_connection(conn_info)
        
        # Return non-idle connections to pool
        for conn_info in connections_to_keep:
            try:
                self._pool.put(conn_info, block=False)
            except Full:
                # Pool is full, close excess connections
                self._close_connection(conn_info)
        
        if connections_to_close:
            self.logger.info(f"Cleaned up {len(connections_to_close)} idle connections")
    
    def _cleanup_all_connections(self):
        """Clean up all connections (called during shutdown)."""
        self.logger.info("Shutting down connection manager...")
        self._shutdown = True
        
        # Close all connections in pool
        while not self._pool.empty():
            try:
                conn_info = self._pool.get_nowait()
                self._close_connection(conn_info)
            except Empty:
                break
        
        # Close all active connections
        with self._pool_lock:
            for conn_info in list(self._active_connections.values()):
                self._close_connection(conn_info)
            self._active_connections.clear()
        
        self.logger.info("Connection manager shutdown complete")
    
    def shutdown(self):
        """Explicitly shutdown the connection manager."""
        self._cleanup_all_connections()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()