"""
Enhanced Database Resilience Manager for Issue #150
Implements robust connection pooling, health monitoring, and graceful degradation
"""

import logging
import threading
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from typing import Dict, Any, Optional, Generator, List, Callable
from pathlib import Path
from enum import Enum
import weakref

import duckdb

from knowledge.database.database_config import DatabaseConfig


class DatabaseHealthState(Enum):
    """Database health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ConnectionState(Enum):
    """Connection states."""
    ACTIVE = "active"
    IDLE = "idle"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"


@dataclass
class ConnectionMetrics:
    """Metrics for a database connection."""
    connection_id: str
    created_at: float
    last_used: float
    use_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    state: ConnectionState = ConnectionState.IDLE
    thread_id: Optional[int] = None
    avg_response_time: float = 0.0
    total_response_time: float = 0.0


@dataclass 
class HealthMetrics:
    """System-wide database health metrics."""
    state: DatabaseHealthState = DatabaseHealthState.HEALTHY
    active_connections: int = 0
    failed_connections: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0
    circuit_breaker_open: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, block requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for database connections."""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    half_open_max_calls: int = 3


class DatabaseResilienceManager:
    """
    Enhanced database connection manager with resilience features.
    
    Features:
    - Advanced connection pooling with health monitoring
    - Circuit breaker pattern for fault tolerance
    - Graceful degradation with fallback mechanisms
    - Comprehensive metrics and monitoring
    - Automatic error recovery and retry logic
    - Connection state management and lifecycle
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None, 
                 fallback_mode_enabled: bool = True):
        self.config = config or DatabaseConfig()
        self.logger = logging.getLogger(__name__)
        self.fallback_mode_enabled = fallback_mode_enabled
        
        # Core connection management
        self._pool: Queue[ConnectionMetrics] = Queue(maxsize=self.config.max_connections)
        self._active_connections: Dict[str, ConnectionMetrics] = {}
        self._connection_counter = 0
        self._pool_lock = threading.RLock()
        self._shutdown = False
        
        # Health monitoring
        self._health_metrics = HealthMetrics()
        self._health_lock = threading.Lock()
        self._health_check_interval = 30.0  # 30 seconds
        self._last_health_check = 0.0
        self._start_time = time.time()
        
        # Circuit breaker
        self._circuit_breaker = CircuitBreaker()
        self._circuit_breaker_lock = threading.Lock()
        
        # Error tracking and recovery
        self._error_history: List[Dict[str, Any]] = []
        self._error_history_lock = threading.Lock()
        self._max_error_history = 1000
        
        # Fallback mechanisms
        self._fallback_data_cache: Dict[str, Any] = {}
        self._fallback_operations: Dict[str, Callable] = {}
        
        # Initialize the system
        self._initialize_resilience_system()
        
        # Register cleanup
        weakref.finalize(self, self._cleanup_all_resources)
    
    def _initialize_resilience_system(self):
        """Initialize the resilience management system."""
        self.logger.info("Initializing Database Resilience Manager")
        
        # Ensure database directory exists
        db_path = Path(self.config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup fallback operations
        self._setup_fallback_operations()
        
        # Create initial connection pool
        self._initialize_connection_pool()
        
        # Start background monitoring
        self._start_background_monitoring()
        
        self.logger.info(f"Resilience Manager initialized - Pool: {self._pool.qsize()} connections")
    
    def _setup_fallback_operations(self):
        """Setup fallback operations for when database is unavailable."""
        self._fallback_operations.update({
            'get_entity': self._fallback_get_entity,
            'search_entities': self._fallback_search_entities,
            'store_entity': self._fallback_store_entity,
            'get_database_stats': self._fallback_get_stats,
        })
    
    def _initialize_connection_pool(self):
        """Initialize the connection pool with health checks."""
        initial_size = min(2, self.config.max_connections)
        successful_connections = 0
        
        for i in range(initial_size):
            try:
                conn_metrics = self._create_connection_with_metrics()
                if conn_metrics and not self._pool.full():
                    self._pool.put(conn_metrics, block=False)
                    successful_connections += 1
            except Exception as e:
                self.logger.warning(f"Failed to create initial connection {i}: {e}")
                self._record_error("connection_creation", str(e))
        
        if successful_connections == 0:
            self._update_health_state(DatabaseHealthState.OFFLINE)
            self.logger.error("Failed to create any initial connections - entering offline mode")
        elif successful_connections < initial_size:
            self._update_health_state(DatabaseHealthState.DEGRADED)
            self.logger.warning(f"Only {successful_connections}/{initial_size} initial connections created")
    
    def _create_connection_with_metrics(self) -> Optional[ConnectionMetrics]:
        """Create a new database connection with comprehensive metrics."""
        connection_id = f"conn_{self._connection_counter}"
        self._connection_counter += 1
        start_time = time.time()
        
        try:
            # Create the connection
            conn = duckdb.connect(
                database=self.config.get_connection_string(),
                read_only=False
            )
            
            # Apply configuration
            self._apply_connection_settings(conn)
            
            # Load extensions
            if self.config.enable_vss:
                self._load_vss_extension(conn)
            
            # Initialize schema
            self._ensure_schema_initialized(conn)
            
            # Test connection health
            conn.execute("SELECT 1").fetchone()
            
            # Create metrics
            creation_time = time.time()
            metrics = ConnectionMetrics(
                connection_id=connection_id,
                created_at=creation_time,
                last_used=creation_time,
                state=ConnectionState.ACTIVE,
                thread_id=threading.get_ident(),
                avg_response_time=creation_time - start_time
            )
            
            # Store the actual connection in metrics
            metrics.connection = conn
            
            self.logger.debug(f"Created connection {connection_id} in {creation_time - start_time:.3f}s")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to create connection {connection_id}: {e}")
            self._record_error("connection_creation", str(e))
            self._circuit_breaker_record_failure()
            return None
    
    def _apply_connection_settings(self, conn: duckdb.DuckDBPyConnection):
        """Apply configuration settings with error resilience."""
        config_dict = self.config.get_config_dict()
        
        for setting, value in config_dict.items():
            try:
                if setting == 'wal_autocheckpoint' and value > 0:
                    try:
                        conn.execute(f"PRAGMA wal_autocheckpoint={value}")
                    except Exception:
                        try:
                            conn.execute(f"SET wal_autocheckpoint={value}")
                        except Exception as e:
                            self.logger.debug(f"WAL autocheckpoint setting failed: {e}")
                elif setting in ['memory_limit', 'max_memory'] and value:
                    conn.execute(f"SET {setting}='{value}'")
                elif setting == 'threads' and value:
                    conn.execute(f"SET {setting}={value}")
                elif setting == 'temp_directory' and value:
                    conn.execute(f"SET temp_directory='{value}'")
                elif setting in ['enable_progress_bar', 'autoinstall_known_extensions', 'autoload_known_extensions']:
                    conn.execute(f"SET {setting}={str(value).lower()}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to set {setting}={value}: {e}")
                self._record_error("configuration", f"Setting {setting}: {e}")
    
    def _load_vss_extension(self, conn: duckdb.DuckDBPyConnection):
        """Load VSS extension with error handling."""
        try:
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
            
            # Enable HNSW persistence
            try:
                conn.execute("SET hnsw_enable_experimental_persistence=true")
            except Exception as e:
                self.logger.debug(f"HNSW persistence setting failed: {e}")
            
            # Verify extension
            result = conn.execute("""
                SELECT extension_name, loaded, installed 
                FROM duckdb_extensions() 
                WHERE extension_name = 'vss'
            """).fetchone()
            
            if not (result and result[1] and result[2]):
                raise Exception("VSS extension verification failed")
                
        except Exception as e:
            self.logger.warning(f"VSS extension loading failed: {e}")
            self._record_error("vss_extension", str(e))
    
    def _ensure_schema_initialized(self, conn: duckdb.DuckDBPyConnection):
        """Ensure schema is initialized with error handling."""
        if not self.config.auto_create_schema:
            return
        
        try:
            for schema_file in self.config.schema_files:
                schema_path = Path(schema_file)
                if schema_path.exists():
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()
                    
                    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                    for statement in statements:
                        try:
                            conn.execute(statement)
                        except Exception as e:
                            self.logger.debug(f"Schema statement failed: {e}")
                            
        except Exception as e:
            self.logger.warning(f"Schema initialization failed: {e}")
            self._record_error("schema_initialization", str(e))
    
    @contextmanager
    def get_resilient_connection(self, timeout: Optional[float] = None, 
                                allow_fallback: bool = True) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """
        Get a database connection with resilience features.
        
        Args:
            timeout: Maximum time to wait for connection
            allow_fallback: Whether to use fallback mechanisms if database is unavailable
        
        Yields:
            Database connection or raises exception if unavailable and fallback disabled
        """
        if self._shutdown:
            raise RuntimeError("Database resilience manager is shutdown")
        
        # Check circuit breaker
        if not self._circuit_breaker_allow_request():
            if allow_fallback and self.fallback_mode_enabled:
                yield self._create_fallback_connection()
                return
            else:
                raise RuntimeError("Database circuit breaker is open - service unavailable")
        
        timeout = timeout or self.config.connection_timeout
        conn_metrics = None
        start_time = time.time()
        
        try:
            # Get connection from pool or create new one
            conn_metrics = self._acquire_connection(timeout)
            
            if not conn_metrics:
                if allow_fallback and self.fallback_mode_enabled:
                    self.logger.warning("No database connection available, using fallback mode")
                    yield self._create_fallback_connection()
                    return
                else:
                    raise RuntimeError("No database connection available and fallback disabled")
            
            # Test connection health before use
            if not self._test_connection_health(conn_metrics):
                self._mark_connection_failed(conn_metrics)
                if allow_fallback and self.fallback_mode_enabled:
                    yield self._create_fallback_connection()
                    return
                else:
                    raise RuntimeError("Database connection failed health check")
            
            # Update metrics
            conn_metrics.last_used = time.time()
            conn_metrics.use_count += 1
            conn_metrics.state = ConnectionState.ACTIVE
            
            # Yield the connection
            query_start = time.time()
            yield conn_metrics.connection
            query_end = time.time()
            
            # Record successful operation
            response_time = query_end - query_start
            self._update_connection_metrics(conn_metrics, response_time)
            self._circuit_breaker_record_success()
            self._update_health_metrics(response_time, success=True)
            
        except Exception as e:
            self.logger.error(f"Database operation failed: {e}")
            
            if conn_metrics:
                self._record_connection_error(conn_metrics, str(e))
            
            self._record_error("database_operation", str(e))
            self._circuit_breaker_record_failure()
            self._update_health_metrics(time.time() - start_time, success=False)
            
            if allow_fallback and self.fallback_mode_enabled:
                self.logger.info("Database operation failed, attempting fallback")
                yield self._create_fallback_connection()
                return
            else:
                raise
        
        finally:
            if conn_metrics:
                self._return_connection(conn_metrics)
    
    def _acquire_connection(self, timeout: float) -> Optional[ConnectionMetrics]:
        """Acquire a connection from the pool or create a new one."""
        try:
            # Try to get from pool first
            try:
                return self._pool.get(timeout=min(timeout, 5.0))
            except Empty:
                pass
            
            # Pool is empty, try to create new connection if under limit
            with self._pool_lock:
                if len(self._active_connections) < self.config.max_connections:
                    return self._create_connection_with_metrics()
            
            # Wait for a connection to become available
            return self._pool.get(timeout=timeout - 5.0 if timeout > 5.0 else timeout)
            
        except Exception as e:
            self.logger.warning(f"Failed to acquire connection: {e}")
            return None
    
    def _test_connection_health(self, conn_metrics: ConnectionMetrics) -> bool:
        """Test if a connection is healthy and responsive."""
        try:
            start_time = time.time()
            conn_metrics.connection.execute("SELECT 1").fetchone()
            response_time = time.time() - start_time
            
            # Check if response time is acceptable (< 5 seconds)
            if response_time > 5.0:
                self.logger.warning(f"Connection {conn_metrics.connection_id} slow: {response_time:.2f}s")
                return False
            
            # Check connection age
            age = time.time() - conn_metrics.created_at
            if age > 3600:  # 1 hour
                self.logger.debug(f"Connection {conn_metrics.connection_id} too old: {age:.0f}s")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Connection health check failed for {conn_metrics.connection_id}: {e}")
            return False
    
    def _update_connection_metrics(self, conn_metrics: ConnectionMetrics, response_time: float):
        """Update metrics for a connection after successful operation."""
        conn_metrics.total_response_time += response_time
        if conn_metrics.use_count > 0:
            conn_metrics.avg_response_time = conn_metrics.total_response_time / conn_metrics.use_count
        conn_metrics.state = ConnectionState.IDLE
    
    def _record_connection_error(self, conn_metrics: ConnectionMetrics, error: str):
        """Record an error for a specific connection."""
        conn_metrics.error_count += 1
        conn_metrics.last_error = error
        conn_metrics.state = ConnectionState.UNHEALTHY
    
    def _mark_connection_failed(self, conn_metrics: ConnectionMetrics):
        """Mark a connection as permanently failed."""
        conn_metrics.state = ConnectionState.FAILED
        try:
            if hasattr(conn_metrics, 'connection') and conn_metrics.connection:
                conn_metrics.connection.close()
        except Exception:
            pass
    
    def _return_connection(self, conn_metrics: ConnectionMetrics):
        """Return a connection to the pool or close it."""
        with self._pool_lock:
            conn_id = conn_metrics.connection_id
            self._active_connections.pop(conn_id, None)
            
            if conn_metrics.state == ConnectionState.FAILED:
                self._close_connection(conn_metrics)
                return
            
            try:
                self._pool.put(conn_metrics, block=False)
            except Full:
                self._close_connection(conn_metrics)
    
    def _close_connection(self, conn_metrics: ConnectionMetrics):
        """Safely close a database connection."""
        try:
            if hasattr(conn_metrics, 'connection') and conn_metrics.connection:
                conn_metrics.connection.close()
                self.logger.debug(f"Closed connection {conn_metrics.connection_id}")
        except Exception as e:
            self.logger.warning(f"Error closing connection {conn_metrics.connection_id}: {e}")
    
    # Circuit Breaker Implementation
    def _circuit_breaker_allow_request(self) -> bool:
        """Check if circuit breaker allows the request."""
        with self._circuit_breaker_lock:
            now = time.time()
            
            if self._circuit_breaker.state == CircuitBreakerState.OPEN:
                if now - self._circuit_breaker.last_failure_time > self._circuit_breaker.recovery_timeout:
                    self._circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                    self._circuit_breaker.success_count = 0
                    self.logger.info("Circuit breaker entering HALF_OPEN state")
                    return True
                return False
            
            elif self._circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                return self._circuit_breaker.success_count < self._circuit_breaker.half_open_max_calls
            
            return True  # CLOSED state
    
    def _circuit_breaker_record_success(self):
        """Record a successful operation for circuit breaker."""
        with self._circuit_breaker_lock:
            if self._circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                self._circuit_breaker.success_count += 1
                if self._circuit_breaker.success_count >= self._circuit_breaker.half_open_max_calls:
                    self._circuit_breaker.state = CircuitBreakerState.CLOSED
                    self._circuit_breaker.failure_count = 0
                    self.logger.info("Circuit breaker returning to CLOSED state")
            elif self._circuit_breaker.state == CircuitBreakerState.CLOSED:
                self._circuit_breaker.failure_count = max(0, self._circuit_breaker.failure_count - 1)
    
    def _circuit_breaker_record_failure(self):
        """Record a failed operation for circuit breaker."""
        with self._circuit_breaker_lock:
            self._circuit_breaker.failure_count += 1
            self._circuit_breaker.last_failure_time = time.time()
            
            if self._circuit_breaker.failure_count >= self._circuit_breaker.failure_threshold:
                if self._circuit_breaker.state != CircuitBreakerState.OPEN:
                    self._circuit_breaker.state = CircuitBreakerState.OPEN
                    self.logger.warning("Circuit breaker OPENED due to excessive failures")
                    self._update_health_state(DatabaseHealthState.CRITICAL)
    
    # Health Monitoring
    def _update_health_state(self, new_state: DatabaseHealthState):
        """Update the overall health state."""
        with self._health_lock:
            if self._health_metrics.state != new_state:
                old_state = self._health_metrics.state
                self._health_metrics.state = new_state
                self.logger.info(f"Database health state changed: {old_state.value} -> {new_state.value}")
    
    def _update_health_metrics(self, response_time: float, success: bool):
        """Update health metrics after an operation."""
        with self._health_lock:
            self._health_metrics.total_queries += 1
            
            if success:
                # Update response time average
                total_time = (self._health_metrics.avg_response_time * 
                            (self._health_metrics.total_queries - 1) + response_time)
                self._health_metrics.avg_response_time = total_time / self._health_metrics.total_queries
            else:
                self._health_metrics.failed_queries += 1
            
            # Calculate error rate
            if self._health_metrics.total_queries > 0:
                self._health_metrics.error_rate = (
                    self._health_metrics.failed_queries / self._health_metrics.total_queries
                )
            
            # Update health state based on error rate
            if self._health_metrics.error_rate > 0.5:  # More than 50% errors
                self._update_health_state(DatabaseHealthState.CRITICAL)
            elif self._health_metrics.error_rate > 0.2:  # More than 20% errors
                self._update_health_state(DatabaseHealthState.DEGRADED)
            elif self._circuit_breaker.state == CircuitBreakerState.CLOSED:
                self._update_health_state(DatabaseHealthState.HEALTHY)
    
    def _start_background_monitoring(self):
        """Start background health monitoring thread."""
        def monitor():
            while not self._shutdown:
                try:
                    self._perform_health_check()
                    time.sleep(self._health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(5)  # Short sleep on error
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("Started background health monitoring")
    
    def _perform_health_check(self):
        """Perform comprehensive health check."""
        now = time.time()
        
        with self._health_lock:
            self._health_metrics.last_health_check = now
            self._health_metrics.uptime = now - self._start_time
            self._health_metrics.active_connections = len(self._active_connections)
            self._health_metrics.circuit_breaker_open = (
                self._circuit_breaker.state == CircuitBreakerState.OPEN
            )
        
        # Test a connection if available
        try:
            with self.get_resilient_connection(timeout=5.0, allow_fallback=False):
                pass  # Connection test successful
        except Exception as e:
            self.logger.warning(f"Health check connection test failed: {e}")
    
    # Error Management
    def _record_error(self, error_type: str, error_message: str):
        """Record an error in the error history."""
        with self._error_history_lock:
            error_record = {
                'timestamp': time.time(),
                'error_type': error_type,
                'error_message': error_message,
                'circuit_breaker_state': self._circuit_breaker.state.value,
                'health_state': self._health_metrics.state.value
            }
            
            self._error_history.append(error_record)
            
            # Limit history size
            if len(self._error_history) > self._max_error_history:
                self._error_history.pop(0)
    
    # Fallback Mechanisms
    def _create_fallback_connection(self):
        """Create a fallback connection object for offline operations."""
        return FallbackConnection(self._fallback_operations, self._fallback_data_cache)
    
    def _fallback_get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Fallback implementation for getting an entity."""
        return self._fallback_data_cache.get(f"entity_{entity_id}")
    
    def _fallback_search_entities(self, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation for searching entities."""
        # Return cached results if available
        cache_key = f"search_{hash(str(sorted(kwargs.items())))}"
        return self._fallback_data_cache.get(cache_key, [])
    
    def _fallback_store_entity(self, entity_data: Dict[str, Any]) -> str:
        """Fallback implementation for storing an entity."""
        # Generate a temporary ID and cache the entity
        entity_id = f"temp_{int(time.time())}_{hash(str(entity_data))}"
        self._fallback_data_cache[f"entity_{entity_id}"] = entity_data
        return entity_id
    
    def _fallback_get_stats(self) -> Dict[str, Any]:
        """Fallback implementation for getting database statistics."""
        return {
            'mode': 'fallback',
            'cached_entities': len([k for k in self._fallback_data_cache.keys() if k.startswith('entity_')]),
            'health_state': self._health_metrics.state.value,
            'circuit_breaker_state': self._circuit_breaker.state.value
        }
    
    # Public API
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics."""
        with self._health_lock:
            return {
                'health_state': self._health_metrics.state.value,
                'active_connections': self._health_metrics.active_connections,
                'failed_connections': self._health_metrics.failed_connections,
                'total_queries': self._health_metrics.total_queries,
                'failed_queries': self._health_metrics.failed_queries,
                'error_rate': round(self._health_metrics.error_rate, 4),
                'avg_response_time': round(self._health_metrics.avg_response_time, 4),
                'uptime': round(self._health_metrics.uptime, 2),
                'circuit_breaker': {
                    'state': self._circuit_breaker.state.value,
                    'failure_count': self._circuit_breaker.failure_count,
                    'success_count': self._circuit_breaker.success_count
                },
                'pool_stats': {
                    'pool_size': self._pool.qsize(),
                    'active_connections': len(self._active_connections),
                    'max_connections': self.config.max_connections
                }
            }
    
    def get_error_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error history."""
        with self._error_history_lock:
            return self._error_history[-limit:] if limit > 0 else self._error_history.copy()
    
    def force_circuit_breaker_reset(self):
        """Force reset of circuit breaker (admin operation)."""
        with self._circuit_breaker_lock:
            self._circuit_breaker.state = CircuitBreakerState.CLOSED
            self._circuit_breaker.failure_count = 0
            self._circuit_breaker.success_count = 0
            self.logger.info("Circuit breaker forcibly reset to CLOSED state")
    
    def _cleanup_all_resources(self):
        """Clean up all resources during shutdown."""
        self.logger.info("Shutting down Database Resilience Manager...")
        self._shutdown = True
        
        # Close all connections
        while not self._pool.empty():
            try:
                conn_metrics = self._pool.get_nowait()
                self._close_connection(conn_metrics)
            except Empty:
                break
        
        with self._pool_lock:
            for conn_metrics in list(self._active_connections.values()):
                self._close_connection(conn_metrics)
            self._active_connections.clear()
        
        self.logger.info("Database Resilience Manager shutdown complete")
    
    def shutdown(self):
        """Explicitly shutdown the resilience manager."""
        self._cleanup_all_resources()


class FallbackConnection:
    """Fallback connection that provides limited functionality when database is offline."""
    
    def __init__(self, fallback_operations: Dict[str, Callable], cache: Dict[str, Any]):
        self.fallback_operations = fallback_operations
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def execute(self, query: str, parameters=None):
        """Mock execute method that logs the query but doesn't execute it."""
        self.logger.warning(f"Database offline - query logged but not executed: {query}")
        
        # Return a mock result for SELECT 1 health checks
        if query.strip().upper() == "SELECT 1":
            return MockResult([1])
        
        return MockResult([])
    
    def fetchone(self):
        """Mock fetchone method."""
        return None
    
    def fetchall(self):
        """Mock fetchall method.""" 
        return []


class MockResult:
    """Mock result object for fallback operations."""
    
    def __init__(self, data):
        self.data = data
        self.rowcount = len(data) if isinstance(data, list) else 1
    
    def fetchone(self):
        return self.data[0] if self.data else None
    
    def fetchall(self):
        return self.data if isinstance(self.data, list) else [self.data]