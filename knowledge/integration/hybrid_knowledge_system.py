"""
HybridKnowledgeSystem - Master Coordination Controller

This is the main integration controller that orchestrates all four pipeline components:
- Entity Extraction (Issue #30)
- Relationship Detection (Issue #31) 
- Vector Embeddings (Issue #32)
- Query Planning (Issue #33)

Implements the Master Coordination Plan from Issue #40 with:
- Resource management and coordination
- Component health monitoring
- Unified system state management
- Error recovery and fallback strategies
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

# Component imports
from ..extraction.entity_extractor import EntityExtractor
from ..relationships.relationship_detector import RelationshipDetector
from ..embeddings.embedding_pipeline import EmbeddingPipeline
from ..query import QueryPlanner
from ..database.connection_manager import DuckDBConnectionManager
from .system_monitor import SystemMonitor
from .integration_controller import IntegrationController


@dataclass 
class SystemStatus:
    """Represents the current system state and health."""
    healthy: bool
    components_ready: Dict[str, bool]
    resource_usage: Dict[str, Any]
    last_updated: float
    error_count: int
    warnings: List[str]


class HybridKnowledgeSystem:
    """
    Master coordination controller for the hybrid knowledge pipeline.
    
    Coordinates all four core components with:
    - Resource management within 2GB memory budget
    - Performance monitoring and optimization 
    - Component health tracking and recovery
    - Unified API for all knowledge operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the hybrid knowledge system with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Resource limits from master plan
        self.memory_limit_mb = config.get('memory_limit_mb', 2048)
        self.cpu_cores = config.get('cpu_cores', 4)
        
        # System state
        self._initialized = False
        self._components = {}
        self._lock = threading.RLock()
        
        # Initialize monitoring and coordination
        self.system_monitor = SystemMonitor(config)
        self.integration_controller = IntegrationController(config)
        
        # Performance tracking
        self.metrics = {
            'queries_processed': 0,
            'entities_extracted': 0,
            'relationships_detected': 0,
            'embeddings_generated': 0,
            'cache_hits': 0,
            'error_count': 0,
            'avg_query_time_ms': 0.0
        }
        
        self.logger.info(f"HybridKnowledgeSystem initialized with {self.memory_limit_mb}MB memory limit, {self.cpu_cores} CPU cores")
    
    def initialize(self, database_path: Optional[str] = None) -> bool:
        """
        Initialize all pipeline components in the correct order.
        
        Follows the master coordination plan:
        1. Database connection and schema validation
        2. Entity extraction system (Issue #30)
        3. Parallel initialization of relationships and embeddings (Issues #31, #32)
        4. Query planning system (Issue #33)
        5. System validation and health checks
        
        Args:
            database_path: Path to DuckDB database file
            
        Returns:
            bool: True if initialization successful
        """
        with self._lock:
            if self._initialized:
                self.logger.warning("System already initialized")
                return True
            
            try:
                self.logger.info("Initializing Hybrid Knowledge System...")
                start_time = time.time()
                
                # Step 1: Database and resource setup
                self._initialize_database(database_path)
                
                # Step 2: Initialize monitoring
                self.system_monitor.start_monitoring()
                
                # Step 3: Initialize components in dependency order
                self._initialize_entity_extraction()
                self._initialize_parallel_components()
                self._initialize_query_planning()
                
                # Step 4: System validation
                if not self._validate_system_health():
                    raise RuntimeError("System health validation failed")
                
                # Step 5: Final setup
                self._initialized = True
                init_time = time.time() - start_time
                
                self.logger.info(f"Hybrid Knowledge System initialized successfully in {init_time:.2f}s")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize system: {e}")
                self._cleanup_partial_initialization()
                return False
    
    def _initialize_database(self, database_path: Optional[str]):
        """Initialize database connection and validate schema."""
        self.logger.info("Initializing database connection...")
        
        db_path = database_path or self.config.get('database_path', 'knowledge/database/entities.duckdb')
        self.db_manager = ConnectionManager(db_path)
        
        # Validate schema exists and is complete
        if not self.db_manager.validate_schema():
            raise RuntimeError("Database schema validation failed")
        
        self._components['database'] = self.db_manager
        self.logger.info("Database connection established and schema validated")
    
    def _initialize_entity_extraction(self):
        """Initialize entity extraction system (Issue #30 foundation)."""
        self.logger.info("Initializing entity extraction system...")
        
        try:
            self.entity_extractor = EntityExtractor(
                database=self.db_manager,
                config={
                    'memory_limit_mb': 200,  # From master plan allocation
                    'cache_size': 100,
                    'supported_languages': ['javascript', 'python', 'go', 'rust']
                }
            )
            
            # Validate component is ready
            if not self.entity_extractor.is_healthy():
                raise RuntimeError("Entity extractor failed health check")
                
            self._components['entity_extraction'] = self.entity_extractor
            self.logger.info("Entity extraction system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize entity extraction: {e}")
            raise
    
    def _initialize_parallel_components(self):
        """Initialize relationship detection and embeddings in parallel (Issues #31, #32)."""
        self.logger.info("Initializing relationship detection and embeddings in parallel...")
        
        import concurrent.futures
        
        def init_relationships():
            """Initialize relationship detection system."""
            try:
                self.relationship_detector = RelationshipDetector(
                    entity_extractor=self.entity_extractor,
                    database=self.db_manager,
                    config={
                        'memory_limit_mb': 300,  # From master plan allocation
                        'cpu_cores': 2,
                        'supported_languages': ['javascript', 'python', 'go', 'rust']
                    }
                )
                
                if not self.relationship_detector.is_healthy():
                    raise RuntimeError("Relationship detector failed health check")
                
                return ('relationships', self.relationship_detector)
                
            except Exception as e:
                self.logger.error(f"Failed to initialize relationship detection: {e}")
                raise
        
        def init_embeddings():
            """Initialize embedding generation system.""" 
            try:
                self.embedding_pipeline = EmbeddingPipeline(
                    entity_extractor=self.entity_extractor,
                    database=self.db_manager,
                    config={
                        'memory_limit_mb': 400,  # From master plan allocation
                        'model_type': 'tfidf',
                        'embedding_dim': 384,
                        'batch_size': 100
                    }
                )
                
                if not self.embedding_pipeline.is_healthy():
                    raise RuntimeError("Embedding pipeline failed health check")
                
                return ('embeddings', self.embedding_pipeline)
                
            except Exception as e:
                self.logger.error(f"Failed to initialize embeddings: {e}")
                raise
        
        # Execute parallel initialization
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(init_relationships),
                executor.submit(init_embeddings)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    component_name, component = future.result()
                    self._components[component_name] = component
                    self.logger.info(f"Parallel component '{component_name}' initialized successfully")
                except Exception as e:
                    self.logger.error(f"Parallel component initialization failed: {e}")
                    raise
        
        self.logger.info("Parallel components initialized successfully")
    
    def _initialize_query_planning(self):
        """Initialize query planning system (Issue #33 integration)."""
        self.logger.info("Initializing query planning system...")
        
        try:
            self.query_planner = QueryPlanner(
                entity_extractor=self.entity_extractor,
                relationship_detector=self.relationship_detector,
                embedding_pipeline=self.embedding_pipeline,
                database=self.db_manager,
                config={
                    'memory_limit_mb': 400,  # From master plan allocation
                    'cache_size': 1000,
                    'performance_mode': self.config.get('performance_mode', 'BALANCED'),
                    'max_latency_ms': 100  # P95 target for simple queries
                }
            )
            
            if not self.query_planner.is_healthy():
                raise RuntimeError("Query planner failed health check")
                
            self._components['query_planning'] = self.query_planner
            self.logger.info("Query planning system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize query planning: {e}")
            raise
    
    def _validate_system_health(self) -> bool:
        """Validate that all components are healthy and integrated properly."""
        self.logger.info("Validating system health...")
        
        # Check all required components are present
        required_components = ['database', 'entity_extraction', 'relationships', 'embeddings', 'query_planning']
        for component in required_components:
            if component not in self._components:
                self.logger.error(f"Required component '{component}' not initialized")
                return False
        
        # Check component health
        unhealthy_components = []
        for name, component in self._components.items():
            if hasattr(component, 'is_healthy') and not component.is_healthy():
                unhealthy_components.append(name)
        
        if unhealthy_components:
            self.logger.error(f"Unhealthy components: {unhealthy_components}")
            return False
        
        # Check resource usage
        memory_usage = self.system_monitor.get_memory_usage_mb()
        if memory_usage > self.memory_limit_mb * 0.95:  # 95% threshold
            self.logger.error(f"Memory usage too high: {memory_usage}MB > {self.memory_limit_mb * 0.95}MB")
            return False
        
        # Test basic integration
        try:
            self._test_basic_integration()
        except Exception as e:
            self.logger.error(f"Integration test failed: {e}")
            return False
        
        self.logger.info("System health validation successful")
        return True
    
    def _test_basic_integration(self):
        """Test basic integration between components."""
        # Create a small test file for validation
        test_content = '''
def test_function():
    """Test function for integration validation."""
    return "test"
        '''
        
        with self._temporary_file(test_content, suffix='.py') as test_file:
            # Test entity extraction
            entities = self.entity_extractor.extract_from_file(test_file)
            if not entities:
                raise RuntimeError("Entity extraction test failed - no entities found")
            
            # Test relationship detection
            relationships = self.relationship_detector.detect_from_file(test_file)
            # Note: relationships may be empty for simple test case
            
            # Test query processing
            results = self.query_planner.process_query("find test_function")
            if not results or len(results.results) == 0:
                raise RuntimeError("Query processing test failed - no results")
        
        self.logger.info("Basic integration test successful")
    
    @contextmanager
    def _temporary_file(self, content: str, suffix: str = '.py'):
        """Create a temporary file for testing."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            yield temp_path
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status and health information."""
        with self._lock:
            if not self._initialized:
                return SystemStatus(
                    healthy=False,
                    components_ready={},
                    resource_usage={},
                    last_updated=time.time(),
                    error_count=0,
                    warnings=["System not initialized"]
                )
            
            # Check component health
            components_ready = {}
            for name, component in self._components.items():
                if hasattr(component, 'is_healthy'):
                    components_ready[name] = component.is_healthy()
                else:
                    components_ready[name] = True
            
            # Get resource usage
            resource_usage = {
                'memory_mb': self.system_monitor.get_memory_usage_mb(),
                'memory_limit_mb': self.memory_limit_mb,
                'cpu_usage_percent': self.system_monitor.get_cpu_usage_percent(),
                'query_cache_size': getattr(self.query_planner, 'cache_size', 0),
                'database_connections': self.db_manager.active_connections if hasattr(self.db_manager, 'active_connections') else 1
            }
            
            # Overall health
            healthy = all(components_ready.values()) and resource_usage['memory_mb'] < self.memory_limit_mb * 0.95
            
            return SystemStatus(
                healthy=healthy,
                components_ready=components_ready,
                resource_usage=resource_usage,
                last_updated=time.time(),
                error_count=self.metrics['error_count'],
                warnings=[]
            )
    
    def process_query(self, query: str, **kwargs) -> Any:
        """
        Process a natural language query using the hybrid knowledge system.
        
        Args:
            query: Natural language query string
            **kwargs: Additional query parameters
            
        Returns:
            Query results from the hybrid search system
        """
        if not self._initialized:
            raise RuntimeError("System not initialized - call initialize() first")
        
        start_time = time.time()
        try:
            # Update metrics
            self.metrics['queries_processed'] += 1
            
            # Process query through the query planner
            results = self.query_planner.process_query(query, **kwargs)
            
            # Update performance metrics
            query_time_ms = (time.time() - start_time) * 1000
            self._update_query_metrics(query_time_ms)
            
            self.logger.debug(f"Query processed in {query_time_ms:.2f}ms: {query}")
            return results
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a file through the complete pipeline.
        
        Args:
            file_path: Path to the file to process
            **kwargs: Additional processing parameters
            
        Returns:
            Dict containing processing results for all components
        """
        if not self._initialized:
            raise RuntimeError("System not initialized - call initialize() first")
        
        start_time = time.time()
        results = {}
        
        try:
            # Extract entities
            entities = self.entity_extractor.extract_from_file(file_path)
            results['entities'] = entities
            self.metrics['entities_extracted'] += len(entities)
            
            # Detect relationships
            relationships = self.relationship_detector.detect_from_file(file_path)
            results['relationships'] = relationships
            self.metrics['relationships_detected'] += len(relationships)
            
            # Generate embeddings
            embeddings_result = self.embedding_pipeline.process_entities_by_file(file_path)
            results['embeddings'] = embeddings_result
            if embeddings_result:
                self.metrics['embeddings_generated'] += len(embeddings_result.get('generated', []))
            
            processing_time = time.time() - start_time
            results['processing_time_seconds'] = processing_time
            results['success'] = True
            
            self.logger.info(f"File processed successfully in {processing_time:.2f}s: {file_path}")
            return results
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.logger.error(f"File processing failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def _update_query_metrics(self, query_time_ms: float):
        """Update query performance metrics with exponential moving average."""
        if self.metrics['avg_query_time_ms'] == 0:
            self.metrics['avg_query_time_ms'] = query_time_ms
        else:
            # EMA with alpha = 0.1
            self.metrics['avg_query_time_ms'] = 0.9 * self.metrics['avg_query_time_ms'] + 0.1 * query_time_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        return {
            **self.metrics,
            'system_uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            'components_healthy': sum(1 for name, component in self._components.items() 
                                    if hasattr(component, 'is_healthy') and component.is_healthy()),
            'total_components': len(self._components)
        }
    
    def shutdown(self):
        """Gracefully shutdown the hybrid knowledge system."""
        with self._lock:
            if not self._initialized:
                return
            
            self.logger.info("Shutting down Hybrid Knowledge System...")
            
            # Stop monitoring
            if hasattr(self.system_monitor, 'stop_monitoring'):
                self.system_monitor.stop_monitoring()
            
            # Shutdown components in reverse order
            for component_name in reversed(list(self._components.keys())):
                component = self._components[component_name]
                if hasattr(component, 'shutdown'):
                    try:
                        component.shutdown()
                        self.logger.info(f"Component '{component_name}' shut down successfully")
                    except Exception as e:
                        self.logger.error(f"Error shutting down component '{component_name}': {e}")
            
            # Clear state
            self._components.clear()
            self._initialized = False
            
            self.logger.info("Hybrid Knowledge System shutdown complete")
    
    def _cleanup_partial_initialization(self):
        """Clean up partially initialized components."""
        self.logger.info("Cleaning up partial initialization...")
        
        for component_name, component in self._components.items():
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                except Exception as e:
                    self.logger.error(f"Error cleaning up component '{component_name}': {e}")
        
        self._components.clear()
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()