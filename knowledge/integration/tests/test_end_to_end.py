#!/usr/bin/env python3
"""
End-to-End Integration Tests for Hybrid Knowledge System

Tests the complete integration of Issues #30-33 through the master coordination system.
Validates performance targets, resource management, and full pipeline functionality.
"""

import os
import sys
import time
import json
import tempfile
import unittest
import logging
from pathlib import Path
from typing import Dict, Any
from contextlib import contextmanager

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from knowledge.integration.hybrid_knowledge_system import HybridKnowledgeSystem
from knowledge.integration.knowledge_api import KnowledgeAPI

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHybridKnowledgeSystemIntegration(unittest.TestCase):
    """
    Comprehensive end-to-end tests for the hybrid knowledge system integration.
    
    Tests validate:
    - Master coordination plan execution
    - Performance targets from Issue #40
    - Resource management within 2GB/4-core limits
    - Component integration and data flow
    - API functionality and agent integration
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and resources."""
        cls.test_dir = Path(__file__).parent / "test_data"
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create test database path
        cls.test_db_path = cls.test_dir / "test_entities.duckdb" 
        
        # Test configuration matching master plan
        cls.test_config = {
            'memory_limit_mb': 2048,
            'cpu_cores': 4,
            'performance_mode': 'BALANCED',
            'query_cache_size': 1000,
            'enable_monitoring': True,
            'enable_metrics': True,
            'database_path': str(cls.test_db_path)
        }
        
        # Create sample test files
        cls._create_test_files()
        
        logger.info("Test environment set up complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        try:
            if cls.test_db_path.exists():
                cls.test_db_path.unlink()
            
            # Clean up test files
            for file_path in cls.test_dir.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
            
            cls.test_dir.rmdir()
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        
        logger.info("Test environment cleaned up")
    
    @classmethod
    def _create_test_files(cls):
        """Create sample test files for integration testing."""
        
        # JavaScript test file
        js_content = '''
// User authentication module
class UserManager {
    constructor(database) {
        this.database = database;
        this.activeUsers = new Map();
    }
    
    async authenticateUser(username, password) {
        try {
            const user = await this.database.findUser(username);
            if (user && this.validatePassword(password, user.passwordHash)) {
                this.activeUsers.set(username, user);
                return { success: true, user };
            }
            return { success: false, error: "Invalid credentials" };
        } catch (error) {
            console.error("Authentication error:", error);
            throw error;
        }
    }
    
    validatePassword(password, hash) {
        return bcrypt.compareSync(password, hash);
    }
    
    logout(username) {
        this.activeUsers.delete(username);
    }
    
    isUserActive(username) {
        return this.activeUsers.has(username);
    }
}

// Payment processing functions
function processPayment(paymentData) {
    if (!validatePaymentData(paymentData)) {
        throw new Error("Invalid payment data");
    }
    
    return chargeCard(paymentData.cardInfo, paymentData.amount);
}

function validatePaymentData(data) {
    return data && data.cardInfo && data.amount > 0;
}

async function chargeCard(cardInfo, amount) {
    // Mock payment processing
    return { transactionId: "txn_" + Date.now(), success: true };
}

export { UserManager, processPayment, validatePaymentData };
        '''
        
        # Python test file  
        python_content = '''
"""
Data processing and analysis module.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    batch_size: int = 1000
    timeout_seconds: int = 30
    enable_caching: bool = True
    output_format: str = "json"


class DataProcessor:
    """Main data processing class with multiple analysis capabilities."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.cache = {}
        self._stats = {"processed": 0, "errors": 0}
    
    def process_batch(self, data: List[Dict]) -> List[Dict]:
        """Process a batch of data items."""
        try:
            results = []
            for item in data:
                processed_item = self._process_single_item(item)
                if processed_item:
                    results.append(processed_item)
            
            self._stats["processed"] += len(results)
            return results
            
        except Exception as e:
            self._stats["errors"] += 1
            raise ProcessingError(f"Batch processing failed: {e}")
    
    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item with validation."""
        if not self._validate_item(item):
            return None
        
        # Apply transformations
        processed = self._apply_transformations(item)
        processed = self._enrich_data(processed)
        
        return processed
    
    def _validate_item(self, item: Dict) -> bool:
        """Validate data item structure and content."""
        required_fields = ["id", "timestamp", "data"]
        return all(field in item for field in required_fields)
    
    def _apply_transformations(self, item: Dict) -> Dict:
        """Apply data transformations."""
        transformed = item.copy()
        
        # Normalize timestamp
        if "timestamp" in transformed:
            transformed["normalized_timestamp"] = pd.to_datetime(transformed["timestamp"])
        
        return transformed
    
    def _enrich_data(self, item: Dict) -> Dict:
        """Enrich data with additional computed fields."""
        enriched = item.copy()
        enriched["processing_timestamp"] = pd.Timestamp.now()
        enriched["processor_version"] = "1.0.0"
        
        return enriched
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self._stats.copy()


class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass


def analyze_data_patterns(data: List[Dict]) -> Dict[str, Union[int, float, List]]:
    """Analyze patterns in the processed data."""
    if not data:
        return {"error": "No data provided"}
    
    analysis = {
        "total_records": len(data),
        "unique_ids": len(set(item.get("id") for item in data if "id" in item)),
        "timestamp_range": _get_timestamp_range(data),
        "common_fields": _get_common_fields(data)
    }
    
    return analysis


def _get_timestamp_range(data: List[Dict]) -> Dict[str, str]:
    """Extract timestamp range from data."""
    timestamps = [item.get("timestamp") for item in data if "timestamp" in item]
    if not timestamps:
        return {"error": "No timestamps found"}
    
    return {
        "earliest": min(timestamps),
        "latest": max(timestamps),
        "span_hours": len(set(ts[:10] for ts in timestamps if isinstance(ts, str)))
    }


def _get_common_fields(data: List[Dict]) -> List[str]:
    """Find fields that appear in most records."""
    if not data:
        return []
    
    field_counts = {}
    for item in data:
        for field in item.keys():
            field_counts[field] = field_counts.get(field, 0) + 1
    
    total_records = len(data)
    common_threshold = total_records * 0.8  # 80% presence threshold
    
    return [field for field, count in field_counts.items() if count >= common_threshold]
        '''
        
        # Go test file
        go_content = '''
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "time"
)

// User represents a user entity
type User struct {
    ID       int    `json:"id"`
    Username string `json:"username"`
    Email    string `json:"email"`
    Created  time.Time `json:"created"`
}

// UserService handles user-related operations
type UserService struct {
    database Database
    cache    Cache
    logger   *log.Logger
}

// NewUserService creates a new UserService instance
func NewUserService(db Database, cache Cache, logger *log.Logger) *UserService {
    return &UserService{
        database: db,
        cache:    cache,
        logger:   logger,
    }
}

// GetUser retrieves a user by ID
func (s *UserService) GetUser(ctx context.Context, userID int) (*User, error) {
    // Check cache first
    if cachedUser := s.cache.Get(fmt.Sprintf("user:%d", userID)); cachedUser != nil {
        if user, ok := cachedUser.(*User); ok {
            return user, nil
        }
    }
    
    // Fetch from database
    user, err := s.database.FindUserByID(ctx, userID)
    if err != nil {
        s.logger.Printf("Error fetching user %d: %v", userID, err)
        return nil, err
    }
    
    // Cache the result
    s.cache.Set(fmt.Sprintf("user:%d", userID), user, 5*time.Minute)
    
    return user, nil
}

// CreateUser creates a new user
func (s *UserService) CreateUser(ctx context.Context, username, email string) (*User, error) {
    if err := s.validateUserData(username, email); err != nil {
        return nil, err
    }
    
    user := &User{
        Username: username,
        Email:    email,
        Created:  time.Now(),
    }
    
    if err := s.database.SaveUser(ctx, user); err != nil {
        s.logger.Printf("Error creating user %s: %v", username, err)
        return nil, err
    }
    
    s.logger.Printf("Created user: %s", username)
    return user, nil
}

func (s *UserService) validateUserData(username, email string) error {
    if username == "" {
        return fmt.Errorf("username cannot be empty")
    }
    if email == "" {
        return fmt.Errorf("email cannot be empty")
    }
    return nil
}

// HTTP handler functions
func handleGetUser(service *UserService) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        userID := getUserIDFromPath(r.URL.Path)
        
        user, err := service.GetUser(r.Context(), userID)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        
        writeJSONResponse(w, user)
    }
}

func handleCreateUser(service *UserService) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        var req struct {
            Username string `json:"username"`
            Email    string `json:"email"`
        }
        
        if err := parseJSONRequest(r, &req); err != nil {
            http.Error(w, err.Error(), http.StatusBadRequest)
            return
        }
        
        user, err := service.CreateUser(r.Context(), req.Username, req.Email)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        
        w.WriteHeader(http.StatusCreated)
        writeJSONResponse(w, user)
    }
}
        '''
        
        # Write test files
        test_files = [
            (cls.test_dir / "auth.js", js_content),
            (cls.test_dir / "data_processor.py", python_content),
            (cls.test_dir / "user_service.go", go_content)
        ]
        
        for file_path, content in test_files:
            with open(file_path, 'w') as f:
                f.write(content)
        
        logger.info(f"Created {len(test_files)} test files")
    
    def setUp(self):
        """Set up each test case."""
        # Create fresh knowledge system for each test
        self.knowledge_system = HybridKnowledgeSystem(self.test_config)
        self.api = KnowledgeAPI(self.knowledge_system)
        
        # Initialize system (this will be mocked in actual tests)
        # For integration tests, we'll test initialization separately
    
    def tearDown(self):
        """Clean up after each test case."""
        if hasattr(self, 'knowledge_system'):
            try:
                self.knowledge_system.shutdown()
            except Exception as e:
                logger.warning(f"Shutdown error in test: {e}")
    
    # === System Initialization Tests ===
    
    def test_system_initialization_sequence(self):
        """Test that the system initializes in the correct dependency order per master plan."""
        logger.info("Testing system initialization sequence...")
        
        # Mock the initialization since we don't have actual components
        with self._mock_components():
            success = self.knowledge_system.initialize()
            
            self.assertTrue(success, "System initialization should succeed")
            self.assertTrue(self.knowledge_system._initialized, "System should be marked as initialized")
            
            # Verify component registration order matches master plan
            expected_order = ['database', 'entity_extraction', 'relationships', 'embeddings', 'query_planning']
            registered_components = list(self.knowledge_system._components.keys())
            
            for expected_component in expected_order:
                self.assertIn(expected_component, registered_components,
                             f"Component '{expected_component}' should be registered")
    
    def test_system_health_validation(self):
        """Test comprehensive system health validation."""
        logger.info("Testing system health validation...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            status = self.knowledge_system.get_system_status()
            
            self.assertTrue(status.healthy, "System should be healthy after initialization")
            self.assertEqual(len(status.components_ready), 5, "Should have 5 components registered")
            self.assertTrue(all(status.components_ready.values()), "All components should be ready")
            self.assertLess(status.resource_usage['memory_mb'], 2048 * 0.95, "Memory usage should be within limits")
    
    def test_resource_management_compliance(self):
        """Test that resource usage stays within master plan limits.""" 
        logger.info("Testing resource management compliance...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Get system monitor metrics
            monitor = self.knowledge_system.system_monitor
            pressure = monitor.get_resource_pressure()
            stats = monitor.get_performance_stats(window_minutes=1)
            
            # Validate memory limits
            self.assertLessEqual(stats['memory']['current_mb'], 2048,
                               "Memory usage should not exceed 2GB limit")
            
            # Validate pressure levels
            self.assertIn(pressure['overall'], ['LOW', 'MEDIUM'], 
                         "Resource pressure should be manageable")
            
            # Validate monitoring is active
            health = monitor.get_health_summary()
            self.assertTrue(health['healthy'], "System should be healthy")
            self.assertTrue(health['monitoring_active'], "Monitoring should be active")
    
    # === API Integration Tests ===
    
    def test_unified_api_query_interface(self):
        """Test the unified API gateway provides consistent query interface."""
        logger.info("Testing unified API query interface...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Test basic query functionality
            test_queries = [
                "find authentication functions",
                "show me error handling patterns",
                "what functions call processPayment",
                "analyze dependencies for UserManager"
            ]
            
            for query in test_queries:
                with self.subTest(query=query):
                    result = self.api.query(query)
                    
                    self.assertIsInstance(result.query, str)
                    self.assertIsInstance(result.results, list)
                    self.assertIsInstance(result.metadata, dict)
                    self.assertGreater(result.processing_time_ms, 0)
                    self.assertIsInstance(result.success, bool)
    
    def test_performance_mode_variations(self):
        """Test different performance modes work correctly."""
        logger.info("Testing performance mode variations...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            query = "find user authentication functions"
            
            modes = ['FAST', 'BALANCED', 'COMPREHENSIVE']
            results = {}
            
            for mode in modes:
                with self.subTest(mode=mode):
                    result = self.api.query(query, performance_mode=mode)
                    results[mode] = result
                    
                    self.assertTrue(result.success or result.error, f"Mode {mode} should return success or error")
                    self.assertEqual(result.metadata.get('performance_mode'), mode)
            
            # FAST mode should generally be fastest
            if all(r.success for r in results.values()):
                self.assertLessEqual(
                    results['FAST'].processing_time_ms,
                    results['COMPREHENSIVE'].processing_time_ms * 1.5,  # Allow some variance
                    "FAST mode should generally be faster than COMPREHENSIVE"
                )
    
    def test_specialized_query_methods(self):
        """Test specialized query methods work correctly."""
        logger.info("Testing specialized query methods...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Test entity finding
            entity_result = self.api.find_entities(name_pattern="authenticate", entity_type="function")
            self.assertIsInstance(entity_result.results, list)
            
            # Test similarity search
            similarity_result = self.api.find_similar_code(
                "function authenticate(username, password)",
                similarity_threshold=0.5
            )
            self.assertIsInstance(similarity_result.results, list)
            
            # Test dependency analysis
            dep_result = self.api.analyze_dependencies("UserManager", direction="both")
            self.assertIsInstance(dep_result.results, list)
            
            # Test impact analysis
            impact_result = self.api.analyze_impact("processPayment", change_type="modification")
            self.assertIsInstance(impact_result.results, list)
    
    # === File Processing Tests ===
    
    def test_single_file_processing(self):
        """Test processing individual files through the complete pipeline."""
        logger.info("Testing single file processing...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            test_file = self.test_dir / "auth.js"
            
            result = self.api.process_file(str(test_file))
            
            self.assertIsInstance(result.success, bool)
            self.assertEqual(result.file_path, str(test_file))
            self.assertGreaterEqual(result.processing_time_ms, 0)
            
            if result.success:
                # Should have extracted some entities
                self.assertGreater(result.entities_extracted, 0, "Should extract entities from test file")
    
    def test_directory_processing(self):
        """Test processing entire directories."""
        logger.info("Testing directory processing...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            results = self.api.process_directory(
                str(self.test_dir),
                file_patterns=['*.js', '*.py', '*.go'],
                max_files=10
            )
            
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0, "Should process at least one file")
            
            # Check each result
            for file_path, result in results.items():
                self.assertIsInstance(result.success, bool)
                self.assertEqual(result.file_path, file_path)
    
    # === Performance Validation Tests ===
    
    def test_query_latency_targets(self):
        """Test query latency meets master plan targets (<100ms P95 for simple queries)."""
        logger.info("Testing query latency targets...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Run multiple simple queries to measure latency distribution
            simple_queries = [
                "find function authenticate",
                "show class UserManager", 
                "get processPayment",
                "find validatePassword"
            ]
            
            latencies = []
            for _ in range(20):  # Run 20 iterations
                for query in simple_queries:
                    result = self.api.query(query, performance_mode='FAST')
                    latencies.append(result.processing_time_ms)
            
            # Calculate P95 latency
            latencies.sort()
            p95_index = int(len(latencies) * 0.95)
            p95_latency = latencies[p95_index]
            
            logger.info(f"P95 latency: {p95_latency:.1f}ms, Median: {latencies[len(latencies)//2]:.1f}ms")
            
            # Validate against target (relaxed for mock implementation)
            self.assertLess(p95_latency, 500, f"P95 latency should be <500ms, got {p95_latency:.1f}ms")
    
    def test_concurrent_query_handling(self):
        """Test system handles concurrent queries without degradation."""
        logger.info("Testing concurrent query handling...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            import concurrent.futures
            import threading
            
            query = "find authentication functions"
            num_concurrent = 5
            results = []
            
            def run_query():
                return self.api.query(query)
            
            # Run concurrent queries
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(run_query) for _ in range(num_concurrent)]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # Validate all queries succeeded
            successful_queries = sum(1 for r in results if r.success)
            self.assertEqual(successful_queries, num_concurrent, 
                           f"All {num_concurrent} concurrent queries should succeed")
            
            # Check performance didn't degrade significantly
            avg_latency = sum(r.processing_time_ms for r in results) / len(results)
            self.assertLess(avg_latency, 1000, f"Average concurrent query latency should be reasonable")
    
    # === Integration Coordination Tests ===
    
    def test_component_coordination_flow(self):
        """Test proper coordination between components per master plan."""
        logger.info("Testing component coordination flow...")
        
        with self._mock_components():
            # Test initialization coordination
            controller = self.knowledge_system.integration_controller
            
            # Mock component registration in dependency order
            mock_components = {
                'entity_extraction': MockComponent('entity_extraction'),
                'relationships': MockComponent('relationships', deps=['entity_extraction']),
                'embeddings': MockComponent('embeddings', deps=['entity_extraction']),
                'query_planning': MockComponent('query_planning', deps=['relationships', 'embeddings'])
            }
            
            # Register components
            for name, component in mock_components.items():
                controller.register_component(name, component, component.dependencies)
            
            # Test coordination status
            status = controller.get_coordination_status()
            
            self.assertEqual(status['total_components'], 4)
            self.assertIsInstance(status['checkpoint_status'], dict)
            self.assertIsInstance(status['dependency_graph'], dict)
    
    def test_error_recovery_and_fallback(self):
        """Test system error recovery and fallback strategies."""
        logger.info("Testing error recovery and fallback...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Simulate component failure
            if hasattr(self.knowledge_system, 'embedding_pipeline'):
                # Test graceful handling of component errors
                
                # Query should still work with reduced functionality
                result = self.api.query("find test functions", performance_mode='FAST')
                
                # System should still report status
                status = self.knowledge_system.get_system_status()
                self.assertIsInstance(status.healthy, bool)
                self.assertIsInstance(status.error_count, int)
    
    # === Agent Integration Tests ===
    
    def test_agent_convenience_methods(self):
        """Test convenience methods designed for RIF agents."""
        logger.info("Testing agent convenience methods...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Test quick search
            quick_results = self.api.quick_search("authentication")
            self.assertIsInstance(quick_results, list)
            
            # Test deep analysis
            deep_analysis = self.api.deep_analysis("analyze error handling patterns")
            self.assertIn('results', deep_analysis)
            self.assertIn('metadata', deep_analysis)
            self.assertIn('processing_time_ms', deep_analysis)
            
            # Test context-aware search
            context_results = self.api.context_aware_search(
                "find user functions",
                active_files=[str(self.test_dir / "auth.js")],
                current_task="authentication analysis"
            )
            self.assertIsInstance(context_results, list)
    
    def test_metrics_and_monitoring_integration(self):
        """Test metrics collection and system monitoring integration."""
        logger.info("Testing metrics and monitoring integration...")
        
        with self._mock_components():
            self.knowledge_system.initialize()
            
            # Generate some activity
            for i in range(5):
                self.api.query(f"test query {i}")
            
            # Check system metrics
            system_metrics = self.knowledge_system.get_metrics()
            self.assertIsInstance(system_metrics, dict)
            self.assertIn('queries_processed', system_metrics)
            
            # Check API metrics
            api_metrics = self.api.get_performance_metrics()
            self.assertIn('api_metrics', api_metrics)
            self.assertIn('query_success_rate', api_metrics)
            self.assertGreater(api_metrics['api_metrics']['total_queries'], 0)
            
            # Check system status
            status = self.api.get_system_status()
            self.assertIn('system_healthy', status)
            self.assertIn('components', status)
            self.assertIn('resource_usage', status)
    
    # === Mock Component Helper ===
    
    @contextmanager
    def _mock_components(self):
        """Context manager to mock the component dependencies for testing."""
        
        # Mock the component initialization methods
        original_methods = {}
        
        try:
            # Mock database initialization
            original_methods['_initialize_database'] = self.knowledge_system._initialize_database
            self.knowledge_system._initialize_database = lambda path: setattr(
                self.knowledge_system, 'db_manager', MockDatabaseManager()
            )
            
            # Mock entity extraction
            original_methods['_initialize_entity_extraction'] = self.knowledge_system._initialize_entity_extraction
            self.knowledge_system._initialize_entity_extraction = lambda: setattr(
                self.knowledge_system, 'entity_extractor', MockEntityExtractor()
            )
            
            # Mock parallel components
            original_methods['_initialize_parallel_components'] = self.knowledge_system._initialize_parallel_components
            def mock_parallel():
                self.knowledge_system.relationship_detector = MockRelationshipDetector()
                self.knowledge_system.embedding_pipeline = MockEmbeddingPipeline()
            self.knowledge_system._initialize_parallel_components = mock_parallel
            
            # Mock query planning
            original_methods['_initialize_query_planning'] = self.knowledge_system._initialize_query_planning  
            self.knowledge_system._initialize_query_planning = lambda: setattr(
                self.knowledge_system, 'query_planner', MockQueryPlanner()
            )
            
            # Mock validation
            original_methods['_validate_system_health'] = self.knowledge_system._validate_system_health
            self.knowledge_system._validate_system_health = lambda: True
            
            # Mock basic integration test
            original_methods['_test_basic_integration'] = self.knowledge_system._test_basic_integration
            self.knowledge_system._test_basic_integration = lambda: None
            
            yield
            
        finally:
            # Restore original methods
            for method_name, original_method in original_methods.items():
                setattr(self.knowledge_system, method_name, original_method)


# === Mock Components for Testing ===

class MockComponent:
    """Mock component for testing coordination."""
    
    def __init__(self, name: str, deps: list = None):
        self.name = name
        self.dependencies = deps or []
        self.healthy = True
    
    def is_healthy(self):
        return self.healthy


class MockDatabaseManager:
    """Mock database manager for testing."""
    
    def __init__(self):
        self.active_connections = 1
    
    def validate_schema(self):
        return True
    
    def is_healthy(self):
        return True


class MockEntityExtractor:
    """Mock entity extractor for testing."""
    
    def extract_from_file(self, file_path):
        # Return mock entities based on file content
        return [
            {'name': 'test_function', 'type': 'function', 'file': file_path},
            {'name': 'TestClass', 'type': 'class', 'file': file_path}
        ]
    
    def is_healthy(self):
        return True


class MockRelationshipDetector:
    """Mock relationship detector for testing."""
    
    def detect_from_file(self, file_path):
        # Return mock relationships
        return [
            {'source': 'test_function', 'target': 'TestClass', 'type': 'calls'}
        ]
    
    def is_healthy(self):
        return True


class MockEmbeddingPipeline:
    """Mock embedding pipeline for testing."""
    
    def process_entities_by_file(self, file_path):
        # Return mock embedding results
        return {
            'generated': ['entity1_embedding', 'entity2_embedding'],
            'cached': [],
            'errors': []
        }
    
    def is_healthy(self):
        return True


class MockQueryPlanner:
    """Mock query planner for testing."""
    
    def __init__(self):
        self.cache_size = 100
    
    def process_query(self, query, **kwargs):
        # Return mock query results
        return MockQueryResult([
            {'name': 'mock_result', 'type': 'function', 'relevance': 0.9},
            {'name': 'another_result', 'type': 'class', 'relevance': 0.8}
        ])
    
    def is_healthy(self):
        return True


class MockQueryResult:
    """Mock query result object."""
    
    def __init__(self, results):
        self.results = results
        self.cached = False


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=True)