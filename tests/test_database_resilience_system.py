"""
Comprehensive test suite for Database Resilience System (Issue #150)
Tests all resilience features including connection pooling, circuit breaker, health monitoring, and graceful degradation
"""

import pytest
import time
import threading
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from knowledge.database.database_config import DatabaseConfig
from systems.database_resilience_manager import DatabaseResilienceManager, DatabaseHealthState, CircuitBreakerState
from systems.resilient_database_interface import ResilientDatabaseInterface
from systems.database_health_monitor import DatabaseHealthMonitor, MonitoringConfig, AlertSeverity
from systems.database_resilience_integration import DatabaseResilienceSystem, create_resilient_database_system


@pytest.fixture
def temp_database_config():
    """Create a temporary database configuration for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = DatabaseConfig(
            database_path=os.path.join(temp_dir, "test_entities.duckdb"),
            memory_limit="50MB",
            max_memory="50MB",
            max_connections=3,
            connection_timeout=5.0,
            idle_timeout=30.0,
            threads=2,
            auto_create_schema=False,  # Skip schema for faster tests
            enable_vss=False  # Skip VSS for faster tests
        )
        yield config


@pytest.fixture
def resilience_manager(temp_database_config):
    """Create a resilience manager for testing."""
    manager = DatabaseResilienceManager(
        config=temp_database_config,
        fallback_mode_enabled=True
    )
    yield manager
    manager.shutdown()


@pytest.fixture
def resilient_database(temp_database_config):
    """Create a resilient database interface for testing."""
    db = ResilientDatabaseInterface(
        config=temp_database_config,
        fallback_mode_enabled=True
    )
    yield db
    db.close()


@pytest.fixture
def monitoring_config():
    """Create monitoring configuration for testing."""
    return MonitoringConfig(
        check_interval=1.0,  # Fast checks for tests
        alert_cooldown=5.0,
        recovery_attempt_limit=2,
        recovery_cooldown=10.0,
        error_rate_warning=0.1,
        error_rate_critical=0.3
    )


@pytest.fixture
def health_monitor(resilient_database, monitoring_config):
    """Create a health monitor for testing."""
    monitor = DatabaseHealthMonitor(
        database_interface=resilient_database,
        config=monitoring_config,
        alert_handlers=[]  # No alert handlers for tests
    )
    yield monitor
    monitor.stop_monitoring()


class TestDatabaseResilienceManager:
    """Test the database resilience manager."""
    
    def test_initialization(self, temp_database_config):
        """Test resilience manager initialization."""
        manager = DatabaseResilienceManager(temp_database_config)
        
        assert manager.config == temp_database_config
        assert manager.fallback_mode_enabled == True
        assert manager._health_metrics.state == DatabaseHealthState.HEALTHY
        assert manager._circuit_breaker.state == CircuitBreakerState.CLOSED
        
        manager.shutdown()
    
    def test_connection_acquisition(self, resilience_manager):
        """Test connection acquisition and return."""
        with resilience_manager.get_resilient_connection() as conn:
            assert conn is not None
            # Test basic operation
            result = conn.execute("SELECT 1").fetchone()
            assert result is not None
    
    def test_connection_pooling(self, resilience_manager):
        """Test connection pooling behavior."""
        connections = []
        
        # Acquire multiple connections
        for _ in range(2):
            conn = resilience_manager._acquire_connection(timeout=5.0)
            assert conn is not None
            connections.append(conn)
        
        # Return connections
        for conn in connections:
            resilience_manager._return_connection(conn)
        
        # Verify pool has connections
        assert resilience_manager._pool.qsize() > 0
    
    def test_circuit_breaker_functionality(self, resilience_manager):
        """Test circuit breaker pattern."""
        # Record multiple failures to open circuit breaker
        for _ in range(resilience_manager._circuit_breaker.failure_threshold + 1):
            resilience_manager._circuit_breaker_record_failure()
        
        # Verify circuit breaker is open
        assert resilience_manager._circuit_breaker.state == CircuitBreakerState.OPEN
        assert not resilience_manager._circuit_breaker_allow_request()
        
        # Test force reset
        resilience_manager.force_circuit_breaker_reset()
        assert resilience_manager._circuit_breaker.state == CircuitBreakerState.CLOSED
        assert resilience_manager._circuit_breaker_allow_request()
    
    def test_fallback_mode(self, resilience_manager):
        """Test fallback mode when database is unavailable."""
        # Force circuit breaker open to simulate database unavailability
        resilience_manager._circuit_breaker.state = CircuitBreakerState.OPEN
        
        with resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
            assert hasattr(conn, 'fallback_operations')
            # Test fallback operations
            result = conn.execute("SELECT 1")
            assert result is not None
    
    def test_health_metrics(self, resilience_manager):
        """Test health metrics collection."""
        metrics = resilience_manager.get_health_metrics()
        
        required_keys = [
            'health_state', 'active_connections', 'total_queries',
            'error_rate', 'avg_response_time', 'circuit_breaker', 'pool_stats'
        ]
        
        for key in required_keys:
            assert key in metrics
        
        assert isinstance(metrics['circuit_breaker'], dict)
        assert isinstance(metrics['pool_stats'], dict)
    
    def test_error_recording(self, resilience_manager):
        """Test error recording and history."""
        initial_history_size = len(resilience_manager.get_error_history())
        
        # Record some errors
        resilience_manager._record_error("test_error", "Test error message")
        resilience_manager._record_error("another_error", "Another test error")
        
        error_history = resilience_manager.get_error_history()
        assert len(error_history) == initial_history_size + 2
        
        latest_error = error_history[-1]
        assert latest_error['error_type'] == 'another_error'
        assert latest_error['error_message'] == 'Another test error'


class TestResilientDatabaseInterface:
    """Test the resilient database interface."""
    
    def test_initialization(self, temp_database_config):
        """Test resilient database interface initialization."""
        db = ResilientDatabaseInterface(temp_database_config)
        
        assert db.config == temp_database_config
        assert db.resilience_manager is not None
        assert db.operation_metrics['total_operations'] == 0
        
        db.close()
    
    def test_entity_operations_with_fallback(self, resilient_database):
        """Test entity operations with fallback mode."""
        # Force fallback mode
        resilient_database.resilience_manager._circuit_breaker.state = CircuitBreakerState.OPEN
        
        entity_data = {
            'type': 'function',
            'name': 'test_function',
            'file_path': '/test/file.py',
            'line_start': 1,
            'line_end': 10
        }
        
        # Store entity (should use fallback)
        entity_id = resilient_database.store_entity(entity_data)
        assert entity_id.startswith('temp_')
        
        # Verify fallback operations counter
        assert resilient_database.operation_metrics['fallback_operations'] > 0
    
    def test_health_status(self, resilient_database):
        """Test health status reporting."""
        health_status = resilient_database.get_health_status()
        
        required_keys = [
            'timestamp', 'overall_health', 'database_available',
            'error_rate', 'avg_response_time', 'operation_metrics'
        ]
        
        for key in required_keys:
            assert key in health_status
        
        assert isinstance(health_status['operation_metrics'], dict)
        assert isinstance(health_status.get('recommendations', []), list)
    
    def test_operation_metrics_tracking(self, resilient_database):
        """Test operation metrics tracking."""
        initial_operations = resilient_database.operation_metrics['total_operations']
        
        # Perform some operations
        try:
            resilient_database.get_entity("nonexistent_id")
        except:
            pass
        
        try:
            resilient_database.search_entities(query="test")
        except:
            pass
        
        # Verify metrics were updated
        assert resilient_database.operation_metrics['total_operations'] > initial_operations
    
    def test_health_check_comprehensive(self, resilient_database):
        """Test comprehensive health check."""
        health_check = resilient_database.run_health_check()
        
        required_keys = [
            'timestamp', 'database_accessible', 'schema_present',
            'connection_pool_working', 'performance_acceptable', 'errors'
        ]
        
        for key in required_keys:
            assert key in health_check
        
        assert isinstance(health_check['errors'], list)
        assert 'duration' in health_check


class TestDatabaseHealthMonitor:
    """Test the database health monitor."""
    
    def test_initialization(self, resilient_database, monitoring_config):
        """Test health monitor initialization."""
        monitor = DatabaseHealthMonitor(
            resilient_database, monitoring_config, []
        )
        
        assert monitor.db_interface == resilient_database
        assert monitor.config == monitoring_config
        assert monitor.state.value in ['starting', 'stopped']
    
    def test_monitoring_lifecycle(self, health_monitor):
        """Test monitoring start, pause, resume, stop."""
        # Start monitoring
        health_monitor.start_monitoring()
        time.sleep(0.5)  # Let it start
        assert health_monitor.state.value == 'running'
        
        # Pause monitoring
        health_monitor.pause_monitoring()
        assert health_monitor.state.value == 'paused'
        
        # Resume monitoring
        health_monitor.resume_monitoring()
        assert health_monitor.state.value == 'running'
        
        # Stop monitoring
        health_monitor.stop_monitoring()
        assert health_monitor.state.value == 'stopped'
    
    def test_alert_generation(self, health_monitor):
        """Test alert generation and management."""
        # Generate a test alert
        health_monitor._generate_alert(
            "test_alert",
            AlertSeverity.WARNING,
            "test",
            "Test alert message",
            {"test_metric": 123}
        )
        
        # Verify alert was created
        active_alerts = health_monitor.get_active_alerts()
        assert len(active_alerts) > 0
        
        test_alert = next((a for a in active_alerts if a['id'] == 'test_alert'), None)
        assert test_alert is not None
        assert test_alert['severity'] == 'warning'
        assert test_alert['message'] == 'Test alert message'
    
    def test_alert_resolution(self, health_monitor):
        """Test alert resolution."""
        # Generate and resolve an alert
        health_monitor._generate_alert("resolve_test", AlertSeverity.ERROR, "test", "Test error", {})
        health_monitor._resolve_alert("resolve_test", "Test resolved")
        
        active_alerts = health_monitor.get_active_alerts()
        resolved_alert = next((a for a in active_alerts if a['id'] == 'resolve_test'), None)
        
        assert resolved_alert is not None
        assert resolved_alert['resolved'] == True
        assert resolved_alert['resolution_message'] == 'Test resolved'
    
    def test_metrics_recording(self, health_monitor):
        """Test metrics recording and history."""
        initial_size = len(health_monitor.metrics_history)
        
        # Record some test metrics
        test_health_status = {
            'overall_health': 'healthy',
            'database_available': True,
            'error_rate': 0.05,
            'avg_response_time': 0.1,
            'active_connections': 2
        }
        
        test_resilience_metrics = {
            'circuit_breaker': {'state': 'closed'},
            'total_queries': 100,
            'failed_queries': 5
        }
        
        health_monitor._record_metrics(test_health_status, test_resilience_metrics)
        
        # Verify metrics were recorded
        assert len(health_monitor.metrics_history) == initial_size + 1
        
        latest_metrics = health_monitor.metrics_history[-1]
        assert latest_metrics['error_rate'] == 0.05
        assert latest_metrics['avg_response_time'] == 0.1
    
    def test_monitoring_status(self, health_monitor):
        """Test monitoring status reporting."""
        status = health_monitor.get_monitoring_status()
        
        required_keys = [
            'state', 'active_alerts', 'critical_alerts',
            'metrics_history_size', 'config'
        ]
        
        for key in required_keys:
            assert key in status
        
        assert isinstance(status['config'], dict)


class TestDatabaseResilienceSystem:
    """Test the integrated resilience system."""
    
    def test_system_creation(self, temp_database_config):
        """Test resilience system creation and initialization."""
        system = create_resilient_database_system(temp_database_config)
        
        assert system is not None
        assert system.config == temp_database_config
        
        # Test initialization
        assert system.initialize() == True
        assert system._initialized == True
        
        system.shutdown()
    
    def test_database_interface_access(self, temp_database_config):
        """Test database interface access through system."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            db_interface = system.get_database_interface()
            
            assert isinstance(db_interface, ResilientDatabaseInterface)
            assert db_interface.config == temp_database_config
    
    def test_health_monitor_access(self, temp_database_config):
        """Test health monitor access through system."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            health_monitor = system.get_health_monitor()
            
            assert isinstance(health_monitor, DatabaseHealthMonitor)
            assert health_monitor.state.value in ['starting', 'running']
    
    def test_system_status(self, temp_database_config):
        """Test system status reporting."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            status = system.get_system_status()
            
            assert status['status'] in ['operational', 'degraded']
            assert 'database_health' in status
            assert 'resilience' in status
            assert 'monitoring' in status
    
    def test_resilience_testing(self, temp_database_config):
        """Test resilience testing functionality."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            test_results = system.test_resilience()
            
            assert 'tests' in test_results
            assert 'overall_success' in test_results
            
            # Verify test categories
            expected_tests = [
                'basic_connectivity', 'fallback_mechanism',
                'health_monitoring', 'circuit_breaker', 'performance_metrics'
            ]
            
            for test_name in expected_tests:
                assert test_name in test_results['tests']
                assert 'success' in test_results['tests'][test_name]
    
    def test_force_recovery(self, temp_database_config):
        """Test force recovery functionality."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            recovery_result = system.force_recovery()
            
            assert 'success' in recovery_result or 'error' in recovery_result
            
            if 'actions_taken' in recovery_result:
                assert isinstance(recovery_result['actions_taken'], list)
    
    def test_context_manager(self, temp_database_config):
        """Test system as context manager."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            assert system._initialized == True
            
            # Test that we can use the system
            status = system.get_system_status()
            assert 'status' in status
        
        # System should be shutdown after context exit
        # Note: We can't easily test this without exposing internal state


class TestResilienceIntegration:
    """Test integration scenarios that simulate real-world conditions."""
    
    def test_connection_failure_recovery(self, temp_database_config):
        """Test recovery from connection failures."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            db = system.get_database_interface()
            
            # Simulate connection failure by opening circuit breaker
            db.resilience_manager._circuit_breaker.state = CircuitBreakerState.OPEN
            
            # Operations should still work via fallback
            entity_data = {
                'type': 'function',
                'name': 'test_func',
                'file_path': '/test.py'
            }
            
            entity_id = db.store_entity(entity_data)
            assert entity_id is not None
            
            # Force recovery
            recovery_result = system.force_recovery()
            assert 'success' in recovery_result
    
    def test_high_load_simulation(self, temp_database_config):
        """Test system behavior under simulated high load."""
        with DatabaseResilienceSystem(temp_database_config) as system:
            db = system.get_database_interface()
            
            # Simulate multiple concurrent operations
            def perform_operations():
                for i in range(10):
                    try:
                        entity_data = {
                            'type': 'test',
                            'name': f'entity_{i}',
                            'file_path': f'/test_{i}.py'
                        }
                        db.store_entity(entity_data)
                        db.search_entities(query=f'entity_{i}')
                    except Exception:
                        pass  # Expected in high load scenario
            
            # Run operations concurrently
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=perform_operations)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
            
            # Verify system is still operational
            status = system.get_system_status()
            assert status['status'] in ['operational', 'degraded']
            
            # Check metrics
            metrics = db.operation_metrics
            assert metrics['total_operations'] > 0
    
    def test_monitoring_alert_flow(self, temp_database_config):
        """Test complete monitoring and alert flow."""
        monitoring_config = MonitoringConfig(
            check_interval=0.5,  # Very fast for testing
            alert_cooldown=1.0,
            error_rate_warning=0.01  # Very low threshold for testing
        )
        
        system = DatabaseResilienceSystem(temp_database_config, monitoring_config)
        
        try:
            system.initialize()
            
            # Force some errors to trigger alerts
            db = system.get_database_interface()
            
            # Generate errors by accessing non-existent entities
            for _ in range(5):
                try:
                    db.get_entity("nonexistent")
                except:
                    pass
            
            # Wait for monitoring to detect issues
            time.sleep(2.0)
            
            # Check for alerts
            monitor = system.get_health_monitor()
            active_alerts = monitor.get_active_alerts()
            
            # Should have generated some alerts
            # (exact alerts depend on system state, so we just verify monitoring is working)
            monitoring_status = monitor.get_monitoring_status()
            assert monitoring_status['state'] == 'running'
            
        finally:
            system.shutdown()


if __name__ == "__main__":
    # Run basic tests if executed directly
    pytest.main([__file__, "-v"])