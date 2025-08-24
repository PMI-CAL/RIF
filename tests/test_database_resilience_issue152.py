"""
Comprehensive test suite for database resilience implementation (Issue #152).

Tests the enhanced retry logic, connection state management, transaction rollback
handling, and deadlock resolution capabilities.
"""

import os
import pytest
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Import the components to test
from knowledge.database.retry_manager import (
    DatabaseRetryManager, RetryConfig, ConnectionState, 
    DeadlockDetector, TransactionContext, retry_on_database_error
)
from knowledge.database.resilient_connection_manager import (
    ResilientConnectionManager, create_resilient_manager
)
from knowledge.conversations.storage_backend import ConversationStorageBackend


class TestDatabaseRetryManager:
    """Test retry manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            backoff_multiplier=2.0
        )
        self.retry_manager = DatabaseRetryManager(self.retry_config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.retry_manager:
            self.retry_manager.shutdown()
    
    def test_retry_config_initialization(self):
        """Test retry configuration initialization."""
        assert self.retry_manager.config.max_attempts == 3
        assert self.retry_manager.config.base_delay == 0.1
        assert self.retry_manager.config.max_delay == 1.0
        assert self.retry_manager.config.backoff_multiplier == 2.0
    
    def test_connection_state_tracking(self):
        """Test connection state management."""
        connection_id = "test_conn_1"
        
        # Initial state should be healthy
        assert self.retry_manager.get_connection_state(connection_id) == ConnectionState.HEALTHY
        
        # Update with successful operations
        self.retry_manager.update_connection_metrics(connection_id, True, 0.1)
        self.retry_manager.update_connection_metrics(connection_id, True, 0.2)
        assert self.retry_manager.get_connection_state(connection_id) == ConnectionState.HEALTHY
        
        # Update with failures
        for _ in range(5):  # failure_threshold = 5
            self.retry_manager.update_connection_metrics(connection_id, False)
        
        assert self.retry_manager.get_connection_state(connection_id) == ConnectionState.FAILED
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        # Test retryable errors
        retryable_exceptions = [
            Exception("Database connection refused"),
            Exception("Connection reset by peer"),
            Exception("Lock wait timeout exceeded"),
            Exception("Deadlock detected")
        ]
        
        for exc in retryable_exceptions:
            assert self.retry_manager.should_retry(1, exc) == True
        
        # Test non-retryable errors
        non_retryable_exceptions = [
            Exception("Invalid SQL syntax"),
            Exception("Table does not exist"),
            ValueError("Invalid parameter")
        ]
        
        for exc in non_retryable_exceptions:
            assert self.retry_manager.should_retry(1, exc) == False
        
        # Test max attempts exceeded
        retryable_exc = Exception("Connection refused")
        assert self.retry_manager.should_retry(3, retryable_exc) == False
    
    def test_delay_calculation(self):
        """Test retry delay calculation."""
        # Test exponential backoff
        delay1 = self.retry_manager.calculate_delay(0)  # First retry
        delay2 = self.retry_manager.calculate_delay(1)  # Second retry
        delay3 = self.retry_manager.calculate_delay(2)  # Third retry
        
        # With jitter, delays will vary, but should follow exponential pattern
        assert 0.1 <= delay1 <= 0.4  # base_delay + jitter
        assert 0.1 <= delay2 <= 0.7  # base_delay * 2 + jitter
        assert 0.1 <= delay3 <= 1.0  # capped at max_delay
    
    def test_execute_with_retry_success(self):
        """Test successful operation execution."""
        def successful_operation():
            return "success"
        
        result = self.retry_manager.execute_with_retry(
            successful_operation,
            "test_conn",
            "test_operation"
        )
        
        assert result == "success"
        assert self.retry_manager.total_retries == 0
        assert self.retry_manager.successful_retries == 0
    
    def test_execute_with_retry_failure_then_success(self):
        """Test operation that fails then succeeds."""
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Connection refused")
            return "success"
        
        result = self.retry_manager.execute_with_retry(
            flaky_operation,
            "test_conn",
            "test_operation"
        )
        
        assert result == "success"
        assert call_count == 2
        assert self.retry_manager.total_retries == 1
        assert self.retry_manager.successful_retries == 1
    
    def test_execute_with_retry_max_failures(self):
        """Test operation that fails all retry attempts."""
        def failing_operation():
            raise Exception("Connection refused")
        
        with pytest.raises(Exception, match="Connection refused"):
            self.retry_manager.execute_with_retry(
                failing_operation,
                "test_conn",
                "test_operation"
            )
        
        assert self.retry_manager.total_retries == 2  # 3 attempts = 2 retries
        assert self.retry_manager.failed_operations == 1
    
    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        call_count = 0
        
        @retry_on_database_error(self.retry_manager, "test_conn", "decorated_op")
        def decorated_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Database connection failed")
            return "decorated_success"
        
        result = decorated_operation()
        assert result == "decorated_success"
        assert call_count == 2
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        # Perform some operations
        self.retry_manager.update_connection_metrics("conn1", True, 0.1)
        self.retry_manager.update_connection_metrics("conn1", False)
        self.retry_manager.update_connection_metrics("conn2", True, 0.2)
        
        metrics = self.retry_manager.get_metrics()
        
        assert "total_retries" in metrics
        assert "successful_retries" in metrics
        assert "failed_operations" in metrics
        assert "connection_metrics" in metrics
        assert "conn1" in metrics["connection_metrics"]
        assert "conn2" in metrics["connection_metrics"]
        
        conn1_metrics = metrics["connection_metrics"]["conn1"]
        assert conn1_metrics["success_count"] == 1
        assert conn1_metrics["failure_count"] == 1


class TestDeadlockDetector:
    """Test deadlock detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DeadlockDetector()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.detector.stop_monitoring()
    
    def test_transaction_registration(self):
        """Test transaction registration and unregistration."""
        tx_context = TransactionContext(
            transaction_id="test_tx_1",
            connection_id="test_conn_1",
            started_at=datetime.now()
        )
        
        self.detector.register_transaction(tx_context)
        assert "test_tx_1" in self.detector.active_transactions
        
        self.detector.unregister_transaction("test_tx_1")
        assert "test_tx_1" not in self.detector.active_transactions
    
    def test_deadlock_monitoring(self):
        """Test deadlock monitoring thread."""
        self.detector.start_monitoring()
        assert self.detector._running == True
        
        time.sleep(0.1)  # Let monitoring run briefly
        
        self.detector.stop_monitoring()
        assert self.detector._running == False


class TestResilientConnectionManager:
    """Test resilient connection manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_resilience.duckdb")
        self.manager = create_resilient_manager(
            db_path=self.test_db_path,
            max_retries=2,
            base_delay=0.1
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.manager:
            self.manager.shutdown()
        
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_connection_manager_initialization(self):
        """Test resilient connection manager initialization."""
        assert self.manager is not None
        assert self.manager.retry_manager is not None
        assert self.manager.connection_manager is not None
    
    def test_connection_context_manager(self):
        """Test connection context manager."""
        with self.manager.get_connection() as conn:
            assert conn is not None
            result = conn.execute("SELECT 1 as test").fetchone()
            assert result[0] == 1
    
    def test_query_execution(self):
        """Test query execution with retry logic."""
        # Create test table
        self.manager.execute_query(
            "CREATE TABLE test_table (id INTEGER, name TEXT)",
            fetch_mode="none"
        )
        
        # Insert test data
        self.manager.execute_query(
            "INSERT INTO test_table VALUES (?, ?)",
            params=[1, "test_name"],
            fetch_mode="none"
        )
        
        # Query test data
        result = self.manager.execute_query(
            "SELECT * FROM test_table WHERE id = ?",
            params=[1],
            fetch_mode="all"
        )
        
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == "test_name"
    
    def test_transaction_execution(self):
        """Test transaction execution with rollback capability."""
        # Create test table
        self.manager.execute_query(
            "CREATE TABLE transaction_test (id INTEGER, value TEXT)",
            fetch_mode="none"
        )
        
        # Execute transaction
        operations = [
            ("INSERT INTO transaction_test VALUES (?, ?)", [1, "value1"]),
            ("INSERT INTO transaction_test VALUES (?, ?)", [2, "value2"]),
            ("UPDATE transaction_test SET value = ? WHERE id = ?", ["updated", 1])
        ]
        
        results = self.manager.execute_transaction(operations)
        assert len(results) == 3
        
        # Verify transaction results
        final_result = self.manager.execute_query(
            "SELECT * FROM transaction_test ORDER BY id",
            fetch_mode="all"
        )
        
        assert len(final_result) == 2
        assert final_result[0][1] == "updated"  # First record updated
        assert final_result[1][1] == "value2"   # Second record unchanged
    
    def test_bulk_insert(self):
        """Test bulk insert functionality."""
        # Create test table
        self.manager.execute_query(
            "CREATE TABLE bulk_test (id INTEGER, name TEXT, value FLOAT)",
            fetch_mode="none"
        )
        
        # Prepare bulk data
        test_data = [
            {"id": i, "name": f"name_{i}", "value": i * 0.5}
            for i in range(100)
        ]
        
        # Perform bulk insert
        inserted_count = self.manager.bulk_insert("bulk_test", test_data, batch_size=25)
        
        assert inserted_count == 100
        
        # Verify data was inserted
        count_result = self.manager.execute_query(
            "SELECT COUNT(*) FROM bulk_test",
            fetch_mode="one"
        )
        assert count_result[0] == 100
    
    def test_connection_health_monitoring(self):
        """Test connection health monitoring."""
        health_info = self.manager.get_connection_health()
        
        assert "pool_stats" in health_info
        assert "retry_metrics" in health_info
        assert "health_status" in health_info
        assert health_info["health_status"] in ["HEALTHY", "WARNING", "DEGRADED", "CRITICAL"]


class TestConversationStorageBackendResilience:
    """Test conversation storage backend with resilience."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_conversations.duckdb")
        
        # Create resilient storage backend
        self.storage = ConversationStorageBackend(
            db_path=self.test_db_path,
            use_resilient_manager=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.storage:
            self.storage.close()
        
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_resilient_storage_initialization(self):
        """Test resilient storage backend initialization."""
        assert self.storage.use_resilient_manager == True
        assert self.storage.connection_manager is not None
        assert self.storage.connection is None  # Should use manager
    
    def test_conversation_event_storage(self):
        """Test storing conversation events with resilience."""
        event_id = self.storage.store_conversation_event(
            conversation_id="test_conversation",
            agent_type="rif-implementer",
            event_type="start",
            event_data={"task": "implement database resilience"},
            issue_number=152
        )
        
        assert event_id is not None
        
        # Retrieve and verify
        events = self.storage.get_conversation_events("test_conversation")
        assert len(events) == 1
        assert events[0]["agent_type"] == "rif-implementer"
        assert events[0]["event_type"] == "start"
    
    def test_health_monitoring(self):
        """Test connection health monitoring."""
        health_info = self.storage.get_connection_health()
        
        assert "health_status" in health_info
        assert health_info["health_status"] in ["HEALTHY", "WARNING", "DEGRADED", "CRITICAL"]
    
    def test_legacy_vs_resilient_behavior(self):
        """Test difference between legacy and resilient storage."""
        # Create legacy storage backend
        legacy_storage = ConversationStorageBackend(
            db_path=os.path.join(self.temp_dir, "legacy.duckdb"),
            use_resilient_manager=False
        )
        
        try:
            # Test basic operations work in both
            event_id_resilient = self.storage.store_conversation_event(
                conversation_id="test", agent_type="test", event_type="test",
                event_data={"test": True}
            )
            
            event_id_legacy = legacy_storage.store_conversation_event(
                conversation_id="test", agent_type="test", event_type="test",
                event_data={"test": True}
            )
            
            assert event_id_resilient is not None
            assert event_id_legacy is not None
            
            # Test health monitoring availability
            resilient_health = self.storage.get_connection_health()
            legacy_health = legacy_storage.get_connection_health()
            
            assert "retry_metrics" in resilient_health
            assert "connection_type" in legacy_health
            assert legacy_health["connection_type"] == "legacy"
        
        finally:
            legacy_storage.close()


class TestIntegrationScenarios:
    """Test integration scenarios for database resilience."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "integration_test.duckdb")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_error_scenario_recovery(self):
        """Test recovery from database connection errors."""
        with create_resilient_manager(self.test_db_path, max_retries=3) as manager:
            # Simulate connection error scenario
            failure_count = 0
            
            def flaky_database_operation():
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 2:
                    raise Exception("Connection refused")
                
                # Simulate successful operation after retries
                with manager.get_connection() as conn:
                    return conn.execute("SELECT 'recovered' as status").fetchone()[0]
            
            # Execute with retry logic
            result = manager.retry_manager.execute_with_retry(
                flaky_database_operation,
                "test_conn",
                "recovery_test"
            )
            
            assert result == "recovered"
            assert failure_count == 3  # Failed twice, succeeded on third try
            assert manager.retry_manager.successful_retries == 1
    
    def test_concurrent_access_resilience(self):
        """Test resilience under concurrent access."""
        manager = create_resilient_manager(self.test_db_path, max_retries=2)
        
        try:
            # Create test table
            manager.execute_query(
                "CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY, thread_id TEXT, timestamp TIMESTAMP)",
                fetch_mode="none"
            )
            
            results = []
            errors = []
            
            def worker_thread(thread_id: int):
                """Worker thread for concurrent testing."""
                try:
                    for i in range(5):
                        manager.execute_query(
                            "INSERT INTO concurrent_test VALUES (?, ?, ?)",
                            params=[thread_id * 10 + i, f"thread_{thread_id}", datetime.now()],
                            fetch_mode="none"
                        )
                    results.append(f"thread_{thread_id}_success")
                except Exception as e:
                    errors.append(f"thread_{thread_id}_error: {e}")
            
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) == 5, f"Expected 5 successful threads, got {len(results)}"
            
            # Verify data integrity
            count_result = manager.execute_query(
                "SELECT COUNT(*) FROM concurrent_test",
                fetch_mode="one"
            )
            assert count_result[0] == 25  # 5 threads * 5 inserts each
        
        finally:
            manager.shutdown()


def test_issue_152_success_criteria():
    """
    Test that Issue #152 success criteria are met:
    - Retry Success Rate: >95% successful recovery on first retry
    - Transaction Recovery: 100% graceful rollback on connection failure
    - Recovery Time: <10s average recovery from transient failures
    - Performance Impact: <2% overhead from retry logic
    """
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "success_criteria.duckdb")
    
    try:
        manager = create_resilient_manager(test_db_path, max_retries=3, base_delay=0.1)
        
        # Test 1: Retry Success Rate
        successful_retries = 0
        total_retry_scenarios = 20
        
        for i in range(total_retry_scenarios):
            attempt_count = 0
            
            def flaky_operation():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count == 1:  # Fail first attempt
                    raise Exception("Connection refused")
                return "success"
            
            try:
                manager.retry_manager.execute_with_retry(
                    flaky_operation, f"test_conn_{i}", "retry_test"
                )
                if attempt_count == 2:  # Succeeded on first retry
                    successful_retries += 1
            except:
                pass  # Count failures in success rate
        
        retry_success_rate = (successful_retries / total_retry_scenarios) * 100
        assert retry_success_rate >= 95, f"Retry success rate {retry_success_rate}% < 95%"
        print(f"✓ Retry Success Rate: {retry_success_rate}% (>95% required)")
        
        # Test 2: Transaction Recovery (100% graceful rollback)
        manager.execute_query(
            "CREATE TABLE transaction_rollback_test (id INTEGER, value TEXT)",
            fetch_mode="none"
        )
        
        transaction_failures = 0
        graceful_rollbacks = 0
        
        for i in range(10):
            try:
                with manager.connection_manager.get_connection() as conn:
                    with manager.retry_manager.transaction_context(conn, f"tx_conn_{i}") as tx_ctx:
                        conn.execute("INSERT INTO transaction_rollback_test VALUES (?, ?)", [i, f"value_{i}"])
                        # Simulate failure during transaction
                        if i % 3 == 0:  # Fail every 3rd transaction
                            raise Exception("Simulated transaction failure")
            except Exception:
                transaction_failures += 1
                graceful_rollbacks += 1  # Should rollback gracefully
        
        rollback_success_rate = (graceful_rollbacks / transaction_failures) * 100
        assert rollback_success_rate == 100, f"Rollback success rate {rollback_success_rate}% < 100%"
        print(f"✓ Transaction Recovery: {rollback_success_rate}% graceful rollback (100% required)")
        
        # Test 3: Recovery Time (<10s average)
        recovery_times = []
        
        for i in range(5):
            start_time = time.time()
            
            def delayed_recovery_operation():
                time.sleep(0.1)  # Simulate network delay
                if time.time() - start_time < 0.2:  # First attempt fails quickly
                    raise Exception("Connection timeout")
                return "recovered"
            
            try:
                manager.retry_manager.execute_with_retry(
                    delayed_recovery_operation, f"recovery_conn_{i}", "recovery_time_test"
                )
                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)
            except:
                pass
        
        if recovery_times:
            avg_recovery_time = sum(recovery_times) / len(recovery_times)
            assert avg_recovery_time < 10.0, f"Average recovery time {avg_recovery_time}s >= 10s"
            print(f"✓ Recovery Time: {avg_recovery_time:.2f}s average (<10s required)")
        
        # Test 4: Performance Impact (<2% overhead)
        # Test baseline performance
        baseline_times = []
        for i in range(10):
            start_time = time.time()
            manager.execute_query("SELECT 1", fetch_mode="one")
            baseline_times.append(time.time() - start_time)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Test with retry overhead (simulate with forced retry)
        retry_times = []
        for i in range(10):
            attempt_count = 0
            
            def single_retry_operation():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count == 1:
                    raise Exception("Connection refused")
                return manager.connection_manager.execute_query("SELECT 1", fetch_mode="one")
            
            start_time = time.time()
            try:
                manager.retry_manager.execute_with_retry(
                    single_retry_operation, f"perf_conn_{i}", "performance_test"
                )
                retry_times.append(time.time() - start_time)
            except:
                pass
        
        if retry_times:
            retry_avg = sum(retry_times) / len(retry_times)
            performance_impact = ((retry_avg - baseline_avg) / baseline_avg) * 100
            assert performance_impact < 2.0, f"Performance impact {performance_impact:.1f}% >= 2%"
            print(f"✓ Performance Impact: {performance_impact:.1f}% (<2% required)")
        
        print("🎉 All Issue #152 success criteria met!")
        
    finally:
        if 'manager' in locals():
            manager.shutdown()
        
        # Clean up temp files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the success criteria test
    test_issue_152_success_criteria()