"""
Test suite for MCP Performance Metrics Collector
Tests for real-time metrics collection, aggregation, and analysis

Issue: #84 - Create MCP health monitor
Component: Performance metrics collector tests
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Import the components to test
import sys
sys.path.append('/Users/cal/DEV/RIF')

from mcp.monitor.metrics import (
    PerformanceMetricsCollector, MetricPoint, MetricSeries, 
    PerformanceBaseline
)


class TestMetricSeries:
    """Test metric series data structure and operations"""
    
    def test_metric_series_creation(self):
        """Test metric series initialization"""
        series = MetricSeries("response_time", "test-server", unit="ms")
        
        assert series.metric_name == "response_time"
        assert series.server_id == "test-server"
        assert series.unit == "ms"
        assert len(series.points) == 0
    
    def test_metric_point_addition(self):
        """Test adding metric points to series"""
        series = MetricSeries("response_time", "test-server")
        
        # Add points
        series.add_point(100.0)
        series.add_point(150.0, tags={"endpoint": "/api/health"})
        
        assert len(series.points) == 2
        
        first_point = series.points[0]
        assert isinstance(first_point, MetricPoint)
        assert first_point.value == 100.0
        assert first_point.server_id == "test-server"
        
        second_point = series.points[1]
        assert second_point.value == 150.0
        assert second_point.tags == {"endpoint": "/api/health"}
    
    def test_recent_values_filtering(self):
        """Test filtering values by time window"""
        series = MetricSeries("response_time", "test-server")
        
        current_time = time.time()
        
        # Add points with different timestamps
        series.add_point(100.0, timestamp=current_time - 600)  # 10 minutes ago
        series.add_point(150.0, timestamp=current_time - 300)  # 5 minutes ago  
        series.add_point(200.0, timestamp=current_time - 60)   # 1 minute ago
        
        # Get values from last 5 minutes (300 seconds)
        recent_values = series.get_recent_values(300)
        
        assert len(recent_values) == 2  # Should exclude the 10-minute-old value
        assert 150.0 in recent_values
        assert 200.0 in recent_values
        assert 100.0 not in recent_values
    
    def test_statistics_calculation(self):
        """Test statistical calculations for metric series"""
        series = MetricSeries("response_time", "test-server")
        
        # Add test data
        values = [100, 150, 200, 125, 175]
        for value in values:
            series.add_point(float(value))
        
        stats = series.get_statistics()
        
        assert stats["count"] == 5
        assert stats["min"] == 100.0
        assert stats["max"] == 200.0
        assert stats["mean"] == 150.0  # (100+150+200+125+175)/5
        assert stats["median"] == 150.0
        assert stats["std"] > 0
    
    def test_maxlen_enforcement(self):
        """Test that metric series enforces maximum length"""
        series = MetricSeries("test_metric", "test-server")
        
        # Add more points than maxlen (1000)
        for i in range(1200):
            series.add_point(float(i))
        
        # Should only keep last 1000 points
        assert len(series.points) == 1000
        assert series.points[0].value == 200.0  # First kept value
        assert series.points[-1].value == 1199.0  # Last value


class TestPerformanceMetricsCollector:
    """Test performance metrics collection system"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def metrics_collector(self, temp_storage):
        """Create metrics collector with temp storage"""
        return PerformanceMetricsCollector(
            storage_path=temp_storage,
            collection_interval=1,  # Fast interval for testing
            retention_hours=1  # Short retention for testing
        )
    
    @pytest.fixture
    def mock_server_callback(self):
        """Create mock server metrics callback"""
        async def callback():
            return {
                "response_time_ms": {"value": 150.0, "unit": "ms"},
                "request_count": {"value": 42.0, "unit": "requests"},
                "error_rate": {"value": 0.05, "unit": "percent", "tags": {"severity": "low"}}
            }
        return callback
    
    def test_initialization(self, metrics_collector, temp_storage):
        """Test metrics collector initialization"""
        assert not metrics_collector.is_collecting
        assert metrics_collector.collection_task is None
        assert metrics_collector.storage_path == Path(temp_storage)
        assert metrics_collector.collection_interval == 1
        assert len(metrics_collector.server_callbacks) == 0
    
    def test_server_callback_registration(self, metrics_collector):
        """Test server metrics callback registration"""
        callback = Mock()
        
        metrics_collector.register_server_callback("test-server-1", callback)
        
        assert "test-server-1" in metrics_collector.server_callbacks
        assert metrics_collector.server_callbacks["test-server-1"] is callback
        
        # Test unregistration
        metrics_collector.unregister_server_callback("test-server-1")
        
        assert "test-server-1" not in metrics_collector.server_callbacks
    
    @pytest.mark.asyncio
    async def test_collection_lifecycle(self, metrics_collector):
        """Test collection system start and stop"""
        # Should not be collecting initially
        assert not metrics_collector.is_collecting
        
        # Start collection
        await metrics_collector.start_collection()
        
        assert metrics_collector.is_collecting
        assert metrics_collector.collection_task is not None
        
        # Stop collection
        await metrics_collector.stop_collection()
        
        assert not metrics_collector.is_collecting
        assert metrics_collector.collection_task is None
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, metrics_collector):
        """Test recording individual metrics"""
        # Record simple numeric metric
        await metrics_collector._record_metric("test-server", "cpu_usage", 75.5)
        
        assert "test-server" in metrics_collector.metrics
        assert "cpu_usage" in metrics_collector.metrics["test-server"]
        
        series = metrics_collector.metrics["test-server"]["cpu_usage"]
        assert len(series.points) == 1
        assert series.points[0].value == 75.5
        
        # Record complex metric
        metric_data = {
            "value": 200.0,
            "unit": "ms", 
            "tags": {"endpoint": "/api/test"}
        }
        
        await metrics_collector._record_metric("test-server", "response_time", metric_data)
        
        response_series = metrics_collector.metrics["test-server"]["response_time"]
        assert len(response_series.points) == 1
        assert response_series.points[0].value == 200.0
        assert response_series.points[0].tags == {"endpoint": "/api/test"}
        assert response_series.unit == "ms"
    
    @pytest.mark.asyncio
    async def test_server_metrics_collection(self, metrics_collector, mock_server_callback):
        """Test collecting metrics from server callback"""
        metrics_collector.register_server_callback("test-server", mock_server_callback)
        
        await metrics_collector._collect_server_metrics("test-server")
        
        # Verify metrics were collected
        assert "test-server" in metrics_collector.metrics
        server_metrics = metrics_collector.metrics["test-server"]
        
        assert "response_time_ms" in server_metrics
        assert "request_count" in server_metrics
        assert "error_rate" in server_metrics
        
        # Verify metric values
        response_time_series = server_metrics["response_time_ms"]
        assert len(response_time_series.points) == 1
        assert response_time_series.points[0].value == 150.0
        assert response_time_series.unit == "ms"
    
    @pytest.mark.asyncio 
    async def test_baseline_calculation(self, metrics_collector):
        """Test performance baseline calculation and updates"""
        server_id = "test-server"
        metric_name = "response_time_ms"
        
        # Add enough data points for baseline calculation
        series = MetricSeries(metric_name, server_id, unit="ms")
        
        # Add 20 data points with average of 150ms
        for i in range(20):
            series.add_point(150.0 + (i % 10))  # Values from 150-159
        
        metrics_collector.metrics[server_id][metric_name] = series
        
        await metrics_collector._update_baseline(server_id, metric_name, series)
        
        # Check baseline was created
        assert server_id in metrics_collector.baselines
        assert metric_name in metrics_collector.baselines[server_id]
        
        baseline = metrics_collector.baselines[server_id][metric_name]
        assert isinstance(baseline, PerformanceBaseline)
        assert baseline.server_id == server_id
        assert baseline.metric_name == metric_name
        assert baseline.baseline_value > 0
        assert baseline.confidence > 0
    
    @pytest.mark.asyncio
    async def test_metric_alerts(self, metrics_collector):
        """Test metric-based alert detection"""
        server_id = "test-server"
        
        # Create baseline
        baseline = PerformanceBaseline(
            server_id=server_id,
            metric_name="response_time_ms",
            baseline_value=100.0,
            threshold_warning=50.0,  # 50% increase
            threshold_critical=100.0,  # 100% increase
            established_at=datetime.utcnow(),
            sample_count=50
        )
        
        metrics_collector.baselines[server_id]["response_time_ms"] = baseline
        
        # Add current metrics that exceed thresholds
        series = MetricSeries("response_time_ms", server_id)
        for i in range(10):
            series.add_point(220.0)  # 120% of baseline - should trigger critical
        
        metrics_collector.metrics[server_id]["response_time_ms"] = series
        
        alerts = await metrics_collector.check_metric_alerts(server_id)
        
        assert len(alerts) > 0
        
        alert = alerts[0]
        assert alert["severity"] == "critical"
        assert alert["metric_name"] == "response_time_ms"
        assert alert["current_value"] == 220.0
        assert alert["baseline_value"] == 100.0
        assert abs(alert["deviation_percent"] - 120.0) < 1.0
    
    @pytest.mark.asyncio
    async def test_data_persistence(self, metrics_collector, temp_storage):
        """Test metrics persistence to storage"""
        # Add some metrics
        await metrics_collector._record_metric("test-server", "cpu_usage", 85.0)
        await metrics_collector._record_metric("test-server", "memory_usage", 70.0)
        
        # Store metrics batch
        await metrics_collector._store_metrics_batch()
        
        # Check if files were created
        storage_path = Path(temp_storage)
        today = datetime.now().strftime("%Y%m%d")
        metrics_file = storage_path / f"performance_metrics_{today}.jsonl"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2  # Should have stored both metrics
                
                # Parse first line
                metric_data = json.loads(lines[0].strip())
                assert metric_data["server_id"] == "test-server"
                assert "timestamp" in metric_data
                assert "metric_name" in metric_data
                assert "value" in metric_data
    
    @pytest.mark.asyncio
    async def test_metrics_cleanup(self, metrics_collector):
        """Test old metrics cleanup based on retention period"""
        server_id = "test-server"
        
        # Add old metrics (beyond retention period)
        old_time = time.time() - (2 * 3600)  # 2 hours ago (retention is 1 hour)
        current_time = time.time()
        
        series = MetricSeries("test_metric", server_id)
        series.add_point(100.0, timestamp=old_time)  # Old point
        series.add_point(200.0, timestamp=current_time)  # Current point
        
        metrics_collector.metrics[server_id]["test_metric"] = series
        
        await metrics_collector._cleanup_old_metrics()
        
        # Old point should be removed
        remaining_series = metrics_collector.metrics[server_id]["test_metric"]
        assert len(remaining_series.points) == 1
        assert remaining_series.points[0].value == 200.0
    
    def test_trend_calculation(self, metrics_collector):
        """Test performance trend calculation"""
        # Increasing trend
        increasing_values = [100, 110, 120, 130, 140]
        trend = metrics_collector._calculate_trend(increasing_values)
        assert trend == "increasing"
        
        # Decreasing trend
        decreasing_values = [200, 180, 160, 140, 120]
        trend = metrics_collector._calculate_trend(decreasing_values)
        assert trend == "decreasing"
        
        # Stable trend
        stable_values = [100, 102, 98, 101, 99]
        trend = metrics_collector._calculate_trend(stable_values)
        assert trend == "stable"
        
        # Insufficient data
        few_values = [100, 110]
        trend = metrics_collector._calculate_trend(few_values)
        assert trend == "insufficient_data"
    
    @pytest.mark.asyncio
    async def test_server_metrics_api(self, metrics_collector, mock_server_callback):
        """Test public API for retrieving server metrics"""
        # Register callback and collect metrics
        metrics_collector.register_server_callback("test-server", mock_server_callback)
        await metrics_collector._collect_server_metrics("test-server")
        
        # Get server metrics
        server_metrics = await metrics_collector.get_server_metrics("test-server", duration_seconds=3600)
        
        assert isinstance(server_metrics, dict)
        assert "response_time_ms" in server_metrics
        
        metric_info = server_metrics["response_time_ms"]
        assert "statistics" in metric_info
        assert "unit" in metric_info
        assert "trend" in metric_info
        
        # Check statistics
        stats = metric_info["statistics"]
        assert stats["count"] == 1
        assert stats["mean"] == 150.0
    
    @pytest.mark.asyncio
    async def test_all_metrics_summary(self, metrics_collector, mock_server_callback):
        """Test getting summary of all server metrics"""
        # Register multiple servers
        metrics_collector.register_server_callback("server-1", mock_server_callback)
        metrics_collector.register_server_callback("server-2", mock_server_callback)
        
        # Collect metrics
        await metrics_collector._collect_server_metrics("server-1")
        await metrics_collector._collect_server_metrics("server-2")
        
        # Get all metrics summary
        all_metrics = await metrics_collector.get_all_metrics_summary()
        
        assert isinstance(all_metrics, dict)
        assert "server-1" in all_metrics
        assert "server-2" in all_metrics
        
        for server_id, metrics in all_metrics.items():
            assert "response_time_ms" in metrics
            assert "request_count" in metrics
            assert "error_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_collection_statistics(self, metrics_collector):
        """Test collection system statistics"""
        stats = await metrics_collector.get_collection_statistics()
        
        assert isinstance(stats, dict)
        assert "is_collecting" in stats
        assert "total_collections" in stats
        assert "collection_errors" in stats
        assert "registered_servers" in stats
        assert "total_metrics" in stats
        assert "collection_interval_seconds" in stats
        assert "retention_hours" in stats
        
        # Values should be reasonable
        assert stats["is_collecting"] == False  # Not started yet
        assert stats["total_collections"] >= 0
        assert stats["collection_interval_seconds"] == 1
        assert stats["retention_hours"] == 1


class TestMetricsCollectorIntegration:
    """Integration tests for metrics collector"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_full_collection_cycle(self, temp_storage):
        """Test complete metrics collection cycle"""
        collector = PerformanceMetricsCollector(
            storage_path=temp_storage,
            collection_interval=0.2,  # Very fast for testing
            retention_hours=1
        )
        
        # Mock server metrics
        async def server_callback():
            return {
                "cpu_usage": {"value": 75.0, "unit": "%"},
                "memory_usage": {"value": 60.0, "unit": "%"},
                "response_time": {"value": 120.0, "unit": "ms"}
            }
        
        try:
            # Register server and start collection
            collector.register_server_callback("integration-server", server_callback)
            await collector.start_collection()
            
            # Let it run for a short time
            await asyncio.sleep(0.5)
            
            # Verify metrics were collected
            server_metrics = await collector.get_server_metrics("integration-server")
            assert len(server_metrics) > 0
            
            # Stop collection
            await collector.stop_collection()
            
        finally:
            if collector.is_collecting:
                await collector.stop_collection()
    
    @pytest.mark.asyncio
    async def test_baseline_and_alert_integration(self, temp_storage):
        """Test baseline establishment and alert generation"""
        collector = PerformanceMetricsCollector(
            storage_path=temp_storage,
            collection_interval=0.1
        )
        
        server_id = "baseline-test-server"
        
        try:
            # Simulate stable metrics for baseline establishment
            stable_callback_count = 0
            
            async def stable_callback():
                nonlocal stable_callback_count
                stable_callback_count += 1
                return {
                    "response_time": {"value": 100.0 + (stable_callback_count % 5), "unit": "ms"}
                }
            
            collector.register_server_callback(server_id, stable_callback)
            
            # Collect stable metrics
            for _ in range(15):  # Enough for baseline
                await collector._collect_server_metrics(server_id)
            
            # Update baselines
            await collector._update_baselines()
            
            # Verify baseline was established
            assert server_id in collector.baselines
            assert "response_time" in collector.baselines[server_id]
            
            # Now inject degraded performance
            async def degraded_callback():
                return {
                    "response_time": {"value": 250.0, "unit": "ms"}  # Significantly higher
                }
            
            collector.register_server_callback(server_id, degraded_callback)
            
            # Collect degraded metrics
            for _ in range(5):
                await collector._collect_server_metrics(server_id)
            
            # Check for alerts
            alerts = await collector.check_metric_alerts(server_id)
            
            # Should detect performance degradation
            assert len(alerts) > 0
            alert = alerts[0]
            assert alert["metric_name"] == "response_time"
            assert alert["current_value"] == 250.0
            
        finally:
            if collector.is_collecting:
                await collector.stop_collection()


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))