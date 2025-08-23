#!/usr/bin/env python3
"""
Tests for RIF System Monitoring
Comprehensive test suite for monitoring, metrics, alerts, and dashboard
"""

import unittest
import tempfile
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from claude.commands.system_monitor import (
    MetricData, Alert, MetricsCollector, AnomalyDetector, 
    AlertManager, MonitoringDashboard, SystemMonitor,
    LatencyTracker, initialize_monitoring, track_latency
)

class TestMetricData(unittest.TestCase):
    """Test MetricData dataclass"""
    
    def test_metric_data_creation(self):
        """Test creating a metric data point"""
        timestamp = datetime.now()
        metric = MetricData(
            name="test.metric",
            value=42.5,
            timestamp=timestamp,
            tags={"env": "test"},
            unit="ms"
        )
        
        self.assertEqual(metric.name, "test.metric")
        self.assertEqual(metric.value, 42.5)
        self.assertEqual(metric.timestamp, timestamp)
        self.assertEqual(metric.tags, {"env": "test"})
        self.assertEqual(metric.unit, "ms")
        
    def test_metric_to_dict(self):
        """Test converting metric to dictionary"""
        timestamp = datetime.now()
        metric = MetricData(
            name="test.metric",
            value=42.5,
            timestamp=timestamp,
            tags={"env": "test"},
            unit="ms"
        )
        
        result = metric.to_dict()
        expected = {
            "name": "test.metric",
            "value": 42.5,
            "timestamp": timestamp.isoformat(),
            "tags": {"env": "test"},
            "unit": "ms"
        }
        
        self.assertEqual(result, expected)

class TestAlert(unittest.TestCase):
    """Test Alert dataclass"""
    
    def test_alert_creation(self):
        """Test creating an alert"""
        timestamp = datetime.now()
        alert = Alert(
            name="test_alert",
            severity="warning",
            message="Test alert message",
            timestamp=timestamp,
            metric_name="test.metric",
            metric_value=100.0
        )
        
        self.assertEqual(alert.name, "test_alert")
        self.assertEqual(alert.severity, "warning")
        self.assertFalse(alert.resolved)
        
    def test_alert_to_dict(self):
        """Test converting alert to dictionary"""
        timestamp = datetime.now()
        alert = Alert(
            name="test_alert",
            severity="critical",
            message="Critical alert",
            timestamp=timestamp,
            metric_name="memory.usage",
            metric_value=95.0
        )
        
        result = alert.to_dict()
        self.assertEqual(result["name"], "test_alert")
        self.assertEqual(result["severity"], "critical")
        self.assertEqual(result["metric_value"], 95.0)
        self.assertFalse(result["resolved"])

class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector class"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            "monitoring": {
                "collection": {
                    "metrics_interval": "30s"
                }
            },
            "storage": {
                "paths": {
                    "logs": "/tmp/test_logs"
                }
            }
        }
        self.collector = MetricsCollector(self.config)
        
    def test_parse_interval(self):
        """Test interval parsing"""
        self.assertEqual(self.collector._parse_interval("30s"), 30.0)
        self.assertEqual(self.collector._parse_interval("5m"), 300.0)
        self.assertEqual(self.collector._parse_interval("2h"), 7200.0)
        self.assertEqual(self.collector._parse_interval("60"), 60.0)
        
    def test_add_metric(self):
        """Test adding a metric"""
        initial_count = len(self.collector.metrics_buffer)
        
        self.collector._add_metric("test.metric", 42.0, {"tag": "value"}, "unit")
        
        self.assertEqual(len(self.collector.metrics_buffer), initial_count + 1)
        metric = self.collector.metrics_buffer[-1]
        self.assertEqual(metric.name, "test.metric")
        self.assertEqual(metric.value, 42.0)
        
    def test_track_latency(self):
        """Test latency tracking"""
        self.collector.track_latency("test_operation", 150.0, {"env": "test"})
        
        # Check that metric was added
        metrics = self.collector.flush_metrics()
        latency_metrics = [m for m in metrics if m["name"].startswith("latency.test_operation")]
        self.assertGreater(len(latency_metrics), 0)
        
    def test_track_indexing_performance(self):
        """Test indexing performance tracking"""
        self.collector.track_indexing_performance(100, 10.0)
        
        metrics = self.collector.flush_metrics()
        rate_metrics = [m for m in metrics if m["name"] == "indexing.documents_per_second"]
        self.assertEqual(len(rate_metrics), 1)
        self.assertEqual(rate_metrics[0]["value"], 10.0)  # 100 docs / 10 seconds
        
    def test_flush_metrics(self):
        """Test flushing metrics"""
        self.collector._add_metric("test1", 1.0, {}, "")
        self.collector._add_metric("test2", 2.0, {}, "")
        
        metrics = self.collector.flush_metrics()
        self.assertEqual(len(metrics), 2)
        
        # Buffer should be empty after flush
        metrics2 = self.collector.flush_metrics()
        self.assertEqual(len(metrics2), 0)
        
    @patch('psutil.virtual_memory')
    def test_collect_memory_metrics(self, mock_memory):
        """Test memory metrics collection"""
        # Mock memory data
        mock_memory.return_value = Mock(
            total=16 * 1024 * 1024 * 1024,  # 16 GB
            available=8 * 1024 * 1024 * 1024,  # 8 GB
            percent=50.0
        )
        
        self.collector.collect_memory_metrics()
        metrics = self.collector.flush_metrics()
        
        memory_metrics = [m for m in metrics if m["name"].startswith("memory.system")]
        self.assertGreater(len(memory_metrics), 0)

class TestLatencyTracker(unittest.TestCase):
    """Test LatencyTracker context manager and decorator"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_collector = Mock()
        self.tracker = LatencyTracker(self.mock_collector, "test_operation")
        
    def test_context_manager(self):
        """Test using LatencyTracker as context manager"""
        with self.tracker:
            time.sleep(0.01)  # 10ms
            
        # Verify track_latency was called
        self.mock_collector.track_latency.assert_called_once()
        args = self.mock_collector.track_latency.call_args
        self.assertEqual(args[0][0], "test_operation")  # operation name
        self.assertGreater(args[0][1], 0)  # duration > 0
        
    def test_decorator(self):
        """Test using LatencyTracker as decorator"""
        @self.tracker
        def test_function():
            time.sleep(0.01)
            return "result"
            
        result = test_function()
        self.assertEqual(result, "result")
        self.mock_collector.track_latency.assert_called_once()

class TestAnomalyDetector(unittest.TestCase):
    """Test AnomalyDetector class"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            "anomaly_detection": {
                "enabled": True,
                "algorithms": [
                    {"name": "statistical_threshold", "enabled": True}
                ]
            }
        }
        self.detector = AnomalyDetector(self.config)
        
    def test_add_metric_values(self):
        """Test adding metric values for anomaly detection"""
        # Add some normal values
        for i in range(20):
            self.detector.add_metric_value("test.metric", 10.0 + i * 0.1)
            
        self.assertIn("test.metric", self.detector.metric_history)
        self.assertEqual(len(self.detector.metric_history["test.metric"]), 20)
        
    def test_baseline_calculation(self):
        """Test baseline statistics calculation"""
        # Add values to establish baseline
        values = [10, 12, 11, 13, 9, 14, 10, 12, 11, 13]
        for value in values:
            self.detector.add_metric_value("test.metric", value)
            
        self.assertIn("test.metric", self.detector.baselines)
        baseline = self.detector.baselines["test.metric"]
        
        self.assertAlmostEqual(baseline["mean"], 11.5, places=1)
        self.assertIn("std", baseline)
        self.assertIn("min", baseline)
        self.assertIn("max", baseline)
        
    def test_anomaly_detection(self):
        """Test anomaly detection algorithm"""
        # Establish baseline with normal values
        normal_values = [10, 12, 11, 13, 9, 14, 10, 12, 11, 13]
        for value in normal_values:
            self.detector.add_metric_value("test.metric", value)
            
        # Test normal value (should not be anomaly)
        result = self.detector.check_anomaly("test.metric", 12.0)
        self.assertIsNone(result)
        
        # Test extreme value (should be anomaly)
        result = self.detector.check_anomaly("test.metric", 100.0)
        self.assertIsNotNone(result)
        self.assertIn("standard deviations", result)

class TestAlertManager(unittest.TestCase):
    """Test AlertManager class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "alerts": {
                "enabled": True,
                "channels": {
                    "log_file": {
                        "enabled": True,
                        "path": f"{self.temp_dir}/alerts.log"
                    },
                    "console": {"enabled": True}
                },
                "rules": [
                    {
                        "name": "high_memory",
                        "condition": "memory.usage_percent > 80",
                        "severity": "warning",
                        "frequency": "1m",
                        "message": "Memory usage high: {{value}}%"
                    }
                ]
            }
        }
        self.alert_manager = AlertManager(self.config, Path(self.temp_dir))
        
    def test_parse_interval(self):
        """Test interval parsing in AlertManager"""
        self.assertEqual(self.alert_manager._parse_interval("30s"), 30.0)
        self.assertEqual(self.alert_manager._parse_interval("5m"), 300.0)
        
    def test_evaluate_condition(self):
        """Test condition evaluation"""
        metrics = [
            {"name": "memory.usage_percent", "value": 85.0},
            {"name": "disk.usage_percent", "value": 50.0}
        ]
        
        # Test condition that should trigger
        result = self.alert_manager._evaluate_condition("memory.usage_percent > 80", metrics)
        self.assertTrue(result)
        
        # Test condition that should not trigger
        result = self.alert_manager._evaluate_condition("memory.usage_percent > 90", metrics)
        self.assertFalse(result)
        
    def test_extract_metric_value(self):
        """Test extracting metric values"""
        metrics = [{"name": "memory.usage_percent", "value": 85.0}]
        
        value = self.alert_manager._extract_metric_value("memory.usage_percent > 80", metrics)
        self.assertEqual(value, 85.0)
        
    def test_check_alert_rules(self):
        """Test checking alert rules"""
        metrics = [{"name": "memory.usage_percent", "value": 85.0}]
        
        initial_alert_count = len(self.alert_manager.active_alerts)
        
        # This should trigger the high_memory alert
        self.alert_manager.check_alert_rules(metrics)
        
        # Check that alert was created
        self.assertGreater(len(self.alert_manager.active_alerts), initial_alert_count)
        self.assertIn("high_memory", self.alert_manager.active_alerts)

class TestMonitoringDashboard(unittest.TestCase):
    """Test MonitoringDashboard class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "dashboard": {
                "enabled": True,
                "export": {
                    "enabled": True,
                    "path": f"{self.temp_dir}/reports"
                }
            }
        }
        
        # Create test data directories
        metrics_dir = Path(self.temp_dir) / "knowledge" / "monitoring" / "metrics"
        alerts_dir = Path(self.temp_dir) / "knowledge" / "monitoring" / "alerts"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data
        today = datetime.now().strftime("%Y%m%d")
        metrics_file = metrics_dir / f"metrics_{today}.jsonl"
        
        sample_metrics = [
            {"name": "memory.usage_percent", "value": 75.0, "timestamp": datetime.now().isoformat()},
            {"name": "disk.usage_percent", "value": 45.0, "timestamp": datetime.now().isoformat()}
        ]
        
        with open(metrics_file, 'w') as f:
            for metric in sample_metrics:
                f.write(json.dumps(metric) + "\n")
                
        self.dashboard = MonitoringDashboard(self.config, Path(self.temp_dir))
        
    def test_generate_report(self):
        """Test generating monitoring report"""
        metrics = [
            {"name": "memory.usage_percent", "value": 75.0},
            {"name": "disk.usage_percent", "value": 45.0}
        ]
        alerts = []
        
        report = self.dashboard.generate_report(metrics, alerts)
        
        self.assertIn("summary", report)
        self.assertIn("performance", report)
        self.assertEqual(report["summary"]["metrics_collected"], 2)
        self.assertEqual(report["summary"]["active_alerts"], 0)

class TestSystemMonitor(unittest.TestCase):
    """Test SystemMonitor integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create monitoring config
        config = {
            "monitoring": {
                "enabled": True,
                "collection": {
                    "metrics_interval": "1s"
                }
            },
            "storage": {
                "paths": {
                    "metrics": f"{self.temp_dir}/metrics",
                    "logs": f"{self.temp_dir}/logs",
                    "alerts": f"{self.temp_dir}/alerts"
                }
            },
            "alerts": {"enabled": False},  # Disable for testing
            "anomaly_detection": {"enabled": False},
            "dashboard": {"enabled": False}
        }
        
        config_file = Path(self.temp_dir) / "monitoring.yaml"
        
        # We'll create a simple test config since we need YAML
        with open(config_file, 'w') as f:
            f.write(f"""
monitoring:
  enabled: true
  collection:
    metrics_interval: "1s"
storage:
  paths:
    metrics: "{self.temp_dir}/metrics"
    logs: "{self.temp_dir}/logs"
    alerts: "{self.temp_dir}/alerts"
alerts:
  enabled: false
anomaly_detection:
  enabled: false
dashboard:
  enabled: false
""")
        
        # Patch the config loading to return our test config
        with patch('builtins.open'), patch('yaml.safe_load', return_value=config):
            self.monitor = SystemMonitor(str(config_file))
            
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor.collector)
        self.assertIsNotNone(self.monitor.anomaly_detector)
        self.assertIsNotNone(self.monitor.alert_manager)
        self.assertIsNotNone(self.monitor.dashboard)
        
    def test_latency_tracker_creation(self):
        """Test creating latency trackers"""
        tracker = self.monitor.get_latency_tracker("test_operation")
        self.assertIsInstance(tracker, LatencyTracker)
        self.assertEqual(tracker.operation_name, "test_operation")

class TestIntegration(unittest.TestCase):
    """Integration tests for the monitoring system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # This test would verify the entire flow:
        # 1. Metrics collection
        # 2. Anomaly detection
        # 3. Alert generation
        # 4. Dashboard reporting
        
        # Create a minimal config
        config = {
            "monitoring": {"enabled": True, "collection": {"metrics_interval": "1s"}},
            "storage": {"paths": {"metrics": f"{self.temp_dir}/metrics"}},
            "alerts": {"enabled": True, "rules": []},
            "anomaly_detection": {"enabled": True},
            "dashboard": {"enabled": True}
        }
        
        collector = MetricsCollector(config)
        
        # Collect some metrics
        collector._add_metric("test.metric", 42.0, {}, "unit")
        collector.track_latency("test_op", 100.0)
        
        metrics = collector.flush_metrics()
        
        # Verify metrics were collected
        self.assertGreater(len(metrics), 0)
        
        # Test anomaly detection
        detector = AnomalyDetector(config)
        for metric in metrics:
            detector.add_metric_value(metric["name"], metric["value"])

class TestShadowModeCompatibility(unittest.TestCase):
    """Test shadow mode compatibility for issue #37"""
    
    def test_shadow_mode_tracking(self):
        """Test shadow mode comparison tracking"""
        config = {"monitoring": {"enabled": True}}
        collector = MetricsCollector(config)
        
        # Test shadow mode tracking
        collector.track_shadow_mode_comparison(95.5, 5.2)
        
        metrics = collector.flush_metrics()
        
        # Check that shadow mode metrics were recorded
        shadow_metrics = [m for m in metrics if m["name"].startswith("shadow_mode")]
        self.assertEqual(len(shadow_metrics), 2)
        
        accuracy_metrics = [m for m in metrics if m["name"] == "shadow_mode.response_accuracy"]
        self.assertEqual(len(accuracy_metrics), 1)
        self.assertEqual(accuracy_metrics[0]["value"], 95.5)
        
    def test_shadow_mode_alerts(self):
        """Test alerts for shadow mode divergence"""
        config = {
            "alerts": {
                "enabled": True,
                "channels": {"console": {"enabled": True}},
                "rules": [
                    {
                        "name": "shadow_divergence",
                        "condition": "shadow_mode.response_accuracy < 90",
                        "severity": "warning",
                        "frequency": "5m",
                        "message": "Shadow mode accuracy low: {{value}}%"
                    }
                ]
            }
        }
        
        alert_manager = AlertManager(config, Path("/tmp"))
        
        # Metrics that should trigger alert (accuracy < 90%)
        metrics = [{"name": "shadow_mode.response_accuracy", "value": 85.0}]
        
        alert_manager.check_alert_rules(metrics)
        
        # Check that alert was triggered
        self.assertIn("shadow_divergence", alert_manager.active_alerts)

# Test runner with performance tracking
class MonitoringTestRunner:
    """Custom test runner that tracks its own performance"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def run_tests(self):
        """Run all monitoring tests with performance tracking"""
        self.start_time = time.time()
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        self.end_time = time.time()
        
        # Print performance summary
        duration = self.end_time - self.start_time
        print(f"\n{'='*50}")
        print(f"Test Performance Summary:")
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Tests Run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        print(f"{'='*50}")
        
        return result.wasSuccessful()

if __name__ == "__main__":
    # Run with custom test runner
    runner = MonitoringTestRunner()
    success = runner.run_tests()
    
    sys.exit(0 if success else 1)