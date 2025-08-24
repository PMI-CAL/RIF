#!/usr/bin/env python3
"""
RIF System Monitoring and Metrics Collection
Tracks performance, memory usage, and knowledge system health
"""

import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import yaml
import logging
import statistics
import os

@dataclass
class MetricData:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    unit: str = ""
    
    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }

@dataclass 
class Alert:
    """Alert information"""
    name: str
    severity: str
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    resolved: bool = False
    
    def to_dict(self):
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "resolved": self.resolved
        }

class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer: List[MetricData] = []
        self.buffer_lock = threading.Lock()
        self.logger = self._setup_logging()
        
        # Metric collection intervals
        self.collection_interval = self._parse_interval(
            config.get("monitoring", {}).get("collection", {}).get("metrics_interval", "30s")
        )
        
        # Performance tracking
        self.latency_trackers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor"""
        logger = logging.getLogger("rif_monitor")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path(self.config.get("storage", {}).get("paths", {}).get("logs", "/tmp"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "monitoring.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _parse_interval(self, interval_str: str) -> float:
        """Parse interval string to seconds"""
        if interval_str.endswith("s"):
            return float(interval_str[:-1])
        elif interval_str.endswith("m"):
            return float(interval_str[:-1]) * 60
        elif interval_str.endswith("h"):
            return float(interval_str[:-1]) * 3600
        else:
            return float(interval_str)
            
    def collect_memory_metrics(self):
        """Collect memory usage metrics"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            self._add_metric("memory.system.total", memory.total, {"type": "system"}, "bytes")
            self._add_metric("memory.system.available", memory.available, {"type": "system"}, "bytes")
            self._add_metric("memory.system.percent", memory.percent, {"type": "system"}, "%")
            
            # Process-specific memory (if we can find relevant processes)
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    
                    # Track Python processes (likely our agents/LightRAG)
                    if 'python' in proc_name:
                        mem_mb = proc_info['memory_info'].rss / (1024 * 1024)
                        self._add_metric(
                            "memory.process.rss", 
                            mem_mb,
                            {"process": proc_name, "pid": str(proc_info['pid'])},
                            "MB"
                        )
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error collecting memory metrics: {e}")
            
    def collect_disk_metrics(self):
        """Collect disk usage metrics for knowledge storage"""
        try:
            storage_paths = self.config.get("storage", {}).get("paths", {})
            
            for path_name, path_str in storage_paths.items():
                path = Path(path_str)
                if path.exists():
                    usage = psutil.disk_usage(str(path))
                    
                    self._add_metric(
                        f"disk.{path_name}.total", 
                        usage.total, 
                        {"path": path_str}, 
                        "bytes"
                    )
                    self._add_metric(
                        f"disk.{path_name}.used", 
                        usage.used, 
                        {"path": path_str}, 
                        "bytes"
                    )
                    self._add_metric(
                        f"disk.{path_name}.percent", 
                        (usage.used / usage.total) * 100, 
                        {"path": path_str}, 
                        "%"
                    )
                    
        except Exception as e:
            self.logger.error(f"Error collecting disk metrics: {e}")
            
    def track_latency(self, operation_name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Track latency for an operation"""
        if tags is None:
            tags = {}
            
        tags["operation"] = operation_name
        
        # Store in buffer for alerts/dashboard
        self._add_metric(f"latency.{operation_name}", duration_ms, tags, "ms")
        
        # Store in rolling window for statistics
        self.latency_trackers[operation_name].append(duration_ms)
        
        # Calculate and store statistics
        if len(self.latency_trackers[operation_name]) >= 5:
            values = list(self.latency_trackers[operation_name])
            
            self._add_metric(f"latency.{operation_name}.avg", statistics.mean(values), tags, "ms")
            self._add_metric(f"latency.{operation_name}.p50", statistics.median(values), tags, "ms")
            
            if len(values) >= 10:
                self._add_metric(f"latency.{operation_name}.p95", 
                               statistics.quantiles(values, n=20)[18], tags, "ms")  # 95th percentile
                               
    def track_indexing_performance(self, docs_processed: int, elapsed_time: float):
        """Track knowledge indexing performance"""
        if elapsed_time > 0:
            rate = docs_processed / elapsed_time
            self._add_metric("indexing.documents_per_second", rate, {}, "docs/s")
            
        self._add_metric("indexing.documents_processed", docs_processed, {}, "count")
        self._add_metric("indexing.batch_time", elapsed_time, {}, "seconds")
        
    def track_shadow_mode_comparison(self, accuracy: float, performance_delta: float):
        """Track shadow mode comparison results"""
        self._add_metric("shadow_mode.response_accuracy", accuracy, {}, "%")
        self._add_metric("shadow_mode.performance_delta", performance_delta, {}, "%")
        
    def track_consensus_metrics(self, voting_session: Dict[str, Any]):
        """Track consensus and voting metrics from a voting session"""
        try:
            # Calculate agreement level (0.0-1.0 scale)
            agreement_level = self._calculate_agreement_level(voting_session)
            self._add_metric("consensus.agreement_level", agreement_level, 
                           {"session_id": voting_session.get("id", "unknown")}, "ratio")
            
            # Track decision time
            duration_ms = voting_session.get("duration", 0)
            self._add_metric("consensus.decision_time", duration_ms,
                           {"session_id": voting_session.get("id", "unknown")}, "ms")
            
            # Identify and track dissenting agents
            dissenters = self._identify_dissenters(voting_session)
            self._add_metric("consensus.dissenter_count", len(dissenters),
                           {"session_id": voting_session.get("id", "unknown")}, "count")
            
            # Track confidence distribution
            confidence_stats = self._analyze_confidence_distribution(voting_session)
            self._add_metric("consensus.confidence_avg", confidence_stats.get("average", 0.0),
                           {"session_id": voting_session.get("id", "unknown")}, "ratio")
            self._add_metric("consensus.confidence_std", confidence_stats.get("std_dev", 0.0),
                           {"session_id": voting_session.get("id", "unknown")}, "ratio")
            
            # Track consensus type and outcome
            consensus_type = voting_session.get("consensus_type", "unknown")
            outcome = voting_session.get("outcome", "unknown")
            self._add_metric("consensus.session_outcome", 1.0 if outcome == "agreed" else 0.0,
                           {"consensus_type": consensus_type, "outcome": outcome}, "boolean")
            
            # Store detailed session data for historical analysis
            self._store_consensus_session(voting_session, agreement_level, dissenters, confidence_stats)
            
        except Exception as e:
            self.logger.error(f"Error tracking consensus metrics: {e}")
            
    def _calculate_agreement_level(self, voting_session: Dict[str, Any]) -> float:
        """Calculate agreement level (0.0 = complete disagreement, 1.0 = unanimous)"""
        votes = voting_session.get("votes", [])
        if not votes:
            return 0.0
            
        # Count votes by decision
        vote_counts = {}
        total_weight = 0
        
        for vote in votes:
            decision = vote.get("decision", "abstain")
            weight = vote.get("weight", 1.0)
            
            vote_counts[decision] = vote_counts.get(decision, 0) + weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        # Find the majority decision and calculate agreement level
        if vote_counts:
            max_votes = max(vote_counts.values())
            agreement_level = max_votes / total_weight
            return min(agreement_level, 1.0)
        
        return 0.0
        
    def _identify_dissenters(self, voting_session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify agents who voted against the majority decision"""
        votes = voting_session.get("votes", [])
        if not votes:
            return []
            
        # Find majority decision
        vote_counts = {}
        for vote in votes:
            decision = vote.get("decision", "abstain")
            weight = vote.get("weight", 1.0)
            vote_counts[decision] = vote_counts.get(decision, 0) + weight
            
        if not vote_counts:
            return []
            
        majority_decision = max(vote_counts, key=vote_counts.get)
        
        # Find dissenters
        dissenters = []
        for vote in votes:
            if vote.get("decision") != majority_decision:
                dissenters.append({
                    "agent": vote.get("agent", "unknown"),
                    "decision": vote.get("decision", "unknown"),
                    "confidence": vote.get("confidence", 0.0),
                    "reasoning": vote.get("reasoning", "")
                })
                
        return dissenters
        
    def _analyze_confidence_distribution(self, voting_session: Dict[str, Any]) -> Dict[str, float]:
        """Analyze confidence score distribution across all votes"""
        votes = voting_session.get("votes", [])
        confidences = [vote.get("confidence", 0.0) for vote in votes if "confidence" in vote]
        
        if not confidences:
            return {"average": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
            
        return {
            "average": statistics.mean(confidences),
            "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            "min": min(confidences),
            "max": max(confidences)
        }
        
    def _store_consensus_session(self, voting_session: Dict[str, Any], agreement_level: float, 
                                dissenters: List[Dict[str, Any]], confidence_stats: Dict[str, float]):
        """Store detailed consensus session data for historical analysis"""
        try:
            # Prepare consensus session record
            session_record = {
                "timestamp": datetime.now().isoformat(),
                "session_id": voting_session.get("id", f"session_{int(time.time())}"),
                "issue_number": voting_session.get("issue_number"),
                "consensus_type": voting_session.get("consensus_type", "unknown"),
                "outcome": voting_session.get("outcome", "unknown"),
                "agreement_level": agreement_level,
                "decision_time_ms": voting_session.get("duration", 0),
                "total_votes": len(voting_session.get("votes", [])),
                "dissenter_count": len(dissenters),
                "dissenters": dissenters,
                "confidence_stats": confidence_stats,
                "votes_summary": self._summarize_votes(voting_session.get("votes", []))
            }
            
            # Store in consensus history file
            consensus_dir = Path(self.config.get("storage", {}).get("paths", {}).get("consensus", 
                                                                   "/tmp/consensus"))
            consensus_dir.mkdir(parents=True, exist_ok=True)
            
            today = datetime.now().strftime("%Y%m%d")
            consensus_file = consensus_dir / f"consensus_sessions_{today}.jsonl"
            
            with open(consensus_file, 'a') as f:
                f.write(json.dumps(session_record) + "\n")
                
        except Exception as e:
            self.logger.error(f"Error storing consensus session data: {e}")
            
    def _summarize_votes(self, votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of votes for storage"""
        summary = {
            "decisions": {},
            "average_confidence": 0.0,
            "agents": []
        }
        
        total_confidence = 0
        confidence_count = 0
        
        for vote in votes:
            decision = vote.get("decision", "abstain")
            confidence = vote.get("confidence", 0.0)
            agent = vote.get("agent", "unknown")
            
            # Count decisions
            summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
            
            # Track confidence
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
                
            # Track agents
            summary["agents"].append({
                "agent": agent,
                "decision": decision,
                "confidence": confidence
            })
            
        if confidence_count > 0:
            summary["average_confidence"] = total_confidence / confidence_count
            
        return summary
        
    def _add_metric(self, name: str, value: float, tags: Dict[str, str], unit: str = ""):
        """Add metric to collection buffer"""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags,
            unit=unit
        )
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
            
    def flush_metrics(self) -> List[Dict[str, Any]]:
        """Flush collected metrics and return them"""
        with self.buffer_lock:
            metrics = [m.to_dict() for m in self.metrics_buffer]
            self.metrics_buffer.clear()
            return metrics

class AnomalyDetector:
    """Detects anomalies in metrics using various algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("anomaly_detection", {})
        self.enabled = self.config.get("enabled", False)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baselines: Dict[str, Dict[str, float]] = {}
        
    def add_metric_value(self, metric_name: str, value: float):
        """Add a metric value to the anomaly detection system"""
        if not self.enabled:
            return
            
        self.metric_history[metric_name].append(value)
        self._update_baseline(metric_name)
        
    def _update_baseline(self, metric_name: str):
        """Update baseline statistics for a metric"""
        values = list(self.metric_history[metric_name])
        if len(values) < 10:  # Need minimum data points
            return
            
        self.baselines[metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values)
        }
        
    def check_anomaly(self, metric_name: str, value: float) -> Optional[str]:
        """Check if a metric value is anomalous"""
        if not self.enabled or metric_name not in self.baselines:
            return None
            
        baseline = self.baselines[metric_name]
        
        # Statistical threshold detection
        if "statistical_threshold" in [a["name"] for a in self.config.get("algorithms", [])]:
            threshold_multiplier = 3  # 3 standard deviations
            
            if abs(value - baseline["mean"]) > threshold_multiplier * baseline["std"]:
                return f"Value {value} is {threshold_multiplier}+ standard deviations from mean {baseline['mean']:.2f}"
                
        # Simple range-based detection  
        extreme_threshold = 5  # Very basic threshold
        if value > baseline["max"] * extreme_threshold or value < baseline["min"] / extreme_threshold:
            return f"Value {value} is extremely outside normal range [{baseline['min']:.2f}, {baseline['max']:.2f}]"
            
        return None

class AlertManager:
    """Manages alerts based on metric thresholds and anomalies"""
    
    def __init__(self, config: Dict[str, Any], base_path: Path):
        self.config = config.get("alerts", {})
        self.enabled = self.config.get("enabled", False)
        self.base_path = base_path
        self.alert_history: Dict[str, datetime] = {}  # Prevent spam
        self.active_alerts: Dict[str, Alert] = {}
        
        self.logger = logging.getLogger("rif_alert_manager")
        
        # Setup alert log file
        if self.config.get("channels", {}).get("log_file", {}).get("enabled", False):
            log_path = Path(self.config["channels"]["log_file"]["path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - ALERT - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def check_alert_rules(self, metrics: List[Dict[str, Any]]):
        """Check alert rules against current metrics"""
        if not self.enabled:
            return
            
        rules = self.config.get("rules", [])
        
        for rule in rules:
            rule_name = rule["name"]
            condition = rule["condition"]
            severity = rule["severity"]
            frequency_str = rule.get("frequency", "5m")
            message_template = rule["message"]
            
            # Simple condition parsing (would need more sophisticated parser for production)
            triggered = self._evaluate_condition(condition, metrics)
            
            if triggered:
                # Check frequency throttling
                last_alert = self.alert_history.get(rule_name)
                frequency_seconds = self._parse_interval(frequency_str)
                
                if last_alert and (datetime.now() - last_alert).total_seconds() < frequency_seconds:
                    continue  # Skip due to throttling
                    
                # Create alert
                metric_value = self._extract_metric_value(condition, metrics)
                message = message_template.replace("{{value}}", str(metric_value))
                
                alert = Alert(
                    name=rule_name,
                    severity=severity,
                    message=message,
                    timestamp=datetime.now(),
                    metric_name=self._extract_metric_name(condition),
                    metric_value=metric_value
                )
                
                self._trigger_alert(alert)
                self.alert_history[rule_name] = datetime.now()
                self.active_alerts[rule_name] = alert
                
    def _evaluate_condition(self, condition: str, metrics: List[Dict[str, Any]]) -> bool:
        """Simple condition evaluation"""
        # This is a basic implementation - would need proper parsing for production
        try:
            # Extract metric name and threshold
            if " > " in condition:
                metric_name, threshold_str = condition.split(" > ")
                threshold = float(threshold_str)
                
                # Find the metric in current batch
                for metric in metrics:
                    if metric["name"] == metric_name.strip():
                        return metric["value"] > threshold
                        
            elif " < " in condition:
                metric_name, threshold_str = condition.split(" < ")
                threshold = float(threshold_str)
                
                for metric in metrics:
                    if metric["name"] == metric_name.strip():
                        return metric["value"] < threshold
                        
        except (ValueError, KeyError):
            pass
            
        return False
        
    def _extract_metric_value(self, condition: str, metrics: List[Dict[str, Any]]) -> float:
        """Extract the actual metric value that triggered the condition"""
        metric_name = self._extract_metric_name(condition)
        
        for metric in metrics:
            if metric["name"] == metric_name:
                return metric["value"]
                
        return 0.0
        
    def _extract_metric_name(self, condition: str) -> str:
        """Extract metric name from condition"""
        if " > " in condition:
            return condition.split(" > ")[0].strip()
        elif " < " in condition:
            return condition.split(" < ")[0].strip()
        return ""
        
    def _parse_interval(self, interval_str: str) -> float:
        """Parse interval string to seconds"""
        if interval_str.endswith("s"):
            return float(interval_str[:-1])
        elif interval_str.endswith("m"):
            return float(interval_str[:-1]) * 60
        elif interval_str.endswith("h"):
            return float(interval_str[:-1]) * 3600
        else:
            return float(interval_str)
            
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert through configured channels"""
        # Log to file
        if self.config.get("channels", {}).get("log_file", {}).get("enabled", False):
            self.logger.warning(f"{alert.severity.upper()}: {alert.message}")
            
        # Console output
        if self.config.get("channels", {}).get("console", {}).get("enabled", False):
            print(f"🚨 ALERT [{alert.severity.upper()}]: {alert.message}")
            
        # Save to alerts directory
        alerts_dir = self.base_path / "knowledge" / "monitoring" / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        
        alert_file = alerts_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alert.name}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2)

class MonitoringDashboard:
    """Simple dashboard for monitoring data"""
    
    def __init__(self, config: Dict[str, Any], base_path: Path):
        self.config = config.get("dashboard", {})
        self.enabled = self.config.get("enabled", False)
        self.base_path = base_path
        
    def generate_report(self, metrics: List[Dict[str, Any]], alerts: List[Alert]) -> Dict[str, Any]:
        """Generate a monitoring report"""
        if not self.enabled:
            return {}
            
        # Organize metrics by type
        organized_metrics = defaultdict(list)
        for metric in metrics:
            category = metric["name"].split(".")[0]
            organized_metrics[category].append(metric)
            
        # Calculate summary statistics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "metrics_collected": len(metrics),
            "active_alerts": len([a for a in alerts if not a.resolved]),
            "categories": list(organized_metrics.keys())
        }
        
        # Recent performance
        recent_performance = {}
        for category, cat_metrics in organized_metrics.items():
            if cat_metrics:
                values = [m["value"] for m in cat_metrics]
                recent_performance[category] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
                
        report = {
            "summary": summary,
            "performance": recent_performance,
            "recent_alerts": [a.to_dict() for a in alerts[-10:]],  # Last 10 alerts
            "metrics_by_category": dict(organized_metrics)
        }
        
        # Save report
        if self.config.get("export", {}).get("enabled", False):
            reports_dir = Path(self.config["export"]["path"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report

class LatencyTracker:
    """Decorator and context manager for tracking operation latency"""
    
    def __init__(self, monitor: MetricsCollector, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.monitor.track_latency(self.operation_name, duration_ms)
            
    def __call__(self, func):
        """Use as decorator"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

class SystemMonitor:
    """Main monitoring system coordinator"""
    
    def __init__(self, config_path: str):
        self.base_path = Path(config_path).parent
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.alert_manager = AlertManager(self.config, self.base_path)
        self.dashboard = MonitoringDashboard(self.config, self.base_path)
        
        self.running = False
        self.collection_thread = None
        
        self.logger = logging.getLogger("rif_system_monitor")
        
    def start(self):
        """Start the monitoring system"""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._monitoring_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.logger.info("RIF System Monitor started")
        
    def stop(self):
        """Stop the monitoring system"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
            
        self.logger.info("RIF System Monitor stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                self.collector.collect_memory_metrics()
                self.collector.collect_disk_metrics()
                
                # Flush metrics and check for alerts
                metrics = self.collector.flush_metrics()
                
                # Update anomaly detection
                for metric in metrics:
                    self.anomaly_detector.add_metric_value(metric["name"], metric["value"])
                    
                # Check alert rules
                self.alert_manager.check_alert_rules(metrics)
                
                # Generate dashboard data (periodically)
                if len(metrics) > 0:
                    active_alerts = list(self.alert_manager.active_alerts.values())
                    report = self.dashboard.generate_report(metrics, active_alerts)
                    
                # Store metrics
                self._store_metrics(metrics)
                
                # Wait for next collection
                time.sleep(self.collector.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause on error
                
    def _store_metrics(self, metrics: List[Dict[str, Any]]):
        """Store metrics to file system"""
        if not metrics:
            return
            
        storage_config = self.config.get("storage", {})
        metrics_dir = Path(storage_config.get("paths", {}).get("metrics", "/tmp/metrics"))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Store in daily files
        today = datetime.now().strftime("%Y%m%d")
        metrics_file = metrics_dir / f"metrics_{today}.jsonl"
        
        with open(metrics_file, 'a') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")
                
    def get_latency_tracker(self, operation_name: str) -> LatencyTracker:
        """Get a latency tracker for an operation"""
        return LatencyTracker(self.collector, operation_name)

# Convenience functions for easy integration
_monitor_instance: Optional[SystemMonitor] = None

def initialize_monitoring(config_path: str = "/Users/cal/DEV/RIF/config/monitoring.yaml"):
    """Initialize the global monitoring system"""
    global _monitor_instance
    _monitor_instance = SystemMonitor(config_path)
    _monitor_instance.start()
    return _monitor_instance

def get_monitor() -> Optional[SystemMonitor]:
    """Get the global monitoring instance"""
    return _monitor_instance

def track_latency(operation_name: str):
    """Decorator to track function latency"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if _monitor_instance:
                with _monitor_instance.get_latency_tracker(operation_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_consensus_session(voting_session: Dict[str, Any]):
    """Track metrics from a consensus voting session"""
    if _monitor_instance:
        _monitor_instance.collector.track_consensus_metrics(voting_session)
    else:
        # Log warning if monitoring is not initialized
        import logging
        logging.getLogger("rif_consensus").warning("Monitoring not initialized - consensus metrics not tracked")

class ConsensusMonitor:
    """Standalone consensus monitoring class for direct integration"""
    
    def __init__(self, storage_path: str = "/Users/cal/DEV/RIF/knowledge/monitoring"):
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger("rif_consensus_monitor")
        
    def track_consensus(self, voting_session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for tracking consensus metrics
        Returns a comprehensive report of the consensus session
        """
        try:
            # Calculate core metrics
            agreement_level = self._calculate_agreement_level(voting_session)
            dissenters = self._identify_dissenters(voting_session)
            confidence_distribution = self._analyze_confidence_distribution(voting_session)
            decision_time = voting_session.get("duration", 0)
            
            # Generate comprehensive report
            report = {
                "session_id": voting_session.get("id", f"session_{int(time.time())}"),
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "agreement_level": agreement_level,
                    "dissenting_agents": [d["agent"] for d in dissenters],
                    "confidence_distribution": confidence_distribution,
                    "decision_time": decision_time,
                    "total_participants": len(voting_session.get("votes", [])),
                    "dissenter_count": len(dissenters)
                },
                "analysis": {
                    "consensus_strength": self._assess_consensus_strength(agreement_level, confidence_distribution),
                    "dissent_patterns": self._analyze_dissent_patterns(dissenters),
                    "efficiency_assessment": self._assess_decision_efficiency(decision_time, len(voting_session.get("votes", [])))
                },
                "recommendations": self._generate_recommendations(agreement_level, dissenters, confidence_distribution)
            }
            
            # Store metrics if monitoring is available
            if _monitor_instance:
                _monitor_instance.collector.track_consensus_metrics(voting_session)
            else:
                # Store directly
                self._store_metrics(report)
                
            return report
            
        except Exception as e:
            self.logger.error(f"Error tracking consensus: {e}")
            return {"error": str(e), "session_id": voting_session.get("id", "unknown")}
            
    def _calculate_agreement_level(self, voting_session: Dict[str, Any]) -> float:
        """Calculate agreement level (0.0 = complete disagreement, 1.0 = unanimous)"""
        votes = voting_session.get("votes", [])
        if not votes:
            return 0.0
            
        # Count votes by decision
        vote_counts = {}
        total_weight = 0
        
        for vote in votes:
            decision = vote.get("decision", "abstain")
            weight = vote.get("weight", 1.0)
            
            vote_counts[decision] = vote_counts.get(decision, 0) + weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        # Find the majority decision and calculate agreement level
        if vote_counts:
            max_votes = max(vote_counts.values())
            agreement_level = max_votes / total_weight
            return min(agreement_level, 1.0)
        
        return 0.0
        
    def _identify_dissenters(self, voting_session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify agents who voted against the majority decision"""
        votes = voting_session.get("votes", [])
        if not votes:
            return []
            
        # Find majority decision
        vote_counts = {}
        for vote in votes:
            decision = vote.get("decision", "abstain")
            weight = vote.get("weight", 1.0)
            vote_counts[decision] = vote_counts.get(decision, 0) + weight
            
        if not vote_counts:
            return []
            
        majority_decision = max(vote_counts, key=vote_counts.get)
        
        # Find dissenters
        dissenters = []
        for vote in votes:
            if vote.get("decision") != majority_decision:
                dissenters.append({
                    "agent": vote.get("agent", "unknown"),
                    "decision": vote.get("decision", "unknown"),
                    "confidence": vote.get("confidence", 0.0),
                    "reasoning": vote.get("reasoning", "")
                })
                
        return dissenters
        
    def _analyze_confidence_distribution(self, voting_session: Dict[str, Any]) -> Dict[str, float]:
        """Analyze confidence score distribution across all votes"""
        votes = voting_session.get("votes", [])
        confidences = [vote.get("confidence", 0.0) for vote in votes if "confidence" in vote]
        
        if not confidences:
            return {"average": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}
            
        return {
            "average": statistics.mean(confidences),
            "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            "min": min(confidences),
            "max": max(confidences)
        }
        
    def _assess_consensus_strength(self, agreement_level: float, confidence_dist: Dict[str, float]) -> str:
        """Assess the strength of the consensus"""
        avg_confidence = confidence_dist.get("average", 0.0)
        
        if agreement_level >= 0.9 and avg_confidence >= 0.8:
            return "strong"
        elif agreement_level >= 0.7 and avg_confidence >= 0.6:
            return "moderate"
        elif agreement_level >= 0.5 and avg_confidence >= 0.5:
            return "weak"
        elif agreement_level >= 0.5:
            return "weak"  # Low confidence but some agreement
        else:
            return "insufficient"
            
    def _analyze_dissent_patterns(self, dissenters: List[Dict[str, Any]]) -> List[str]:
        """Analyze patterns in dissenting votes"""
        patterns = []
        
        if not dissenters:
            return ["no_dissent"]
            
        # Check for high-confidence dissenters
        high_confidence_dissenters = [d for d in dissenters if d.get("confidence", 0) > 0.8]
        if high_confidence_dissenters:
            patterns.append("high_confidence_dissent")
            
        # Check for repeated dissenters (would need historical data)
        agent_names = [d.get("agent") for d in dissenters]
        if len(set(agent_names)) < len(agent_names):
            patterns.append("repeated_agent_dissent")
            
        # Check for reasoning patterns
        reasoning_types = [d.get("reasoning", "").split()[0].lower() for d in dissenters if d.get("reasoning")]
        if "security" in reasoning_types:
            patterns.append("security_concerns")
        if "performance" in reasoning_types:
            patterns.append("performance_concerns")
            
        return patterns or ["general_dissent"]
        
    def _assess_decision_efficiency(self, decision_time_ms: float, participant_count: int) -> str:
        """Assess the efficiency of the decision process"""
        if participant_count == 0:
            return "no_participants"
            
        # Calculate time per participant
        time_per_participant = decision_time_ms / participant_count
        
        if time_per_participant < 300000:  # < 5 minutes per participant
            return "efficient"
        elif time_per_participant < 900000:  # < 15 minutes per participant
            return "moderate"
        else:
            return "slow"
            
    def _generate_recommendations(self, agreement_level: float, dissenters: List[Dict[str, Any]], 
                                confidence_dist: Dict[str, float]) -> List[str]:
        """Generate recommendations based on consensus analysis"""
        recommendations = []
        
        # Agreement level recommendations
        if agreement_level < 0.7:  # Lowered threshold to catch moderate disagreement
            recommendations.append("Consider additional discussion or evidence gathering")
        if agreement_level < 0.5:
            recommendations.append("Strong disagreement detected - may need arbitration")
            
        # Confidence recommendations
        avg_confidence = confidence_dist.get("average", 0.0)
        if avg_confidence < 0.5:
            recommendations.append("Low confidence detected - gather more evidence")
            
        # Dissenter recommendations
        if len(dissenters) > 0:
            high_conf_dissenters = [d for d in dissenters if d.get("confidence", 0) > 0.7]
            if high_conf_dissenters:
                recommendations.append("High-confidence dissenters present - review their concerns")
            
            # If there are many dissenters relative to total participants
            if len(dissenters) >= 2:  # 2 or more dissenters suggests significant disagreement
                recommendations.append("Multiple dissenters detected - consider arbitration")
                
        if not recommendations:
            recommendations.append("Consensus appears healthy - proceed as planned")
            
        return recommendations
        
    def _store_metrics(self, report: Dict[str, Any]):
        """Store consensus metrics to file"""
        try:
            consensus_dir = self.storage_path / "consensus"
            consensus_dir.mkdir(parents=True, exist_ok=True)
            
            today = datetime.now().strftime("%Y%m%d")
            consensus_file = consensus_dir / f"consensus_reports_{today}.jsonl"
            
            with open(consensus_file, 'a') as f:
                f.write(json.dumps(report) + "\n")
                
        except Exception as e:
            self.logger.error(f"Error storing consensus metrics: {e}")
            
    def generate_report(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive consensus report"""
        return self.track_consensus(session_data)

if __name__ == "__main__":
    # Basic usage example
    monitor = initialize_monitoring()
    
    try:
        # Simulate some operations
        time.sleep(60)  # Let it collect some data
        
    except KeyboardInterrupt:
        monitor.stop()