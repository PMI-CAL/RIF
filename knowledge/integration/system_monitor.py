"""
SystemMonitor - Resource Management and Health Monitoring

Implements resource monitoring and management for the hybrid knowledge pipeline,
ensuring the system stays within the 2GB memory budget and 4 CPU core allocation
from the Master Coordination Plan.
"""

import os
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    disk_io_mb: float
    network_io_mb: float
    active_threads: int
    open_files: int


class SystemMonitor:
    """
    Real-time system monitoring for resource management and performance tracking.
    
    Features:
    - Memory pressure monitoring with 2GB limit
    - CPU usage tracking with 4-core allocation
    - I/O performance monitoring  
    - Component health tracking
    - Automatic alerts and throttling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize system monitoring with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Resource limits from master coordination plan
        self.memory_limit_mb = config.get('memory_limit_mb', 2048)
        self.cpu_cores = config.get('cpu_cores', 4)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # Resource tracking
        self.history = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.current_snapshot = None
        self.alerts = deque(maxlen=100)   # Keep last 100 alerts
        
        # Performance thresholds
        self.thresholds = {
            'memory_warning_percent': 80,      # 80% of limit
            'memory_critical_percent': 95,     # 95% of limit
            'cpu_warning_percent': 70,         # 70% CPU usage
            'cpu_critical_percent': 90,        # 90% CPU usage
            'disk_io_warning_mb_s': 50,        # 50MB/s sustained
            'response_time_warning_ms': 200,   # 200ms query response
            'error_rate_warning_per_min': 10   # 10 errors/minute
        }
        
        # Get process handle for monitoring
        self.process = psutil.Process()
        self.logger.info(f"SystemMonitor initialized with {self.memory_limit_mb}MB memory limit")
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start continuous system monitoring.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        with self._lock:
            if self._monitoring:
                self.logger.warning("System monitoring already running")
                return
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval_seconds,),
                daemon=True
            )
            self._monitor_thread.start()
            
            self.logger.info(f"System monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        with self._lock:
            if not self._monitoring:
                return
            
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=10)
                if self._monitor_thread.is_alive():
                    self.logger.warning("Monitor thread did not shut down cleanly")
            
            self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop running in background thread."""
        self.logger.info("System monitoring loop started")
        
        while self._monitoring:
            try:
                # Take resource snapshot
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self.history.append(snapshot)
                    self.current_snapshot = snapshot
                
                # Check thresholds and generate alerts
                self._check_thresholds(snapshot)
                
                # Sleep until next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
        
        self.logger.info("System monitoring loop stopped")
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        try:
            # Memory usage (RSS - Resident Set Size)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # CPU usage (percent over last interval)  
            cpu_percent = self.process.cpu_percent()
            
            # I/O statistics
            io_counters = self.process.io_counters()
            disk_io_mb = (io_counters.read_bytes + io_counters.write_bytes) / (1024 * 1024)
            
            # Network I/O (if available)
            network_io_mb = 0  # Placeholder - would need system-level monitoring
            
            # Thread and file handle counts
            active_threads = self.process.num_threads()
            open_files = len(self.process.open_files())
            
            return ResourceSnapshot(
                timestamp=time.time(),
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                disk_io_mb=disk_io_mb,
                network_io_mb=network_io_mb,
                active_threads=active_threads,
                open_files=open_files
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Could not get process info: {e}")
            # Return zero snapshot
            return ResourceSnapshot(
                timestamp=time.time(),
                memory_mb=0,
                cpu_percent=0,
                disk_io_mb=0,
                network_io_mb=0,
                active_threads=0,
                open_files=0
            )
    
    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check resource usage against thresholds and generate alerts."""
        
        # Memory threshold checks
        memory_percent = (snapshot.memory_mb / self.memory_limit_mb) * 100
        if memory_percent > self.thresholds['memory_critical_percent']:
            self._add_alert('CRITICAL', 'MEMORY', 
                           f'Memory usage at {memory_percent:.1f}% of limit ({snapshot.memory_mb:.0f}MB)')
        elif memory_percent > self.thresholds['memory_warning_percent']:
            self._add_alert('WARNING', 'MEMORY',
                           f'Memory usage at {memory_percent:.1f}% of limit ({snapshot.memory_mb:.0f}MB)')
        
        # CPU threshold checks  
        if snapshot.cpu_percent > self.thresholds['cpu_critical_percent']:
            self._add_alert('CRITICAL', 'CPU',
                           f'CPU usage at {snapshot.cpu_percent:.1f}%')
        elif snapshot.cpu_percent > self.thresholds['cpu_warning_percent']:
            self._add_alert('WARNING', 'CPU',
                           f'CPU usage at {snapshot.cpu_percent:.1f}%')
        
        # Thread count check (potential resource leak)
        if snapshot.active_threads > 20:  # Reasonable threshold
            self._add_alert('WARNING', 'THREADS',
                           f'High thread count: {snapshot.active_threads}')
        
        # File handle check (potential resource leak)
        if snapshot.open_files > 100:  # Reasonable threshold  
            self._add_alert('WARNING', 'FILES',
                           f'High open file count: {snapshot.open_files}')
    
    def _add_alert(self, level: str, category: str, message: str):
        """Add an alert to the alert queue."""
        alert = {
            'timestamp': time.time(),
            'level': level,
            'category': category,
            'message': message
        }
        
        with self._lock:
            self.alerts.append(alert)
        
        # Log the alert
        log_method = self.logger.critical if level == 'CRITICAL' else self.logger.warning
        log_method(f"{level} {category}: {message}")
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.current_snapshot:
            return self.current_snapshot.memory_mb
        else:
            # Take immediate snapshot
            snapshot = self._take_snapshot()
            return snapshot.memory_mb
    
    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        if self.current_snapshot:
            return self.current_snapshot.cpu_percent
        else:
            # Take immediate snapshot
            snapshot = self._take_snapshot()
            return snapshot.cpu_percent
    
    def get_resource_pressure(self) -> Dict[str, str]:
        """
        Get current resource pressure levels.
        
        Returns:
            Dict with pressure levels: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        """
        if not self.current_snapshot:
            return {'memory': 'UNKNOWN', 'cpu': 'UNKNOWN', 'overall': 'UNKNOWN'}
        
        snapshot = self.current_snapshot
        
        # Memory pressure
        memory_percent = (snapshot.memory_mb / self.memory_limit_mb) * 100
        if memory_percent > self.thresholds['memory_critical_percent']:
            memory_pressure = 'CRITICAL'
        elif memory_percent > self.thresholds['memory_warning_percent']:
            memory_pressure = 'HIGH'
        elif memory_percent > 50:
            memory_pressure = 'MEDIUM'
        else:
            memory_pressure = 'LOW'
        
        # CPU pressure
        if snapshot.cpu_percent > self.thresholds['cpu_critical_percent']:
            cpu_pressure = 'CRITICAL'
        elif snapshot.cpu_percent > self.thresholds['cpu_warning_percent']:
            cpu_pressure = 'HIGH'
        elif snapshot.cpu_percent > 40:
            cpu_pressure = 'MEDIUM' 
        else:
            cpu_pressure = 'LOW'
        
        # Overall pressure (worst of memory/CPU)
        pressure_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        overall_level = max(pressure_levels[memory_pressure], pressure_levels[cpu_pressure])
        overall_pressure = list(pressure_levels.keys())[overall_level]
        
        return {
            'memory': memory_pressure,
            'cpu': cpu_pressure,
            'overall': overall_pressure
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        with self._lock:
            return list(self.alerts)[-count:] if self.alerts else []
    
    def get_performance_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get performance statistics over a time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dict with performance statistics
        """
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_snapshots = [s for s in self.history if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {
                'window_minutes': window_minutes,
                'samples': 0,
                'memory': {},
                'cpu': {},
                'io': {}
            }
        
        # Calculate statistics
        memory_values = [s.memory_mb for s in recent_snapshots]
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        disk_io_values = [s.disk_io_mb for s in recent_snapshots]
        
        return {
            'window_minutes': window_minutes,
            'samples': len(recent_snapshots),
            'memory': {
                'avg_mb': sum(memory_values) / len(memory_values),
                'max_mb': max(memory_values),
                'min_mb': min(memory_values),
                'current_mb': memory_values[-1],
                'limit_mb': self.memory_limit_mb,
                'usage_percent': (memory_values[-1] / self.memory_limit_mb) * 100
            },
            'cpu': {
                'avg_percent': sum(cpu_values) / len(cpu_values),
                'max_percent': max(cpu_values),
                'min_percent': min(cpu_values),
                'current_percent': cpu_values[-1]
            },
            'io': {
                'avg_disk_mb': sum(disk_io_values) / len(disk_io_values),
                'max_disk_mb': max(disk_io_values),
                'total_disk_mb': disk_io_values[-1]
            }
        }
    
    def should_throttle_processing(self) -> bool:
        """
        Check if processing should be throttled due to resource pressure.
        
        Returns:
            bool: True if processing should be throttled
        """
        pressure = self.get_resource_pressure()
        return pressure['overall'] in ['HIGH', 'CRITICAL']
    
    def get_recommended_batch_size(self, default_size: int = 100) -> int:
        """
        Get recommended batch size based on current resource pressure.
        
        Args:
            default_size: Default batch size
            
        Returns:
            int: Recommended batch size
        """
        pressure = self.get_resource_pressure()
        
        if pressure['overall'] == 'CRITICAL':
            return max(1, default_size // 4)  # Quarter size
        elif pressure['overall'] == 'HIGH':
            return max(10, default_size // 2)  # Half size
        elif pressure['overall'] == 'MEDIUM':
            return max(25, int(default_size * 0.75))  # Three-quarters size
        else:
            return default_size
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary."""
        if not self.current_snapshot:
            return {'healthy': False, 'reason': 'No monitoring data available'}
        
        pressure = self.get_resource_pressure()
        recent_alerts = self.get_recent_alerts(5)
        critical_alerts = [a for a in recent_alerts if a['level'] == 'CRITICAL']
        
        # Health determination
        healthy = (
            pressure['overall'] not in ['CRITICAL'] and
            len(critical_alerts) == 0
        )
        
        health_score = 100
        if pressure['memory'] == 'CRITICAL':
            health_score -= 50
        elif pressure['memory'] == 'HIGH':
            health_score -= 25
        elif pressure['memory'] == 'MEDIUM':
            health_score -= 10
        
        if pressure['cpu'] == 'CRITICAL':
            health_score -= 30
        elif pressure['cpu'] == 'HIGH':
            health_score -= 15
        elif pressure['cpu'] == 'MEDIUM':
            health_score -= 5
        
        health_score -= len(critical_alerts) * 10
        health_score = max(0, health_score)
        
        return {
            'healthy': healthy,
            'health_score': health_score,
            'resource_pressure': pressure,
            'current_usage': {
                'memory_mb': self.current_snapshot.memory_mb,
                'memory_percent': (self.current_snapshot.memory_mb / self.memory_limit_mb) * 100,
                'cpu_percent': self.current_snapshot.cpu_percent,
                'threads': self.current_snapshot.active_threads,
                'open_files': self.current_snapshot.open_files
            },
            'recent_alerts': len(recent_alerts),
            'critical_alerts': len(critical_alerts),
            'monitoring_active': self._monitoring
        }
    
    def __del__(self):
        """Destructor to ensure monitoring is stopped."""
        try:
            self.stop_monitoring()
        except:
            pass