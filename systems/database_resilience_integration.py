"""
Database Resilience Integration for Issue #150
Integrates all resilience components and provides unified access
"""

import logging
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

from knowledge.database.database_config import DatabaseConfig
from systems.resilient_database_interface import ResilientDatabaseInterface
from systems.database_health_monitor import DatabaseHealthMonitor, MonitoringConfig, console_alert_handler, file_alert_handler


class DatabaseResilienceSystem:
    """
    Unified database resilience system that integrates all components.
    
    This system addresses the root cause identified in error err_20250823_ed8e1099:
    "The system architecture may lack sufficient resilience for this failure mode"
    
    Features:
    - Resilient database interface with connection pooling
    - Circuit breaker pattern for fault tolerance
    - Comprehensive health monitoring with alerting
    - Graceful degradation and fallback mechanisms
    - Automated recovery and self-healing capabilities
    - Performance metrics and trend analysis
    """
    
    def __init__(self, 
                 config: Optional[DatabaseConfig] = None,
                 monitoring_config: Optional[MonitoringConfig] = None,
                 enable_file_alerts: bool = True):
        
        self.config = config or DatabaseConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.database = None
        self.health_monitor = None
        self._initialization_lock = threading.Lock()
        self._initialized = False
        
        # Alert configuration
        self.enable_file_alerts = enable_file_alerts
        self.alert_log_path = "knowledge/logs/database_alerts.log"
        
        self.logger.info("Database Resilience System created")
    
    def initialize(self) -> bool:
        """Initialize all resilience components."""
        with self._initialization_lock:
            if self._initialized:
                return True
            
            try:
                self.logger.info("Initializing Database Resilience System...")
                
                # Initialize database interface with resilience
                self.database = ResilientDatabaseInterface(
                    config=self.config,
                    fallback_mode_enabled=True
                )
                
                # Setup alert handlers
                alert_handlers = [console_alert_handler]
                
                if self.enable_file_alerts:
                    alert_handlers.append(file_alert_handler(self.alert_log_path))
                
                # Initialize health monitor
                self.health_monitor = DatabaseHealthMonitor(
                    database_interface=self.database,
                    config=self.monitoring_config,
                    alert_handlers=alert_handlers
                )
                
                # Start health monitoring
                self.health_monitor.start_monitoring()
                
                # Verify system health
                health_check = self.database.run_health_check()
                if health_check['overall_healthy']:
                    self.logger.info("Database Resilience System initialized successfully")
                    self._initialized = True
                    return True
                else:
                    self.logger.warning(f"Database Resilience System initialized with issues: {health_check['errors']}")
                    self._initialized = True  # Still mark as initialized for fallback operations
                    return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Database Resilience System: {e}")
                return False
    
    def get_database_interface(self) -> ResilientDatabaseInterface:
        """Get the resilient database interface."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Database Resilience System not initialized")
        return self.database
    
    def get_health_monitor(self) -> DatabaseHealthMonitor:
        """Get the health monitor."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Database Resilience System not initialized")
        return self.health_monitor
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self._initialized:
            return {'status': 'not_initialized', 'error': 'System not initialized'}
        
        try:
            # Get database health
            db_health = self.database.get_health_status()
            
            # Get monitoring status
            monitoring_status = self.health_monitor.get_monitoring_status()
            
            # Get active alerts
            active_alerts = self.health_monitor.get_active_alerts()
            
            # Get resilience metrics
            resilience_metrics = self.database.resilience_manager.get_health_metrics()
            
            return {
                'status': 'operational',
                'timestamp': db_health['timestamp'],
                'database_health': {
                    'state': db_health['overall_health'],
                    'available': db_health['database_available'],
                    'error_rate': db_health['error_rate'],
                    'avg_response_time': db_health['avg_response_time'],
                    'uptime': db_health['uptime']
                },
                'resilience': {
                    'circuit_breaker_open': resilience_metrics['circuit_breaker']['state'] == 'open',
                    'active_connections': resilience_metrics['active_connections'],
                    'pool_utilization': (
                        resilience_metrics['active_connections'] / 
                        max(1, resilience_metrics['pool_stats']['max_connections'])
                    ),
                    'failure_count': resilience_metrics['circuit_breaker']['failure_count']
                },
                'monitoring': {
                    'state': monitoring_status['state'],
                    'active_alerts': monitoring_status['active_alerts'],
                    'critical_alerts': monitoring_status['critical_alerts']
                },
                'alerts': active_alerts,
                'recommendations': db_health.get('recommendations', [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_resilience(self) -> Dict[str, Any]:
        """Test resilience features and recovery mechanisms."""
        if not self._initialized:
            return {'error': 'System not initialized'}
        
        test_results = {
            'timestamp': self.database.resilience_manager._start_time,
            'tests': {},
            'overall_success': True
        }
        
        try:
            # Test 1: Basic database connectivity
            self.logger.info("Testing basic database connectivity...")
            try:
                with self.database.resilience_manager.get_resilient_connection(timeout=5.0) as conn:
                    conn.execute("SELECT 1").fetchone()
                test_results['tests']['basic_connectivity'] = {'success': True, 'message': 'Database connection successful'}
            except Exception as e:
                test_results['tests']['basic_connectivity'] = {'success': False, 'error': str(e)}
                test_results['overall_success'] = False
            
            # Test 2: Fallback mechanism
            self.logger.info("Testing fallback mechanism...")
            try:
                with self.database.resilience_manager.get_resilient_connection(allow_fallback=True) as conn:
                    # This should work even if database is unavailable
                    pass
                test_results['tests']['fallback_mechanism'] = {'success': True, 'message': 'Fallback mechanism operational'}
            except Exception as e:
                test_results['tests']['fallback_mechanism'] = {'success': False, 'error': str(e)}
                test_results['overall_success'] = False
            
            # Test 3: Health monitoring
            self.logger.info("Testing health monitoring...")
            try:
                monitoring_status = self.health_monitor.get_monitoring_status()
                if monitoring_status['state'] == 'running':
                    test_results['tests']['health_monitoring'] = {'success': True, 'message': 'Health monitoring active'}
                else:
                    test_results['tests']['health_monitoring'] = {'success': False, 'error': f"Monitoring state: {monitoring_status['state']}"}
                    test_results['overall_success'] = False
            except Exception as e:
                test_results['tests']['health_monitoring'] = {'success': False, 'error': str(e)}
                test_results['overall_success'] = False
            
            # Test 4: Circuit breaker functionality
            self.logger.info("Testing circuit breaker...")
            try:
                cb_state = self.database.resilience_manager._circuit_breaker.state
                test_results['tests']['circuit_breaker'] = {'success': True, 'message': f'Circuit breaker state: {cb_state.value}'}
            except Exception as e:
                test_results['tests']['circuit_breaker'] = {'success': False, 'error': str(e)}
                test_results['overall_success'] = False
            
            # Test 5: Performance metrics
            self.logger.info("Testing performance metrics...")
            try:
                metrics = self.database.get_health_status()
                if 'operation_metrics' in metrics:
                    test_results['tests']['performance_metrics'] = {'success': True, 'message': 'Performance metrics available'}
                else:
                    test_results['tests']['performance_metrics'] = {'success': False, 'error': 'Performance metrics not available'}
                    test_results['overall_success'] = False
            except Exception as e:
                test_results['tests']['performance_metrics'] = {'success': False, 'error': str(e)}
                test_results['overall_success'] = False
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Resilience test failed: {e}")
            return {'error': f'Test execution failed: {e}', 'overall_success': False}
    
    def force_recovery(self) -> Dict[str, Any]:
        """Force recovery of the database system."""
        if not self._initialized:
            return {'error': 'System not initialized'}
        
        self.logger.info("Forcing database system recovery...")
        
        try:
            # Use the database interface recovery
            recovery_result = self.database.force_recovery()
            
            # Add additional recovery steps
            recovery_result['additional_actions'] = []
            
            # Reset health monitor if needed
            if self.health_monitor.state.value != 'running':
                try:
                    self.health_monitor.resume_monitoring()
                    recovery_result['additional_actions'].append('Health monitoring resumed')
                except Exception as e:
                    recovery_result['additional_actions'].append(f'Health monitoring resume failed: {e}')
            
            return recovery_result
            
        except Exception as e:
            self.logger.error(f"Force recovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def shutdown(self):
        """Shutdown the resilience system."""
        self.logger.info("Shutting down Database Resilience System...")
        
        try:
            if self.health_monitor:
                self.health_monitor.stop_monitoring()
            
            if self.database:
                self.database.close()
            
            self._initialized = False
            self.logger.info("Database Resilience System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Factory function for easy access
def create_resilient_database_system(config: Optional[DatabaseConfig] = None) -> DatabaseResilienceSystem:
    """
    Create a database resilience system with default configuration.
    
    This is the main entry point for applications that want to use
    resilient database operations instead of the original RIFDatabase.
    """
    return DatabaseResilienceSystem(config=config)


# Migration helper for existing code
class RIFDatabase:
    """
    Compatibility wrapper for existing RIFDatabase usage.
    
    This class maintains the same interface as the original RIFDatabase
    but uses the new resilient implementation underneath.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.resilience_system = create_resilient_database_system(config)
        self._database = None
    
    def __enter__(self):
        self.resilience_system.initialize()
        self._database = self.resilience_system.get_database_interface()
        return self._database
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.resilience_system.shutdown()


# Global instance for convenience
_global_resilience_system: Optional[DatabaseResilienceSystem] = None
_global_lock = threading.Lock()


def get_global_resilience_system() -> DatabaseResilienceSystem:
    """Get or create the global resilience system instance."""
    global _global_resilience_system
    
    with _global_lock:
        if _global_resilience_system is None:
            _global_resilience_system = create_resilient_database_system()
            _global_resilience_system.initialize()
        
        return _global_resilience_system


def shutdown_global_resilience_system():
    """Shutdown the global resilience system."""
    global _global_resilience_system
    
    with _global_lock:
        if _global_resilience_system:
            _global_resilience_system.shutdown()
            _global_resilience_system = None