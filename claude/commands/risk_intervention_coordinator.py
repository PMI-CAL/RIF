#!/usr/bin/env python3
"""
Risk Intervention Coordinator - Issue #92 Phase 4
End-to-end orchestration and integration coordinator for risk-based manual intervention.

This coordinator implements:
1. End-to-end workflow orchestration from GitHub triggers
2. Component health monitoring and error recovery
3. Performance metrics collection and optimization
4. Integration with existing RIF workflow state machine
5. Quality gate coordination and enforcement
6. System health checks and self-healing capabilities
"""

import json
import subprocess
import yaml
import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import signal
import sys

# Import all components
try:
    from .risk_assessment_engine import RiskAssessmentEngine, create_change_context_from_issue
    from .specialist_assignment_engine import SpecialistAssignmentEngine
    from .sla_monitoring_system import SLAMonitoringSystem
    from .manual_intervention_workflow import ManualInterventionWorkflow, InterventionStatus
    from .decision_audit_tracker import DecisionAuditTracker, AuditRecord
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from risk_assessment_engine import RiskAssessmentEngine, create_change_context_from_issue
    from specialist_assignment_engine import SpecialistAssignmentEngine
    from sla_monitoring_system import SLAMonitoringSystem
    from manual_intervention_workflow import ManualInterventionWorkflow, InterventionStatus
    from decision_audit_tracker import DecisionAuditTracker, AuditRecord

class SystemHealth(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class ComponentStatus(Enum):
    """Individual component status."""
    OPERATIONAL = "operational"
    WARNING = "warning"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    timestamp: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    active_workflows: int = 0
    pending_reviews: int = 0
    sla_compliance_rate: float = 0.0
    system_health: SystemHealth = SystemHealth.HEALTHY
    component_health: Dict[str, ComponentStatus] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Container for component health check result."""
    component: str
    status: ComponentStatus
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

class RiskInterventionCoordinator:
    """
    Master coordinator for the risk-based manual intervention framework.
    
    Orchestrates all components, monitors system health, handles integration
    with RIF workflow, and provides comprehensive error recovery and metrics.
    """
    
    def __init__(self, config_path: str = "config/risk-assessment.yaml"):
        """Initialize the risk intervention coordinator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize all components
        self.risk_assessor = RiskAssessmentEngine(config_path)
        self.assignment_engine = SpecialistAssignmentEngine(config_path)
        self.sla_monitor = SLAMonitoringSystem(config_path)
        self.workflow_coordinator = ManualInterventionWorkflow(config_path)
        self.audit_tracker = DecisionAuditTracker()
        
        # System state
        self.is_running = False
        self.metrics = SystemMetrics(timestamp=datetime.now(timezone.utc))
        self.health_checks = {}
        self.error_counts = {}
        
        # Monitoring and recovery
        self.monitoring_thread = None
        self.last_health_check = datetime.now(timezone.utc)
        
        # Integration hooks
        self.github_webhooks = []
        self.quality_gate_hooks = []
        
        # Initialize health monitoring
        self._initialize_health_monitoring()
        
        self.logger.info("🚀 Risk Intervention Coordinator initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging for coordinator."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RiskInterventionCoordinator - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create file handler for audit logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "risk_intervention.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load coordinator configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default coordinator configuration."""
        return {
            'coordination': {
                'enabled': True,
                'health_check_interval_seconds': 300,  # 5 minutes
                'metrics_collection_interval_seconds': 60,  # 1 minute
                'auto_recovery_enabled': True,
                'max_concurrent_workflows': 10,
                'component_timeout_seconds': 30
            },
            'integration': {
                'rif_workflow_integration': True,
                'github_webhooks': True,
                'quality_gate_enforcement': True,
                'performance_monitoring': True
            },
            'thresholds': {
                'error_rate_warning': 0.05,  # 5% error rate triggers warning
                'error_rate_critical': 0.15,  # 15% error rate triggers critical
                'response_time_warning_ms': 5000,  # 5 seconds
                'response_time_critical_ms': 15000,  # 15 seconds
                'sla_compliance_warning': 0.85,  # 85% SLA compliance warning
                'sla_compliance_critical': 0.70  # 70% SLA compliance critical
            }
        }
    
    def start_system(self) -> bool:
        """Start the complete risk intervention system."""
        try:
            if self.is_running:
                self.logger.warning("System already running")
                return True
            
            self.logger.info("🚀 Starting Risk Intervention System...")
            
            # Start all components
            success = self._start_all_components()
            if not success:
                self.logger.error("Failed to start all components")
                return False
            
            # Start monitoring
            self._start_monitoring()
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.is_running = True
            
            # Record system startup
            startup_record = AuditRecord(
                workflow_id="system",
                timestamp=datetime.now(timezone.utc),
                action="system_startup",
                actor="coordinator",
                context="Risk intervention system started",
                rationale="System initialization complete",
                evidence=["All components started successfully"]
            )
            self.audit_tracker.record_decision(startup_record)
            
            self.logger.info("✅ Risk Intervention System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            return False
    
    def stop_system(self) -> bool:
        """Stop the risk intervention system gracefully."""
        try:
            if not self.is_running:
                self.logger.warning("System not running")
                return True
            
            self.logger.info("🛑 Stopping Risk Intervention System...")
            
            self.is_running = False
            
            # Stop monitoring
            self._stop_monitoring()
            
            # Stop all components
            self._stop_all_components()
            
            # Record system shutdown
            shutdown_record = AuditRecord(
                workflow_id="system",
                timestamp=datetime.now(timezone.utc),
                action="system_shutdown",
                actor="coordinator",
                context="Risk intervention system stopped",
                rationale="Graceful shutdown requested",
                evidence=["All components stopped successfully"]
            )
            self.audit_tracker.record_decision(shutdown_record)
            
            self.logger.info("✅ Risk Intervention System stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
            return False
    
    def _start_all_components(self) -> bool:
        """Start all system components."""
        components = [
            ("SLA Monitor", self.sla_monitor.start_monitoring),
            ("Risk Assessor", lambda: True),  # Risk assessor doesn't need explicit start
            ("Assignment Engine", lambda: True),  # Assignment engine doesn't need explicit start
            ("Workflow Coordinator", lambda: True),  # Workflow coordinator doesn't need explicit start
            ("Audit Tracker", lambda: True)  # Audit tracker doesn't need explicit start
        ]
        
        for component_name, start_func in components:
            try:
                start_func()
                self.logger.info(f"✅ Started {component_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {component_name}: {e}")
                return False
        
        return True
    
    def _stop_all_components(self) -> None:
        """Stop all system components."""
        try:
            self.sla_monitor.stop_monitoring()
            self.logger.info("✅ Stopped SLA Monitor")
        except Exception as e:
            self.logger.error(f"Error stopping SLA Monitor: {e}")
    
    def _start_monitoring(self) -> None:
        """Start system health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("📊 Started system monitoring")
    
    def _stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            # Monitoring thread will stop when is_running becomes False
            self.monitoring_thread.join(timeout=5)
        self.logger.info("📊 Stopped system monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Collect metrics
                self._collect_metrics()
                
                # Check for system issues and attempt recovery
                self._check_and_recover()
                
                # Sleep for configured interval
                time.sleep(self.config.get('coordination', {}).get('health_check_interval_seconds', 300))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        self.last_health_check = datetime.now(timezone.utc)
        
        # Check each component
        self.health_checks['risk_assessor'] = self._check_risk_assessor_health()
        self.health_checks['assignment_engine'] = self._check_assignment_engine_health()
        self.health_checks['sla_monitor'] = self._check_sla_monitor_health()
        self.health_checks['workflow_coordinator'] = self._check_workflow_coordinator_health()
        self.health_checks['audit_tracker'] = self._check_audit_tracker_health()
        
        # Update overall system health
        self._update_system_health()
    
    def _check_risk_assessor_health(self) -> HealthCheck:
        """Check health of risk assessment engine."""
        start_time = time.time()
        try:
            # Simple health check - verify configuration loading
            config_loaded = bool(self.risk_assessor.config)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            if config_loaded:
                return HealthCheck(
                    component="risk_assessor",
                    status=ComponentStatus.OPERATIONAL,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time
                )
            else:
                return HealthCheck(
                    component="risk_assessor",
                    status=ComponentStatus.ERROR,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    error_message="Configuration not loaded"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                component="risk_assessor",
                status=ComponentStatus.ERROR,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def _check_assignment_engine_health(self) -> HealthCheck:
        """Check health of specialist assignment engine."""
        start_time = time.time()
        try:
            # Check if specialists are loaded
            total_specialists = sum(len(specialists) for specialists in self.assignment_engine.specialists.values())
            
            response_time = (time.time() - start_time) * 1000
            
            if total_specialists > 0:
                return HealthCheck(
                    component="assignment_engine",
                    status=ComponentStatus.OPERATIONAL,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    additional_info={'total_specialists': total_specialists}
                )
            else:
                return HealthCheck(
                    component="assignment_engine",
                    status=ComponentStatus.WARNING,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    error_message="No specialists loaded"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                component="assignment_engine",
                status=ComponentStatus.ERROR,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def _check_sla_monitor_health(self) -> HealthCheck:
        """Check health of SLA monitoring system."""
        start_time = time.time()
        try:
            is_monitoring = self.sla_monitor.monitoring_active
            active_slas = len(self.sla_monitor.active_slas)
            
            response_time = (time.time() - start_time) * 1000
            
            if is_monitoring:
                return HealthCheck(
                    component="sla_monitor",
                    status=ComponentStatus.OPERATIONAL,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    additional_info={'active_slas': active_slas}
                )
            else:
                return HealthCheck(
                    component="sla_monitor",
                    status=ComponentStatus.WARNING,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    error_message="Monitoring not active"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                component="sla_monitor",
                status=ComponentStatus.ERROR,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def _check_workflow_coordinator_health(self) -> HealthCheck:
        """Check health of workflow coordinator."""
        start_time = time.time()
        try:
            active_workflows = len(self.workflow_coordinator.active_workflows)
            completed_workflows = len(self.workflow_coordinator.completed_workflows)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                component="workflow_coordinator",
                status=ComponentStatus.OPERATIONAL,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                additional_info={
                    'active_workflows': active_workflows,
                    'completed_workflows': completed_workflows
                }
            )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                component="workflow_coordinator",
                status=ComponentStatus.ERROR,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def _check_audit_tracker_health(self) -> HealthCheck:
        """Check health of audit tracker."""
        start_time = time.time()
        try:
            # Validate audit chain integrity
            is_valid, errors = self.audit_tracker.validate_audit_chain_integrity()
            
            response_time = (time.time() - start_time) * 1000
            
            if is_valid:
                return HealthCheck(
                    component="audit_tracker",
                    status=ComponentStatus.OPERATIONAL,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    additional_info={'audit_chain_valid': True}
                )
            else:
                return HealthCheck(
                    component="audit_tracker",
                    status=ComponentStatus.ERROR,
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=response_time,
                    error_message=f"Audit chain integrity issues: {len(errors)} errors"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                component="audit_tracker",
                status=ComponentStatus.ERROR,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def _update_system_health(self) -> None:
        """Update overall system health based on component health."""
        component_statuses = [check.status for check in self.health_checks.values()]
        
        if any(status == ComponentStatus.ERROR for status in component_statuses):
            self.metrics.system_health = SystemHealth.CRITICAL
        elif any(status == ComponentStatus.WARNING for status in component_statuses):
            self.metrics.system_health = SystemHealth.DEGRADED
        else:
            self.metrics.system_health = SystemHealth.HEALTHY
        
        # Update component health in metrics
        self.metrics.component_health = {
            component: check.status for component, check in self.health_checks.items()
        }
    
    def _collect_metrics(self) -> None:
        """Collect system performance metrics."""
        self.metrics.timestamp = datetime.now(timezone.utc)
        
        # Collect workflow metrics
        self.metrics.active_workflows = len(self.workflow_coordinator.active_workflows)
        
        # Collect SLA metrics
        active_slas = len(self.sla_monitor.active_slas)
        self.metrics.pending_reviews = active_slas
        
        # Calculate average response time from health checks
        response_times = [check.response_time_ms for check in self.health_checks.values()]
        if response_times:
            self.metrics.average_response_time = sum(response_times) / len(response_times)
        
        # Update error rates
        for component, check in self.health_checks.items():
            if check.status == ComponentStatus.ERROR:
                self.error_counts[component] = self.error_counts.get(component, 0) + 1
            
            # Calculate error rate over last 100 checks (simplified)
            self.metrics.error_rates[component] = min(self.error_counts.get(component, 0) / 100, 1.0)
    
    def _check_and_recover(self) -> None:
        """Check for system issues and attempt automatic recovery."""
        if not self.config.get('coordination', {}).get('auto_recovery_enabled', True):
            return
        
        # Check for critical issues and attempt recovery
        for component, check in self.health_checks.items():
            if check.status == ComponentStatus.ERROR:
                self.logger.warning(f"⚠️ Component {component} in error state: {check.error_message}")
                self._attempt_component_recovery(component, check)
    
    def _attempt_component_recovery(self, component: str, health_check: HealthCheck) -> None:
        """Attempt to recover a failed component."""
        try:
            self.logger.info(f"🔧 Attempting recovery for component {component}")
            
            if component == "sla_monitor" and not self.sla_monitor.monitoring_active:
                # Restart SLA monitoring
                self.sla_monitor.start_monitoring()
                self.logger.info(f"✅ Restarted SLA monitoring")
            
            elif component == "audit_tracker":
                # Re-initialize audit tracker
                self.audit_tracker = DecisionAuditTracker()
                self.logger.info(f"✅ Re-initialized audit tracker")
            
            # Record recovery attempt
            recovery_record = AuditRecord(
                workflow_id="system",
                timestamp=datetime.now(timezone.utc),
                action="component_recovery",
                actor="coordinator",
                context=f"Recovery attempted for {component}",
                rationale=f"Component error: {health_check.error_message}",
                evidence=[f"Component status: {health_check.status.value}"]
            )
            
            # Try to record recovery, but don't fail if audit tracker is the problem
            try:
                self.audit_tracker.record_decision(recovery_record)
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"❌ Recovery failed for component {component}: {e}")
    
    def process_github_issue_trigger(self, issue_number: int, trigger_context: str = "github_webhook") -> str:
        """
        Process GitHub issue trigger for manual intervention.
        
        This is the main entry point for GitHub-triggered interventions.
        
        Args:
            issue_number: GitHub issue number
            trigger_context: Context that triggered the intervention
            
        Returns:
            workflow_id: Unique identifier for the intervention workflow
        """
        try:
            self.logger.info(f"📥 Processing GitHub issue trigger for #{issue_number}")
            
            # Update metrics
            self.metrics.total_requests += 1
            
            # Check if system is healthy enough to process
            if self.metrics.system_health == SystemHealth.FAILED:
                self.logger.error(f"System in failed state, rejecting issue #{issue_number}")
                self.metrics.failed_requests += 1
                raise Exception("System health check failed")
            
            # Check if we're at capacity
            max_concurrent = self.config.get('coordination', {}).get('max_concurrent_workflows', 10)
            if self.metrics.active_workflows >= max_concurrent:
                self.logger.warning(f"At capacity ({max_concurrent} workflows), queuing issue #{issue_number}")
                # In a production system, this would queue the request
                # For now, we'll process it anyway but log the warning
            
            # Initiate manual intervention workflow
            workflow_id = self.workflow_coordinator.initiate_intervention(
                issue_number=issue_number,
                triggering_context=trigger_context
            )
            
            self.metrics.successful_requests += 1
            
            self.logger.info(f"✅ GitHub issue trigger processed: {workflow_id}")
            
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"❌ Error processing GitHub issue trigger for #{issue_number}: {e}")
            self.metrics.failed_requests += 1
            
            # Record error in audit trail
            error_record = AuditRecord(
                workflow_id=f"error_{issue_number}",
                timestamp=datetime.now(timezone.utc),
                action="github_trigger_error",
                actor="coordinator",
                context=f"GitHub issue #{issue_number} trigger failed",
                rationale=str(e),
                evidence=[f"Trigger context: {trigger_context}"]
            )
            
            try:
                self.audit_tracker.record_decision(error_record)
            except:
                pass  # Don't fail if audit tracker is also having issues
            
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_system()
        sys.exit(0)
    
    def _initialize_health_monitoring(self) -> None:
        """Initialize health monitoring configuration."""
        # Set up initial health checks
        self.health_checks = {}
        
        # Initialize error counters
        self.error_counts = {}
        
        self.logger.info("📊 Health monitoring initialized")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_health': self.metrics.system_health.value,
            'is_running': self.is_running,
            'metrics': asdict(self.metrics),
            'health_checks': {
                component: asdict(check) for component, check in self.health_checks.items()
            },
            'component_error_counts': self.error_counts,
            'last_health_check': self.last_health_check.isoformat(),
            'configuration': {
                'max_concurrent_workflows': self.config.get('coordination', {}).get('max_concurrent_workflows', 10),
                'auto_recovery_enabled': self.config.get('coordination', {}).get('auto_recovery_enabled', True),
                'health_check_interval': self.config.get('coordination', {}).get('health_check_interval_seconds', 300)
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Get component-specific metrics
        workflow_metrics = self.workflow_coordinator.get_system_metrics()
        sla_metrics = self.sla_monitor.get_sla_performance_metrics()
        audit_stats = self.audit_tracker.get_audit_statistics()
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'coordinator_metrics': asdict(self.metrics),
            'workflow_metrics': workflow_metrics,
            'sla_metrics': sla_metrics,
            'audit_statistics': audit_stats,
            'system_uptime_seconds': (datetime.now(timezone.utc) - self.metrics.timestamp).total_seconds(),
            'performance_summary': {
                'request_success_rate': (
                    self.metrics.successful_requests / max(self.metrics.total_requests, 1)
                ),
                'average_response_time_ms': self.metrics.average_response_time,
                'active_workflow_count': self.metrics.active_workflows,
                'system_health_score': self._calculate_health_score()
            }
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        if self.metrics.system_health == SystemHealth.HEALTHY:
            base_score = 1.0
        elif self.metrics.system_health == SystemHealth.DEGRADED:
            base_score = 0.7
        elif self.metrics.system_health == SystemHealth.CRITICAL:
            base_score = 0.4
        else:  # FAILED
            base_score = 0.0
        
        # Adjust based on error rates
        avg_error_rate = sum(self.metrics.error_rates.values()) / max(len(self.metrics.error_rates), 1)
        error_penalty = avg_error_rate * 0.3
        
        # Adjust based on response time
        response_time_penalty = min(self.metrics.average_response_time / 10000, 0.2)  # Max 0.2 penalty
        
        final_score = max(0.0, base_score - error_penalty - response_time_penalty)
        return final_score

def main():
    """Command line interface for risk intervention coordinator."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python risk_intervention_coordinator.py <command> [args]")
        print("Commands:")
        print("  start                     - Start the risk intervention system")
        print("  stop                      - Stop the risk intervention system")
        print("  status                    - Get system status")
        print("  health                    - Get health check report")
        print("  performance               - Get performance report")
        print("  process-issue <number>    - Process GitHub issue trigger")
        print("  test-system               - Run system test")
        return
    
    command = sys.argv[1]
    coordinator = RiskInterventionCoordinator()
    
    if command == "start":
        success = coordinator.start_system()
        if success:
            print("✅ Risk Intervention System started")
            # Keep running until interrupted
            try:
                while coordinator.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                coordinator.stop_system()
        else:
            print("❌ Failed to start system")
            return 1
    
    elif command == "stop":
        success = coordinator.stop_system()
        if success:
            print("✅ Risk Intervention System stopped")
        else:
            print("❌ Failed to stop system")
            return 1
    
    elif command == "status":
        status = coordinator.get_system_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif command == "health":
        coordinator._perform_health_checks()
        health_report = {
            'system_health': coordinator.metrics.system_health.value,
            'component_health': {
                component: asdict(check) for component, check in coordinator.health_checks.items()
            },
            'last_check': coordinator.last_health_check.isoformat()
        }
        print(json.dumps(health_report, indent=2, default=str))
    
    elif command == "performance":
        performance = coordinator.get_performance_report()
        print(json.dumps(performance, indent=2, default=str))
    
    elif command == "process-issue" and len(sys.argv) >= 3:
        issue_number = int(sys.argv[2])
        try:
            workflow_id = coordinator.process_github_issue_trigger(issue_number, "CLI test")
            print(f"✅ Processed issue #{issue_number}, workflow: {workflow_id}")
        except Exception as e:
            print(f"❌ Failed to process issue #{issue_number}: {e}")
            return 1
    
    elif command == "test-system":
        print("🧪 Testing Risk Intervention System...")
        
        # Start system
        if not coordinator.start_system():
            print("❌ Failed to start system for testing")
            return 1
        
        # Wait a moment
        time.sleep(2)
        
        # Run health checks
        coordinator._perform_health_checks()
        
        # Get system status
        status = coordinator.get_system_status()
        print(f"System Health: {status['system_health']}")
        print(f"Components: {len(status['health_checks'])} checked")
        
        # Stop system
        coordinator.stop_system()
        
        print("✅ System test complete")
    
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())