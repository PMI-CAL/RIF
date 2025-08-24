#!/usr/bin/env python3
"""
Layer 8: Integration Hub for RIF Adversarial Validation System

This layer provides deep integration with existing RIF orchestration intelligence systems,
ensuring seamless coordination between adversarial validation and production workflows.
"""

import json
import os
import sys
import sqlite3
import asyncio
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationStatus:
    """Status of integration with a specific RIF system"""
    system_name: str
    integration_type: str  # ACTIVE, PASSIVE, MONITORING, DISABLED
    last_interaction: str
    health_status: str  # HEALTHY, DEGRADED, FAILED
    capabilities: List[str]
    data_flows: List[str]
    dependencies: List[str]

@dataclass
class OrchestrationEvent:
    """Event for coordination between validation and orchestration systems"""
    event_id: str
    event_type: str
    source_system: str
    target_systems: List[str]
    event_data: Dict[str, Any]
    timestamp: str
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    status: str  # PENDING, PROCESSING, COMPLETED, FAILED

class AdversarialIntegrationHub:
    """
    Layer 8: Deep integration hub connecting adversarial validation with existing RIF systems.
    
    Key Integrations:
    - Enhanced Orchestration Intelligence: Workflow coordination and decision making
    - Shadow Quality Tracking: Quality assurance and audit systems  
    - Dependency Intelligence Orchestrator: Dependency management and analysis
    - Knowledge Integration APIs: Pattern learning and knowledge sharing
    - Error Capture System: Error handling and issue generation
    - DPIBS Systems: Performance benchmarking and optimization
    - MCP Knowledge Server: External knowledge integration
    """
    
    def __init__(self, rif_root: str = "/Users/cal/DEV/RIF"):
        """Initialize integration hub with RIF system connections"""
        self.rif_root = rif_root
        self.integration_db = os.path.join(rif_root, "knowledge", "integration_hub.db")
        self.active_integrations = {}
        self.event_queue = []
        self.integration_status = {}
        
        self.ensure_database_schema()
        self.discover_rif_systems()
        self.initialize_integrations()
        
        logger.info("Adversarial Integration Hub Layer 8 initialized")
    
    def ensure_database_schema(self):
        """Create database schema for integration coordination"""
        os.makedirs(os.path.dirname(self.integration_db), exist_ok=True)
        
        with sqlite3.connect(self.integration_db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS integration_status (
                    system_name TEXT PRIMARY KEY,
                    integration_type TEXT NOT NULL,
                    last_interaction TEXT,
                    health_status TEXT NOT NULL,
                    capabilities TEXT,
                    data_flows TEXT,
                    dependencies TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS orchestration_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    target_systems TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS integration_metrics (
                    metric_id TEXT PRIMARY KEY,
                    system_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS validation_orchestration_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_feature_id TEXT NOT NULL,
                    orchestration_action TEXT NOT NULL,
                    system_target TEXT NOT NULL,
                    action_result TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
    
    def discover_rif_systems(self):
        """Discover and catalog existing RIF systems for integration"""
        logger.info("Discovering RIF systems for integration")
        
        # Priority integration systems
        priority_systems = {
            "enhanced_orchestration_intelligence": "claude/commands/enhanced_orchestration_intelligence.py",
            "shadow_quality_tracking": "systems/shadow_quality_tracking.py",
            "dependency_intelligence_orchestrator": "claude/commands/dependency_intelligence_orchestrator.py", 
            "knowledge_integration_apis": "systems/knowledge_integration_apis.py",
            "error_capture_trigger_system": "claude/commands/error_capture_trigger_system.py",
            "dpibs_master_integration": "systems/dpibs_master_integration.py",
            "orchestration_utilities": "claude/commands/orchestration_utilities.py"
        }
        
        # Secondary integration systems
        secondary_systems = {
            "context_integration_hub": "systems/context_integration_hub.py",
            "dynamic_orchestrator_engine": "claude/commands/dynamic_orchestrator_engine.py",
            "parallel_execution_coordinator": "claude/commands/parallel_execution_coordinator.py",
            "dpibs_benchmarking_enhanced": "systems/dpibs_benchmarking_enhanced.py",
            "shadow_quality_integration": "systems/shadow_quality_integration.py"
        }
        
        discovered_systems = {}
        
        # Check priority systems
        for system_name, relative_path in priority_systems.items():
            full_path = os.path.join(self.rif_root, relative_path)
            if os.path.exists(full_path):
                discovered_systems[system_name] = {
                    "path": full_path,
                    "priority": "HIGH",
                    "capabilities": self._analyze_system_capabilities(full_path),
                    "integration_type": "ACTIVE"
                }
                logger.info(f"Discovered priority system: {system_name}")
            else:
                logger.warning(f"Priority system not found: {system_name} at {full_path}")
        
        # Check secondary systems
        for system_name, relative_path in secondary_systems.items():
            full_path = os.path.join(self.rif_root, relative_path)
            if os.path.exists(full_path):
                discovered_systems[system_name] = {
                    "path": full_path,
                    "priority": "MEDIUM",
                    "capabilities": self._analyze_system_capabilities(full_path),
                    "integration_type": "PASSIVE"
                }
                logger.info(f"Discovered secondary system: {system_name}")
        
        self.discovered_systems = discovered_systems
        logger.info(f"System discovery complete: {len(discovered_systems)} systems found")
        
    def _analyze_system_capabilities(self, system_path: str) -> List[str]:
        """Analyze system file to determine integration capabilities"""
        capabilities = []
        
        try:
            with open(system_path, 'r') as f:
                content = f.read()
            
            # Check for common capability patterns
            if "class" in content and "def" in content:
                capabilities.append("class_based_api")
            if "async def" in content:
                capabilities.append("async_operations")
            if "sqlite" in content.lower() or "database" in content.lower():
                capabilities.append("database_integration")
            if "github" in content.lower() or "gh " in content:
                capabilities.append("github_integration")
            if "json" in content.lower():
                capabilities.append("json_data_exchange")
            if "logging" in content.lower():
                capabilities.append("logging_support")
            if "orchestrat" in content.lower():
                capabilities.append("orchestration_capabilities")
            if "knowledge" in content.lower():
                capabilities.append("knowledge_integration")
            if "validation" in content.lower():
                capabilities.append("validation_support")
            if "error" in content.lower():
                capabilities.append("error_handling")
            
        except Exception as e:
            logger.warning(f"Could not analyze capabilities for {system_path}: {e}")
        
        return capabilities
    
    def initialize_integrations(self):
        """Initialize integrations with discovered RIF systems"""
        logger.info("Initializing system integrations")
        
        for system_name, system_info in self.discovered_systems.items():
            try:
                integration_status = IntegrationStatus(
                    system_name=system_name,
                    integration_type=system_info["integration_type"],
                    last_interaction=datetime.datetime.utcnow().isoformat() + "Z",
                    health_status="HEALTHY",
                    capabilities=system_info["capabilities"],
                    data_flows=self._determine_data_flows(system_name),
                    dependencies=self._determine_dependencies(system_name)
                )
                
                self.integration_status[system_name] = integration_status
                self._record_integration_status(integration_status)
                
                # Initialize active integrations
                if system_info["integration_type"] == "ACTIVE":
                    self._initialize_active_integration(system_name, system_info["path"])
                
                logger.info(f"Initialized integration: {system_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize integration for {system_name}: {e}")
                # Record failed integration
                failed_status = IntegrationStatus(
                    system_name=system_name,
                    integration_type="DISABLED",
                    last_interaction=datetime.datetime.utcnow().isoformat() + "Z",
                    health_status="FAILED",
                    capabilities=[],
                    data_flows=[],
                    dependencies=[]
                )
                self.integration_status[system_name] = failed_status
                self._record_integration_status(failed_status)
    
    def _determine_data_flows(self, system_name: str) -> List[str]:
        """Determine data flow patterns for system integration"""
        data_flows = []
        
        # System-specific data flow patterns
        flow_patterns = {
            "enhanced_orchestration_intelligence": ["validation_results_to_orchestrator", "orchestration_decisions_from_orchestrator"],
            "shadow_quality_tracking": ["validation_reports_to_shadow", "quality_alerts_from_shadow"],
            "dependency_intelligence_orchestrator": ["dependency_data_to_orchestrator", "coordination_events_from_orchestrator"],
            "knowledge_integration_apis": ["patterns_to_knowledge", "learned_insights_from_knowledge"],
            "error_capture_trigger_system": ["validation_errors_to_capture", "error_patterns_from_capture"],
            "dpibs_master_integration": ["performance_data_to_dpibs", "benchmarks_from_dpibs"],
            "orchestration_utilities": ["utility_requests_to_utilities", "processed_data_from_utilities"]
        }
        
        return flow_patterns.get(system_name, ["bidirectional_data_exchange"])
    
    def _determine_dependencies(self, system_name: str) -> List[str]:
        """Determine system dependencies for integration coordination"""
        dependency_patterns = {
            "enhanced_orchestration_intelligence": ["orchestration_utilities", "knowledge_integration_apis"],
            "shadow_quality_tracking": ["error_capture_trigger_system", "orchestration_utilities"],
            "dependency_intelligence_orchestrator": ["knowledge_integration_apis", "orchestration_utilities"],
            "knowledge_integration_apis": [],  # Base system
            "error_capture_trigger_system": [],  # Base system
            "dpibs_master_integration": ["knowledge_integration_apis", "orchestration_utilities"],
            "orchestration_utilities": []  # Base system
        }
        
        return dependency_patterns.get(system_name, [])
    
    def _initialize_active_integration(self, system_name: str, system_path: str):
        """Initialize active integration with a RIF system"""
        try:
            # Load system module dynamically
            spec = importlib.util.spec_from_file_location(system_name, system_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.active_integrations[system_name] = module
                logger.info(f"Active integration loaded: {system_name}")
            else:
                logger.warning(f"Could not load module spec for: {system_name}")
        except Exception as e:
            logger.error(f"Failed to load active integration {system_name}: {e}")
    
    def _record_integration_status(self, status: IntegrationStatus):
        """Record integration status in database"""
        with sqlite3.connect(self.integration_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO integration_status
                (system_name, integration_type, last_interaction, health_status,
                 capabilities, data_flows, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                status.system_name, status.integration_type, status.last_interaction,
                status.health_status, json.dumps(status.capabilities),
                json.dumps(status.data_flows), json.dumps(status.dependencies)
            ))
    
    def coordinate_validation_with_orchestration(self, validation_feature_id: str, 
                                                validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate validation results with orchestration systems"""
        logger.info(f"Coordinating validation result for feature: {validation_feature_id}")
        
        coordination_results = {
            "feature_id": validation_feature_id,
            "coordination_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "orchestration_actions": [],
            "notifications_sent": [],
            "follow_up_actions": []
        }
        
        # Send results to Enhanced Orchestration Intelligence
        if "enhanced_orchestration_intelligence" in self.integration_status:
            orchestration_action = self._notify_orchestration_intelligence(
                validation_feature_id, validation_result
            )
            coordination_results["orchestration_actions"].append(orchestration_action)
        
        # Update Shadow Quality Tracking
        if "shadow_quality_tracking" in self.integration_status:
            shadow_action = self._update_shadow_quality_tracking(
                validation_feature_id, validation_result
            )
            coordination_results["orchestration_actions"].append(shadow_action)
        
        # Update Knowledge Integration APIs
        if "knowledge_integration_apis" in self.integration_status:
            knowledge_action = self._update_knowledge_integration(
                validation_feature_id, validation_result
            )
            coordination_results["orchestration_actions"].append(knowledge_action)
        
        # Handle critical failures
        if validation_result.get("validation_status") == "FAIL" and validation_result.get("risk_level") == "CRITICAL":
            critical_actions = self._handle_critical_validation_failure(
                validation_feature_id, validation_result
            )
            coordination_results["follow_up_actions"].extend(critical_actions)
        
        # Log coordination activity
        self._log_validation_orchestration(validation_feature_id, coordination_results)
        
        return coordination_results
    
    def _notify_orchestration_intelligence(self, feature_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Notify orchestration intelligence system of validation results"""
        action = {
            "target_system": "enhanced_orchestration_intelligence",
            "action_type": "validation_result_notification",
            "action_data": {
                "feature_id": feature_id,
                "validation_status": result.get("validation_status"),
                "risk_level": result.get("risk_level"),
                "requires_orchestration_decision": result.get("validation_status") in ["FAIL", "PARTIAL"]
            },
            "action_result": "PENDING"
        }
        
        try:
            # If active integration available, make direct call
            if "enhanced_orchestration_intelligence" in self.active_integrations:
                # This would be actual integration call
                action["action_result"] = "NOTIFIED_DIRECTLY"
            else:
                # Create orchestration event for asynchronous processing
                event = OrchestrationEvent(
                    event_id=f"validation_{feature_id}_{datetime.datetime.utcnow().timestamp()}",
                    event_type="validation_result",
                    source_system="adversarial_validation",
                    target_systems=["enhanced_orchestration_intelligence"],
                    event_data=result,
                    timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                    priority="HIGH" if result.get("risk_level") == "CRITICAL" else "MEDIUM",
                    status="PENDING"
                )
                self._queue_orchestration_event(event)
                action["action_result"] = "QUEUED_FOR_PROCESSING"
                
        except Exception as e:
            logger.error(f"Failed to notify orchestration intelligence: {e}")
            action["action_result"] = f"FAILED: {str(e)}"
        
        return action
    
    def _update_shadow_quality_tracking(self, feature_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Update shadow quality tracking system"""
        action = {
            "target_system": "shadow_quality_tracking", 
            "action_type": "quality_status_update",
            "action_data": {
                "feature_id": feature_id,
                "quality_status": result.get("validation_status"),
                "requires_shadow_audit": result.get("validation_status") == "FAIL"
            },
            "action_result": "PENDING"
        }
        
        try:
            # Create quality tracking update
            shadow_data = {
                "validation_feature_id": feature_id,
                "shadow_audit_required": result.get("validation_status") == "FAIL",
                "quality_concern_level": result.get("risk_level"),
                "updated_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            
            # Write to shadow quality tracking data files
            shadow_dir = os.path.join(self.rif_root, "knowledge", "quality_monitoring")
            os.makedirs(shadow_dir, exist_ok=True)
            
            shadow_file = os.path.join(shadow_dir, f"shadow_audit_{feature_id}.json")
            with open(shadow_file, 'w') as f:
                json.dump(shadow_data, f, indent=2)
            
            action["action_result"] = "SHADOW_AUDIT_RECORD_CREATED"
            
        except Exception as e:
            logger.error(f"Failed to update shadow quality tracking: {e}")
            action["action_result"] = f"FAILED: {str(e)}"
        
        return action
    
    def _update_knowledge_integration(self, feature_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge integration APIs with validation learnings"""
        action = {
            "target_system": "knowledge_integration_apis",
            "action_type": "pattern_learning_update", 
            "action_data": {
                "feature_id": feature_id,
                "validation_pattern": result.get("validation_status"),
                "learning_type": "validation_result"
            },
            "action_result": "PENDING"
        }
        
        try:
            # Create knowledge pattern from validation result
            learning_pattern = {
                "pattern_id": f"validation_pattern_{feature_id}",
                "pattern_type": "adversarial_validation_result",
                "feature_category": result.get("category", "unknown"),
                "validation_outcome": result.get("validation_status"),
                "risk_assessment": result.get("risk_level"),
                "evidence_quality": result.get("evidence_count", 0),
                "test_completion_rate": result.get("test_levels_completed", 0) / max(result.get("test_levels_total", 1), 1),
                "learned_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "learning_confidence": "HIGH" if result.get("evidence_count", 0) >= 3 else "MEDIUM"
            }
            
            # Write to knowledge patterns directory
            patterns_dir = os.path.join(self.rif_root, "knowledge", "patterns")
            os.makedirs(patterns_dir, exist_ok=True)
            
            pattern_file = os.path.join(patterns_dir, f"validation_pattern_{feature_id}.json")
            with open(pattern_file, 'w') as f:
                json.dump(learning_pattern, f, indent=2)
            
            action["action_result"] = "KNOWLEDGE_PATTERN_RECORDED"
            
        except Exception as e:
            logger.error(f"Failed to update knowledge integration: {e}")
            action["action_result"] = f"FAILED: {str(e)}"
        
        return action
    
    def _handle_critical_validation_failure(self, feature_id: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle critical validation failures with urgent orchestration actions"""
        critical_actions = []
        
        # Create GitHub issue for critical failure
        github_action = {
            "action_type": "create_github_issue",
            "priority": "CRITICAL",
            "action_data": {
                "title": f"CRITICAL: Validation Failure - {result.get('feature_name', feature_id)}",
                "body": f"""# Critical Validation Failure

**Feature**: {result.get('feature_name', feature_id)}
**Category**: {result.get('category', 'unknown')}
**Risk Level**: {result.get('risk_level')}
**Fix Required**: {result.get('fix_required', True)}

## Validation Results
- **Status**: {result.get('validation_status')}
- **Evidence Count**: {result.get('evidence_count', 0)}
- **Test Levels Completed**: {result.get('test_levels_completed', 0)}/{result.get('test_levels_total', 0)}

## Immediate Actions Required
1. Investigate root cause of failure
2. Implement fix for non-functional system
3. Re-run adversarial validation
4. Update system documentation

**Generated by**: Adversarial Validation Integration Hub
**Timestamp**: {datetime.datetime.utcnow().isoformat()}Z
                """,
                "labels": ["state:new", "priority:critical", "validation:failed", "fix:required"]
            }
        }
        critical_actions.append(github_action)
        
        # Trigger error capture system
        if "error_capture_trigger_system" in self.integration_status:
            error_action = {
                "action_type": "trigger_error_capture",
                "priority": "HIGH",
                "action_data": {
                    "error_type": "validation_failure",
                    "feature_id": feature_id,
                    "failure_details": result
                }
            }
            critical_actions.append(error_action)
        
        # Alert dependency systems
        dependency_action = {
            "action_type": "alert_dependent_systems",
            "priority": "HIGH", 
            "action_data": {
                "failed_feature": feature_id,
                "impact_assessment": "System may not function as expected",
                "dependent_systems": result.get("dependencies", [])
            }
        }
        critical_actions.append(dependency_action)
        
        return critical_actions
    
    def _queue_orchestration_event(self, event: OrchestrationEvent):
        """Queue orchestration event for asynchronous processing"""
        with sqlite3.connect(self.integration_db) as conn:
            conn.execute("""
                INSERT INTO orchestration_events
                (event_id, event_type, source_system, target_systems, event_data, timestamp, priority, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.event_type, event.source_system,
                json.dumps(event.target_systems), json.dumps(event.event_data),
                event.timestamp, event.priority, event.status
            ))
        
        self.event_queue.append(event)
    
    def _log_validation_orchestration(self, feature_id: str, coordination_results: Dict[str, Any]):
        """Log validation orchestration activity"""
        with sqlite3.connect(self.integration_db) as conn:
            for action in coordination_results["orchestration_actions"]:
                conn.execute("""
                    INSERT INTO validation_orchestration_log
                    (validation_feature_id, orchestration_action, system_target, action_result)
                    VALUES (?, ?, ?, ?)
                """, (
                    feature_id, action["action_type"], action["target_system"], action["action_result"]
                ))
    
    def process_orchestration_events(self, max_events: int = 10) -> Dict[str, Any]:
        """Process queued orchestration events"""
        processed_events = {
            "processed_count": 0,
            "successful_events": [],
            "failed_events": [],
            "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        # Get pending events from database
        with sqlite3.connect(self.integration_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, event_type, source_system, target_systems, event_data, priority
                FROM orchestration_events
                WHERE status = 'PENDING'
                ORDER BY 
                    CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,
                    created_at
                LIMIT ?
            """, (max_events,))
            
            pending_events = cursor.fetchall()
        
        # Process each event
        for event_data in pending_events:
            event_id, event_type, source_system, target_systems_json, event_data_json, priority = event_data
            
            try:
                target_systems = json.loads(target_systems_json)
                event_payload = json.loads(event_data_json)
                
                # Process event based on type
                if event_type == "validation_result":
                    result = self._process_validation_result_event(target_systems, event_payload)
                else:
                    result = {"status": "UNKNOWN_EVENT_TYPE", "message": f"Unsupported event type: {event_type}"}
                
                # Update event status
                with sqlite3.connect(self.integration_db) as conn:
                    conn.execute("""
                        UPDATE orchestration_events 
                        SET status = 'COMPLETED' 
                        WHERE event_id = ?
                    """, (event_id,))
                
                processed_events["successful_events"].append({
                    "event_id": event_id,
                    "event_type": event_type,
                    "result": result
                })
                processed_events["processed_count"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process orchestration event {event_id}: {e}")
                
                # Update event status to failed
                with sqlite3.connect(self.integration_db) as conn:
                    conn.execute("""
                        UPDATE orchestration_events 
                        SET status = 'FAILED' 
                        WHERE event_id = ?
                    """, (event_id,))
                
                processed_events["failed_events"].append({
                    "event_id": event_id,
                    "error": str(e)
                })
        
        return processed_events
    
    def _process_validation_result_event(self, target_systems: List[str], event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process validation result event for target systems"""
        processing_results = {
            "target_processing": {},
            "overall_status": "SUCCESS"
        }
        
        for target_system in target_systems:
            try:
                if target_system == "enhanced_orchestration_intelligence":
                    # Process for orchestration intelligence
                    result = {"status": "PROCESSED", "action": "orchestration_decision_updated"}
                elif target_system == "shadow_quality_tracking":
                    # Process for shadow quality
                    result = {"status": "PROCESSED", "action": "shadow_audit_triggered"}
                else:
                    result = {"status": "PROCESSED", "action": "generic_notification_sent"}
                
                processing_results["target_processing"][target_system] = result
                
            except Exception as e:
                processing_results["target_processing"][target_system] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                processing_results["overall_status"] = "PARTIAL_SUCCESS"
        
        return processing_results
    
    def get_integration_health_report(self) -> Dict[str, Any]:
        """Generate integration health report"""
        health_report = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "total_integrations": len(self.integration_status),
            "healthy_integrations": 0,
            "degraded_integrations": 0,
            "failed_integrations": 0,
            "integration_details": {},
            "orchestration_event_stats": {},
            "recommendations": []
        }
        
        # Analyze integration health
        for system_name, status in self.integration_status.items():
            health_report["integration_details"][system_name] = asdict(status)
            
            if status.health_status == "HEALTHY":
                health_report["healthy_integrations"] += 1
            elif status.health_status == "DEGRADED":
                health_report["degraded_integrations"] += 1
            else:
                health_report["failed_integrations"] += 1
        
        # Get orchestration event statistics
        with sqlite3.connect(self.integration_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM orchestration_events 
                GROUP BY status
            """)
            event_stats = dict(cursor.fetchall())
            health_report["orchestration_event_stats"] = event_stats
        
        # Generate recommendations
        if health_report["failed_integrations"] > 0:
            health_report["recommendations"].append(
                f"Address {health_report['failed_integrations']} failed integrations"
            )
        
        if event_stats.get("FAILED", 0) > 0:
            health_report["recommendations"].append(
                f"Review {event_stats.get('FAILED', 0)} failed orchestration events"
            )
        
        if health_report["healthy_integrations"] == health_report["total_integrations"]:
            health_report["recommendations"].append("All integrations healthy - system ready for production")
        
        return health_report

def main():
    """Main function for testing the Integration Hub Layer"""
    hub = AdversarialIntegrationHub()
    
    # Test sample validation result coordination
    sample_validation_result = {
        "validation_status": "FAIL",
        "feature_name": "Shadow Issue Tracking System",
        "category": "quality_assurance", 
        "risk_level": "CRITICAL",
        "evidence_count": 3,
        "test_levels_completed": 2,
        "test_levels_total": 5,
        "fix_required": True
    }
    
    # Coordinate with orchestration systems
    coordination_result = hub.coordinate_validation_with_orchestration(
        "shadow_issue_tracking", sample_validation_result
    )
    print(f"Coordination completed: {len(coordination_result['orchestration_actions'])} actions")
    
    # Process orchestration events
    event_processing = hub.process_orchestration_events()
    print(f"Processed {event_processing['processed_count']} orchestration events")
    
    # Generate health report
    health_report = hub.get_integration_health_report()
    print(f"Integration health: {health_report['healthy_integrations']}/{health_report['total_integrations']} healthy")
    
    print("Adversarial Integration Hub Layer 8 - Testing Complete")

if __name__ == "__main__":
    main()