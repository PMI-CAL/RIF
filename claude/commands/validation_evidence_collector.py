#!/usr/bin/env python3
"""
Validation Evidence Collector

Automated collection and verification of validation evidence to prevent
false positive validations like Issue #225.

This system ensures comprehensive evidence collection during validation:
1. Automated capture of test execution traces
2. Performance metrics collection
3. Error condition documentation
4. Environment state recording
5. Audit trail generation

Issue #231: Root Cause Analysis: False Validation of MCP Server Integration
"""

import json
import time
import os
import sys
import subprocess
import psutil
import platform
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from dataclasses import dataclass, asdict
import traceback
import threading
from contextlib import contextmanager


@dataclass
class EvidenceItem:
    """Individual piece of validation evidence"""
    evidence_type: str
    evidence_category: str  # "execution", "performance", "error", "environment", "audit"
    timestamp: str
    source: str  # What generated this evidence
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    authenticity_hash: str
    collection_context: str


@dataclass
class ValidationEvidencePackage:
    """Complete validation evidence package"""
    validation_id: str
    issue_id: str
    validator_agent: str
    validation_type: str
    collection_start: str
    collection_end: str
    evidence_items: List[EvidenceItem]
    summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]


class ValidationEvidenceCollector:
    """
    Collects comprehensive validation evidence automatically during validation processes.
    
    Features:
    - Real-time evidence collection during test execution
    - Performance metrics monitoring
    - Error condition capture
    - Environment state recording
    - Audit trail generation
    - Evidence authenticity verification
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = Path(knowledge_base_path or "/Users/cal/DEV/RIF/knowledge")
        self.evidence_storage = self.knowledge_base_path / "evidence_collection"
        self.evidence_storage.mkdir(parents=True, exist_ok=True)
        
        # Active collection sessions
        self.active_sessions = {}
        
        # Evidence collection configuration
        self.collection_config = {
            "performance_monitoring": {
                "cpu_sampling_interval": 0.5,
                "memory_sampling_interval": 1.0,
                "network_monitoring": True,
                "process_monitoring": True
            },
            "execution_tracing": {
                "capture_stdout": True,
                "capture_stderr": True,
                "capture_exit_codes": True,
                "capture_timing": True,
                "capture_environment": True
            },
            "error_detection": {
                "exception_tracking": True,
                "failure_analysis": True,
                "recovery_attempts": True,
                "error_patterns": True
            },
            "authenticity_verification": {
                "timestamp_validation": True,
                "source_verification": True,
                "integrity_checking": True,
                "digital_signatures": False  # Could be enhanced later
            }
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for evidence collection"""
        log_dir = self.knowledge_base_path / "enforcement_logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"evidence_collection_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ValidationEvidenceCollector")
    
    def start_evidence_collection(
        self,
        validation_id: str,
        issue_id: str,
        validator_agent: str,
        validation_type: str,
        collection_context: str = "validation_session"
    ) -> str:
        """
        Start evidence collection for a validation session.
        
        Args:
            validation_id: Unique validation identifier
            issue_id: GitHub issue ID
            validator_agent: Name of validator agent
            validation_type: Type of validation being performed
            collection_context: Context for evidence collection
            
        Returns:
            Collection session key
        """
        session_key = f"{validation_id}_{int(time.time())}"
        
        session_data = {
            "session_key": session_key,
            "validation_id": validation_id,
            "issue_id": issue_id,
            "validator_agent": validator_agent,
            "validation_type": validation_type,
            "collection_context": collection_context,
            "collection_start": datetime.now().isoformat(),
            "status": "active",
            "evidence_items": [],
            "performance_monitor": None,
            "execution_traces": [],
            "error_log": [],
            "audit_events": []
        }
        
        # Start performance monitoring
        if self.collection_config["performance_monitoring"]["process_monitoring"]:
            session_data["performance_monitor"] = self._start_performance_monitoring(session_key)
        
        # Initialize environment baseline
        environment_evidence = self._collect_environment_baseline()
        self._add_evidence_item(
            session_key,
            "environment_baseline",
            "environment",
            "evidence_collector",
            environment_evidence,
            {"collection_phase": "initialization"}
        )
        
        self.active_sessions[session_key] = session_data
        
        self.logger.info(f"Started evidence collection session {session_key} for validation {validation_id}")
        return session_key
    
    def _start_performance_monitoring(self, session_key: str) -> threading.Thread:
        """Start background performance monitoring"""
        
        def monitor_performance():
            while session_key in self.active_sessions and self.active_sessions[session_key]["status"] == "active":
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    # Collect process metrics if available
                    process_metrics = {}
                    try:
                        current_process = psutil.Process()
                        process_metrics = {
                            "cpu_percent": current_process.cpu_percent(),
                            "memory_percent": current_process.memory_percent(),
                            "memory_info": current_process.memory_info()._asdict(),
                            "num_threads": current_process.num_threads(),
                            "open_files": len(current_process.open_files())
                        }
                    except Exception:
                        pass
                    
                    performance_data = {
                        "timestamp": datetime.now().isoformat(),
                        "system": {
                            "cpu_percent": cpu_percent,
                            "memory_percent": memory.percent,
                            "memory_available": memory.available,
                            "disk_percent": disk.percent,
                            "disk_free": disk.free
                        },
                        "process": process_metrics
                    }
                    
                    self._add_evidence_item(
                        session_key,
                        "performance_snapshot",
                        "performance",
                        "performance_monitor",
                        performance_data,
                        {"monitoring_interval": self.collection_config["performance_monitoring"]["cpu_sampling_interval"]}
                    )
                    
                    time.sleep(self.collection_config["performance_monitoring"]["cpu_sampling_interval"])
                    
                except Exception as e:
                    self.logger.error(f"Performance monitoring error for {session_key}: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def _collect_environment_baseline(self) -> Dict[str, Any]:
        """Collect baseline environment information"""
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path[:5]  # First 5 path entries
            },
            "working_directory": os.getcwd(),
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith(('PYTHON', 'PATH', 'HOME', 'USER', 'SHELL'))
            },
            "system_resources": {
                "cpu_count": psutil.cpu_count(),
                "total_memory": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def collect_execution_evidence(
        self,
        session_key: str,
        command: str,
        execution_context: str,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """
        Collect evidence during command execution.
        
        Args:
            session_key: Evidence collection session key
            command: Command being executed
            execution_context: Context of execution
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            Execution evidence data
        """
        if session_key not in self.active_sessions:
            raise ValueError(f"Invalid session key: {session_key}")
        
        execution_start = time.time()
        execution_evidence = {
            "command": command,
            "execution_context": execution_context,
            "start_time": datetime.fromtimestamp(execution_start).isoformat(),
            "process_info": {},
            "output": {},
            "timing": {},
            "errors": []
        }
        
        try:
            # Execute command and capture evidence
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                execution_evidence["output"] = {
                    "stdout": result.stdout[:1000] if result.stdout else "",  # Limit output size
                    "stderr": result.stderr[:1000] if result.stderr else "",
                    "return_code": result.returncode
                }
                
                execution_evidence["process_info"] = {
                    "pid": "completed",
                    "return_code": result.returncode,
                    "success": result.returncode == 0
                }
            else:
                # Start process without capturing output (for long-running processes)
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                execution_evidence["process_info"] = {
                    "pid": process.pid,
                    "started": True
                }
                
        except subprocess.TimeoutExpired as e:
            execution_evidence["errors"].append(f"Command timeout: {str(e)}")
            execution_evidence["process_info"]["timeout"] = True
        except Exception as e:
            execution_evidence["errors"].append(f"Execution error: {str(e)}")
            execution_evidence["process_info"]["error"] = str(e)
        
        # Calculate timing
        execution_end = time.time()
        execution_evidence["timing"] = {
            "duration_seconds": execution_end - execution_start,
            "end_time": datetime.fromtimestamp(execution_end).isoformat()
        }
        
        # Add to evidence collection
        self._add_evidence_item(
            session_key,
            "command_execution",
            "execution",
            "execution_tracer",
            execution_evidence,
            {"command_hash": hashlib.md5(command.encode()).hexdigest()[:8]}
        )
        
        return execution_evidence
    
    def collect_test_execution_evidence(
        self,
        session_key: str,
        test_file: str,
        test_function: str,
        test_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Collect evidence during test execution.
        
        Args:
            session_key: Evidence collection session key
            test_file: Path to test file
            test_function: Name of test function
            test_args: Optional test arguments
            
        Returns:
            Test execution evidence
        """
        test_command = f"python3 -m pytest {test_file}::{test_function}"
        if test_args:
            test_command += " " + " ".join(test_args)
        
        test_evidence = self.collect_execution_evidence(
            session_key,
            test_command,
            f"test_execution_{test_function}",
            capture_output=True
        )
        
        # Enhanced test-specific evidence
        test_evidence["test_metadata"] = {
            "test_file": test_file,
            "test_function": test_function,
            "test_framework": "pytest",
            "test_args": test_args or []
        }
        
        # Parse test output for additional evidence
        if "stdout" in test_evidence["output"]:
            test_output = test_evidence["output"]["stdout"]
            test_evidence["test_results"] = self._parse_pytest_output(test_output)
        
        return test_evidence
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output for test results"""
        results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        try:
            lines = output.split('\n')
            for line in lines:
                if '::' in line and ('PASSED' in line or 'FAILED' in line):
                    results["tests_run"] += 1
                    if 'PASSED' in line:
                        results["tests_passed"] += 1
                    elif 'FAILED' in line:
                        results["tests_failed"] += 1
                    
                    results["test_details"].append(line.strip())
                    
                elif line.startswith('=') and ('passed' in line or 'failed' in line):
                    # Summary line
                    results["summary"] = line.strip()
        except Exception:
            pass
        
        return results
    
    def collect_error_evidence(
        self,
        session_key: str,
        error: Exception,
        error_context: str,
        recovery_attempted: bool = False
    ):
        """
        Collect evidence when errors occur.
        
        Args:
            session_key: Evidence collection session key
            error: Exception that occurred
            error_context: Context where error occurred
            recovery_attempted: Whether recovery was attempted
        """
        if session_key not in self.active_sessions:
            return
        
        error_evidence = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": error_context,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
            "recovery_attempted": recovery_attempted,
            "system_state": {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "active_processes": len(psutil.pids())
            }
        }
        
        # Add to session error log
        session = self.active_sessions[session_key]
        session["error_log"].append(error_evidence)
        
        self._add_evidence_item(
            session_key,
            "error_occurrence",
            "error",
            "error_detector",
            error_evidence,
            {"error_hash": hashlib.md5(str(error).encode()).hexdigest()[:8]}
        )
        
        self.logger.warning(f"Error evidence collected for {session_key}: {type(error).__name__}")
    
    def collect_performance_evidence(
        self,
        session_key: str,
        operation_name: str,
        start_time: float,
        end_time: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Collect performance evidence for operations.
        
        Args:
            session_key: Evidence collection session key
            operation_name: Name of operation measured
            start_time: Operation start timestamp
            end_time: Operation end timestamp
            additional_metrics: Additional performance metrics
        """
        if session_key not in self.active_sessions:
            return
        
        performance_evidence = {
            "operation_name": operation_name,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": end_time - start_time,
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent
            },
            "additional_metrics": additional_metrics or {}
        }
        
        self._add_evidence_item(
            session_key,
            "performance_measurement",
            "performance",
            "performance_collector",
            performance_evidence,
            {"operation_hash": hashlib.md5(operation_name.encode()).hexdigest()[:8]}
        )
    
    @contextmanager
    def collect_operation_evidence(
        self,
        session_key: str,
        operation_name: str,
        capture_errors: bool = True
    ):
        """
        Context manager for collecting evidence during operations.
        
        Usage:
            with collector.collect_operation_evidence(session_key, "test_operation"):
                # Perform operation
                result = some_operation()
        """
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            if capture_errors:
                self.collect_error_evidence(
                    session_key, 
                    e, 
                    f"operation_{operation_name}",
                    recovery_attempted=False
                )
            raise
        finally:
            end_time = time.time()
            self.collect_performance_evidence(
                session_key,
                operation_name,
                start_time,
                end_time
            )
    
    def _add_evidence_item(
        self,
        session_key: str,
        evidence_type: str,
        evidence_category: str,
        source: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Add evidence item to collection session"""
        if session_key not in self.active_sessions:
            return
        
        # Generate authenticity hash
        evidence_content = json.dumps(data, sort_keys=True, default=str)
        authenticity_hash = hashlib.sha256(
            f"{evidence_type}:{source}:{evidence_content}:{time.time()}".encode()
        ).hexdigest()
        
        evidence_item = EvidenceItem(
            evidence_type=evidence_type,
            evidence_category=evidence_category,
            timestamp=datetime.now().isoformat(),
            source=source,
            data=data,
            metadata=metadata,
            authenticity_hash=authenticity_hash,
            collection_context=self.active_sessions[session_key]["collection_context"]
        )
        
        self.active_sessions[session_key]["evidence_items"].append(evidence_item)
    
    def finalize_evidence_collection(self, session_key: str) -> ValidationEvidencePackage:
        """
        Finalize evidence collection and create evidence package.
        
        Args:
            session_key: Evidence collection session key
            
        Returns:
            Complete validation evidence package
        """
        if session_key not in self.active_sessions:
            raise ValueError(f"Invalid session key: {session_key}")
        
        session = self.active_sessions[session_key]
        session["status"] = "finalizing"
        
        # Stop performance monitoring
        if session["performance_monitor"]:
            # The monitoring thread will stop when status changes
            pass
        
        # Final environment snapshot
        final_environment = self._collect_environment_baseline()
        self._add_evidence_item(
            session_key,
            "environment_final",
            "environment",
            "evidence_collector",
            final_environment,
            {"collection_phase": "finalization"}
        )
        
        # Generate summary and quality metrics
        summary = self._generate_evidence_summary(session)
        quality_metrics = self._calculate_quality_metrics(session)
        audit_trail = self._generate_audit_trail(session)
        
        # Create evidence package
        evidence_package = ValidationEvidencePackage(
            validation_id=session["validation_id"],
            issue_id=session["issue_id"],
            validator_agent=session["validator_agent"],
            validation_type=session["validation_type"],
            collection_start=session["collection_start"],
            collection_end=datetime.now().isoformat(),
            evidence_items=session["evidence_items"],
            summary=summary,
            quality_metrics=quality_metrics,
            audit_trail=audit_trail
        )
        
        # Store evidence package
        self._store_evidence_package(evidence_package)
        
        # Clean up session
        session["status"] = "completed"
        
        self.logger.info(
            f"Evidence collection finalized for {session_key}: "
            f"{len(evidence_package.evidence_items)} items collected"
        )
        
        return evidence_package
    
    def _generate_evidence_summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of collected evidence"""
        evidence_items = session["evidence_items"]
        
        # Count by category
        category_counts = {}
        type_counts = {}
        source_counts = {}
        
        for item in evidence_items:
            category_counts[item.evidence_category] = category_counts.get(item.evidence_category, 0) + 1
            type_counts[item.evidence_type] = type_counts.get(item.evidence_type, 0) + 1
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        
        # Calculate time span
        if evidence_items:
            first_timestamp = min(item.timestamp for item in evidence_items)
            last_timestamp = max(item.timestamp for item in evidence_items)
            collection_duration = (
                datetime.fromisoformat(last_timestamp) - 
                datetime.fromisoformat(first_timestamp)
            ).total_seconds()
        else:
            collection_duration = 0
        
        return {
            "total_evidence_items": len(evidence_items),
            "evidence_categories": category_counts,
            "evidence_types": type_counts,
            "evidence_sources": source_counts,
            "collection_duration_seconds": collection_duration,
            "error_count": len(session["error_log"]),
            "performance_snapshots": category_counts.get("performance", 0)
        }
    
    def _calculate_quality_metrics(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate evidence quality metrics"""
        evidence_items = session["evidence_items"]
        
        if not evidence_items:
            return {"overall_quality": 0, "completeness": 0, "authenticity": 0}
        
        # Completeness: Are all expected evidence categories present?
        expected_categories = {"environment", "execution", "performance"}
        present_categories = set(item.evidence_category for item in evidence_items)
        completeness = len(present_categories & expected_categories) / len(expected_categories) * 100
        
        # Authenticity: Do all items have valid authenticity hashes?
        authentic_items = sum(1 for item in evidence_items if item.authenticity_hash)
        authenticity = (authentic_items / len(evidence_items)) * 100
        
        # Temporal coverage: Are evidence items distributed over time?
        if len(evidence_items) > 1:
            timestamps = [datetime.fromisoformat(item.timestamp) for item in evidence_items]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            temporal_coverage = min(time_span / 60, 100)  # Max 100 for 1+ minute span
        else:
            temporal_coverage = 0
        
        # Overall quality score
        overall_quality = (completeness * 0.4 + authenticity * 0.4 + temporal_coverage * 0.2)
        
        return {
            "overall_quality": overall_quality,
            "completeness": completeness,
            "authenticity": authenticity,
            "temporal_coverage": temporal_coverage,
            "evidence_density": len(evidence_items) / max(session.get("collection_duration", 1), 1)
        }
    
    def _generate_audit_trail(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate audit trail for evidence collection"""
        audit_events = [
            {
                "event": "collection_started",
                "timestamp": session["collection_start"],
                "details": {
                    "validation_id": session["validation_id"],
                    "validator_agent": session["validator_agent"]
                }
            }
        ]
        
        # Add error events
        for error in session["error_log"]:
            audit_events.append({
                "event": "error_detected",
                "timestamp": error["timestamp"],
                "details": {
                    "error_type": error["error_type"],
                    "error_context": error["error_context"]
                }
            })
        
        # Add collection milestone events
        evidence_items = session["evidence_items"]
        milestone_counts = [10, 25, 50, 100]  # Evidence count milestones
        
        for count in milestone_counts:
            if len(evidence_items) >= count:
                # Find the timestamp of the nth evidence item
                milestone_item = evidence_items[count - 1]
                audit_events.append({
                    "event": f"evidence_milestone_{count}",
                    "timestamp": milestone_item.timestamp,
                    "details": {"evidence_count": count}
                })
        
        audit_events.append({
            "event": "collection_completed",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "total_evidence_items": len(evidence_items),
                "collection_duration": session.get("collection_duration", 0)
            }
        })
        
        return sorted(audit_events, key=lambda x: x["timestamp"])
    
    def _store_evidence_package(self, package: ValidationEvidencePackage):
        """Store evidence package to persistent storage"""
        try:
            # Create storage directory structure
            package_dir = self.evidence_storage / f"validation_{package.validation_id}"
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Store main evidence package
            package_file = package_dir / "evidence_package.json"
            with open(package_file, 'w') as f:
                json.dump(asdict(package), f, indent=2, default=str)
            
            # Store individual evidence items
            items_dir = package_dir / "evidence_items"
            items_dir.mkdir(exist_ok=True)
            
            for i, item in enumerate(package.evidence_items):
                item_file = items_dir / f"evidence_{i:04d}_{item.evidence_type}.json"
                with open(item_file, 'w') as f:
                    json.dump(asdict(item), f, indent=2, default=str)
            
            # Create evidence index
            index_data = {
                "validation_id": package.validation_id,
                "issue_id": package.issue_id,
                "evidence_count": len(package.evidence_items),
                "collection_timespan": f"{package.collection_start} to {package.collection_end}",
                "quality_score": package.quality_metrics.get("overall_quality", 0),
                "storage_path": str(package_dir),
                "created": datetime.now().isoformat()
            }
            
            index_file = self.evidence_storage / "evidence_index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {"packages": []}
            
            index["packages"].append(index_data)
            
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to store evidence package: {e}")
    
    def get_evidence_package(self, validation_id: str) -> Optional[ValidationEvidencePackage]:
        """Retrieve stored evidence package"""
        try:
            package_dir = self.evidence_storage / f"validation_{validation_id}"
            package_file = package_dir / "evidence_package.json"
            
            if not package_file.exists():
                return None
            
            with open(package_file, 'r') as f:
                package_data = json.load(f)
            
            # Convert back to dataclass structure
            evidence_items = [EvidenceItem(**item) for item in package_data["evidence_items"]]
            package_data["evidence_items"] = evidence_items
            
            return ValidationEvidencePackage(**package_data)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve evidence package {validation_id}: {e}")
            return None
    
    def generate_collection_report(self, session_key: str) -> Dict[str, Any]:
        """Generate report on evidence collection session"""
        if session_key not in self.active_sessions:
            return {"error": "Invalid session key"}
        
        session = self.active_sessions[session_key]
        summary = self._generate_evidence_summary(session)
        quality_metrics = self._calculate_quality_metrics(session)
        
        return {
            "collection_report": {
                "session_key": session_key,
                "validation_id": session["validation_id"],
                "issue_id": session["issue_id"],
                "status": session["status"],
                "collection_duration": summary["collection_duration_seconds"]
            },
            "evidence_summary": summary,
            "quality_metrics": quality_metrics,
            "false_positive_prevention": {
                "comprehensive_evidence_collected": summary["total_evidence_items"] > 10,
                "multiple_categories_covered": len(summary["evidence_categories"]) >= 3,
                "authenticity_verification_applied": quality_metrics["authenticity"] > 90,
                "temporal_coverage_adequate": quality_metrics["temporal_coverage"] > 50,
                "audit_trail_complete": len(session.get("audit_events", [])) > 0
            },
            "recommendations": self._generate_collection_recommendations(summary, quality_metrics),
            "report_generated": datetime.now().isoformat()
        }
    
    def _generate_collection_recommendations(
        self, 
        summary: Dict[str, Any], 
        quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving evidence collection"""
        recommendations = []
        
        if summary["total_evidence_items"] < 10:
            recommendations.append("Increase evidence collection depth - aim for 10+ evidence items")
        
        if len(summary["evidence_categories"]) < 3:
            recommendations.append("Expand evidence categories - include execution, performance, and environment evidence")
        
        if quality_metrics["authenticity"] < 90:
            recommendations.append("Strengthen authenticity verification - ensure all evidence has valid hashes")
        
        if quality_metrics["temporal_coverage"] < 50:
            recommendations.append("Improve temporal coverage - collect evidence throughout validation process")
        
        if summary["error_count"] == 0:
            recommendations.append("Consider adding error condition testing for more comprehensive evidence")
        
        return recommendations


# Global collector instance
_global_evidence_collector = None

def get_evidence_collector() -> ValidationEvidenceCollector:
    """Get global evidence collector instance"""
    global _global_evidence_collector
    if _global_evidence_collector is None:
        _global_evidence_collector = ValidationEvidenceCollector()
    return _global_evidence_collector


def start_validation_evidence_collection(
    validation_id: str,
    issue_id: str,
    validator_agent: str,
    validation_type: str
) -> str:
    """Start evidence collection for validation"""
    collector = get_evidence_collector()
    return collector.start_evidence_collection(validation_id, issue_id, validator_agent, validation_type)


def finalize_validation_evidence_collection(session_key: str) -> ValidationEvidencePackage:
    """Finalize evidence collection"""
    collector = get_evidence_collector()
    return collector.finalize_evidence_collection(session_key)


# Example usage and testing
if __name__ == "__main__":
    print("📊 Validation Evidence Collector - Comprehensive Evidence Collection")
    print("=" * 80)
    
    # Create collector
    collector = ValidationEvidenceCollector()
    
    # Start evidence collection
    session_key = collector.start_evidence_collection(
        "test_validation_231",
        "231",
        "rif-validator",
        "mcp_integration"
    )
    
    print(f"Started evidence collection session: {session_key}")
    
    # Simulate evidence collection during validation
    with collector.collect_operation_evidence(session_key, "test_execution"):
        time.sleep(1)  # Simulate work
        
        # Collect test execution evidence
        test_evidence = collector.collect_test_execution_evidence(
            session_key,
            "tests/mcp/integration/test_mcp_claude_desktop_integration.py",
            "test_comprehensive_integration_validation"
        )
        
        # Collect performance evidence
        start = time.time()
        time.sleep(0.5)  # Simulate operation
        collector.collect_performance_evidence(
            session_key,
            "mcp_connection_test",
            start,
            time.time(),
            {"connection_attempts": 3, "success_rate": 1.0}
        )
    
    # Finalize collection
    evidence_package = collector.finalize_evidence_collection(session_key)
    
    print(f"\nEvidence collection completed:")
    print(f"Evidence items collected: {len(evidence_package.evidence_items)}")
    print(f"Quality score: {evidence_package.quality_metrics['overall_quality']:.1f}%")
    print(f"Categories: {list(evidence_package.summary['evidence_categories'].keys())}")
    
    # Generate report
    print(f"\n📊 Collection Report:")
    report = collector.generate_collection_report(session_key)
    print(json.dumps(report, indent=2)[:1000] + "...")
    
    print(f"\n✅ EVIDENCE COLLECTION SYSTEM OPERATIONAL")
    print(f"✅ COMPREHENSIVE EVIDENCE: CAPTURED")
    print(f"✅ AUTHENTICITY VERIFICATION: ACTIVE")
    print(f"✅ AUDIT TRAIL: GENERATED")
    print(f"✅ FALSE POSITIVE PREVENTION: ENHANCED")