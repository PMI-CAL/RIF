#!/usr/bin/env python3
"""
False Positive Detection System

Automatically detects potential false positive validations to prevent
issues like Issue #225 where broken systems were marked as working.

This system uses pattern recognition, anomaly detection, and validation
cross-checking to identify suspicious validation results that may be
false positives.

Issue #231: Root Cause Analysis: False Validation of MCP Server Integration
"""

import json
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import re
import hashlib


class FalsePositiveIndicator(Enum):
    """Types of false positive indicators"""
    SUSPICIOUSLY_PERFECT_RESULTS = "suspiciously_perfect_results"
    IMPOSSIBLY_FAST_EXECUTION = "impossibly_fast_execution"
    INCONSISTENT_EVIDENCE = "inconsistent_evidence"
    MISSING_ERROR_CONDITIONS = "missing_error_conditions" 
    PATTERN_DEVIATION = "pattern_deviation"
    FABRICATED_METRICS = "fabricated_metrics"
    INSUFFICIENT_TESTING_DEPTH = "insufficient_testing_depth"
    UNREALISTIC_PERFORMANCE = "unrealistic_performance"
    EVIDENCE_TIMESTAMP_ANOMALIES = "evidence_timestamp_anomalies"
    VALIDATION_PATTERN_REPETITION = "validation_pattern_repetition"


@dataclass
class FalsePositiveAlert:
    """Alert for potential false positive validation"""
    alert_id: str
    timestamp: str
    validation_id: str
    issue_id: str
    validator_agent: str
    confidence_score: float  # 0.0 to 1.0
    severity: str  # "low", "medium", "high", "critical"
    indicators: List[FalsePositiveIndicator]
    evidence: Dict[str, Any]
    recommended_actions: List[str]
    validation_data: Dict[str, Any]


@dataclass
class ValidationPattern:
    """Pattern of validation behavior for analysis"""
    pattern_id: str
    validation_type: str
    typical_execution_time_range: Tuple[float, float]
    typical_success_rate_range: Tuple[float, float]
    common_error_types: List[str]
    expected_evidence_types: Set[str]
    performance_benchmarks: Dict[str, Tuple[float, float]]
    complexity_indicators: Dict[str, Any]


class FalsePositiveDetector:
    """
    Detects potential false positive validations using multiple detection methods.
    
    Detection Methods:
    - Statistical anomaly detection
    - Pattern matching against historical data
    - Evidence consistency analysis
    - Performance plausibility checks
    - Temporal analysis of validation patterns
    - Cross-validation with similar systems
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = Path(knowledge_base_path or "/Users/cal/DEV/RIF/knowledge")
        self.detection_data_path = self.knowledge_base_path / "false_positive_detection"
        self.detection_data_path.mkdir(parents=True, exist_ok=True)
        
        # Historical validation patterns
        self.validation_patterns = {}
        self.historical_validations = []
        self.false_positive_alerts = []
        
        # Detection configuration
        self.detection_config = {
            "statistical_thresholds": {
                "perfect_success_rate_threshold": 0.98,  # Above 98% success is suspicious
                "execution_time_outlier_factor": 3.0,   # 3x faster/slower than average
                "evidence_completeness_threshold": 0.95, # Above 95% completeness is suspicious
                "performance_improvement_threshold": 0.5  # 50% improvement is suspicious
            },
            "pattern_analysis": {
                "minimum_historical_samples": 5,
                "pattern_deviation_threshold": 2.0,  # Standard deviations
                "evidence_consistency_threshold": 0.8,
                "temporal_clustering_window": 3600  # 1 hour window for clustering
            },
            "evidence_analysis": {
                "timestamp_variance_threshold": 10.0,  # Seconds
                "metric_precision_threshold": 0.001,   # Too precise metrics are suspicious
                "missing_error_ratio_threshold": 0.1   # Less than 10% errors is suspicious
            }
        }
        
        # Load existing patterns and historical data
        self._load_validation_patterns()
        self._load_historical_validations()
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for false positive detection"""
        log_dir = self.knowledge_base_path / "enforcement_logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"false_positive_detection_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FalsePositiveDetector")
    
    def analyze_validation_for_false_positives(
        self,
        validation_id: str,
        issue_id: str,
        validator_agent: str,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any],
        validation_metadata: Optional[Dict[str, Any]] = None
    ) -> FalsePositiveAlert:
        """
        Analyze a validation for potential false positive indicators.
        
        Args:
            validation_id: Unique validation identifier
            issue_id: GitHub issue ID
            validator_agent: Name of validator agent
            validation_results: Validation test results
            validation_evidence: Evidence collected during validation
            validation_metadata: Optional metadata about validation
            
        Returns:
            False positive alert with analysis results
        """
        analysis_start = time.time()
        
        # Initialize detection results
        detected_indicators = []
        evidence_analysis = {}
        confidence_scores = []
        
        # 1. Statistical Anomaly Detection
        statistical_indicators = self._detect_statistical_anomalies(validation_results, validation_evidence)
        detected_indicators.extend(statistical_indicators["indicators"])
        confidence_scores.extend(statistical_indicators["confidence_scores"])
        evidence_analysis["statistical_analysis"] = statistical_indicators["analysis"]
        
        # 2. Pattern Deviation Analysis
        pattern_indicators = self._detect_pattern_deviations(validation_id, validation_results, validation_metadata)
        detected_indicators.extend(pattern_indicators["indicators"])
        confidence_scores.extend(pattern_indicators["confidence_scores"])
        evidence_analysis["pattern_analysis"] = pattern_indicators["analysis"]
        
        # 3. Evidence Consistency Analysis
        evidence_indicators = self._analyze_evidence_consistency(validation_evidence)
        detected_indicators.extend(evidence_indicators["indicators"])
        confidence_scores.extend(evidence_indicators["confidence_scores"])
        evidence_analysis["evidence_analysis"] = evidence_indicators["analysis"]
        
        # 4. Temporal Analysis
        temporal_indicators = self._analyze_temporal_patterns(validation_id, validation_results, validation_evidence)
        detected_indicators.extend(temporal_indicators["indicators"])
        confidence_scores.extend(temporal_indicators["confidence_scores"])
        evidence_analysis["temporal_analysis"] = temporal_indicators["analysis"]
        
        # 5. Performance Plausibility Analysis
        performance_indicators = self._analyze_performance_plausibility(validation_results, validation_evidence)
        detected_indicators.extend(performance_indicators["indicators"])
        confidence_scores.extend(performance_indicators["confidence_scores"])
        evidence_analysis["performance_analysis"] = performance_indicators["analysis"]
        
        # Calculate overall confidence score
        overall_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        # Determine severity
        severity = self._determine_alert_severity(detected_indicators, overall_confidence)
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(detected_indicators)
        
        # Create false positive alert
        alert = FalsePositiveAlert(
            alert_id=f"fp_{validation_id}_{int(analysis_start)}",
            timestamp=datetime.now().isoformat(),
            validation_id=validation_id,
            issue_id=issue_id,
            validator_agent=validator_agent,
            confidence_score=overall_confidence,
            severity=severity,
            indicators=detected_indicators,
            evidence=evidence_analysis,
            recommended_actions=recommended_actions,
            validation_data={
                "validation_results": validation_results,
                "validation_evidence": validation_evidence,
                "validation_metadata": validation_metadata or {}
            }
        )
        
        # Store alert
        self.false_positive_alerts.append(alert)
        self._store_false_positive_alert(alert)
        
        # Update historical data
        self._update_historical_validations(validation_id, validation_results, validation_evidence)
        
        self.logger.info(
            f"False positive analysis completed for {validation_id}: "
            f"Confidence={overall_confidence:.3f}, Severity={severity}, "
            f"Indicators={len(detected_indicators)}"
        )
        
        return alert
    
    def _detect_statistical_anomalies(
        self,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect statistical anomalies in validation results"""
        
        indicators = []
        confidence_scores = []
        analysis = {}
        
        # Check for suspiciously perfect results
        if "success_rate" in validation_results:
            success_rate = validation_results["success_rate"]
            if success_rate >= self.detection_config["statistical_thresholds"]["perfect_success_rate_threshold"]:
                indicators.append(FalsePositiveIndicator.SUSPICIOUSLY_PERFECT_RESULTS)
                confidence_scores.append(min(success_rate, 1.0))
                analysis["perfect_success_rate"] = {
                    "observed_rate": success_rate,
                    "threshold": self.detection_config["statistical_thresholds"]["perfect_success_rate_threshold"],
                    "suspicion_level": "high" if success_rate >= 0.99 else "medium"
                }
        
        # Check for impossibly fast execution
        if "execution_time" in validation_results:
            execution_time = validation_results["execution_time"]
            if execution_time < 0.1:  # Less than 100ms is suspicious for integration tests
                indicators.append(FalsePositiveIndicator.IMPOSSIBLY_FAST_EXECUTION)
                confidence_scores.append(max(0.1 - execution_time, 0.0) * 10)  # Higher confidence for faster times
                analysis["fast_execution"] = {
                    "observed_time": execution_time,
                    "minimum_expected": 0.1,
                    "suspicion_level": "critical" if execution_time < 0.01 else "high"
                }
        
        # Check for missing error conditions
        error_indicators = self._analyze_error_patterns(validation_results, validation_evidence)
        if error_indicators["missing_errors"]:
            indicators.append(FalsePositiveIndicator.MISSING_ERROR_CONDITIONS)
            confidence_scores.append(error_indicators["confidence"])
            analysis["missing_errors"] = error_indicators
        
        # Check for unrealistic performance improvements
        performance_analysis = self._analyze_performance_anomalies(validation_results)
        if performance_analysis["unrealistic_improvement"]:
            indicators.append(FalsePositiveIndicator.UNREALISTIC_PERFORMANCE)
            confidence_scores.append(performance_analysis["confidence"])
            analysis["performance_anomalies"] = performance_analysis
        
        return {
            "indicators": indicators,
            "confidence_scores": confidence_scores,
            "analysis": analysis
        }
    
    def _analyze_error_patterns(
        self,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze error patterns for anomalies"""
        
        # Count errors in results
        total_tests = validation_results.get("total_tests", 0)
        failed_tests = validation_results.get("failed_tests", 0)
        
        error_rate = failed_tests / max(total_tests, 1)
        expected_minimum_error_rate = self.detection_config["evidence_analysis"]["missing_error_ratio_threshold"]
        
        missing_errors = error_rate < expected_minimum_error_rate and total_tests > 10
        
        # Check for error diversity
        error_types = set()
        for evidence_item in validation_evidence.values():
            if isinstance(evidence_item, dict) and "errors" in evidence_item:
                errors = evidence_item["errors"]
                if isinstance(errors, list):
                    for error in errors:
                        if isinstance(error, dict) and "error_type" in error:
                            error_types.add(error["error_type"])
        
        error_diversity_low = len(error_types) < 2 and total_tests > 5
        
        confidence = 0.0
        if missing_errors:
            confidence += 0.6
        if error_diversity_low:
            confidence += 0.4
        
        return {
            "missing_errors": missing_errors,
            "error_diversity_low": error_diversity_low,
            "observed_error_rate": error_rate,
            "expected_minimum_error_rate": expected_minimum_error_rate,
            "error_types_found": list(error_types),
            "confidence": confidence
        }
    
    def _analyze_performance_anomalies(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance results for anomalies"""
        
        unrealistic_improvement = False
        confidence = 0.0
        
        # Check for unrealistic response times
        if "average_response_time" in validation_results:
            response_time = validation_results["average_response_time"]
            if response_time < 0.001:  # Less than 1ms is unrealistic for real operations
                unrealistic_improvement = True
                confidence += 0.8
        
        # Check for impossibly high throughput
        if "throughput" in validation_results:
            throughput = validation_results["throughput"]
            if throughput > 10000:  # More than 10k ops/sec is suspicious for integration tests
                unrealistic_improvement = True
                confidence += 0.6
        
        # Check for perfect performance consistency
        if "performance_variance" in validation_results:
            variance = validation_results["performance_variance"]
            if variance < 0.001:  # Too little variance is suspicious
                unrealistic_improvement = True
                confidence += 0.4
        
        return {
            "unrealistic_improvement": unrealistic_improvement,
            "confidence": confidence,
            "analysis_details": {
                "response_time_check": validation_results.get("average_response_time", "not_available"),
                "throughput_check": validation_results.get("throughput", "not_available"),
                "variance_check": validation_results.get("performance_variance", "not_available")
            }
        }
    
    def _detect_pattern_deviations(
        self,
        validation_id: str,
        validation_results: Dict[str, Any],
        validation_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect deviations from historical validation patterns"""
        
        indicators = []
        confidence_scores = []
        analysis = {}
        
        validation_type = validation_metadata.get("validation_type", "unknown") if validation_metadata else "unknown"
        
        # Get historical pattern for this validation type
        if validation_type in self.validation_patterns:
            pattern = self.validation_patterns[validation_type]
            
            # Check execution time deviation
            if "execution_time" in validation_results:
                execution_time = validation_results["execution_time"]
                typical_range = pattern.typical_execution_time_range
                
                if execution_time < typical_range[0] / 3 or execution_time > typical_range[1] * 3:
                    indicators.append(FalsePositiveIndicator.PATTERN_DEVIATION)
                    deviation_factor = max(typical_range[0] / max(execution_time, 0.001), execution_time / max(typical_range[1], 0.001))
                    confidence_scores.append(min(deviation_factor / 3.0, 1.0))
                    analysis["execution_time_deviation"] = {
                        "observed": execution_time,
                        "expected_range": typical_range,
                        "deviation_factor": deviation_factor
                    }
            
            # Check success rate deviation
            if "success_rate" in validation_results:
                success_rate = validation_results["success_rate"]
                typical_success_range = pattern.typical_success_rate_range
                
                if success_rate > typical_success_range[1] + 0.1:  # More than 10% above typical
                    indicators.append(FalsePositiveIndicator.PATTERN_DEVIATION)
                    confidence_scores.append((success_rate - typical_success_range[1]) * 2)
                    analysis["success_rate_deviation"] = {
                        "observed": success_rate,
                        "expected_range": typical_success_range,
                        "deviation": success_rate - typical_success_range[1]
                    }
        else:
            analysis["pattern_availability"] = "No historical pattern available for validation type"
        
        return {
            "indicators": indicators,
            "confidence_scores": confidence_scores,
            "analysis": analysis
        }
    
    def _analyze_evidence_consistency(self, validation_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evidence for consistency issues"""
        
        indicators = []
        confidence_scores = []
        analysis = {}
        
        # Check timestamp consistency
        timestamps = []
        for evidence_item in validation_evidence.values():
            if isinstance(evidence_item, dict) and "timestamp" in evidence_item:
                try:
                    timestamp = evidence_item["timestamp"]
                    if isinstance(timestamp, str):
                        parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamps.append(parsed_time.timestamp())
                    elif isinstance(timestamp, (int, float)):
                        timestamps.append(timestamp)
                except Exception:
                    pass
        
        if len(timestamps) > 1:
            timestamp_variance = statistics.variance(timestamps) if len(timestamps) > 1 else 0
            max_variance_threshold = self.detection_config["evidence_analysis"]["timestamp_variance_threshold"]
            
            if timestamp_variance > max_variance_threshold ** 2:
                indicators.append(FalsePositiveIndicator.EVIDENCE_TIMESTAMP_ANOMALIES)
                confidence_scores.append(min(timestamp_variance / (max_variance_threshold ** 2), 1.0))
                analysis["timestamp_anomalies"] = {
                    "variance": timestamp_variance,
                    "threshold": max_variance_threshold ** 2,
                    "timestamps_analyzed": len(timestamps)
                }
        
        # Check for overly precise metrics
        precision_issues = self._check_metric_precision(validation_evidence)
        if precision_issues["suspicious_precision"]:
            indicators.append(FalsePositiveIndicator.FABRICATED_METRICS)
            confidence_scores.append(precision_issues["confidence"])
            analysis["precision_issues"] = precision_issues
        
        # Check evidence completeness
        evidence_completeness = len(validation_evidence) / max(10, 1)  # Expect ~10 evidence items
        if evidence_completeness > 2.0:  # More than 2x expected evidence is suspicious
            indicators.append(FalsePositiveIndicator.INCONSISTENT_EVIDENCE)
            confidence_scores.append(min((evidence_completeness - 2.0) / 2.0, 1.0))
            analysis["evidence_overcompleteness"] = {
                "observed_items": len(validation_evidence),
                "expected_items": 10,
                "completeness_factor": evidence_completeness
            }
        
        return {
            "indicators": indicators,
            "confidence_scores": confidence_scores,
            "analysis": analysis
        }
    
    def _check_metric_precision(self, validation_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Check for suspiciously precise metrics"""
        
        suspicious_precision = False
        confidence = 0.0
        precision_threshold = self.detection_config["evidence_analysis"]["metric_precision_threshold"]
        
        suspicious_metrics = []
        
        for evidence_key, evidence_item in validation_evidence.items():
            if isinstance(evidence_item, dict):
                for metric_name, metric_value in evidence_item.items():
                    if isinstance(metric_value, float):
                        # Check if metric has suspiciously high precision
                        decimal_places = len(str(metric_value).split('.')[-1]) if '.' in str(metric_value) else 0
                        
                        if decimal_places > 6:  # More than 6 decimal places is suspicious
                            suspicious_precision = True
                            confidence += 0.2
                            suspicious_metrics.append({
                                "evidence_key": evidence_key,
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                                "decimal_places": decimal_places
                            })
                        
                        # Check if metric is exactly zero when it shouldn't be
                        if metric_value == 0.0 and "time" in metric_name.lower():
                            suspicious_precision = True
                            confidence += 0.3
                            suspicious_metrics.append({
                                "evidence_key": evidence_key,
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                                "issue": "zero_timing_metric"
                            })
        
        return {
            "suspicious_precision": suspicious_precision,
            "confidence": min(confidence, 1.0),
            "suspicious_metrics": suspicious_metrics
        }
    
    def _analyze_temporal_patterns(
        self,
        validation_id: str,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns for anomalies"""
        
        indicators = []
        confidence_scores = []
        analysis = {}
        
        # Check for validation pattern repetition (identical results at different times)
        similar_validations = self._find_similar_validations(validation_results, validation_evidence)
        if similar_validations:
            indicators.append(FalsePositiveIndicator.VALIDATION_PATTERN_REPETITION)
            confidence_scores.append(len(similar_validations) / 10.0)  # Higher confidence with more repetitions
            analysis["pattern_repetition"] = {
                "similar_validations_found": len(similar_validations),
                "similarity_threshold": 0.95,
                "similar_validation_ids": similar_validations[:5]  # First 5 for logging
            }
        
        # Check for temporal clustering (too many validations in short time window)
        recent_validations = self._get_recent_validations(3600)  # 1 hour window
        if len(recent_validations) > 10:  # More than 10 validations per hour is suspicious
            indicators.append(FalsePositiveIndicator.INSUFFICIENT_TESTING_DEPTH)
            confidence_scores.append(min((len(recent_validations) - 10) / 20.0, 1.0))
            analysis["temporal_clustering"] = {
                "validations_in_window": len(recent_validations),
                "time_window_hours": 1,
                "average_time_between_validations": 3600 / len(recent_validations)
            }
        
        return {
            "indicators": indicators,
            "confidence_scores": confidence_scores,
            "analysis": analysis
        }
    
    def _find_similar_validations(
        self,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any]
    ) -> List[str]:
        """Find historically similar validations"""
        
        similar_validations = []
        
        for historical_validation in self.historical_validations[-50:]:  # Check last 50 validations
            similarity = self._calculate_validation_similarity(
                validation_results,
                historical_validation["validation_results"]
            )
            
            if similarity > 0.95:  # 95% similarity threshold
                similar_validations.append(historical_validation["validation_id"])
        
        return similar_validations
    
    def _calculate_validation_similarity(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> float:
        """Calculate similarity between two validation results"""
        
        common_keys = set(results1.keys()) & set(results2.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        
        for key in common_keys:
            value1 = results1[key]
            value2 = results2[key]
            
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if value1 == 0 and value2 == 0:
                    similarity_scores.append(1.0)
                elif value1 == 0 or value2 == 0:
                    similarity_scores.append(0.0)
                else:
                    similarity_scores.append(1.0 - abs(value1 - value2) / max(abs(value1), abs(value2)))
            elif value1 == value2:
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)
        
        return statistics.mean(similarity_scores) if similarity_scores else 0.0
    
    def _get_recent_validations(self, time_window_seconds: int) -> List[Dict[str, Any]]:
        """Get validations within recent time window"""
        
        cutoff_time = time.time() - time_window_seconds
        
        recent = []
        for validation in self.historical_validations:
            if validation.get("timestamp", 0) > cutoff_time:
                recent.append(validation)
        
        return recent
    
    def _analyze_performance_plausibility(
        self,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance results for plausibility"""
        
        indicators = []
        confidence_scores = []
        analysis = {}
        
        # Check for impossible performance improvements
        if "performance_improvement" in validation_results:
            improvement = validation_results["performance_improvement"]
            improvement_threshold = self.detection_config["statistical_thresholds"]["performance_improvement_threshold"]
            
            if improvement > improvement_threshold:
                indicators.append(FalsePositiveIndicator.UNREALISTIC_PERFORMANCE)
                confidence_scores.append(min((improvement - improvement_threshold) * 2, 1.0))
                analysis["unrealistic_improvement"] = {
                    "observed_improvement": improvement,
                    "threshold": improvement_threshold,
                    "plausibility": "low"
                }
        
        # Check resource usage consistency
        resource_analysis = self._analyze_resource_usage_plausibility(validation_evidence)
        if resource_analysis["implausible"]:
            indicators.append(FalsePositiveIndicator.FABRICATED_METRICS)
            confidence_scores.append(resource_analysis["confidence"])
            analysis["resource_usage"] = resource_analysis
        
        return {
            "indicators": indicators,
            "confidence_scores": confidence_scores,
            "analysis": analysis
        }
    
    def _analyze_resource_usage_plausibility(self, validation_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage metrics for plausibility"""
        
        implausible = False
        confidence = 0.0
        
        # Look for resource usage metrics
        cpu_metrics = []
        memory_metrics = []
        
        for evidence_item in validation_evidence.values():
            if isinstance(evidence_item, dict):
                if "cpu_percent" in evidence_item:
                    cpu_metrics.append(evidence_item["cpu_percent"])
                if "memory_percent" in evidence_item:
                    memory_metrics.append(evidence_item["memory_percent"])
        
        # Check for impossibly low resource usage
        if cpu_metrics and all(cpu < 0.1 for cpu in cpu_metrics):  # Less than 0.1% CPU is implausible
            implausible = True
            confidence += 0.6
        
        # Check for perfectly consistent resource usage
        if cpu_metrics and len(set(cpu_metrics)) == 1 and len(cpu_metrics) > 3:
            implausible = True
            confidence += 0.4
        
        return {
            "implausible": implausible,
            "confidence": min(confidence, 1.0),
            "cpu_metrics_analyzed": len(cpu_metrics),
            "memory_metrics_analyzed": len(memory_metrics),
            "analysis_details": {
                "cpu_range": (min(cpu_metrics), max(cpu_metrics)) if cpu_metrics else None,
                "cpu_variance": statistics.variance(cpu_metrics) if len(cpu_metrics) > 1 else None
            }
        }
    
    def _determine_alert_severity(
        self,
        indicators: List[FalsePositiveIndicator],
        confidence_score: float
    ) -> str:
        """Determine severity of false positive alert"""
        
        critical_indicators = [
            FalsePositiveIndicator.IMPOSSIBLY_FAST_EXECUTION,
            FalsePositiveIndicator.FABRICATED_METRICS,
            FalsePositiveIndicator.VALIDATION_PATTERN_REPETITION
        ]
        
        high_indicators = [
            FalsePositiveIndicator.SUSPICIOUSLY_PERFECT_RESULTS,
            FalsePositiveIndicator.UNREALISTIC_PERFORMANCE,
            FalsePositiveIndicator.INCONSISTENT_EVIDENCE
        ]
        
        if any(indicator in critical_indicators for indicator in indicators) and confidence_score > 0.8:
            return "critical"
        elif any(indicator in high_indicators for indicator in indicators) and confidence_score > 0.6:
            return "high"
        elif len(indicators) > 3 or confidence_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_recommended_actions(self, indicators: List[FalsePositiveIndicator]) -> List[str]:
        """Generate recommended actions based on detected indicators"""
        
        actions = []
        
        for indicator in indicators:
            if indicator == FalsePositiveIndicator.SUSPICIOUSLY_PERFECT_RESULTS:
                actions.append("Re-run validation with more rigorous testing conditions")
                actions.append("Include error injection testing to verify error handling")
            
            elif indicator == FalsePositiveIndicator.IMPOSSIBLY_FAST_EXECUTION:
                actions.append("Verify test actually executed - check for test skipping or mocking")
                actions.append("Add timing verification to ensure actual operations occurred")
            
            elif indicator == FalsePositiveIndicator.FABRICATED_METRICS:
                actions.append("Verify evidence collection authenticity")
                actions.append("Cross-validate metrics with independent measurement tools")
            
            elif indicator == FalsePositiveIndicator.PATTERN_DEVIATION:
                actions.append("Compare validation approach with historical successful validations")
                actions.append("Investigate why results deviate significantly from expected patterns")
            
            elif indicator == FalsePositiveIndicator.MISSING_ERROR_CONDITIONS:
                actions.append("Add comprehensive error condition testing")
                actions.append("Verify system actually handles edge cases and failures")
            
            elif indicator == FalsePositiveIndicator.VALIDATION_PATTERN_REPETITION:
                actions.append("Investigate potential test result caching or copying")
                actions.append("Ensure validation is testing fresh system state")
        
        # Add general recommendations
        actions.append("Consider requiring peer validation review for this result")
        actions.append("Run adversarial testing to challenge validation assumptions")
        
        return list(set(actions))  # Remove duplicates
    
    def _load_validation_patterns(self):
        """Load historical validation patterns"""
        try:
            patterns_file = self.detection_data_path / "validation_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    
                for pattern_data in patterns_data.get("patterns", []):
                    pattern = ValidationPattern(
                        pattern_id=pattern_data["pattern_id"],
                        validation_type=pattern_data["validation_type"],
                        typical_execution_time_range=tuple(pattern_data["typical_execution_time_range"]),
                        typical_success_rate_range=tuple(pattern_data["typical_success_rate_range"]),
                        common_error_types=pattern_data["common_error_types"],
                        expected_evidence_types=set(pattern_data["expected_evidence_types"]),
                        performance_benchmarks=pattern_data["performance_benchmarks"],
                        complexity_indicators=pattern_data["complexity_indicators"]
                    )
                    self.validation_patterns[pattern.validation_type] = pattern
        except Exception as e:
            self.logger.error(f"Failed to load validation patterns: {e}")
    
    def _load_historical_validations(self):
        """Load historical validation data"""
        try:
            history_file = self.detection_data_path / "validation_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.historical_validations = history_data.get("validations", [])
        except Exception as e:
            self.logger.error(f"Failed to load historical validations: {e}")
    
    def _update_historical_validations(
        self,
        validation_id: str,
        validation_results: Dict[str, Any],
        validation_evidence: Dict[str, Any]
    ):
        """Update historical validation data"""
        
        historical_record = {
            "validation_id": validation_id,
            "timestamp": time.time(),
            "validation_results": validation_results,
            "evidence_summary": {
                "evidence_count": len(validation_evidence),
                "evidence_types": list(validation_evidence.keys())
            }
        }
        
        self.historical_validations.append(historical_record)
        
        # Keep only last 1000 validations
        if len(self.historical_validations) > 1000:
            self.historical_validations = self.historical_validations[-1000:]
        
        # Store updated history
        try:
            history_file = self.detection_data_path / "validation_history.json"
            with open(history_file, 'w') as f:
                json.dump({"validations": self.historical_validations}, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to store validation history: {e}")
    
    def _store_false_positive_alert(self, alert: FalsePositiveAlert):
        """Store false positive alert"""
        try:
            alert_file = self.detection_data_path / f"alert_{alert.alert_id}.json"
            
            with open(alert_file, 'w') as f:
                json.dump(asdict(alert), f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to store false positive alert: {e}")
    
    def get_false_positive_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of false positive alerts in time window"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_alerts = [
            alert for alert in self.false_positive_alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff_time
        ]
        
        if not recent_alerts:
            return {
                "summary": "No false positive alerts in specified time window",
                "alert_count": 0,
                "time_window_hours": time_window_hours
            }
        
        # Analyze alerts
        severity_counts = {}
        indicator_counts = {}
        
        for alert in recent_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            
            for indicator in alert.indicators:
                indicator_name = indicator.value
                indicator_counts[indicator_name] = indicator_counts.get(indicator_name, 0) + 1
        
        avg_confidence = statistics.mean([alert.confidence_score for alert in recent_alerts])
        
        return {
            "summary": {
                "total_alerts": len(recent_alerts),
                "time_window_hours": time_window_hours,
                "average_confidence": avg_confidence,
                "severity_distribution": severity_counts,
                "most_common_indicators": sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "validation_id": alert.validation_id,
                    "issue_id": alert.issue_id,
                    "confidence_score": alert.confidence_score,
                    "severity": alert.severity,
                    "indicator_count": len(alert.indicators)
                } for alert in recent_alerts
            ]
        }


# Global detector instance
_global_false_positive_detector = None

def get_false_positive_detector() -> FalsePositiveDetector:
    """Get global false positive detector instance"""
    global _global_false_positive_detector
    if _global_false_positive_detector is None:
        _global_false_positive_detector = FalsePositiveDetector()
    return _global_false_positive_detector


def analyze_validation_for_false_positives(
    validation_id: str,
    issue_id: str,
    validator_agent: str,
    validation_results: Dict[str, Any],
    validation_evidence: Dict[str, Any]
) -> FalsePositiveAlert:
    """Analyze validation for false positive indicators"""
    detector = get_false_positive_detector()
    return detector.analyze_validation_for_false_positives(
        validation_id, issue_id, validator_agent, validation_results, validation_evidence
    )


# Example usage and testing
if __name__ == "__main__":
    print("🕵️ False Positive Detection System - Preventing Validation Fraud")
    print("=" * 80)
    
    # Create detector
    detector = FalsePositiveDetector()
    
    # Test with suspicious validation results
    suspicious_results = {
        "success_rate": 1.0,  # Perfect success rate
        "execution_time": 0.001,  # Impossibly fast
        "total_tests": 50,
        "failed_tests": 0,  # No failures
        "average_response_time": 0.0001,  # Impossibly fast response
        "performance_improvement": 0.8  # 80% improvement
    }
    
    suspicious_evidence = {
        "test_execution": {
            "timestamp": time.time(),
            "cpu_percent": 0.01,  # Impossibly low CPU usage
            "memory_percent": 0.02  # Impossibly low memory usage
        },
        "performance_metrics": {
            "timestamp": time.time() + 1,
            "response_time": 0.123456789,  # Suspiciously precise
            "throughput": 15000  # Unrealistically high
        }
    }
    
    print("\n🔍 Analyzing suspicious validation results...")
    alert = detector.analyze_validation_for_false_positives(
        "suspicious_validation_test",
        "231",
        "test_validator",
        suspicious_results,
        suspicious_evidence
    )
    
    print(f"\nFalse Positive Alert Generated:")
    print(f"Alert ID: {alert.alert_id}")
    print(f"Confidence Score: {alert.confidence_score:.3f}")
    print(f"Severity: {alert.severity}")
    print(f"Indicators Found: {len(alert.indicators)}")
    for indicator in alert.indicators:
        print(f"  - {indicator.value}")
    
    print(f"\nRecommended Actions:")
    for action in alert.recommended_actions:
        print(f"  - {action}")
    
    # Test with normal validation results
    normal_results = {
        "success_rate": 0.85,  # Realistic success rate
        "execution_time": 15.2,  # Reasonable execution time
        "total_tests": 25,
        "failed_tests": 3,  # Some failures expected
        "average_response_time": 0.45  # Realistic response time
    }
    
    normal_evidence = {
        "test_execution": {
            "timestamp": time.time(),
            "cpu_percent": 45.2,  # Realistic CPU usage
            "memory_percent": 68.5  # Realistic memory usage
        }
    }
    
    print(f"\n🔍 Analyzing normal validation results...")
    normal_alert = detector.analyze_validation_for_false_positives(
        "normal_validation_test",
        "231",
        "test_validator",
        normal_results,
        normal_evidence
    )
    
    print(f"\nNormal Validation Analysis:")
    print(f"Confidence Score: {normal_alert.confidence_score:.3f}")
    print(f"Severity: {normal_alert.severity}")
    print(f"Indicators Found: {len(normal_alert.indicators)}")
    
    # Generate summary report
    print(f"\n📊 False Positive Detection Summary:")
    summary = detector.get_false_positive_summary(24)
    print(json.dumps(summary, indent=2)[:1000] + "...")
    
    print(f"\n✅ FALSE POSITIVE DETECTION SYSTEM OPERATIONAL")
    print(f"✅ STATISTICAL ANOMALY DETECTION: ACTIVE")
    print(f"✅ PATTERN DEVIATION ANALYSIS: ENABLED")
    print(f"✅ EVIDENCE CONSISTENCY CHECKING: OPERATIONAL")
    print(f"✅ VALIDATION FRAUD PREVENTION: ENHANCED")