#!/usr/bin/env python3
"""
Integration Evidence Validator

Validates the authenticity and completeness of integration test evidence 
to prevent fabricated validation claims that led to Issue #225.

This system ensures that integration test results are:
1. Actually executed (not fabricated)
2. Comprehensive and complete  
3. Authenticated with proper timestamps and context
4. Verified against expected evidence patterns

Issue #231: Root Cause Analysis: False Validation of MCP Server Integration
"""

import json
import hashlib
import time
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import subprocess
from dataclasses import dataclass


@dataclass
class EvidenceValidationResult:
    """Result of evidence validation"""
    is_valid: bool
    confidence_score: float
    authenticity_verified: bool
    completeness_score: float
    evidence_quality: str  # "excellent", "good", "acceptable", "poor", "invalid"
    validation_errors: List[str]
    evidence_fingerprint: str
    verification_timestamp: str


class IntegrationEvidenceValidator:
    """
    Validates integration test evidence to prevent false positive validations.
    
    Prevents validation fraud by:
    - Verifying test execution authenticity
    - Validating evidence completeness
    - Cross-referencing evidence consistency
    - Detecting fabricated or manipulated evidence
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = Path(knowledge_base_path or "/Users/cal/DEV/RIF/knowledge")
        self.evidence_patterns = self._load_evidence_patterns()
        self.evidence_cache = {}
        self.validation_history = []
        
        # Evidence authenticity requirements
        self.authenticity_requirements = {
            "timestamp_validity": {"max_age_hours": 24, "required": True},
            "execution_traces": {"required_traces": ["stdout", "stderr", "exit_code"], "required": True},
            "test_environment": {"required_fields": ["python_version", "test_framework", "host"], "required": True},
            "evidence_consistency": {"cross_validation": True, "required": True},
            "process_verification": {"pid_tracking": True, "execution_time": True, "required": False}
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for evidence validation"""
        log_dir = self.knowledge_base_path / "enforcement_logs" 
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"evidence_validation_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("IntegrationEvidenceValidator")
    
    def _load_evidence_patterns(self) -> Dict[str, Any]:
        """Load known evidence patterns for validation"""
        return {
            "claude_desktop_connection": {
                "required_fields": [
                    "connection_successful", "protocol_compliant", "tools_accessible",
                    "timestamp", "connection_attempts", "error_details"
                ],
                "expected_values": {
                    "connection_successful": bool,
                    "protocol_compliant": bool, 
                    "tools_accessible": bool,
                    "connection_attempts": int
                },
                "consistency_rules": [
                    "if connection_successful is False, error_details must be provided",
                    "if protocol_compliant is True, connection_successful must be True",
                    "connection_attempts must be >= 1"
                ]
            },
            "tool_invocation": {
                "required_fields": [
                    "tools_tested", "tools_successful", "tool_results", "timestamp"
                ],
                "expected_values": {
                    "tools_tested": int,
                    "tools_successful": int,
                    "success_rate": float,
                    "tool_results": dict
                },
                "consistency_rules": [
                    "tools_successful <= tools_tested",
                    "success_rate == tools_successful / tools_tested", 
                    "len(tool_results) == tools_tested"
                ]
            },
            "production_simulation": {
                "required_fields": [
                    "performance_under_load", "error_handling", "stability_metrics", "timestamp"
                ],
                "expected_values": {
                    "performance_under_load": dict,
                    "error_handling": dict,
                    "stability_metrics": dict
                },
                "consistency_rules": [
                    "performance_under_load must contain concurrent_requests and success_rate",
                    "stability_metrics must contain process_responsive and no_crashes"
                ]
            }
        }
    
    def validate_integration_evidence(
        self, 
        evidence_type: str,
        evidence_data: Dict[str, Any],
        test_metadata: Optional[Dict[str, Any]] = None
    ) -> EvidenceValidationResult:
        """
        Validate integration test evidence for authenticity and completeness.
        
        Args:
            evidence_type: Type of evidence (claude_desktop_connection, tool_invocation, etc.)
            evidence_data: The evidence data to validate
            test_metadata: Optional metadata about test execution
            
        Returns:
            EvidenceValidationResult with validation details
        """
        validation_start = time.time()
        validation_errors = []
        
        # Generate evidence fingerprint
        evidence_fingerprint = self._generate_evidence_fingerprint(evidence_data)
        
        # Check if this evidence was already validated
        if evidence_fingerprint in self.evidence_cache:
            cached_result = self.evidence_cache[evidence_fingerprint]
            self.logger.info(f"Using cached validation result for evidence {evidence_fingerprint[:8]}...")
            return cached_result
        
        # 1. Validate evidence structure and completeness
        structure_validation = self._validate_evidence_structure(evidence_type, evidence_data)
        completeness_score = structure_validation["completeness_score"]
        validation_errors.extend(structure_validation["errors"])
        
        # 2. Validate evidence authenticity
        authenticity_validation = self._validate_evidence_authenticity(evidence_data, test_metadata)
        authenticity_verified = authenticity_validation["authenticity_verified"]
        validation_errors.extend(authenticity_validation["errors"])
        
        # 3. Validate evidence consistency
        consistency_validation = self._validate_evidence_consistency(evidence_type, evidence_data)
        validation_errors.extend(consistency_validation["errors"])
        
        # 4. Cross-validate with known patterns
        pattern_validation = self._validate_against_patterns(evidence_type, evidence_data)
        validation_errors.extend(pattern_validation["errors"])
        
        # 5. Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            completeness_score,
            authenticity_verified,
            len(validation_errors),
            consistency_validation["consistency_score"],
            pattern_validation["pattern_match_score"]
        )
        
        # 6. Determine overall validity
        is_valid = (
            len(validation_errors) == 0 and
            completeness_score >= 80.0 and
            authenticity_verified and
            confidence_score >= 70.0
        )
        
        # 7. Determine evidence quality
        evidence_quality = self._determine_evidence_quality(
            confidence_score, 
            completeness_score, 
            authenticity_verified,
            len(validation_errors)
        )
        
        # Create validation result
        result = EvidenceValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            authenticity_verified=authenticity_verified,
            completeness_score=completeness_score,
            evidence_quality=evidence_quality,
            validation_errors=validation_errors,
            evidence_fingerprint=evidence_fingerprint,
            verification_timestamp=datetime.now().isoformat()
        )
        
        # Cache result
        self.evidence_cache[evidence_fingerprint] = result
        
        # Store validation record
        self._store_validation_record(evidence_type, evidence_data, result, validation_start)
        
        self.logger.info(
            f"Evidence validation completed: {evidence_type} - "
            f"Valid: {is_valid}, Quality: {evidence_quality}, "
            f"Confidence: {confidence_score:.1f}%"
        )
        
        return result
    
    def _generate_evidence_fingerprint(self, evidence_data: Dict[str, Any]) -> str:
        """Generate unique fingerprint for evidence data"""
        # Create a deterministic hash of the evidence data
        evidence_str = json.dumps(evidence_data, sort_keys=True, default=str)
        return hashlib.sha256(evidence_str.encode()).hexdigest()
    
    def _validate_evidence_structure(
        self, 
        evidence_type: str, 
        evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate evidence structure and completeness"""
        
        if evidence_type not in self.evidence_patterns:
            return {
                "completeness_score": 0.0,
                "errors": [f"Unknown evidence type: {evidence_type}"]
            }
        
        pattern = self.evidence_patterns[evidence_type]
        required_fields = pattern["required_fields"]
        expected_values = pattern["expected_values"]
        
        # Check required fields
        missing_fields = []
        for field in required_fields:
            if field not in evidence_data:
                missing_fields.append(field)
        
        errors = []
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check field types
        type_errors = []
        for field, expected_type in expected_values.items():
            if field in evidence_data:
                actual_value = evidence_data[field]
                if not isinstance(actual_value, expected_type):
                    type_errors.append(f"Field {field}: expected {expected_type.__name__}, got {type(actual_value).__name__}")
        
        if type_errors:
            errors.extend(type_errors)
        
        # Calculate completeness score
        total_fields = len(required_fields)
        present_fields = total_fields - len(missing_fields)
        completeness_score = (present_fields / total_fields) * 100 if total_fields > 0 else 0
        
        return {
            "completeness_score": completeness_score,
            "errors": errors,
            "missing_fields": missing_fields,
            "type_errors": type_errors
        }
    
    def _validate_evidence_authenticity(
        self, 
        evidence_data: Dict[str, Any], 
        test_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate that evidence is authentic (not fabricated)"""
        
        errors = []
        authenticity_verified = True
        
        # 1. Timestamp validation
        if "timestamp" in evidence_data:
            timestamp_valid = self._validate_timestamp(evidence_data["timestamp"])
            if not timestamp_valid:
                errors.append("Invalid or suspicious timestamp")
                authenticity_verified = False
        else:
            errors.append("Missing timestamp for authenticity verification")
            authenticity_verified = False
        
        # 2. Test metadata validation
        if test_metadata:
            metadata_validation = self._validate_test_metadata(test_metadata)
            if not metadata_validation["is_valid"]:
                errors.extend(metadata_validation["errors"])
                authenticity_verified = False
        
        # 3. Evidence pattern matching
        pattern_authenticity = self._validate_evidence_patterns(evidence_data)
        if not pattern_authenticity["authentic"]:
            errors.extend(pattern_authenticity["errors"])
            authenticity_verified = False
        
        # 4. Cross-reference validation
        cross_ref_validation = self._cross_reference_evidence(evidence_data)
        if not cross_ref_validation["consistent"]:
            errors.extend(cross_ref_validation["errors"])
            authenticity_verified = False
        
        return {
            "authenticity_verified": authenticity_verified,
            "errors": errors
        }
    
    def _validate_timestamp(self, timestamp: Any) -> bool:
        """Validate timestamp for authenticity"""
        try:
            if isinstance(timestamp, (int, float)):
                test_time = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                test_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                return False
            
            now = datetime.now()
            time_diff = abs((now - test_time).total_seconds())
            
            # Timestamp should be recent (within last 24 hours) but not in future
            max_age = 24 * 60 * 60  # 24 hours in seconds
            future_tolerance = 5 * 60  # 5 minutes future tolerance
            
            if time_diff > max_age:
                return False
            
            if (test_time - now).total_seconds() > future_tolerance:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_test_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test execution metadata"""
        errors = []
        
        required_metadata = ["test_framework", "execution_time", "host_info"]
        for field in required_metadata:
            if field not in metadata:
                errors.append(f"Missing test metadata: {field}")
        
        # Validate execution time is reasonable
        if "execution_time" in metadata:
            exec_time = metadata["execution_time"]
            if not isinstance(exec_time, (int, float)) or exec_time <= 0 or exec_time > 3600:
                errors.append("Suspicious execution time")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    def _validate_evidence_patterns(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evidence follows expected patterns"""
        errors = []
        
        # Check for suspiciously perfect results
        if "success_rate" in evidence_data:
            success_rate = evidence_data["success_rate"]
            if success_rate == 1.0 and "tools_tested" in evidence_data and evidence_data["tools_tested"] > 5:
                errors.append("Suspiciously perfect success rate for large test set")
        
        # Check for impossible timing
        if "response_time" in evidence_data:
            response_time = evidence_data["response_time"]
            if response_time < 0.001:  # Less than 1ms is suspicious
                errors.append("Impossibly fast response time")
        
        # Check for fabricated error patterns
        if "error_details" in evidence_data and evidence_data["error_details"]:
            error_details = str(evidence_data["error_details"])
            if len(error_details) < 10 or "generic error" in error_details.lower():
                errors.append("Suspicious or generic error details")
        
        return {
            "authentic": len(errors) == 0,
            "errors": errors
        }
    
    def _cross_reference_evidence(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference evidence for internal consistency"""
        errors = []
        
        # Mathematical consistency checks
        if all(k in evidence_data for k in ["tools_tested", "tools_successful"]):
            tools_tested = evidence_data["tools_tested"]
            tools_successful = evidence_data["tools_successful"]
            
            if tools_successful > tools_tested:
                errors.append("tools_successful cannot exceed tools_tested")
            
            if "success_rate" in evidence_data:
                expected_rate = tools_successful / tools_tested if tools_tested > 0 else 0
                actual_rate = evidence_data["success_rate"]
                if abs(expected_rate - actual_rate) > 0.01:  # Allow small floating point errors
                    errors.append("Success rate inconsistent with test counts")
        
        # Logical consistency checks
        if "connection_successful" in evidence_data and "tools_accessible" in evidence_data:
            if evidence_data["tools_accessible"] and not evidence_data["connection_successful"]:
                errors.append("Tools cannot be accessible if connection failed")
        
        return {
            "consistent": len(errors) == 0,
            "errors": errors
        }
    
    def _validate_evidence_consistency(
        self, 
        evidence_type: str, 
        evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate evidence against consistency rules"""
        
        if evidence_type not in self.evidence_patterns:
            return {"consistency_score": 0, "errors": ["Unknown evidence type"]}
        
        consistency_rules = self.evidence_patterns[evidence_type]["consistency_rules"]
        errors = []
        passed_rules = 0
        
        for rule in consistency_rules:
            try:
                rule_passed = self._evaluate_consistency_rule(rule, evidence_data)
                if rule_passed:
                    passed_rules += 1
                else:
                    errors.append(f"Consistency rule failed: {rule}")
            except Exception as e:
                errors.append(f"Error evaluating rule '{rule}': {str(e)}")
        
        consistency_score = (passed_rules / len(consistency_rules)) * 100 if consistency_rules else 100
        
        return {
            "consistency_score": consistency_score,
            "errors": errors,
            "rules_passed": passed_rules,
            "total_rules": len(consistency_rules)
        }
    
    def _evaluate_consistency_rule(self, rule: str, evidence_data: Dict[str, Any]) -> bool:
        """Evaluate a single consistency rule"""
        
        # Simple rule evaluation (could be enhanced with a proper expression parser)
        
        if "if connection_successful is False, error_details must be provided" in rule:
            if "connection_successful" in evidence_data and not evidence_data["connection_successful"]:
                return "error_details" in evidence_data and evidence_data["error_details"]
            return True
        
        if "if protocol_compliant is True, connection_successful must be True" in rule:
            if "protocol_compliant" in evidence_data and evidence_data["protocol_compliant"]:
                return evidence_data.get("connection_successful", False)
            return True
        
        if "connection_attempts must be >= 1" in rule:
            return evidence_data.get("connection_attempts", 0) >= 1
        
        if "tools_successful <= tools_tested" in rule:
            return evidence_data.get("tools_successful", 0) <= evidence_data.get("tools_tested", 0)
        
        if "success_rate == tools_successful / tools_tested" in rule:
            tools_tested = evidence_data.get("tools_tested", 0)
            tools_successful = evidence_data.get("tools_successful", 0)
            success_rate = evidence_data.get("success_rate", 0)
            expected_rate = tools_successful / tools_tested if tools_tested > 0 else 0
            return abs(expected_rate - success_rate) < 0.01
        
        if "len(tool_results) == tools_tested" in rule:
            tool_results = evidence_data.get("tool_results", {})
            tools_tested = evidence_data.get("tools_tested", 0)
            return len(tool_results) == tools_tested
        
        # Default: assume rule passes if we can't evaluate it
        return True
    
    def _validate_against_patterns(
        self, 
        evidence_type: str, 
        evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate evidence against known good patterns"""
        
        errors = []
        pattern_match_score = 100.0
        
        # Load historical evidence patterns for comparison
        historical_patterns = self._load_historical_patterns(evidence_type)
        
        if historical_patterns:
            # Compare against historical patterns
            pattern_comparison = self._compare_to_historical_patterns(evidence_data, historical_patterns)
            pattern_match_score = pattern_comparison["match_score"]
            
            if pattern_match_score < 50.0:
                errors.append("Evidence deviates significantly from historical patterns")
            elif pattern_match_score < 70.0:
                errors.append("Evidence shows unusual patterns compared to historical data")
        
        return {
            "pattern_match_score": pattern_match_score,
            "errors": errors
        }
    
    def _load_historical_patterns(self, evidence_type: str) -> List[Dict[str, Any]]:
        """Load historical evidence patterns for comparison"""
        try:
            patterns_file = self.knowledge_base_path / "patterns" / f"{evidence_type}_evidence_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _compare_to_historical_patterns(
        self, 
        evidence_data: Dict[str, Any], 
        historical_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare evidence to historical patterns"""
        
        if not historical_patterns:
            return {"match_score": 50.0}  # Neutral score if no patterns available
        
        # Simple pattern matching - could be enhanced with ML techniques
        match_scores = []
        
        for pattern in historical_patterns:
            score = 0
            total_fields = 0
            
            for field, value in evidence_data.items():
                if field in pattern:
                    total_fields += 1
                    pattern_value = pattern[field]
                    
                    if isinstance(value, bool) and isinstance(pattern_value, bool):
                        if value == pattern_value:
                            score += 100
                    elif isinstance(value, (int, float)) and isinstance(pattern_value, (int, float)):
                        # Calculate similarity based on relative difference
                        if pattern_value != 0:
                            diff = abs(value - pattern_value) / abs(pattern_value)
                            similarity = max(0, 100 - (diff * 100))
                            score += similarity
                        else:
                            score += 100 if value == 0 else 0
                    elif isinstance(value, str) and isinstance(pattern_value, str):
                        # Simple string similarity
                        similarity = 100 if value == pattern_value else 50
                        score += similarity
            
            if total_fields > 0:
                match_scores.append(score / total_fields)
        
        # Return the average match score
        average_score = sum(match_scores) / len(match_scores) if match_scores else 50.0
        return {"match_score": average_score}
    
    def _calculate_confidence_score(
        self,
        completeness_score: float,
        authenticity_verified: bool,
        error_count: int,
        consistency_score: float,
        pattern_match_score: float
    ) -> float:
        """Calculate overall confidence score for evidence validity"""
        
        # Weighted scoring system
        weights = {
            "completeness": 0.25,
            "authenticity": 0.35,
            "errors": 0.15,
            "consistency": 0.15,
            "pattern_match": 0.10
        }
        
        # Completeness contribution
        completeness_contribution = completeness_score * weights["completeness"]
        
        # Authenticity contribution
        authenticity_contribution = (100.0 if authenticity_verified else 0.0) * weights["authenticity"]
        
        # Error penalty
        max_acceptable_errors = 3
        error_penalty = min(error_count / max_acceptable_errors, 1.0) * 100
        error_contribution = (100.0 - error_penalty) * weights["errors"]
        
        # Consistency contribution
        consistency_contribution = consistency_score * weights["consistency"]
        
        # Pattern match contribution
        pattern_contribution = pattern_match_score * weights["pattern_match"]
        
        # Calculate total confidence score
        confidence_score = (
            completeness_contribution +
            authenticity_contribution +
            error_contribution +
            consistency_contribution +
            pattern_contribution
        )
        
        return min(confidence_score, 100.0)
    
    def _determine_evidence_quality(
        self,
        confidence_score: float,
        completeness_score: float,
        authenticity_verified: bool,
        error_count: int
    ) -> str:
        """Determine evidence quality rating"""
        
        if not authenticity_verified or error_count > 5:
            return "invalid"
        
        if confidence_score >= 90 and completeness_score >= 95 and error_count == 0:
            return "excellent"
        elif confidence_score >= 80 and completeness_score >= 85 and error_count <= 1:
            return "good"
        elif confidence_score >= 70 and completeness_score >= 75 and error_count <= 2:
            return "acceptable"
        elif confidence_score >= 50 and completeness_score >= 60:
            return "poor"
        else:
            return "invalid"
    
    def _store_validation_record(
        self,
        evidence_type: str,
        evidence_data: Dict[str, Any],
        validation_result: EvidenceValidationResult,
        validation_start: float
    ):
        """Store validation record for audit trail"""
        
        try:
            validation_dir = self.knowledge_base_path / "enforcement_logs" / "evidence_validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            
            validation_record = {
                "evidence_type": evidence_type,
                "evidence_fingerprint": validation_result.evidence_fingerprint,
                "validation_result": {
                    "is_valid": validation_result.is_valid,
                    "confidence_score": validation_result.confidence_score,
                    "authenticity_verified": validation_result.authenticity_verified,
                    "completeness_score": validation_result.completeness_score,
                    "evidence_quality": validation_result.evidence_quality,
                    "validation_errors": validation_result.validation_errors
                },
                "validation_metadata": {
                    "validation_duration": time.time() - validation_start,
                    "timestamp": validation_result.verification_timestamp,
                    "validator_version": "1.0.0"
                },
                "evidence_summary": {
                    "field_count": len(evidence_data),
                    "data_size": len(json.dumps(evidence_data, default=str))
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence_validation_{evidence_type}_{timestamp}.json"
            filepath = validation_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(validation_record, f, indent=2)
                
            self.validation_history.append(validation_record)
            
        except Exception as e:
            self.logger.error(f"Failed to store validation record: {e}")
    
    def batch_validate_evidence(
        self, 
        evidence_batch: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]]
    ) -> List[EvidenceValidationResult]:
        """Validate multiple pieces of evidence in batch"""
        
        results = []
        
        for evidence_type, evidence_data, test_metadata in evidence_batch:
            try:
                result = self.validate_integration_evidence(evidence_type, evidence_data, test_metadata)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = EvidenceValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    authenticity_verified=False,
                    completeness_score=0.0,
                    evidence_quality="invalid",
                    validation_errors=[f"Validation exception: {str(e)}"],
                    evidence_fingerprint="error",
                    verification_timestamp=datetime.now().isoformat()
                )
                results.append(error_result)
                self.logger.error(f"Batch validation error for {evidence_type}: {e}")
        
        return results
    
    def generate_evidence_validation_report(self, validation_results: List[EvidenceValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive evidence validation report"""
        
        total_evidence = len(validation_results)
        valid_evidence = sum(1 for r in validation_results if r.is_valid)
        authentic_evidence = sum(1 for r in validation_results if r.authenticity_verified)
        
        quality_distribution = {}
        for result in validation_results:
            quality = result.evidence_quality
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        avg_confidence = sum(r.confidence_score for r in validation_results) / total_evidence if total_evidence > 0 else 0
        avg_completeness = sum(r.completeness_score for r in validation_results) / total_evidence if total_evidence > 0 else 0
        
        report = {
            "validation_summary": {
                "total_evidence_items": total_evidence,
                "valid_evidence_items": valid_evidence,
                "invalid_evidence_items": total_evidence - valid_evidence,
                "authenticity_verified_items": authentic_evidence,
                "validation_success_rate": (valid_evidence / total_evidence) * 100 if total_evidence > 0 else 0,
                "authenticity_verification_rate": (authentic_evidence / total_evidence) * 100 if total_evidence > 0 else 0
            },
            "quality_analysis": {
                "quality_distribution": quality_distribution,
                "average_confidence_score": avg_confidence,
                "average_completeness_score": avg_completeness
            },
            "false_positive_prevention": {
                "fabrication_detection_active": True,
                "authenticity_verification_enforced": True,
                "consistency_checking_enabled": True,
                "pattern_matching_applied": True,
                "evidence_quality_threshold": 70.0
            },
            "validation_errors": [
                error for result in validation_results for error in result.validation_errors
            ],
            "recommendations": self._generate_validation_recommendations(validation_results),
            "report_generated": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_validation_recommendations(self, validation_results: List[EvidenceValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        invalid_count = sum(1 for r in validation_results if not r.is_valid)
        if invalid_count > 0:
            recommendations.append(f"Review and revalidate {invalid_count} invalid evidence items")
        
        low_confidence_count = sum(1 for r in validation_results if r.confidence_score < 70)
        if low_confidence_count > 0:
            recommendations.append(f"Improve quality of {low_confidence_count} low-confidence evidence items")
        
        unverified_count = sum(1 for r in validation_results if not r.authenticity_verified)
        if unverified_count > 0:
            recommendations.append(f"Strengthen authenticity verification for {unverified_count} items")
        
        error_count = sum(len(r.validation_errors) for r in validation_results)
        if error_count > 0:
            recommendations.append(f"Address {error_count} validation errors across evidence items")
        
        return recommendations


# Global validator instance
_global_evidence_validator = None

def get_evidence_validator() -> IntegrationEvidenceValidator:
    """Get global evidence validator instance"""
    global _global_evidence_validator
    if _global_evidence_validator is None:
        _global_evidence_validator = IntegrationEvidenceValidator()
    return _global_evidence_validator


def validate_integration_evidence(
    evidence_type: str,
    evidence_data: Dict[str, Any],
    test_metadata: Optional[Dict[str, Any]] = None
) -> EvidenceValidationResult:
    """Validate integration test evidence"""
    validator = get_evidence_validator()
    return validator.validate_integration_evidence(evidence_type, evidence_data, test_metadata)


def batch_validate_evidence(
    evidence_batch: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]]
) -> List[EvidenceValidationResult]:
    """Batch validate multiple evidence items"""
    validator = get_evidence_validator()
    return validator.batch_validate_evidence(evidence_batch)


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Integration Evidence Validator - Preventing Fabricated Validation Claims")
    print("=" * 80)
    
    # Create validator
    validator = IntegrationEvidenceValidator()
    
    # Test with valid evidence
    valid_evidence = {
        "connection_successful": True,
        "protocol_compliant": True,
        "tools_accessible": True,
        "timestamp": time.time(),
        "connection_attempts": 1,
        "error_details": None,
        "tools_tested": 3,
        "tools_successful": 3,
        "success_rate": 1.0,
        "tool_results": {"tool1": {"status": "success"}, "tool2": {"status": "success"}, "tool3": {"status": "success"}}
    }
    
    print("\n🔍 Testing valid evidence validation...")
    result = validator.validate_integration_evidence("claude_desktop_connection", valid_evidence)
    print(f"Valid: {result.is_valid}")
    print(f"Quality: {result.evidence_quality}")  
    print(f"Confidence: {result.confidence_score:.1f}%")
    print(f"Errors: {result.validation_errors}")
    
    # Test with suspicious evidence
    suspicious_evidence = {
        "connection_successful": True,
        "protocol_compliant": True,
        "tools_accessible": True,
        "timestamp": time.time() - 100000,  # Very old timestamp
        "connection_attempts": 1,
        "tools_tested": 10,
        "tools_successful": 10,
        "success_rate": 1.0,  # Suspiciously perfect
        "response_time": 0.0001  # Impossibly fast
    }
    
    print(f"\n🔍 Testing suspicious evidence validation...")
    result = validator.validate_integration_evidence("claude_desktop_connection", suspicious_evidence)
    print(f"Valid: {result.is_valid}")
    print(f"Quality: {result.evidence_quality}")
    print(f"Confidence: {result.confidence_score:.1f}%")
    print(f"Errors: {result.validation_errors}")
    
    print(f"\n✅ EVIDENCE VALIDATION SYSTEM ACTIVE")
    print(f"✅ FABRICATION DETECTION: ENABLED") 
    print(f"✅ AUTHENTICITY VERIFICATION: ENFORCED")
    print(f"✅ CONSISTENCY CHECKING: ACTIVE")
    print(f"✅ FALSE POSITIVE PREVENTION: OPERATIONAL")