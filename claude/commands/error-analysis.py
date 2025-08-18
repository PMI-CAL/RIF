#!/usr/bin/env python3
"""
RIF Error Analysis System - Comprehensive error tracking and analysis
Implements Five Whys, Fishbone diagrams, and continuous improvement
"""

import sys
import json
import datetime
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ErrorType(Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UNKNOWN = "unknown"

class ErrorSource(Enum):
    USER_CODE = "user_code"
    RIF_SYSTEM = "rif_system"
    CLAUDE_CODE = "claude_code"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

@dataclass
class ErrorEvent:
    """Core error event structure"""
    id: str
    timestamp: str
    severity: ErrorSeverity
    error_type: ErrorType
    source: ErrorSource
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None

@dataclass
class RootCauseAnalysis:
    """Five Whys analysis structure"""
    error_id: str
    why_1: str
    answer_1: str
    why_2: str
    answer_2: str
    why_3: str
    answer_3: str
    why_4: str
    answer_4: str
    why_5: str
    answer_5: str
    root_cause: str
    confidence: float

@dataclass
class FishboneDiagram:
    """Fishbone diagram structure"""
    error_id: str
    people_factors: List[str]
    process_factors: List[str]
    technology_factors: List[str]
    environment_factors: List[str]
    primary_cause_category: str

@dataclass
class ErrorSolution:
    """Error solution tracking"""
    error_id: str
    immediate_fix: str
    long_term_prevention: str
    testing_strategy: str
    implementation_notes: str
    effectiveness_score: Optional[float] = None

class ErrorAnalyzer:
    """Main error analysis engine"""
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.kb_path = Path(knowledge_base_path)
        self.errors_path = self.kb_path / "errors"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.errors_path / "patterns",
            self.errors_path / "solutions", 
            self.errors_path / "rootcauses",
            self.errors_path / "metrics",
            self.errors_path / "logs",
            self.errors_path / "analysis"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def capture_error(self, 
                     message: str,
                     context: Dict[str, Any],
                     command: Optional[str] = None,
                     exit_code: Optional[int] = None,
                     stack_trace: Optional[str] = None) -> ErrorEvent:
        """Capture and classify a new error"""
        
        error_id = self._generate_error_id(message, context)
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        # Classify error
        severity = self._classify_severity(message, exit_code, context)
        error_type = self._classify_type(message, stack_trace, context)
        source = self._classify_source(context, stack_trace)
        
        error_event = ErrorEvent(
            id=error_id,
            timestamp=timestamp,
            severity=severity,
            error_type=error_type,
            source=source,
            message=message,
            context=context,
            command=command,
            exit_code=exit_code,
            stack_trace=stack_trace
        )
        
        # Store error event
        self._store_error_event(error_event)
        
        # Trigger analysis for high severity errors
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.analyze_error(error_event)
        
        return error_event
    
    def analyze_error(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Perform comprehensive error analysis"""
        
        analysis = {
            "error_id": error_event.id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "five_whys": self._perform_five_whys(error_event),
            "fishbone": self._create_fishbone_diagram(error_event),
            "pattern_match": self._find_similar_patterns(error_event),
            "adversarial_analysis": self._adversarial_analysis(error_event),
            "recommendations": self._generate_recommendations(error_event)
        }
        
        # Store analysis
        analysis_file = self.errors_path / "analysis" / f"{error_event.id}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _generate_error_id(self, message: str, context: Dict[str, Any]) -> str:
        """Generate unique error ID"""
        error_signature = f"{message}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(error_signature.encode()).hexdigest()[:12]
    
    def _classify_severity(self, message: str, exit_code: Optional[int], context: Dict[str, Any]) -> ErrorSeverity:
        """Classify error severity based on patterns"""
        
        critical_patterns = [
            r"segmentation fault",
            r"out of memory",
            r"permission denied.*root",
            r"database.*corrupt",
            r"security.*breach"
        ]
        
        high_patterns = [
            r"connection.*refused",
            r"timeout.*exceeded",
            r"authentication.*failed",
            r"file not found.*critical"
        ]
        
        message_lower = message.lower()
        
        # Check for critical patterns
        for pattern in critical_patterns:
            if re.search(pattern, message_lower):
                return ErrorSeverity.CRITICAL
        
        # Check exit codes
        if exit_code is not None:
            if exit_code in [1, 2, 126, 127, 128]:  # Common critical exit codes
                return ErrorSeverity.HIGH
            elif exit_code != 0:
                return ErrorSeverity.MEDIUM
        
        # Check for high patterns  
        for pattern in high_patterns:
            if re.search(pattern, message_lower):
                return ErrorSeverity.HIGH
        
        return ErrorSeverity.MEDIUM
    
    def _classify_type(self, message: str, stack_trace: Optional[str], context: Dict[str, Any]) -> ErrorType:
        """Classify error type"""
        
        message_lower = message.lower()
        
        # Syntax errors
        if any(term in message_lower for term in ["syntax error", "parsing", "invalid syntax"]):
            return ErrorType.SYNTAX
        
        # Runtime errors
        if any(term in message_lower for term in ["null pointer", "index out", "division by zero"]):
            return ErrorType.RUNTIME
        
        # Integration errors
        if any(term in message_lower for term in ["connection", "timeout", "api", "service"]):
            return ErrorType.INTEGRATION
        
        # Performance errors
        if any(term in message_lower for term in ["timeout", "memory", "cpu", "performance"]):
            return ErrorType.PERFORMANCE
        
        # Security errors
        if any(term in message_lower for term in ["authentication", "authorization", "security", "permission"]):
            return ErrorType.SECURITY
        
        return ErrorType.UNKNOWN
    
    def _classify_source(self, context: Dict[str, Any], stack_trace: Optional[str]) -> ErrorSource:
        """Classify error source"""
        
        if stack_trace:
            if "/claude/" in stack_trace or "claude_code" in stack_trace:
                return ErrorSource.CLAUDE_CODE
            elif "/rif/" in stack_trace.lower() or "rif" in stack_trace.lower():
                return ErrorSource.RIF_SYSTEM
        
        # Check context for source hints
        if context.get("source") == "user_code":
            return ErrorSource.USER_CODE
        elif context.get("source") == "rif_system":
            return ErrorSource.RIF_SYSTEM
        elif context.get("source") == "claude_code":
            return ErrorSource.CLAUDE_CODE
        
        return ErrorSource.UNKNOWN
    
    def _perform_five_whys(self, error_event: ErrorEvent) -> RootCauseAnalysis:
        """Automated Five Whys analysis"""
        
        # This is a template - in practice, this would involve more sophisticated analysis
        # For now, we provide a framework that can be enhanced with AI/ML
        
        analysis = RootCauseAnalysis(
            error_id=error_event.id,
            why_1="Why did this error occur?",
            answer_1=f"The immediate cause was: {error_event.message}",
            why_2="Why did the immediate cause happen?",
            answer_2="System state or input caused the condition",
            why_3="Why was the system in that state?",
            answer_3="Configuration or process led to the state",
            why_4="Why was the configuration/process inadequate?",
            answer_4="Design or implementation gap",
            why_5="Why was there a design/implementation gap?",
            answer_5="Insufficient requirements or testing",
            root_cause="Inadequate error handling and validation in system design",
            confidence=0.7
        )
        
        return analysis
    
    def _create_fishbone_diagram(self, error_event: ErrorEvent) -> FishboneDiagram:
        """Create fishbone diagram analysis"""
        
        diagram = FishboneDiagram(
            error_id=error_event.id,
            people_factors=[
                "Insufficient error handling knowledge",
                "Inadequate testing practices",
                "Missing code review"
            ],
            process_factors=[
                "No error monitoring in place",
                "Insufficient validation procedures",
                "Lack of error handling standards"
            ],
            technology_factors=[
                "Inadequate error logging",
                "Missing error recovery mechanisms",
                "Insufficient monitoring tools"
            ],
            environment_factors=[
                "External dependency failures",
                "Resource constraints",
                "Network connectivity issues"
            ],
            primary_cause_category="process_factors"
        )
        
        return diagram
    
    def _find_similar_patterns(self, error_event: ErrorEvent) -> List[Dict[str, Any]]:
        """Find similar error patterns from history"""
        
        patterns = []
        patterns_dir = self.errors_path / "patterns"
        
        if patterns_dir.exists():
            for pattern_file in patterns_dir.glob("*.json"):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern = json.load(f)
                        
                    # Simple similarity check (can be enhanced with ML)
                    if self._calculate_similarity(error_event, pattern) > 0.7:
                        patterns.append(pattern)
                        
                except Exception as e:
                    continue
        
        return patterns
    
    def _calculate_similarity(self, error_event: ErrorEvent, pattern: Dict[str, Any]) -> float:
        """Calculate similarity between error and pattern"""
        
        similarity_score = 0.0
        
        # Message similarity (simple word overlap)
        error_words = set(error_event.message.lower().split())
        pattern_words = set(pattern.get("message", "").lower().split())
        
        if error_words and pattern_words:
            overlap = len(error_words.intersection(pattern_words))
            union = len(error_words.union(pattern_words))
            similarity_score += (overlap / union) * 0.4
        
        # Type similarity
        if error_event.error_type.value == pattern.get("error_type"):
            similarity_score += 0.3
        
        # Source similarity
        if error_event.source.value == pattern.get("source"):
            similarity_score += 0.3
        
        return similarity_score
    
    def _adversarial_analysis(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Perform adversarial analysis"""
        
        analysis = {
            "risk_assessment": self._assess_risk(error_event),
            "attack_vectors": self._identify_attack_vectors(error_event),
            "assumptions_tested": self._test_assumptions(error_event),
            "edge_cases": self._find_edge_cases(error_event)
        }
        
        return analysis
    
    def _assess_risk(self, error_event: ErrorEvent) -> Dict[str, Any]:
        """Assess security and stability risks"""
        
        risk_level = "medium"
        
        if error_event.error_type == ErrorType.SECURITY:
            risk_level = "high"
        elif error_event.severity == ErrorSeverity.CRITICAL:
            risk_level = "high"
        elif error_event.error_type == ErrorType.INTEGRATION:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "security_impact": error_event.error_type == ErrorType.SECURITY,
            "stability_impact": error_event.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH],
            "data_exposure_risk": "authentication" in error_event.message.lower() or "permission" in error_event.message.lower()
        }
    
    def _identify_attack_vectors(self, error_event: ErrorEvent) -> List[str]:
        """Identify potential attack vectors"""
        
        vectors = []
        
        message_lower = error_event.message.lower()
        
        if "sql" in message_lower:
            vectors.append("SQL injection potential")
        if "script" in message_lower:
            vectors.append("Cross-site scripting risk")
        if "permission" in message_lower:
            vectors.append("Privilege escalation risk")
        if "authentication" in message_lower:
            vectors.append("Authentication bypass risk")
        if "path" in message_lower:
            vectors.append("Path traversal risk")
        
        return vectors
    
    def _test_assumptions(self, error_event: ErrorEvent) -> List[str]:
        """Test system assumptions"""
        
        assumptions = [
            "Input validation is comprehensive",
            "Error messages don't expose sensitive information", 
            "System gracefully handles unexpected inputs",
            "Dependencies are always available",
            "Network connections are reliable"
        ]
        
        return assumptions
    
    def _find_edge_cases(self, error_event: ErrorEvent) -> List[str]:
        """Identify edge cases to test"""
        
        edge_cases = [
            "Empty input handling",
            "Maximum input size limits",
            "Special character processing",
            "Concurrent access scenarios",
            "Resource exhaustion conditions"
        ]
        
        return edge_cases
    
    def _generate_recommendations(self, error_event: ErrorEvent) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Type-specific recommendations
        if error_event.error_type == ErrorType.SYNTAX:
            recommendations.extend([
                "Implement syntax validation pre-checks",
                "Add linting to CI/CD pipeline",
                "Improve IDE/editor syntax highlighting"
            ])
        
        if error_event.error_type == ErrorType.RUNTIME:
            recommendations.extend([
                "Add null checks and validation",
                "Implement graceful error handling",
                "Add runtime monitoring and alerts"
            ])
        
        if error_event.error_type == ErrorType.INTEGRATION:
            recommendations.extend([
                "Implement circuit breaker patterns",
                "Add retry mechanisms with backoff",
                "Improve dependency health monitoring"
            ])
        
        # Severity-specific recommendations
        if error_event.severity == ErrorSeverity.CRITICAL:
            recommendations.extend([
                "Implement immediate alerting",
                "Add automated rollback mechanisms",
                "Create emergency response procedures"
            ])
        
        return recommendations
    
    def _store_error_event(self, error_event: ErrorEvent):
        """Store error event to knowledge base"""
        
        # Store in logs
        log_file = self.errors_path / "logs" / f"{datetime.date.today().isoformat()}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(error_event), default=str) + '\n')
        
        # Update metrics
        self._update_metrics(error_event)
    
    def _update_metrics(self, error_event: ErrorEvent):
        """Update error metrics"""
        
        metrics_file = self.errors_path / "metrics" / "current.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {
                "total_errors": 0,
                "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "by_type": {"syntax": 0, "runtime": 0, "logic": 0, "integration": 0, "performance": 0, "security": 0, "unknown": 0},
                "by_source": {"user_code": 0, "rif_system": 0, "claude_code": 0, "external": 0, "unknown": 0},
                "last_updated": None
            }
        
        metrics["total_errors"] += 1
        metrics["by_severity"][error_event.severity.value] += 1
        metrics["by_type"][error_event.error_type.value] += 1
        metrics["by_source"][error_event.source.value] += 1
        metrics["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    """CLI interface for error analysis"""
    
    if len(sys.argv) < 2:
        print("Usage: error-analysis.py <command> [options]")
        print("Commands:")
        print("  --capture <message> [--context <json>] [--command <cmd>] [--exit-code <code>]")
        print("  --analyze <error_id>")
        print("  --analyze-exit-code <exit_code> --command <command>")
        print("  --metrics")
        sys.exit(1)
    
    analyzer = ErrorAnalyzer()
    
    if sys.argv[1] == "--capture":
        message = sys.argv[2] if len(sys.argv) > 2 else "Unknown error"
        context = {}
        command = None
        exit_code = None
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--context" and i + 1 < len(sys.argv):
                try:
                    context = json.loads(sys.argv[i + 1])
                except:
                    context = {"raw": sys.argv[i + 1]}
                i += 2
            elif sys.argv[i] == "--command" and i + 1 < len(sys.argv):
                command = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--exit-code" and i + 1 < len(sys.argv):
                exit_code = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        error_event = analyzer.capture_error(message, context, command, exit_code)
        print(f"Error captured: {error_event.id}")
        print(f"Severity: {error_event.severity.value}")
        print(f"Type: {error_event.error_type.value}")
    
    elif sys.argv[1] == "--analyze-exit-code":
        if len(sys.argv) < 3:
            print("Error: Exit code required")
            sys.exit(1)
        
        exit_code = int(sys.argv[2])
        command = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--command" else "Unknown command"
        
        if exit_code != 0:
            message = f"Command failed with exit code {exit_code}: {command}"
            context = {"command": command, "exit_code": exit_code, "source": "command_execution"}
            
            error_event = analyzer.capture_error(message, context, command, exit_code)
            print(f"Command failure analyzed: {error_event.id}")
    
    elif sys.argv[1] == "--metrics":
        metrics_file = Path("/Users/cal/DEV/RIF/knowledge/errors/metrics/current.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(json.dumps(metrics, indent=2))
        else:
            print("No metrics available")
    
    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)

if __name__ == "__main__":
    main()