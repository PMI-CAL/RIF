#!/usr/bin/env python3
"""
RIF Error Analysis Integration - Connects error handling with RIF agent system
Provides bridge between session error handler and RIF knowledge base/orchestration
"""

import sys
import json
import os
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

class ErrorRIFIntegrator:
    """Integrates error handling with RIF orchestration system"""
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.kb_path = Path(knowledge_base_path)
        self.errors_path = self.kb_path / "errors"
        self.patterns_path = self.kb_path / "patterns"
        self.issues_path = self.kb_path / "issues"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            self.errors_path / "rif_integration",
            self.patterns_path,
            self.issues_path
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_error_for_rif(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Process error for RIF agent system integration"""
        
        # Find the error in session logs
        error_data = self._find_error_in_logs(error_id)
        if not error_data:
            return None
        
        # Create RIF-compatible issue analysis
        rif_analysis = self._convert_error_to_rif_format(error_data)
        
        # Store for RIF agents
        self._store_rif_analysis(rif_analysis)
        
        # Check if RIF agent should be triggered
        if self._should_trigger_rif_agent(error_data):
            self._trigger_rif_error_analyst(rif_analysis)
        
        return rif_analysis
    
    def _find_error_in_logs(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Find error in session logs"""
        
        logs_dir = self.errors_path / "logs"
        if not logs_dir.exists():
            return None
        
        # Search in today's logs first
        today = datetime.date.today().isoformat()
        today_log = logs_dir / f"{today}.jsonl"
        
        if today_log.exists():
            error = self._search_log_file(today_log, error_id)
            if error:
                return error
        
        # Search in session logs
        for session_log in logs_dir.glob("session_*.jsonl"):
            error = self._search_log_file(session_log, error_id)
            if error:
                return error
        
        return None
    
    def _search_log_file(self, log_file: Path, error_id: str) -> Optional[Dict[str, Any]]:
        """Search for error in specific log file"""
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        error = json.loads(line.strip())
                        if error.get("id") == error_id:
                            return error
                    except:
                        continue
        except Exception:
            pass
        
        return None
    
    def _convert_error_to_rif_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert error data to RIF-compatible analysis format"""
        
        rif_analysis = {
            "id": f"error_analysis_{error_data['id']}",
            "type": "error_analysis",
            "source": "session_error_handler",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "error_details": error_data,
            "analysis_status": "pending",
            
            # RIF-specific fields
            "complexity": self._assess_error_complexity(error_data),
            "priority": self._map_severity_to_priority(error_data.get("severity", "medium")),
            "suggested_agent": self._suggest_rif_agent(error_data),
            "estimated_effort": self._estimate_effort(error_data),
            
            # Error-specific analysis
            "root_cause_hypothesis": self._generate_root_cause_hypothesis(error_data),
            "similar_patterns": self._find_similar_error_patterns(error_data),
            "prevention_opportunities": self._identify_prevention_opportunities(error_data),
            "knowledge_gaps": self._identify_knowledge_gaps(error_data),
            
            # Integration metadata
            "rif_state": "error:analyzing",
            "requires_human_attention": error_data.get("severity") in ["critical", "high"],
            "auto_resolution_possible": self._can_auto_resolve(error_data),
            "learning_value": self._assess_learning_value(error_data)
        }
        
        return rif_analysis
    
    def _assess_error_complexity(self, error_data: Dict[str, Any]) -> str:
        """Assess complexity for RIF planning"""
        
        error_type = error_data.get("error_type", "unknown")
        severity = error_data.get("severity", "medium")
        context = error_data.get("context", {})
        
        # High complexity indicators
        if severity == "critical":
            return "high"
        if error_type in ["integration", "security"]:
            return "high"
        if context.get("source") == "rif_system":
            return "high"
        
        # Medium complexity indicators
        if error_type in ["runtime", "performance"]:
            return "medium"
        if severity == "high":
            return "medium"
        
        return "low"
    
    def _map_severity_to_priority(self, severity: str) -> str:
        """Map error severity to RIF priority"""
        mapping = {
            "critical": "critical",
            "high": "high",
            "medium": "normal",
            "low": "low"
        }
        return mapping.get(severity, "normal")
    
    def _suggest_rif_agent(self, error_data: Dict[str, Any]) -> str:
        """Suggest which RIF agent should handle the error"""
        
        error_type = error_data.get("error_type", "unknown")
        severity = error_data.get("severity", "medium")
        source = error_data.get("source", "unknown")
        
        # Critical errors get error analyst
        if severity == "critical":
            return "rif-error-analyst"
        
        # Type-based agent selection
        if error_type == "integration":
            return "rif-architect"  # May need system design changes
        elif error_type == "security":
            return "rif-error-analyst"  # Security needs specialist analysis
        elif error_type in ["syntax", "runtime"]:
            return "rif-implementer"  # Code-level fixes
        elif error_type == "performance":
            return "rif-analyst"  # Performance analysis needed
        elif source == "rif_system":
            return "rif-error-analyst"  # RIF system errors need specialist
        
        # Default to error analyst for unknown patterns
        return "rif-error-analyst"
    
    def _estimate_effort(self, error_data: Dict[str, Any]) -> str:
        """Estimate effort required for error resolution"""
        
        complexity = self._assess_error_complexity(error_data)
        error_type = error_data.get("error_type", "unknown")
        
        if complexity == "high":
            return "large"
        elif error_type in ["syntax", "runtime"] and complexity == "low":
            return "small"
        else:
            return "medium"
    
    def _generate_root_cause_hypothesis(self, error_data: Dict[str, Any]) -> List[str]:
        """Generate initial root cause hypotheses"""
        
        hypotheses = []
        error_type = error_data.get("error_type", "unknown")
        message = error_data.get("message", "").lower()
        
        if error_type == "syntax":
            hypotheses.extend([
                "Invalid syntax in user code",
                "Missing dependencies or imports",
                "Configuration file syntax errors"
            ])
        elif error_type == "runtime":
            hypotheses.extend([
                "Null pointer or undefined variable access",
                "Type mismatch or conversion error",
                "Resource exhaustion or limits exceeded"
            ])
        elif error_type == "integration":
            hypotheses.extend([
                "External service unavailable or timeout",
                "Authentication or authorization failure",
                "Network connectivity or DNS issues"
            ])
        elif error_type == "security":
            hypotheses.extend([
                "Insufficient permissions or access control",
                "Security policy violation",
                "Credential or token expiration"
            ])
        elif error_type == "performance":
            hypotheses.extend([
                "Resource bottleneck or contention",
                "Inefficient algorithm or query",
                "Memory leak or resource exhaustion"
            ])
        
        # Add message-specific hypotheses
        if "not found" in message:
            hypotheses.append("Missing file, module, or resource")
        if "timeout" in message:
            hypotheses.append("Operation exceeded time limit")
        if "permission" in message:
            hypotheses.append("Insufficient access permissions")
        
        return hypotheses[:5]  # Limit to top 5 hypotheses
    
    def _find_similar_error_patterns(self, error_data: Dict[str, Any]) -> List[str]:
        """Find similar error patterns from knowledge base"""
        
        similar_patterns = []
        
        # Search in patterns directory
        if self.patterns_path.exists():
            for pattern_file in self.patterns_path.glob("*error*.json"):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern = json.load(f)
                    
                    if self._is_similar_pattern(error_data, pattern):
                        similar_patterns.append(pattern.get("name", pattern_file.stem))
                        
                except Exception:
                    continue
        
        # Search in error patterns
        patterns_dir = self.errors_path / "patterns"
        if patterns_dir.exists():
            for pattern_file in patterns_dir.glob("*.json"):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern = json.load(f)
                    
                    if self._is_similar_error_pattern(error_data, pattern):
                        similar_patterns.append(pattern.get("pattern_id", pattern_file.stem))
                        
                except Exception:
                    continue
        
        return similar_patterns[:3]  # Limit to top 3 similar patterns
    
    def _is_similar_pattern(self, error_data: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if error matches a general pattern"""
        
        # Simple keyword matching
        error_words = set(error_data.get("message", "").lower().split())
        pattern_words = set(pattern.get("description", "").lower().split())
        
        intersection = len(error_words.intersection(pattern_words))
        return intersection >= 2
    
    def _is_similar_error_pattern(self, error_data: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if error matches a specific error pattern"""
        
        # Type and severity matching
        if (error_data.get("error_type") == pattern.get("error_type") and
            error_data.get("severity") == pattern.get("severity")):
            return True
        
        # Message keyword matching
        error_words = set(error_data.get("message", "").lower().split())
        pattern_keywords = set(pattern.get("message_keywords", []))
        
        return len(error_words.intersection(pattern_keywords)) >= 2
    
    def _identify_prevention_opportunities(self, error_data: Dict[str, Any]) -> List[str]:
        """Identify opportunities to prevent similar errors"""
        
        opportunities = []
        error_type = error_data.get("error_type", "unknown")
        
        if error_type == "syntax":
            opportunities.extend([
                "Add pre-commit syntax validation",
                "Improve IDE configuration and linting",
                "Add automated code formatting"
            ])
        elif error_type == "runtime":
            opportunities.extend([
                "Add input validation and type checking",
                "Implement defensive programming practices",
                "Add comprehensive unit tests"
            ])
        elif error_type == "integration":
            opportunities.extend([
                "Add circuit breaker patterns",
                "Implement retry and backoff strategies", 
                "Add service health monitoring"
            ])
        elif error_type == "security":
            opportunities.extend([
                "Add security scanning to CI/CD",
                "Implement automated security testing",
                "Add access control validation"
            ])
        elif error_type == "performance":
            opportunities.extend([
                "Add performance monitoring and alerting",
                "Implement automated performance testing",
                "Add resource usage monitoring"
            ])
        
        return opportunities
    
    def _identify_knowledge_gaps(self, error_data: Dict[str, Any]) -> List[str]:
        """Identify knowledge gaps that contributed to the error"""
        
        gaps = []
        error_type = error_data.get("error_type", "unknown")
        context = error_data.get("context", {})
        
        # Common knowledge gaps by error type
        if error_type == "syntax":
            gaps.append("Language syntax and best practices")
        elif error_type == "runtime":
            gaps.append("Runtime behavior and error handling patterns")
        elif error_type == "integration":
            gaps.append("External service integration patterns")
        elif error_type == "security":
            gaps.append("Security best practices and threat models")
        elif error_type == "performance":
            gaps.append("Performance optimization techniques")
        
        # Context-specific gaps
        if context.get("source") == "claude_code":
            gaps.append("Claude Code tool usage and limitations")
        elif context.get("source") == "rif_system":
            gaps.append("RIF system architecture and operations")
        
        return gaps
    
    def _can_auto_resolve(self, error_data: Dict[str, Any]) -> bool:
        """Determine if error can be automatically resolved"""
        
        error_type = error_data.get("error_type", "unknown")
        severity = error_data.get("severity", "medium")
        
        # Cannot auto-resolve critical or security errors
        if severity == "critical" or error_type == "security":
            return False
        
        # Can potentially auto-resolve simple syntax and runtime errors
        if error_type in ["syntax", "runtime"] and severity in ["low", "medium"]:
            return True
        
        return False
    
    def _assess_learning_value(self, error_data: Dict[str, Any]) -> str:
        """Assess the learning value of this error for knowledge base"""
        
        error_type = error_data.get("error_type", "unknown")
        severity = error_data.get("severity", "medium")
        
        # High learning value
        if severity == "critical":
            return "high"
        if error_type in ["security", "integration"]:
            return "high"
        
        # Medium learning value
        if error_type in ["performance", "runtime"]:
            return "medium"
        
        return "low"
    
    def _should_trigger_rif_agent(self, error_data: Dict[str, Any]) -> bool:
        """Determine if a RIF agent should be triggered"""
        
        severity = error_data.get("severity", "medium")
        
        # Always trigger for critical and high severity
        if severity in ["critical", "high"]:
            return True
        
        # Trigger for medium severity if it's a recurring pattern
        if severity == "medium":
            return self._is_recurring_error(error_data)
        
        return False
    
    def _is_recurring_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a recurring error pattern"""
        
        # Simple check for similar errors in recent logs
        # This could be enhanced with more sophisticated pattern matching
        return False  # For now, conservative approach
    
    def _store_rif_analysis(self, analysis: Dict[str, Any]):
        """Store analysis for RIF agents"""
        
        analysis_file = self.errors_path / "rif_integration" / f"{analysis['id']}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _trigger_rif_error_analyst(self, analysis: Dict[str, Any]):
        """Trigger RIF Error Analyst agent"""
        
        print(f"RIF: Triggering error analyst for {analysis['error_details']['id']}")
        
        # Create a GitHub issue if one doesn't exist yet
        error_data = analysis["error_details"]
        if not error_data.get("auto_handled"):
            self._ensure_github_issue_exists(analysis)
    
    def _ensure_github_issue_exists(self, analysis: Dict[str, Any]):
        """Ensure GitHub issue exists for error analysis"""
        
        error_data = analysis["error_details"]
        
        # Check if issue already exists
        links_file = self.errors_path / "github_issues" / "error_issue_links.jsonl"
        if links_file.exists():
            with open(links_file, 'r') as f:
                for line in f:
                    try:
                        link = json.loads(line.strip())
                        if link.get("error_id") == error_data["id"]:
                            return  # Issue already exists
                    except:
                        continue
        
        # Create issue using session error handler
        try:
            subprocess.run([
                "python3", 
                str(Path(__file__).parent / "session_error_handler.py"),
                "--capture-generic",
                error_data["message"],
                "--context", json.dumps(error_data.get("context", {}))
            ], check=True, capture_output=True)
            
        except Exception as e:
            print(f"RIF: Failed to create GitHub issue: {e}")

def main():
    """CLI interface for RIF integration"""
    
    if len(sys.argv) < 2:
        print("Usage: error_rif_integration.py <command> [options]")
        print("Commands:")
        print("  --process-error <error_id>")
        print("  --analyze-session <session_id>")
        sys.exit(1)
    
    integrator = ErrorRIFIntegrator()
    
    command = sys.argv[1]
    
    if command == "--process-error":
        if len(sys.argv) < 3:
            print("Error: Error ID required")
            sys.exit(1)
        
        error_id = sys.argv[2]
        analysis = integrator.process_error_for_rif(error_id)
        
        if analysis:
            print(f"RIF analysis created: {analysis['id']}")
            print(json.dumps(analysis, indent=2))
        else:
            print(f"Error not found: {error_id}")
    
    elif command == "--analyze-session":
        if len(sys.argv) < 3:
            print("Error: Session ID required")
            sys.exit(1)
        
        session_id = sys.argv[2]
        print(f"Session analysis for {session_id} - functionality to be implemented")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()