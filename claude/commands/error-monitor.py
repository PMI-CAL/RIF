#!/usr/bin/env python3
"""
RIF Error Monitoring and Continuous Improvement System
Monitors error patterns, triggers analyses, and implements improvements
"""

import sys
import json
import time
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ErrorPattern:
    """Error pattern definition"""
    pattern_id: str
    frequency: int
    severity: str
    root_cause: str
    solution: str
    prevention_measures: List[str]
    last_occurrence: str
    effectiveness_score: float

class ErrorMonitor:
    """Continuous error monitoring and improvement"""
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.kb_path = Path(knowledge_base_path)
        self.errors_path = self.kb_path / "errors"
        self.patterns_path = self.errors_path / "patterns"
        self.solutions_path = self.errors_path / "solutions"
        self.metrics_path = self.errors_path / "metrics"
        
        self.monitoring_config = {
            "check_interval": 300,  # 5 minutes
            "pattern_threshold": 3,  # Minimum occurrences to trigger analysis
            "critical_threshold": 1,  # Immediate analysis for critical errors
            "improvement_threshold": 0.7  # Effectiveness score for improvement recommendations
        }
    
    def start_monitoring(self):
        """Start continuous error monitoring"""
        print("RIF Error Monitor starting...")
        
        while True:
            try:
                self.check_for_new_errors()
                self.analyze_error_patterns()
                self.check_improvement_opportunities()
                self.update_monitoring_metrics()
                
                time.sleep(self.monitoring_config["check_interval"])
                
            except KeyboardInterrupt:
                print("\\nRIF Error Monitor stopping...")
                break
            except Exception as e:
                self.log_monitor_error(str(e))
                time.sleep(60)  # Wait 1 minute before retrying
    
    def check_for_new_errors(self):
        """Check for new errors that need analysis"""
        today = datetime.date.today().isoformat()
        log_file = self.errors_path / "logs" / f"{today}.jsonl"
        
        if not log_file.exists():
            return
        
        # Read recent errors (last 5 minutes)
        cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=self.monitoring_config["check_interval"]
        )
        
        recent_errors = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    error = json.loads(line.strip())
                    error_time = datetime.datetime.fromisoformat(
                        error["timestamp"].replace("Z", "+00:00")
                    )
                    
                    if error_time > cutoff_time:
                        recent_errors.append(error)
                        
                except Exception as e:
                    continue
        
        # Process recent errors
        for error in recent_errors:
            self.process_new_error(error)
    
    def process_new_error(self, error: Dict[str, Any]):
        """Process a newly detected error"""
        
        # Check if critical error needs immediate analysis
        if error.get("severity") == "critical":
            self.trigger_immediate_analysis(error)
        
        # Check for error patterns
        pattern_id = self.identify_error_pattern(error)
        if pattern_id:
            self.update_error_pattern(pattern_id, error)
        else:
            self.create_new_pattern(error)
    
    def trigger_immediate_analysis(self, error: Dict[str, Any]):
        """Trigger immediate analysis for critical errors"""
        
        print(f"Critical error detected: {error['id']}")
        
        # Run error analysis
        try:
            result = subprocess.run([
                "python3", 
                str(self.kb_path.parent / "claude" / "commands" / "error-analysis.py"),
                "--analyze", error["id"]
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Analysis completed for error {error['id']}")
                
                # Create GitHub issue for critical errors
                self.create_github_issue(error)
            else:
                print(f"Analysis failed for error {error['id']}: {result.stderr}")
                
        except Exception as e:
            print(f"Failed to analyze critical error {error['id']}: {e}")
    
    def identify_error_pattern(self, error: Dict[str, Any]) -> Optional[str]:
        """Identify if error matches existing pattern"""
        
        if not self.patterns_path.exists():
            return None
        
        for pattern_file in self.patterns_path.glob("*.json"):
            try:
                with open(pattern_file, 'r') as f:
                    pattern = json.load(f)
                
                # Simple pattern matching (can be enhanced)
                if self.matches_pattern(error, pattern):
                    return pattern["pattern_id"]
                    
            except Exception as e:
                continue
        
        return None
    
    def matches_pattern(self, error: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if error matches a pattern"""
        
        # Message similarity
        error_words = set(error["message"].lower().split())
        pattern_words = set(pattern.get("message_keywords", []))
        
        word_overlap = len(error_words.intersection(pattern_words)) / max(len(error_words), 1)
        
        # Type and severity match
        type_match = error.get("error_type") == pattern.get("error_type")
        severity_match = error.get("severity") == pattern.get("severity")
        
        # Combined scoring
        score = (word_overlap * 0.5) + (type_match * 0.3) + (severity_match * 0.2)
        
        return score > 0.6
    
    def update_error_pattern(self, pattern_id: str, error: Dict[str, Any]):
        """Update existing error pattern with new occurrence"""
        
        pattern_file = self.patterns_path / f"{pattern_id}.json"
        
        try:
            with open(pattern_file, 'r') as f:
                pattern = json.load(f)
            
            pattern["frequency"] += 1
            pattern["last_occurrence"] = error["timestamp"]
            pattern["recent_errors"].append(error["id"])
            
            # Keep only last 10 error IDs
            pattern["recent_errors"] = pattern["recent_errors"][-10:]
            
            with open(pattern_file, 'w') as f:
                json.dump(pattern, f, indent=2)
            
            # Check if pattern needs analysis
            if pattern["frequency"] >= self.monitoring_config["pattern_threshold"]:
                self.trigger_pattern_analysis(pattern)
                
        except Exception as e:
            print(f"Failed to update pattern {pattern_id}: {e}")
    
    def create_new_pattern(self, error: Dict[str, Any]):
        """Create new error pattern"""
        
        pattern_id = f"pattern_{error['id'][:8]}"
        
        pattern = {
            "pattern_id": pattern_id,
            "error_type": error.get("error_type"),
            "severity": error.get("severity"),
            "message_keywords": error["message"].lower().split()[:5],
            "frequency": 1,
            "first_occurrence": error["timestamp"],
            "last_occurrence": error["timestamp"],
            "recent_errors": [error["id"]],
            "analysis_status": "pending",
            "root_cause": None,
            "solution": None,
            "prevention_measures": []
        }
        
        pattern_file = self.patterns_path / f"{pattern_id}.json"
        with open(pattern_file, 'w') as f:
            json.dump(pattern, f, indent=2)
    
    def trigger_pattern_analysis(self, pattern: Dict[str, Any]):
        """Trigger analysis for error pattern"""
        
        if pattern.get("analysis_status") == "completed":
            return
        
        print(f"Analyzing error pattern: {pattern['pattern_id']}")
        
        # Perform root cause analysis for the pattern
        analysis = self.analyze_error_pattern(pattern)
        
        # Update pattern with analysis results
        pattern["analysis_status"] = "completed"
        pattern["root_cause"] = analysis.get("root_cause")
        pattern["solution"] = analysis.get("solution")
        pattern["prevention_measures"] = analysis.get("prevention_measures", [])
        
        pattern_file = self.patterns_path / f"{pattern['pattern_id']}.json"
        with open(pattern_file, 'w') as f:
            json.dump(pattern, f, indent=2)
        
        # Implement prevention measures
        self.implement_prevention_measures(pattern)
    
    def analyze_error_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error pattern to identify root cause and solutions"""
        
        analysis = {
            "root_cause": "Pattern-based analysis needed",
            "solution": "Implement pattern-specific fix",
            "prevention_measures": []
        }
        
        # Type-specific analysis
        error_type = pattern.get("error_type")
        
        if error_type == "integration":
            analysis["root_cause"] = "Service dependency or communication failure"
            analysis["solution"] = "Implement circuit breaker and retry logic"
            analysis["prevention_measures"] = [
                "Add service health monitoring",
                "Implement graceful degradation",
                "Add timeout configurations"
            ]
        
        elif error_type == "runtime":
            analysis["root_cause"] = "Runtime state or validation issue"
            analysis["solution"] = "Add comprehensive input validation"
            analysis["prevention_measures"] = [
                "Implement null checks",
                "Add boundary validation",
                "Improve error handling"
            ]
        
        elif error_type == "performance":
            analysis["root_cause"] = "Resource exhaustion or inefficient algorithms"
            analysis["solution"] = "Optimize performance bottlenecks"
            analysis["prevention_measures"] = [
                "Add performance monitoring",
                "Implement resource limits",
                "Add caching strategies"
            ]
        
        elif error_type == "security":
            analysis["root_cause"] = "Security control bypass or vulnerability"
            analysis["solution"] = "Strengthen security controls"
            analysis["prevention_measures"] = [
                "Add security auditing",
                "Implement additional validation",
                "Add access control monitoring"
            ]
        
        return analysis
    
    def implement_prevention_measures(self, pattern: Dict[str, Any]):
        """Implement prevention measures for error pattern"""
        
        prevention_measures = pattern.get("prevention_measures", [])
        
        for measure in prevention_measures:
            print(f"Implementing prevention measure: {measure}")
            
            # Create implementation task
            task = {
                "id": f"prevention_{pattern['pattern_id']}_{len(prevention_measures)}",
                "pattern_id": pattern["pattern_id"],
                "measure": measure,
                "status": "pending",
                "created": datetime.datetime.utcnow().isoformat() + "Z"
            }
            
            # Store prevention task
            prevention_file = self.errors_path / "prevention_tasks.jsonl"
            with open(prevention_file, 'a') as f:
                f.write(json.dumps(task) + '\\n')
    
    def check_improvement_opportunities(self):
        """Check for continuous improvement opportunities"""
        
        # Look for patterns with low effectiveness scores
        improvement_opportunities = []
        
        if self.patterns_path.exists():
            for pattern_file in self.patterns_path.glob("*.json"):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern = json.load(f)
                    
                    effectiveness = pattern.get("effectiveness_score", 1.0)
                    if effectiveness < self.monitoring_config["improvement_threshold"]:
                        improvement_opportunities.append(pattern)
                        
                except Exception as e:
                    continue
        
        # Process improvement opportunities
        for pattern in improvement_opportunities:
            self.create_improvement_recommendation(pattern)
    
    def create_improvement_recommendation(self, pattern: Dict[str, Any]):
        """Create improvement recommendation"""
        
        recommendation = {
            "id": f"improvement_{pattern['pattern_id']}",
            "pattern_id": pattern["pattern_id"],
            "current_effectiveness": pattern.get("effectiveness_score", 0.0),
            "target_effectiveness": 0.9,
            "recommendations": self.generate_improvement_recommendations(pattern),
            "created": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "pending"
        }
        
        # Store recommendation
        improvement_file = self.errors_path / "improvements.jsonl"
        with open(improvement_file, 'a') as f:
            f.write(json.dumps(recommendation) + '\\n')
    
    def generate_improvement_recommendations(self, pattern: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations"""
        
        recommendations = []
        
        frequency = pattern.get("frequency", 0)
        error_type = pattern.get("error_type")
        
        if frequency > 10:
            recommendations.append("High frequency pattern - implement proactive monitoring")
            recommendations.append("Add automated detection and prevention")
        
        if error_type == "integration":
            recommendations.append("Implement service mesh for better resilience")
            recommendations.append("Add distributed tracing for better debugging")
        
        elif error_type == "runtime":
            recommendations.append("Add comprehensive unit testing")
            recommendations.append("Implement runtime validation framework")
        
        elif error_type == "performance":
            recommendations.append("Add performance regression testing")
            recommendations.append("Implement real-time performance monitoring")
        
        return recommendations
    
    def create_github_issue(self, error: Dict[str, Any]):
        """Create GitHub issue for critical errors"""
        
        title = f"Critical Error: {error['message'][:50]}..."
        
        body = f"""## Critical Error Detected
        
**Error ID**: {error['id']}
**Timestamp**: {error['timestamp']}
**Severity**: {error['severity']}
**Type**: {error.get('error_type', 'unknown')}
**Source**: {error.get('source', 'unknown')}

### Error Details
```
{error['message']}
```

### Context
```json
{json.dumps(error.get('context', {}), indent=2)}
```

### Automated Analysis
- Root cause analysis: Pending
- Impact assessment: High
- Priority: Immediate

### Action Required
1. Investigate root cause
2. Implement immediate fix
3. Add prevention measures
4. Update monitoring

**Auto-generated by RIF Error Analysis System**
"""
        
        try:
            result = subprocess.run([
                "gh", "issue", "create",
                "--title", title,
                "--body", body,
                "--label", "error:critical,priority:high,state:new"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"GitHub issue created for critical error {error['id']}")
            else:
                print(f"Failed to create GitHub issue: {result.stderr}")
                
        except Exception as e:
            print(f"Failed to create GitHub issue: {e}")
    
    def update_monitoring_metrics(self):
        """Update monitoring system metrics"""
        
        metrics = {
            "last_check": datetime.datetime.utcnow().isoformat() + "Z",
            "patterns_detected": len(list(self.patterns_path.glob("*.json"))) if self.patterns_path.exists() else 0,
            "critical_errors_today": self.count_todays_critical_errors(),
            "monitoring_status": "active"
        }
        
        metrics_file = self.metrics_path / "monitoring.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def count_todays_critical_errors(self) -> int:
        """Count critical errors from today"""
        
        today = datetime.date.today().isoformat()
        log_file = self.errors_path / "logs" / f"{today}.jsonl"
        
        if not log_file.exists():
            return 0
        
        count = 0
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    error = json.loads(line.strip())
                    if error.get("severity") == "critical":
                        count += 1
                except:
                    continue
        
        return count
    
    def log_monitor_error(self, error_message: str):
        """Log errors in the monitoring system itself"""
        
        error_log = self.errors_path / "logs" / "monitor-errors.log"
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        with open(error_log, 'a') as f:
            f.write(f"{timestamp}: {error_message}\\n")

def main():
    """CLI interface for error monitoring"""
    
    if len(sys.argv) < 2:
        print("Usage: error-monitor.py <command>")
        print("Commands:")
        print("  --start-monitoring    Start continuous monitoring")
        print("  --check-patterns      Check error patterns once")
        print("  --status              Show monitoring status")
        sys.exit(1)
    
    monitor = ErrorMonitor()
    
    if sys.argv[1] == "--start-monitoring":
        monitor.start_monitoring()
    
    elif sys.argv[1] == "--check-patterns":
        monitor.check_for_new_errors()
        monitor.analyze_error_patterns()
        monitor.check_improvement_opportunities()
        print("Pattern check completed")
    
    elif sys.argv[1] == "--status":
        metrics_file = Path("/Users/cal/DEV/RIF/knowledge/errors/metrics/monitoring.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print(json.dumps(metrics, indent=2))
        else:
            print("No monitoring status available")
    
    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)

if __name__ == "__main__":
    main()