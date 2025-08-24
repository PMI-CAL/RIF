#!/usr/bin/env python3
"""
Error Capture Trigger System - Issue #47
Hooks into error events to trigger automatic Five Whys analysis and store error patterns.

This system implements:
1. Error event detection and capture
2. Automatic Five Whys analysis trigger
3. Error pattern extraction and storage
4. Resolution tracking and learning
5. Integration with RIF workflow automation
6. Error classification and severity assessment
7. Automated recovery suggestions
"""

import json
import subprocess
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import traceback
import re

# Add knowledge directory to path
knowledge_path = Path(__file__).parent.parent / "knowledge"
sys.path.insert(0, str(knowledge_path))

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    USER_ERROR = "user_error"
    UNKNOWN = "unknown"

@dataclass
class ErrorEvent:
    """Container for error event information."""
    error_id: str
    conversation_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    error_context: Dict[str, Any]
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: Optional[str] = None
    affected_components: List[str] = None
    recovery_suggestions: List[str] = None

@dataclass
class FiveWhysAnalysis:
    """Container for Five Whys root cause analysis."""
    error_id: str
    analysis_id: str
    questions_and_answers: List[Dict[str, str]]
    root_cause: str
    contributing_factors: List[str]
    prevention_measures: List[str]
    confidence_score: float

@dataclass
class ErrorPattern:
    """Container for identified error patterns."""
    pattern_id: str
    pattern_name: str
    error_signature: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    affected_systems: List[str]
    resolution_success_rate: float
    recommended_actions: List[str]

class ErrorCaptureSystem:
    """
    Comprehensive error capture and analysis system.
    """
    
    def __init__(self, storage_dir: str = "knowledge/errors"):
        """Initialize error capture system."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / "events").mkdir(exist_ok=True)
        (self.storage_dir / "patterns").mkdir(exist_ok=True)
        (self.storage_dir / "analysis").mkdir(exist_ok=True)
        (self.storage_dir / "logs").mkdir(exist_ok=True)
        
        self.setup_logging()
        self.error_patterns = self._load_error_patterns()
        
        self.logger.info(f"🚨 Error Capture System initialized with storage: {self.storage_dir}")
    
    def setup_logging(self):
        """Setup logging for error capture system."""
        log_file = self.storage_dir / "logs" / f"error_capture_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ErrorCaptureSystem - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def capture_error(self, error_data: Dict[str, Any], conversation_id: Optional[str] = None) -> str:
        """
        Capture an error event and trigger analysis.
        
        Args:
            error_data: Error information dictionary
            conversation_id: Optional conversation context
            
        Returns:
            Error ID for tracking
        """
        try:
            # Generate unique error ID
            error_id = self._generate_error_id(error_data)
            
            # Classify the error
            severity = self._classify_severity(error_data)
            category = self._classify_category(error_data)
            
            # Extract context and suggestions
            context = self._extract_error_context(error_data)
            recovery_suggestions = self._generate_recovery_suggestions(error_data, category)
            
            # Create error event
            error_event = ErrorEvent(
                error_id=error_id,
                conversation_id=conversation_id or "unknown",
                timestamp=datetime.now(timezone.utc),
                error_type=error_data.get('type', 'unknown'),
                error_message=error_data.get('message', 'No message provided'),
                error_context=context,
                severity=severity,
                category=category,
                stack_trace=error_data.get('stack_trace'),
                affected_components=self._identify_affected_components(error_data),
                recovery_suggestions=recovery_suggestions
            )
            
            # Store error event
            self._store_error_event(error_event)
            
            # Trigger analysis for critical/high severity errors
            if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                self._trigger_five_whys_analysis(error_event)
            
            # Update error patterns
            self._update_error_patterns(error_event)
            
            # Log the capture
            self.logger.info(f"🚨 Captured {severity.value} {category.value} error: {error_id}")
            
            return error_id
            
        except Exception as e:
            self.logger.error(f"Error capturing error event: {e}")
            return f"capture_failed_{datetime.now().timestamp()}"
    
    def _generate_error_id(self, error_data: Dict[str, Any]) -> str:
        """Generate unique error ID."""
        # Create hash from error signature
        signature_parts = [
            error_data.get('type', 'unknown'),
            error_data.get('message', '')[:100],  # First 100 chars
            str(datetime.now().date())  # Include date for uniqueness
        ]
        
        signature = '|'.join(signature_parts)
        hash_value = hashlib.md5(signature.encode()).hexdigest()[:8]
        
        return f"err_{datetime.now().strftime('%Y%m%d')}_{hash_value}"
    
    def _classify_severity(self, error_data: Dict[str, Any]) -> ErrorSeverity:
        """Classify error severity based on error characteristics."""
        error_type = error_data.get('type', '').lower()
        error_message = error_data.get('message', '').lower()
        
        # Critical severity indicators
        critical_keywords = [
            'system crash', 'data corruption', 'security breach',
            'authentication failed', 'database connection lost',
            'critical system failure', 'service unavailable'
        ]
        
        if any(keyword in error_message for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL
        
        # High severity indicators
        high_keywords = [
            'connection refused', 'permission denied', 'timeout',
            'validation failed', 'authentication error', 'access denied',
            'command failed', 'operation failed'
        ]
        
        if any(keyword in error_message for keyword in high_keywords):
            return ErrorSeverity.HIGH
        
        # Medium severity indicators
        medium_keywords = [
            'warning', 'deprecated', 'retry', 'fallback',
            'configuration error', 'missing file'
        ]
        
        if any(keyword in error_message for keyword in medium_keywords):
            return ErrorSeverity.MEDIUM
        
        # Check for specific error types
        if 'error' in error_type or 'exception' in error_type:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _classify_category(self, error_data: Dict[str, Any]) -> ErrorCategory:
        """Classify error category for better analysis."""
        error_message = error_data.get('message', '').lower()
        error_context = error_data.get('context', {})
        
        # Database errors
        if any(keyword in error_message for keyword in ['database', 'sql', 'connection', 'duckdb', 'sqlite']):
            return ErrorCategory.DATABASE
        
        # Network errors
        if any(keyword in error_message for keyword in ['network', 'connection refused', 'timeout', 'dns']):
            return ErrorCategory.NETWORK
        
        # Authentication errors
        if any(keyword in error_message for keyword in ['authentication', 'login', 'credentials', 'token', 'auth']):
            return ErrorCategory.AUTHENTICATION
        
        # Permission errors
        if any(keyword in error_message for keyword in ['permission', 'access denied', 'forbidden', 'unauthorized']):
            return ErrorCategory.PERMISSION
        
        # Validation errors
        if any(keyword in error_message for keyword in ['validation', 'invalid', 'format', 'parse', 'syntax']):
            return ErrorCategory.VALIDATION
        
        # Integration errors
        if any(keyword in error_message for keyword in ['github', 'api', 'webhook', 'integration', 'service']):
            return ErrorCategory.INTEGRATION
        
        # Performance errors
        if any(keyword in error_message for keyword in ['performance', 'slow', 'memory', 'cpu', 'resource']):
            return ErrorCategory.PERFORMANCE
        
        # System errors
        if any(keyword in error_message for keyword in ['system', 'os', 'file', 'directory', 'path']):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    def _extract_error_context(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context from error data."""
        context = {}
        
        # Copy provided context
        if 'context' in error_data:
            context.update(error_data['context'])
        
        # Add system context
        context['system'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'working_directory': os.getcwd(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # Add process context if available
        if 'pid' in error_data:
            context['process'] = {'pid': error_data['pid']}
        
        # Add environment context
        context['environment'] = {
            'user': os.environ.get('USER', 'unknown'),
            'home': os.environ.get('HOME', 'unknown'),
            'path': os.environ.get('PATH', '')[:200]  # First 200 chars of PATH
        }
        
        return context
    
    def _identify_affected_components(self, error_data: Dict[str, Any]) -> List[str]:
        """Identify system components affected by the error."""
        components = set()
        error_message = error_data.get('message', '').lower()
        stack_trace = error_data.get('stack_trace', '')
        
        # Check for specific component mentions
        component_keywords = {
            'github': ['gh', 'github', 'git'],
            'database': ['duckdb', 'sqlite', 'database', 'db'],
            'claude': ['claude', 'anthropic'],
            'rif': ['rif', 'orchestrator', 'workflow'],
            'quality': ['quality', 'gate', 'enforcement'],
            'specialist': ['specialist', 'assignment', 'sla'],
            'conversation': ['conversation', 'storage', 'backend']
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in error_message or keyword in stack_trace.lower() for keyword in keywords):
                components.add(component)
        
        return list(components)
    
    def _generate_recovery_suggestions(self, error_data: Dict[str, Any], category: ErrorCategory) -> List[str]:
        """Generate automated recovery suggestions based on error type."""
        suggestions = []
        error_message = error_data.get('message', '').lower()
        
        # Category-specific suggestions
        if category == ErrorCategory.DATABASE:
            suggestions.extend([
                "Check database connection configuration",
                "Verify database file permissions",
                "Consider database repair or recreation"
            ])
        
        elif category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check internet connectivity",
                "Verify API endpoints are accessible",
                "Consider retry with exponential backoff"
            ])
        
        elif category == ErrorCategory.AUTHENTICATION:
            suggestions.extend([
                "Verify authentication credentials",
                "Check token expiration",
                "Re-authenticate with service"
            ])
        
        elif category == ErrorCategory.PERMISSION:
            suggestions.extend([
                "Check file/directory permissions",
                "Verify user access rights",
                "Run with appropriate privileges"
            ])
        
        elif category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Check input format and syntax",
                "Validate data against schema",
                "Review validation rules"
            ])
        
        elif category == ErrorCategory.INTEGRATION:
            suggestions.extend([
                "Check API availability",
                "Verify integration configuration",
                "Review service status"
            ])
        
        # Message-specific suggestions
        if 'command not found' in error_message:
            suggestions.append("Install missing command or check PATH")
        
        if 'file not found' in error_message:
            suggestions.append("Verify file path and existence")
        
        if 'timeout' in error_message:
            suggestions.append("Increase timeout value or check service responsiveness")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _store_error_event(self, error_event: ErrorEvent):
        """Store error event to filesystem."""
        try:
            event_file = self.storage_dir / "events" / f"{error_event.error_id}.json"
            
            # Convert to serializable format
            event_data = asdict(error_event)
            event_data['timestamp'] = error_event.timestamp.isoformat()
            event_data['severity'] = error_event.severity.value
            event_data['category'] = error_event.category.value
            
            with open(event_file, 'w') as f:
                json.dump(event_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error storing error event: {e}")
    
    def _trigger_five_whys_analysis(self, error_event: ErrorEvent):
        """Trigger Five Whys root cause analysis for critical errors."""
        try:
            self.logger.info(f"🔍 Starting Five Whys analysis for {error_event.error_id}")
            
            # Generate analysis questions
            questions = self._generate_five_whys_questions(error_event)
            
            # Attempt automated analysis
            analysis = self._perform_automated_five_whys(error_event, questions)
            
            # Store analysis
            self._store_five_whys_analysis(analysis)
            
            # Create GitHub issue for manual review if needed
            if error_event.severity == ErrorSeverity.CRITICAL:
                self._create_error_investigation_issue(error_event, analysis)
                
        except Exception as e:
            self.logger.error(f"Error in Five Whys analysis: {e}")
    
    def _generate_five_whys_questions(self, error_event: ErrorEvent) -> List[str]:
        """Generate appropriate Five Whys questions for the error."""
        base_questions = [
            f"Why did the {error_event.category.value} error occur?",
            "Why was this condition not prevented?",
            "Why was this scenario not anticipated?",
            "Why are the current safeguards insufficient?",
            "Why is the system vulnerable to this type of failure?"
        ]
        
        return base_questions
    
    def _perform_automated_five_whys(self, error_event: ErrorEvent, questions: List[str]) -> FiveWhysAnalysis:
        """Perform automated Five Whys analysis based on error data."""
        analysis_id = f"analysis_{error_event.error_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate automated answers based on error context
        qa_pairs = []
        for i, question in enumerate(questions):
            answer = self._generate_automated_answer(error_event, question, i)
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'confidence': 0.6  # Moderate confidence for automated analysis
            })
        
        # Identify root cause
        root_cause = self._identify_root_cause(error_event, qa_pairs)
        
        # Generate contributing factors
        contributing_factors = self._identify_contributing_factors(error_event)
        
        # Generate prevention measures
        prevention_measures = self._generate_prevention_measures(error_event, root_cause)
        
        return FiveWhysAnalysis(
            error_id=error_event.error_id,
            analysis_id=analysis_id,
            questions_and_answers=qa_pairs,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            prevention_measures=prevention_measures,
            confidence_score=0.6  # Automated analysis has moderate confidence
        )
    
    def _generate_automated_answer(self, error_event: ErrorEvent, question: str, level: int) -> str:
        """Generate automated answer for Five Whys question."""
        answers = {
            0: f"The {error_event.category.value} error occurred due to: {error_event.error_message}",
            1: "The system lacked proper validation or error handling for this scenario",
            2: "The error condition was not identified during testing or deployment",
            3: "Current monitoring and safeguards did not detect this condition early enough",
            4: "The system architecture may lack sufficient resilience for this failure mode"
        }
        
        # Customize answer based on error category
        category_specific = {
            ErrorCategory.DATABASE: f"Database connection or query execution failed: {error_event.error_message}",
            ErrorCategory.NETWORK: f"Network communication failed: {error_event.error_message}",
            ErrorCategory.AUTHENTICATION: f"Authentication process failed: {error_event.error_message}",
            ErrorCategory.PERMISSION: f"Insufficient permissions to access resource: {error_event.error_message}",
            ErrorCategory.VALIDATION: f"Input validation failed: {error_event.error_message}"
        }
        
        if level == 0 and error_event.category in category_specific:
            return category_specific[error_event.category]
        
        return answers.get(level, "Further investigation needed to determine root cause")
    
    def _identify_root_cause(self, error_event: ErrorEvent, qa_pairs: List[Dict[str, str]]) -> str:
        """Identify the root cause from Five Whys analysis."""
        # For automated analysis, use the last answer as root cause
        if qa_pairs:
            return qa_pairs[-1]['answer']
        
        return f"Root cause analysis needed for {error_event.category.value} error"
    
    def _identify_contributing_factors(self, error_event: ErrorEvent) -> List[str]:
        """Identify contributing factors to the error."""
        factors = []
        
        # Add context-based factors
        if error_event.affected_components:
            factors.append(f"Affected components: {', '.join(error_event.affected_components)}")
        
        # Add timing factors
        factors.append(f"Error occurred at: {error_event.timestamp.isoformat()}")
        
        # Add severity factors
        if error_event.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            factors.append("High severity indicates significant system impact")
        
        return factors
    
    def _generate_prevention_measures(self, error_event: ErrorEvent, root_cause: str) -> List[str]:
        """Generate prevention measures based on root cause analysis."""
        measures = []
        
        # Category-specific prevention measures
        category_measures = {
            ErrorCategory.DATABASE: [
                "Implement database connection pooling and retry logic",
                "Add database health checks and monitoring",
                "Implement graceful degradation for database failures"
            ],
            ErrorCategory.NETWORK: [
                "Add network timeout and retry mechanisms",
                "Implement circuit breaker patterns",
                "Add network connectivity monitoring"
            ],
            ErrorCategory.AUTHENTICATION: [
                "Implement robust token refresh mechanisms",
                "Add authentication state monitoring",
                "Implement fallback authentication methods"
            ],
            ErrorCategory.PERMISSION: [
                "Implement permission validation before operations",
                "Add permission monitoring and alerting",
                "Create self-healing permission mechanisms"
            ],
            ErrorCategory.VALIDATION: [
                "Strengthen input validation and sanitization",
                "Add comprehensive schema validation",
                "Implement early validation in processing pipeline"
            ]
        }
        
        if error_event.category in category_measures:
            measures.extend(category_measures[error_event.category])
        
        # General prevention measures
        measures.extend([
            "Add comprehensive error handling and logging",
            "Implement monitoring and alerting for this error type",
            "Add automated testing for this scenario"
        ])
        
        return measures[:5]  # Limit to top 5 measures
    
    def _store_five_whys_analysis(self, analysis: FiveWhysAnalysis):
        """Store Five Whys analysis results."""
        try:
            analysis_file = self.storage_dir / "analysis" / f"{analysis.analysis_id}.json"
            
            # Convert to serializable format
            analysis_data = asdict(analysis)
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
                
            self.logger.info(f"📊 Stored Five Whys analysis: {analysis.analysis_id}")
                
        except Exception as e:
            self.logger.error(f"Error storing Five Whys analysis: {e}")
    
    def _create_error_investigation_issue(self, error_event: ErrorEvent, analysis: FiveWhysAnalysis):
        """Create GitHub issue for critical error investigation."""
        try:
            title = f"🚨 Critical Error Investigation: {error_event.error_id}"
            
            body = f"""# Critical Error Investigation Required
            
## Error Details
- **Error ID**: {error_event.error_id}
- **Severity**: {error_event.severity.value.upper()}
- **Category**: {error_event.category.value}
- **Timestamp**: {error_event.timestamp.isoformat()}

## Error Message
```
{error_event.error_message}
```

## Five Whys Analysis
"""
            
            for i, qa in enumerate(analysis.questions_and_answers, 1):
                body += f"\n**Why #{i}**: {qa['question']}\n**Answer**: {qa['answer']}\n"
            
            body += f"""
## Root Cause
{analysis.root_cause}

## Contributing Factors
{chr(10).join([f"- {factor}" for factor in analysis.contributing_factors])}

## Prevention Measures
{chr(10).join([f"- {measure}" for measure in analysis.prevention_measures])}

## Recovery Suggestions
{chr(10).join([f"- {suggestion}" for suggestion in error_event.recovery_suggestions])}

## Next Steps
1. Investigate the root cause identified above
2. Implement prevention measures
3. Test the fixes thoroughly
4. Update monitoring and alerting
5. Document lessons learned

---
*Automatically generated by RIF Error Capture System*
"""
            
            # Create GitHub issue
            result = subprocess.run([
                'gh', 'issue', 'create',
                '--title', title,
                '--body', body,
                '--label', 'error:critical,severity:high,error:auto-detected'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"🎟️ Created GitHub issue for critical error: {error_event.error_id}")
            else:
                self.logger.warning(f"Failed to create GitHub issue: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error creating GitHub issue: {e}")
    
    def _load_error_patterns(self) -> Dict[str, ErrorPattern]:
        """Load existing error patterns from storage."""
        patterns = {}
        patterns_dir = self.storage_dir / "patterns"
        
        if patterns_dir.exists():
            for pattern_file in patterns_dir.glob("*.json"):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern_data = json.load(f)
                        pattern = ErrorPattern(**pattern_data)
                        patterns[pattern.pattern_id] = pattern
                except Exception as e:
                    self.logger.debug(f"Error loading pattern {pattern_file}: {e}")
        
        return patterns
    
    def _update_error_patterns(self, error_event: ErrorEvent):
        """Update error patterns with new error event."""
        try:
            # Generate error signature
            signature = self._generate_error_signature(error_event)
            
            # Check if pattern exists
            pattern_id = hashlib.md5(signature.encode()).hexdigest()[:12]
            
            if pattern_id in self.error_patterns:
                # Update existing pattern
                pattern = self.error_patterns[pattern_id]
                pattern.frequency += 1
                pattern.last_seen = error_event.timestamp
                if error_event.affected_components:
                    pattern.affected_systems.extend(error_event.affected_components)
                    pattern.affected_systems = list(set(pattern.affected_systems))  # Remove duplicates
            else:
                # Create new pattern
                pattern = ErrorPattern(
                    pattern_id=pattern_id,
                    pattern_name=f"{error_event.category.value}_{error_event.error_type}",
                    error_signature=signature,
                    frequency=1,
                    first_seen=error_event.timestamp,
                    last_seen=error_event.timestamp,
                    affected_systems=error_event.affected_components or [],
                    resolution_success_rate=0.0,  # Will be updated as resolutions are tracked
                    recommended_actions=error_event.recovery_suggestions or []
                )
                self.error_patterns[pattern_id] = pattern
            
            # Store updated pattern
            self._store_error_pattern(pattern)
            
        except Exception as e:
            self.logger.error(f"Error updating error patterns: {e}")
    
    def _generate_error_signature(self, error_event: ErrorEvent) -> str:
        """Generate error signature for pattern matching."""
        # Normalize error message for pattern matching
        normalized_message = re.sub(r'\d+', 'N', error_event.error_message)  # Replace numbers
        normalized_message = re.sub(r'[\'"][^\'\"]*[\'"]', 'STR', normalized_message)  # Replace strings
        normalized_message = re.sub(r'[a-f0-9]{8,}', 'HASH', normalized_message)  # Replace hashes
        
        signature_parts = [
            error_event.category.value,
            error_event.error_type,
            normalized_message[:100]  # First 100 chars of normalized message
        ]
        
        return '|'.join(signature_parts)
    
    def _store_error_pattern(self, pattern: ErrorPattern):
        """Store error pattern to filesystem."""
        try:
            pattern_file = self.storage_dir / "patterns" / f"{pattern.pattern_id}.json"
            
            # Convert to serializable format
            pattern_data = asdict(pattern)
            pattern_data['first_seen'] = pattern.first_seen.isoformat()
            pattern_data['last_seen'] = pattern.last_seen.isoformat()
            
            with open(pattern_file, 'w') as f:
                json.dump(pattern_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error storing error pattern: {e}")
    
    def get_error_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            events_dir = self.storage_dir / "events"
            
            stats = {
                'period_days': days,
                'total_errors': 0,
                'by_severity': {s.value: 0 for s in ErrorSeverity},
                'by_category': {c.value: 0 for c in ErrorCategory},
                'top_patterns': [],
                'resolution_trends': []
            }
            
            # Process error events
            if events_dir.exists():
                for event_file in events_dir.glob("*.json"):
                    try:
                        with open(event_file, 'r') as f:
                            event_data = json.load(f)
                            
                        event_timestamp = datetime.fromisoformat(event_data['timestamp'])
                        if event_timestamp >= cutoff_date:
                            stats['total_errors'] += 1
                            stats['by_severity'][event_data['severity']] += 1
                            stats['by_category'][event_data['category']] += 1
                            
                    except Exception as e:
                        self.logger.debug(f"Error processing event file {event_file}: {e}")
            
            # Add pattern statistics
            pattern_stats = []
            for pattern in self.error_patterns.values():
                if pattern.last_seen >= cutoff_date:
                    pattern_stats.append({
                        'pattern_name': pattern.pattern_name,
                        'frequency': pattern.frequency,
                        'affected_systems': pattern.affected_systems,
                        'resolution_success_rate': pattern.resolution_success_rate
                    })
            
            # Sort by frequency and take top 10
            pattern_stats.sort(key=lambda x: x['frequency'], reverse=True)
            stats['top_patterns'] = pattern_stats[:10]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting error statistics: {e}")
            return {'error': str(e)}

def main():
    """Command line interface for error capture system."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python error_capture_trigger_system.py <command> [args]")
        print("Commands:")
        print("  capture <error_message> [severity] [category]    - Capture error event")
        print("  stats [days]                                     - Show error statistics")
        print("  test-capture                                     - Test error capture")
        print("  analyze <error_id>                              - Show error analysis")
        return
    
    command = sys.argv[1]
    system = ErrorCaptureSystem()
    
    if command == "capture":
        if len(sys.argv) < 3:
            print("Usage: capture <error_message> [severity] [category]")
            return 1
        
        error_message = sys.argv[2]
        severity = sys.argv[3] if len(sys.argv) > 3 else "medium"
        category = sys.argv[4] if len(sys.argv) > 4 else "unknown"
        
        error_data = {
            'type': 'manual_capture',
            'message': error_message,
            'context': {'manual_capture': True},
            'severity': severity,
            'category': category
        }
        
        error_id = system.capture_error(error_data, "manual_session")
        print(f"✅ Captured error: {error_id}")
        
    elif command == "stats":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        stats = system.get_error_statistics(days)
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return 1
        
        print(f"📊 Error Statistics ({days} days)")
        print(f"Total Errors: {stats['total_errors']}")
        
        print("\nBy Severity:")
        for severity, count in stats['by_severity'].items():
            if count > 0:
                print(f"  {severity}: {count}")
        
        print("\nBy Category:")
        for category, count in stats['by_category'].items():
            if count > 0:
                print(f"  {category}: {count}")
        
        if stats['top_patterns']:
            print("\nTop Error Patterns:")
            for i, pattern in enumerate(stats['top_patterns'][:5], 1):
                print(f"  {i}. {pattern['pattern_name']} (frequency: {pattern['frequency']})")
    
    elif command == "test-capture":
        print("🧪 Testing error capture system...")
        
        # Test different error types
        test_errors = [
            {
                'type': 'database_error',
                'message': 'Database connection failed: Connection refused',
                'context': {'database': 'conversations.duckdb'},
                'severity': 'high'
            },
            {
                'type': 'network_error', 
                'message': 'GitHub API timeout after 30 seconds',
                'context': {'api_endpoint': 'https://api.github.com'},
                'severity': 'medium'
            },
            {
                'type': 'validation_error',
                'message': 'Invalid JSON format in configuration file',
                'context': {'config_file': 'rif-workflow.yaml'},
                'severity': 'low'
            }
        ]
        
        for error_data in test_errors:
            error_id = system.capture_error(error_data, "test_session")
            print(f"  Captured {error_data['type']}: {error_id}")
        
        print("✅ Test capture complete")
    
    elif command == "analyze":
        if len(sys.argv) < 3:
            print("Usage: analyze <error_id>")
            return 1
        
        error_id = sys.argv[2]
        
        # Look for error event
        event_file = system.storage_dir / "events" / f"{error_id}.json"
        if not event_file.exists():
            print(f"Error event not found: {error_id}")
            return 1
        
        with open(event_file, 'r') as f:
            event_data = json.load(f)
        
        print(f"🔍 Error Analysis: {error_id}")
        print(f"Severity: {event_data['severity']}")
        print(f"Category: {event_data['category']}")
        print(f"Message: {event_data['error_message']}")
        print(f"Timestamp: {event_data['timestamp']}")
        
        if event_data.get('affected_components'):
            print(f"Affected Components: {', '.join(event_data['affected_components'])}")
        
        if event_data.get('recovery_suggestions'):
            print("Recovery Suggestions:")
            for suggestion in event_data['recovery_suggestions']:
                print(f"  - {suggestion}")
        
        # Look for analysis
        analysis_files = list((system.storage_dir / "analysis").glob(f"analysis_{error_id}_*.json"))
        if analysis_files:
            print(f"\n📊 Five Whys Analysis Available: {len(analysis_files)} analyses")
    
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())