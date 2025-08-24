#!/usr/bin/env python3
"""
Real-time Decision Tracker - Issue #94 Phase 1
Tracks quality gate decisions in real-time as they occur during development sessions.

This component:
- Integrates with Claude Code hooks for automatic data capture
- Tracks all quality gate evaluations during active sessions
- Provides real-time visibility into quality gate decisions
- Maintains session-based tracking compatible with Claude Code workflow
- Supports both manual and automated decision tracking
"""

import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

# Import the collector for integration
try:
    from quality_metrics_collector import QualityMetricsCollector, QualityGateType, QualityDecisionType
except ImportError:
    # Fallback for standalone execution
    sys.path.append(str(Path(__file__).parent))
    from quality_metrics_collector import QualityMetricsCollector, QualityGateType, QualityDecisionType


@dataclass
class RealTimeDecision:
    """Real-time quality gate decision with session context."""
    session_id: str
    timestamp: str
    issue_number: int
    gate_type: str
    decision: str
    context: Dict[str, Any]
    agent_name: Optional[str]
    processing_time_ms: float
    tool_context: Optional[str]
    file_changes: List[str]


class RealTimeDecisionTracker:
    """
    Tracks quality gate decisions in real-time during Claude Code sessions.
    Integrates with Claude Code hooks for automatic capture.
    """
    
    def __init__(self, storage_path: str = "knowledge/quality_metrics"):
        """Initialize the real-time decision tracker."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.session_id = self._get_session_id()
        
        # Initialize real-time tracking storage
        self.realtime_path = self.storage_path / "realtime"
        self.realtime_path.mkdir(exist_ok=True)
        
        # Initialize metrics collector for integration
        self.metrics_collector = QualityMetricsCollector(str(storage_path))
        
        self.setup_logging()
        self._initialize_session()
    
    def setup_logging(self):
        """Setup logging for real-time tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RealTimeDecisionTracker - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_session_id(self) -> str:
        """Get or create session ID compatible with Claude Code."""
        # Try to get session ID from environment (Claude Code sets this)
        session_id = os.environ.get('CLAUDE_SESSION_ID')
        if session_id:
            return session_id
        
        # Fallback to timestamp-based ID
        return f"session_{int(time.time())}"
    
    def _initialize_session(self) -> None:
        """Initialize real-time tracking for the current session."""
        session_file = self.realtime_path / f"{self.session_id}.json"
        
        if not session_file.exists():
            session_data = {
                'session_id': self.session_id,
                'start_time': datetime.now().isoformat(),
                'decisions': [],
                'summary': {
                    'total_decisions': 0,
                    'decisions_by_gate': {},
                    'decisions_by_result': {},
                    'average_processing_time': 0.0
                }
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Initialized real-time tracking session: {self.session_id}")
    
    def track_quality_decision_realtime(
        self,
        issue_number: int,
        gate_type: Union[str, QualityGateType],
        decision: Union[str, QualityDecisionType],
        context: Dict[str, Any] = None,
        agent_name: Optional[str] = None,
        tool_context: Optional[str] = None,
        file_changes: List[str] = None,
        threshold_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        confidence_score: Optional[float] = None,
        evidence: List[str] = None
    ) -> str:
        """
        Track a quality gate decision in real-time.
        
        Args:
            issue_number: GitHub issue number
            gate_type: Type of quality gate
            decision: Gate decision
            context: Additional context
            agent_name: Name of the agent making the decision
            tool_context: Context about which tool triggered the decision
            file_changes: List of files that changed
            threshold_value: Gate threshold value
            actual_value: Actual measured value
            confidence_score: Confidence in the decision
            evidence: Supporting evidence
            
        Returns:
            Decision ID for tracking
        """
        start_time = time.time()
        
        try:
            # Convert enums to strings if necessary
            gate_type_str = gate_type.value if isinstance(gate_type, QualityGateType) else str(gate_type)
            decision_str = decision.value if isinstance(decision, QualityDecisionType) else str(decision)
            
            # Create real-time decision record
            rt_decision = RealTimeDecision(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                issue_number=issue_number,
                gate_type=gate_type_str,
                decision=decision_str,
                context=context or {},
                agent_name=agent_name,
                processing_time_ms=0,  # Will be updated
                tool_context=tool_context,
                file_changes=file_changes or []
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            rt_decision.processing_time_ms = processing_time
            
            # Store in real-time session
            self._update_session_with_decision(rt_decision)
            
            # Also store in persistent metrics collector
            try:
                gate_enum = QualityGateType(gate_type_str) if hasattr(QualityGateType, gate_type_str.upper()) else None
                decision_enum = QualityDecisionType(decision_str) if hasattr(QualityDecisionType, decision_str.upper()) else None
                
                if gate_enum and decision_enum:
                    decision_id = self.metrics_collector.record_quality_decision(
                        issue_number=issue_number,
                        gate_type=gate_enum,
                        decision=decision_enum,
                        threshold_value=threshold_value,
                        actual_value=actual_value,
                        context=context,
                        agent_type=agent_name,
                        confidence_score=confidence_score,
                        evidence=evidence
                    )
                else:
                    decision_id = f"rt_{int(time.time())}"
            except Exception as e:
                self.logger.warning(f"Error storing to metrics collector: {e}")
                decision_id = f"rt_{int(time.time())}"
            
            self.logger.info(
                f"Real-time decision tracked: {decision_str} for {gate_type_str} "
                f"on issue #{issue_number} (session: {self.session_id})"
            )
            
            return decision_id
            
        except Exception as e:
            self.logger.error(f"Error tracking real-time decision: {e}")
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get real-time summary of current session's quality gate activity."""
        try:
            session_file = self.realtime_path / f"{self.session_id}.json"
            
            if not session_file.exists():
                return {
                    'session_id': self.session_id,
                    'status': 'no_data',
                    'message': 'No quality gate decisions recorded in this session'
                }
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Calculate real-time metrics
            decisions = session_data.get('decisions', [])
            if not decisions:
                return {
                    'session_id': self.session_id,
                    'status': 'empty',
                    'start_time': session_data.get('start_time'),
                    'current_time': datetime.now().isoformat()
                }
            
            # Calculate summary statistics
            total_decisions = len(decisions)
            decisions_by_gate = {}
            decisions_by_result = {}
            total_processing_time = 0
            
            for decision in decisions:
                gate_type = decision['gate_type']
                result = decision['decision']
                processing_time = decision.get('processing_time_ms', 0)
                
                decisions_by_gate[gate_type] = decisions_by_gate.get(gate_type, 0) + 1
                decisions_by_result[result] = decisions_by_result.get(result, 0) + 1
                total_processing_time += processing_time
            
            avg_processing_time = total_processing_time / total_decisions if total_decisions > 0 else 0
            
            # Get recent activity (last 10 decisions)
            recent_decisions = sorted(decisions, key=lambda x: x['timestamp'])[-10:]
            
            return {
                'session_id': self.session_id,
                'status': 'active',
                'start_time': session_data.get('start_time'),
                'current_time': datetime.now().isoformat(),
                'summary': {
                    'total_decisions': total_decisions,
                    'decisions_by_gate': decisions_by_gate,
                    'decisions_by_result': decisions_by_result,
                    'average_processing_time_ms': avg_processing_time,
                    'performance_warning': avg_processing_time > 50
                },
                'recent_activity': recent_decisions,
                'active_issues': list(set(d['issue_number'] for d in decisions))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session summary: {e}")
            return {
                'session_id': self.session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get live dashboard data for real-time monitoring."""
        try:
            # Get current session summary
            session_summary = self.get_session_summary()
            
            # Get recent decisions across all sessions
            recent_decisions = self._get_recent_decisions_all_sessions(minutes_back=60)
            
            # Calculate live metrics
            live_metrics = self._calculate_live_metrics(recent_decisions)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_session': session_summary,
                'live_metrics': live_metrics,
                'recent_activity': recent_decisions[-20:],  # Last 20 decisions
                'status': 'active' if recent_decisions else 'idle'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def hook_integration_track_decision(
        self,
        hook_data: Dict[str, Any]
    ) -> bool:
        """
        Integration point for Claude Code hooks to automatically track decisions.
        
        Args:
            hook_data: Data from Claude Code hook
            
        Returns:
            True if successfully tracked, False otherwise
        """
        try:
            # Extract information from hook data
            issue_number = self._extract_issue_number(hook_data)
            if not issue_number:
                return False
            
            # Determine gate type from hook context
            gate_type = self._determine_gate_type_from_hook(hook_data)
            if not gate_type:
                return False
            
            # Determine decision from hook context
            decision = self._determine_decision_from_hook(hook_data)
            if not decision:
                return False
            
            # Extract additional context
            context = self._extract_hook_context(hook_data)
            agent_name = hook_data.get('agent_name')
            tool_context = hook_data.get('tool_name')
            file_changes = hook_data.get('file_changes', [])
            
            # Track the decision
            decision_id = self.track_quality_decision_realtime(
                issue_number=issue_number,
                gate_type=gate_type,
                decision=decision,
                context=context,
                agent_name=agent_name,
                tool_context=tool_context,
                file_changes=file_changes
            )
            
            return bool(decision_id)
            
        except Exception as e:
            self.logger.error(f"Error in hook integration: {e}")
            return False
    
    def _update_session_with_decision(self, decision: RealTimeDecision) -> None:
        """Update session file with new decision."""
        session_file = self.realtime_path / f"{self.session_id}.json"
        
        try:
            # Load existing session data
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Add the new decision
            session_data['decisions'].append(asdict(decision))
            
            # Update summary
            decisions = session_data['decisions']
            session_data['summary']['total_decisions'] = len(decisions)
            
            # Update gate type counts
            gate_counts = {}
            result_counts = {}
            total_time = 0
            
            for d in decisions:
                gate_type = d['gate_type']
                result = d['decision']
                processing_time = d.get('processing_time_ms', 0)
                
                gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
                result_counts[result] = result_counts.get(result, 0) + 1
                total_time += processing_time
            
            session_data['summary']['decisions_by_gate'] = gate_counts
            session_data['summary']['decisions_by_result'] = result_counts
            session_data['summary']['average_processing_time'] = total_time / len(decisions)
            
            # Write updated session data
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating session with decision: {e}")
    
    def _get_recent_decisions_all_sessions(self, minutes_back: int = 60) -> List[Dict[str, Any]]:
        """Get recent decisions from all sessions."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
        recent_decisions = []
        
        # Check all session files
        for session_file in self.realtime_path.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                decisions = session_data.get('decisions', [])
                for decision in decisions:
                    decision_time = datetime.fromisoformat(decision['timestamp'])
                    if decision_time >= cutoff_time:
                        recent_decisions.append(decision)
                        
            except Exception as e:
                self.logger.warning(f"Error reading session file {session_file}: {e}")
                continue
        
        # Sort by timestamp
        recent_decisions.sort(key=lambda x: x['timestamp'])
        return recent_decisions
    
    def _calculate_live_metrics(self, recent_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate live metrics from recent decisions."""
        if not recent_decisions:
            return {
                'decision_rate_per_hour': 0,
                'pass_rate': 0,
                'average_processing_time': 0,
                'most_active_gate': 'none',
                'performance_status': 'good'
            }
        
        # Calculate decision rate (decisions per hour)
        if len(recent_decisions) >= 2:
            first_time = datetime.fromisoformat(recent_decisions[0]['timestamp'])
            last_time = datetime.fromisoformat(recent_decisions[-1]['timestamp'])
            time_span_hours = (last_time - first_time).total_seconds() / 3600
            decision_rate = len(recent_decisions) / max(time_span_hours, 0.0167)  # Min 1 minute
        else:
            decision_rate = 0
        
        # Calculate pass rate
        pass_count = sum(1 for d in recent_decisions if d['decision'] == 'pass')
        pass_rate = (pass_count / len(recent_decisions)) * 100
        
        # Calculate average processing time
        total_time = sum(d.get('processing_time_ms', 0) for d in recent_decisions)
        avg_processing_time = total_time / len(recent_decisions)
        
        # Find most active gate type
        gate_counts = {}
        for decision in recent_decisions:
            gate_type = decision['gate_type']
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
        
        most_active_gate = max(gate_counts.items(), key=lambda x: x[1])[0] if gate_counts else 'none'
        
        # Determine performance status
        if avg_processing_time > 100:
            performance_status = 'poor'
        elif avg_processing_time > 50:
            performance_status = 'warning'
        else:
            performance_status = 'good'
        
        return {
            'decision_rate_per_hour': decision_rate,
            'pass_rate': pass_rate,
            'average_processing_time': avg_processing_time,
            'most_active_gate': most_active_gate,
            'performance_status': performance_status,
            'total_recent_decisions': len(recent_decisions)
        }
    
    def _extract_issue_number(self, hook_data: Dict[str, Any]) -> Optional[int]:
        """Extract issue number from Claude Code hook data."""
        # Try various sources for issue number
        if 'issue_number' in hook_data:
            return int(hook_data['issue_number'])
        
        # Try to extract from tool context or command
        command = hook_data.get('command', '')
        tool_name = hook_data.get('tool_name', '')
        
        # Look for patterns like "issue #94" or "#94"
        import re
        patterns = [r'issue\s*#(\d+)', r'#(\d+)']
        
        for pattern in patterns:
            for text in [command, tool_name]:
                match = re.search(pattern, str(text), re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return None
    
    def _determine_gate_type_from_hook(self, hook_data: Dict[str, Any]) -> Optional[str]:
        """Determine quality gate type from hook context."""
        tool_name = hook_data.get('tool_name', '').lower()
        command = hook_data.get('command', '').lower()
        
        # Map tool/command patterns to gate types
        gate_mappings = {
            'bash': 'linting',  # Often used for running tests/lints
            'edit': 'code_coverage',  # Code changes affect coverage
            'multiedit': 'code_coverage',
            'write': 'documentation',
            'grep': 'security_scan',  # Searching for security patterns
            'read': 'evidence_requirements'  # Reading to gather evidence
        }
        
        # Check tool name first
        if tool_name in gate_mappings:
            return gate_mappings[tool_name]
        
        # Check command patterns
        if any(keyword in command for keyword in ['test', 'coverage', 'pytest']):
            return 'code_coverage'
        elif any(keyword in command for keyword in ['lint', 'flake8', 'black']):
            return 'linting'
        elif any(keyword in command for keyword in ['security', 'bandit', 'safety']):
            return 'security_scan'
        elif any(keyword in command for keyword in ['docs', 'documentation']):
            return 'documentation'
        
        # Default to quality_score for general tool usage
        return 'quality_score'
    
    def _determine_decision_from_hook(self, hook_data: Dict[str, Any]) -> Optional[str]:
        """Determine decision result from hook context."""
        # Check for explicit decision in hook data
        if 'decision' in hook_data:
            return hook_data['decision']
        
        # Check command output for success/failure indicators
        output = hook_data.get('output', '')
        if output:
            if any(indicator in output.lower() for indicator in ['pass', 'success', 'ok', '✓']):
                return 'pass'
            elif any(indicator in output.lower() for indicator in ['fail', 'error', 'failed', '✗']):
                return 'fail'
            elif any(indicator in output.lower() for indicator in ['warn', 'warning']):
                return 'warning'
        
        # Default to pass for successful tool execution
        return 'pass'
    
    def _extract_hook_context(self, hook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from hook data."""
        context = {}
        
        # Copy relevant fields
        for field in ['tool_name', 'command', 'output', 'duration', 'session_id']:
            if field in hook_data:
                context[field] = hook_data[field]
        
        # Add timestamp
        context['hook_timestamp'] = datetime.now().isoformat()
        
        return context


# Hook integration functions for Claude Code
def on_tool_use_complete(tool_name: str, command: str, output: str, duration: float) -> None:
    """
    Claude Code hook called after tool use completes.
    Automatically tracks quality gate decisions.
    """
    try:
        tracker = RealTimeDecisionTracker()
        
        hook_data = {
            'tool_name': tool_name,
            'command': command,
            'output': output,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        tracker.hook_integration_track_decision(hook_data)
        
    except Exception as e:
        # Don't let hook errors break the main workflow
        print(f"Warning: Quality tracking hook failed: {e}", file=sys.stderr)


def on_session_start() -> None:
    """Claude Code hook called at session start."""
    try:
        tracker = RealTimeDecisionTracker()
        tracker.logger.info(f"Quality tracking initialized for session: {tracker.session_id}")
    except Exception as e:
        print(f"Warning: Quality tracking initialization failed: {e}", file=sys.stderr)


def main():
    """Command line interface for real-time decision tracker."""
    if len(sys.argv) < 2:
        print("Usage: python real_time_decision_tracker.py <command> [args]")
        print("Commands:")
        print("  track <issue_number> <gate_type> <decision>")
        print("  session-summary")
        print("  dashboard")
        print("  test-hook")
        return 1
    
    tracker = RealTimeDecisionTracker()
    command = sys.argv[1]
    
    if command == "track" and len(sys.argv) >= 5:
        issue_num = int(sys.argv[2])
        gate_type = sys.argv[3]
        decision = sys.argv[4]
        
        decision_id = tracker.track_quality_decision_realtime(
            issue_number=issue_num,
            gate_type=gate_type,
            decision=decision,
            agent_name="manual"
        )
        print(f"Decision tracked: {decision_id}")
        
    elif command == "session-summary":
        summary = tracker.get_session_summary()
        print(json.dumps(summary, indent=2))
        
    elif command == "dashboard":
        dashboard = tracker.get_live_dashboard_data()
        print(json.dumps(dashboard, indent=2))
        
    elif command == "test-hook":
        # Test hook integration
        test_hook_data = {
            'tool_name': 'bash',
            'command': 'pytest tests/ --cov=. --cov-report=term',
            'output': 'PASSED - Coverage: 85%',
            'duration': 2.5,
            'issue_number': 94
        }
        
        success = tracker.hook_integration_track_decision(test_hook_data)
        print(f"Hook integration test: {'success' if success else 'failed'}")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())