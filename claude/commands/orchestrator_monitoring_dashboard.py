#!/usr/bin/env python3
"""
RIF Orchestrator Monitoring Dashboard
Issue #56: Create orchestrator monitoring dashboard

This module provides real-time monitoring and visualization for RIF orchestration
workflows, including state transitions, performance metrics, and historical analysis.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("DuckDB not available - install with: pip install duckdb")
    duckdb = None

from orchestrator_state_persistence import OrchestratorStatePersistence

class OrchestratorMonitoringDashboard:
    """
    Real-time monitoring dashboard for orchestration workflows.
    Provides visualization data, metrics tracking, and historical analysis.
    """
    
    def __init__(self, persistence_system: Optional[OrchestratorStatePersistence] = None,
                 db_path: str = 'knowledge/orchestration.duckdb'):
        """
        Initialize the monitoring dashboard.
        
        Args:
            persistence_system: Existing persistence system instance
            db_path: Database path if creating new persistence system
        """
        self.persistence = persistence_system or OrchestratorStatePersistence(db_path)
        self.active_workflows = {}
        self.metrics_cache = defaultdict(lambda: {'count': 0, 'total_time': 0})
        self.real_time_events = deque(maxlen=1000)  # Keep last 1000 events
        self.state_transitions = defaultdict(int)
        self.agent_performance = defaultdict(list)
        self.refresh_interval = 1.0  # 1 second refresh rate
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data snapshot.
        
        Returns:
            Complete dashboard data structure
        """
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'refresh_interval': self.refresh_interval,
                'active_workflows': self.get_active_workflows(),
                'state_distribution': self.get_state_distribution(),
                'transition_metrics': self.get_transition_metrics(),
                'performance_stats': self.get_performance_stats(),
                'recent_decisions': self.get_recent_decisions(),
                'agent_status': self.get_agent_status(),
                'system_health': self.get_system_health(),
                'real_time_events': list(self.real_time_events)[-50:]  # Last 50 events
            }
            
            return dashboard_data
            
        except Exception as e:
            return {
                'error': f"Failed to generate dashboard data: {e}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """
        Get current active workflows with detailed status.
        
        Returns:
            List of active workflow information
        """
        try:
            active_sessions = self.persistence.get_active_sessions()
            workflows = []
            
            for session in active_sessions:
                # Get detailed state for each session
                state = self.persistence.recover_state(session['session_id'])
                if state:
                    # Calculate workflow progress
                    history = state.get('history', [])
                    progress = self._calculate_workflow_progress(state['current_state'], history)
                    
                    workflow_info = {
                        'session_id': session['session_id'],
                        'current_state': session['current_state'],
                        'workflow_type': session['workflow_type'],
                        'priority': session['priority'],
                        'progress_percentage': progress,
                        'duration_minutes': self._calculate_duration_minutes(session['created_at']),
                        'active_agents': self._extract_active_agents(state),
                        'issues_count': len(state.get('context', {}).get('github_issues', [])),
                        'last_decision': self._get_last_decision(session['session_id']),
                        'created_at': session['created_at'],
                        'updated_at': session['updated_at']
                    }
                    workflows.append(workflow_info)
            
            return workflows
            
        except Exception as e:
            return [{'error': f"Failed to get active workflows: {e}"}]
    
    def get_state_distribution(self) -> Dict[str, Any]:
        """
        Get distribution of workflow states for visualization.
        
        Returns:
            State distribution data for charts
        """
        try:
            # Query state distribution from database
            results = self.persistence.db.execute(
                """SELECT current_state, COUNT(*) as count, workflow_type
                   FROM orchestration_state 
                   WHERE current_state NOT IN ('completed', 'failed', 'cancelled')
                   GROUP BY current_state, workflow_type
                   ORDER BY count DESC"""
            ).fetchall()
            
            distribution = {
                'total_active': sum(row[1] for row in results),
                'by_state': {},
                'by_workflow_type': defaultdict(dict),
                'chart_data': []
            }
            
            for state, count, workflow_type in results:
                distribution['by_state'][state] = distribution['by_state'].get(state, 0) + count
                distribution['by_workflow_type'][workflow_type][state] = count
                distribution['chart_data'].append({
                    'state': state,
                    'count': count,
                    'workflow_type': workflow_type
                })
            
            return distribution
            
        except Exception as e:
            return {'error': f"Failed to get state distribution: {e}"}
    
    def get_transition_metrics(self) -> Dict[str, Any]:
        """
        Get state transition metrics and patterns.
        
        Returns:
            Transition metrics for analysis
        """
        try:
            # Get transition data from decisions table
            results = self.persistence.db.execute(
                """SELECT from_state, to_state, COUNT(*) as count,
                          AVG(execution_time_ms) as avg_time,
                          AVG(confidence_score) as avg_confidence,
                          SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
                   FROM orchestration_decisions 
                   WHERE timestamp >= (NOW() - INTERVAL 24 HOUR)
                   GROUP BY from_state, to_state
                   ORDER BY count DESC"""
            ).fetchall()
            
            transitions = {
                'total_transitions': sum(row[2] for row in results),
                'transition_matrix': {},
                'common_paths': [],
                'performance_by_transition': {}
            }
            
            for from_state, to_state, count, avg_time, avg_confidence, successes in results:
                transition_key = f"{from_state}->{to_state}"
                
                # Build transition matrix
                if from_state not in transitions['transition_matrix']:
                    transitions['transition_matrix'][from_state] = {}
                transitions['transition_matrix'][from_state][to_state] = count
                
                # Record performance metrics
                transitions['performance_by_transition'][transition_key] = {
                    'count': count,
                    'avg_execution_time_ms': float(avg_time) if avg_time else 0,
                    'avg_confidence_score': float(avg_confidence) if avg_confidence else 0,
                    'success_rate': successes / count if count > 0 else 0
                }
                
                # Track common paths
                transitions['common_paths'].append({
                    'path': transition_key,
                    'count': count,
                    'avg_time_ms': float(avg_time) if avg_time else 0
                })
            
            # Sort common paths by frequency
            transitions['common_paths'].sort(key=lambda x: x['count'], reverse=True)
            
            return transitions
            
        except Exception as e:
            return {'error': f"Failed to get transition metrics: {e}"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get system performance statistics.
        
        Returns:
            Performance metrics and statistics
        """
        try:
            # Get overall performance stats
            overall_stats = self.persistence.get_performance_stats()
            
            # Get recent performance trends
            trend_results = self.persistence.db.execute(
                """SELECT DATE(timestamp) as date,
                          COUNT(*) as decisions,
                          AVG(execution_time_ms) as avg_time,
                          AVG(confidence_score) as avg_confidence,
                          SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
                   FROM orchestration_decisions 
                   WHERE timestamp >= (NOW() - INTERVAL 7 DAYS)
                   GROUP BY DATE(timestamp)
                   ORDER BY date DESC"""
            ).fetchall()
            
            trends = []
            for date, decisions, avg_time, avg_confidence, successes in trend_results:
                trends.append({
                    'date': str(date),
                    'decisions': decisions,
                    'avg_execution_time_ms': float(avg_time) if avg_time else 0,
                    'avg_confidence': float(avg_confidence) if avg_confidence else 0,
                    'success_rate': successes / decisions if decisions > 0 else 0
                })
            
            return {
                'overall': overall_stats,
                'trends': trends,
                'current_load': self._calculate_current_load(),
                'bottlenecks': self._identify_bottlenecks()
            }
            
        except Exception as e:
            return {'error': f"Failed to get performance stats: {e}"}
    
    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent orchestration decisions for timeline view.
        
        Args:
            limit: Number of recent decisions to return
            
        Returns:
            List of recent decision records
        """
        try:
            results = self.persistence.db.execute(
                """SELECT d.decision_id, d.session_id, d.from_state, d.to_state,
                          d.decision_reason, d.agents_selected, d.confidence_score,
                          d.execution_time_ms, d.success, d.error_details, d.timestamp,
                          s.workflow_type, s.priority
                   FROM orchestration_decisions d
                   JOIN orchestration_state s ON d.session_id = s.session_id
                   ORDER BY d.timestamp DESC
                   LIMIT ?""",
                [limit]
            ).fetchall()
            
            decisions = []
            for row in results:
                (decision_id, session_id, from_state, to_state, reason, agents_json,
                 confidence, exec_time, success, error_details, timestamp,
                 workflow_type, priority) = row
                
                decisions.append({
                    'decision_id': decision_id,
                    'session_id': session_id,
                    'transition': f"{from_state} → {to_state}",
                    'reason': reason,
                    'agents': json.loads(agents_json) if agents_json else [],
                    'confidence_score': confidence,
                    'execution_time_ms': exec_time,
                    'success': success,
                    'error_details': error_details,
                    'timestamp': str(timestamp),
                    'workflow_type': workflow_type,
                    'priority': priority
                })
            
            return decisions
            
        except Exception as e:
            return [{'error': f"Failed to get recent decisions: {e}"}]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents in the system.
        
        Returns:
            Agent status and performance information
        """
        try:
            # Get agent performance from decisions
            results = self.persistence.db.execute(
                """SELECT agents_selected, COUNT(*) as total_tasks,
                          AVG(execution_time_ms) as avg_time,
                          SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                          AVG(confidence_score) as avg_confidence
                   FROM orchestration_decisions
                   WHERE timestamp >= (NOW() - INTERVAL 24 HOUR)
                   GROUP BY agents_selected"""
            ).fetchall()
            
            agent_stats = {}
            total_agents_active = 0
            
            for agents_json, total_tasks, avg_time, successes, avg_confidence in results:
                try:
                    agents = json.loads(agents_json) if agents_json else []
                    for agent in agents:
                        if agent not in agent_stats:
                            agent_stats[agent] = {
                                'total_tasks': 0,
                                'avg_execution_time_ms': 0,
                                'success_rate': 0,
                                'avg_confidence': 0,
                                'status': 'active' if total_tasks > 0 else 'idle'
                            }
                        
                        agent_stats[agent]['total_tasks'] += total_tasks
                        agent_stats[agent]['avg_execution_time_ms'] = float(avg_time) if avg_time else 0
                        agent_stats[agent]['success_rate'] = successes / total_tasks if total_tasks > 0 else 0
                        agent_stats[agent]['avg_confidence'] = float(avg_confidence) if avg_confidence else 0
                        
                        if total_tasks > 0:
                            total_agents_active += 1
                            
                except json.JSONDecodeError:
                    continue
            
            return {
                'total_agents': len(agent_stats),
                'active_agents': total_agents_active,
                'agent_details': agent_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f"Failed to get agent status: {e}"}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        
        Returns:
            System health information
        """
        try:
            # Calculate various health metrics
            active_sessions = len(self.persistence.get_active_sessions())
            
            # Get error rate over last hour
            error_results = self.persistence.db.execute(
                """SELECT COUNT(*) as total, 
                          SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as errors
                   FROM orchestration_decisions 
                   WHERE timestamp >= (NOW() - INTERVAL 1 HOUR)"""
            ).fetchone()
            
            total_decisions, error_count = error_results if error_results else (0, 0)
            error_rate = error_count / total_decisions if total_decisions > 0 else 0
            
            # Check database connectivity
            db_health = self._check_database_health()
            
            # Determine overall health status
            health_score = 100
            if error_rate > 0.1:  # More than 10% errors
                health_score -= 30
            if active_sessions > 100:  # High load
                health_score -= 20
            if not db_health:
                health_score -= 50
            
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "warning"
            else:
                status = "critical"
            
            return {
                'status': status,
                'health_score': max(0, health_score),
                'active_sessions': active_sessions,
                'error_rate': error_rate,
                'database_health': db_health,
                'last_check': datetime.now().isoformat(),
                'alerts': self._generate_health_alerts(error_rate, active_sessions)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to check system health: {e}",
                'last_check': datetime.now().isoformat()
            }
    
    def visualize_workflow(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate workflow visualization data for a specific session.
        
        Args:
            session_id: Session to visualize
            
        Returns:
            Workflow visualization data or None if session not found
        """
        try:
            # Get session state
            workflow_state = self.persistence.recover_state(session_id)
            if not workflow_state:
                return None
            
            # Get decision history for this session
            decisions = self.persistence.get_decision_history(session_id)
            
            # Build nodes (states)
            visited_states = set()
            nodes = []
            
            # Add states from history
            for decision in decisions:
                visited_states.add(decision['from_state'])
                visited_states.add(decision['to_state'])
            
            # Add current state
            visited_states.add(workflow_state['current_state'])
            
            for state in visited_states:
                node = {
                    'id': state,
                    'label': state.replace('_', ' ').title(),
                    'status': 'current' if state == workflow_state['current_state'] else 'visited',
                    'agents': self._get_agents_for_state(state, decisions)
                }
                nodes.append(node)
            
            # Build edges (transitions)
            edges = []
            for decision in decisions:
                edge = {
                    'from': decision['from_state'],
                    'to': decision['to_state'],
                    'label': decision.get('reason', '')[:50] + ('...' if len(decision.get('reason', '')) > 50 else ''),
                    'timestamp': decision['timestamp'],
                    'success': decision['success'],
                    'confidence': decision['confidence_score'],
                    'execution_time_ms': decision['execution_time_ms']
                }
                edges.append(edge)
            
            return {
                'session_id': session_id,
                'workflow_type': workflow_state.get('workflow_type', 'unknown'),
                'current_state': workflow_state['current_state'],
                'nodes': nodes,
                'edges': edges,
                'metadata': {
                    'created_at': workflow_state.get('created_at'),
                    'updated_at': workflow_state.get('updated_at'),
                    'total_states': len(nodes),
                    'total_transitions': len(edges)
                }
            }
            
        except Exception as e:
            return {'error': f"Failed to visualize workflow: {e}"}
    
    def track_metrics(self, event_type: str, data: Dict[str, Any]):
        """
        Track real-time orchestration metrics.
        
        Args:
            event_type: Type of event to track
            data: Event data
        """
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'data': data
            }
            self.real_time_events.append(event)
            
            # Update cached metrics based on event type
            if event_type == 'state_transition':
                transition_key = f"{data['from']}->{data['to']}"
                self.metrics_cache[transition_key]['count'] += 1
                self.metrics_cache[transition_key]['total_time'] += data.get('duration', 0)
                
            elif event_type == 'agent_launch':
                agent_key = f"agent_{data['agent']}"
                self.metrics_cache[agent_key]['count'] += 1
                
            elif event_type == 'workflow_complete':
                self.metrics_cache['completions']['count'] += 1
                self.metrics_cache['completions']['total_time'] += data.get('total_duration', 0)
                
            elif event_type == 'error_occurred':
                self.metrics_cache['errors']['count'] += 1
                
        except Exception as e:
            print(f"Error tracking metrics: {e}")
    
    def generate_report(self, session_id: Optional[str] = None, 
                       timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            session_id: Specific session to report on, or None for all
            timeframe_hours: Timeframe for the report in hours
            
        Returns:
            Comprehensive monitoring report
        """
        try:
            report = {
                'report_id': f"report_{int(time.time())}",
                'generated_at': datetime.now().isoformat(),
                'timeframe_hours': timeframe_hours,
                'session_id': session_id,
                'summary': {},
                'performance': {},
                'recommendations': []
            }
            
            # Generate summary
            if session_id:
                workflow_viz = self.visualize_workflow(session_id)
                report['summary'] = {
                    'session_focus': session_id,
                    'workflow_type': workflow_viz.get('workflow_type') if workflow_viz else 'unknown',
                    'current_state': workflow_viz.get('current_state') if workflow_viz else 'unknown',
                    'total_transitions': len(workflow_viz.get('edges', [])) if workflow_viz else 0
                }
            else:
                active_workflows = self.get_active_workflows()
                report['summary'] = {
                    'active_workflows': len(active_workflows),
                    'total_states_represented': len(set(w['current_state'] for w in active_workflows)),
                    'avg_workflow_duration': sum(w.get('duration_minutes', 0) for w in active_workflows) / len(active_workflows) if active_workflows else 0
                }
            
            # Add performance data
            report['performance'] = self.get_performance_stats()
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            return {'error': f"Failed to generate report: {e}"}
    
    def _calculate_workflow_progress(self, current_state: str, history: List[Dict]) -> float:
        """Calculate workflow progress percentage based on current state and history."""
        # Define typical workflow stages and their progress values
        stage_progress = {
            'initialized': 10,
            'analyzing': 25,
            'planning': 40,
            'architecting': 55,
            'implementing': 75,
            'validating': 90,
            'completed': 100,
            'failed': 0,
            'cancelled': 0
        }
        
        return stage_progress.get(current_state, 50)  # Default to 50% if unknown
    
    def _calculate_duration_minutes(self, created_at: str) -> float:
        """Calculate duration in minutes from creation timestamp."""
        try:
            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            duration = datetime.now().replace(tzinfo=created_time.tzinfo) - created_time
            return duration.total_seconds() / 60
        except:
            return 0
    
    def _extract_active_agents(self, state: Dict[str, Any]) -> List[str]:
        """Extract list of currently active agents from state."""
        agent_assignments = state.get('agent_assignments', {})
        return [agent for agent, status in agent_assignments.items() if status == 'active']
    
    def _get_last_decision(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent decision for a session."""
        decisions = self.persistence.get_decision_history(session_id, limit=1)
        return decisions[0] if decisions else None
    
    def _calculate_current_load(self) -> Dict[str, Any]:
        """Calculate current system load metrics."""
        active_sessions = self.persistence.get_active_sessions()
        return {
            'active_sessions': len(active_sessions),
            'high_priority_sessions': len([s for s in active_sessions if s['priority'] > 5]),
            'load_level': 'high' if len(active_sessions) > 50 else 'medium' if len(active_sessions) > 20 else 'low'
        }
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify potential system bottlenecks."""
        bottlenecks = []
        
        try:
            # Check for slow transitions
            slow_transitions = self.persistence.db.execute(
                """SELECT from_state, to_state, AVG(execution_time_ms) as avg_time
                   FROM orchestration_decisions 
                   WHERE timestamp >= (NOW() - INTERVAL 1 HOUR)
                   GROUP BY from_state, to_state
                   HAVING AVG(execution_time_ms) > 10000
                   ORDER BY avg_time DESC"""
            ).fetchall()
            
            for from_state, to_state, avg_time in slow_transitions:
                bottlenecks.append({
                    'type': 'slow_transition',
                    'description': f"Transition {from_state} → {to_state} is slow",
                    'avg_time_ms': float(avg_time),
                    'severity': 'high' if avg_time > 30000 else 'medium'
                })
        
        except Exception as e:
            print(f"Error identifying bottlenecks: {e}")
        
        return bottlenecks
    
    def _check_database_health(self) -> bool:
        """Check database connectivity and health."""
        try:
            result = self.persistence.db.execute("SELECT 1").fetchone()
            return result is not None
        except:
            return False
    
    def _generate_health_alerts(self, error_rate: float, active_sessions: int) -> List[Dict[str, Any]]:
        """Generate health alerts based on current metrics."""
        alerts = []
        
        if error_rate > 0.1:
            alerts.append({
                'severity': 'high',
                'message': f"High error rate detected: {error_rate:.1%}",
                'action': 'Review recent failed decisions'
            })
        
        if active_sessions > 100:
            alerts.append({
                'severity': 'medium', 
                'message': f"High number of active sessions: {active_sessions}",
                'action': 'Consider scaling or optimizing workflows'
            })
        
        return alerts
    
    def _get_agents_for_state(self, state: str, decisions: List[Dict]) -> List[str]:
        """Get agents associated with a specific state from decisions."""
        agents = set()
        for decision in decisions:
            if decision['from_state'] == state or decision['to_state'] == state:
                agents.update(decision.get('agents_selected', []))
        return list(agents)
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on report data."""
        recommendations = []
        
        performance = report.get('performance', {})
        overall_stats = performance.get('overall', {})
        
        # Performance-based recommendations
        if overall_stats.get('avg_execution_time_ms', 0) > 10000:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Optimize Execution Time',
                'description': 'Average execution time is high. Consider optimizing agent workflows or increasing system resources.'
            })
        
        success_rate = overall_stats.get('success_rate', 1.0)
        if success_rate < 0.9:
            recommendations.append({
                'type': 'reliability',
                'priority': 'high', 
                'title': 'Improve Success Rate',
                'description': f'Success rate is {success_rate:.1%}. Review failed decisions and improve error handling.'
            })
        
        return recommendations

# Utility functions
def create_monitoring_dashboard(db_path: str = 'knowledge/orchestration.duckdb') -> OrchestratorMonitoringDashboard:
    """Create and initialize a monitoring dashboard instance."""
    return OrchestratorMonitoringDashboard(db_path=db_path)

def test_monitoring_dashboard() -> bool:
    """Test the monitoring dashboard functionality."""
    try:
        from orchestrator_state_persistence import OrchestratorStatePersistence
        
        # Create test persistence system with in-memory database
        persistence = OrchestratorStatePersistence(':memory:')
        dashboard = OrchestratorMonitoringDashboard(persistence)
        
        # Create a test session
        session_id = persistence.start_session('test_monitoring')
        
        # Add some test data
        persistence.save_state({
            'current_state': 'testing',
            'context': {'test': True},
            'history': [{'action': 'test'}],
            'agent_assignments': {'test_agent': 'active'}
        })
        
        persistence.record_decision('initialized', 'testing', 'Test decision', ['test_agent'], 0.9)
        
        # Test dashboard functionality
        dashboard_data = dashboard.get_dashboard_data()
        assert 'timestamp' in dashboard_data
        assert 'active_workflows' in dashboard_data
        
        # Test workflow visualization
        workflow_viz = dashboard.visualize_workflow(session_id)
        assert workflow_viz is not None
        assert 'nodes' in workflow_viz
        assert 'edges' in workflow_viz
        
        # Test report generation
        report = dashboard.generate_report(session_id)
        assert 'report_id' in report
        assert 'summary' in report
        
        persistence.close()
        print("✅ Monitoring dashboard test passed")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring dashboard test failed: {e}")
        return False

if __name__ == "__main__":
    # Run basic tests
    test_monitoring_dashboard()