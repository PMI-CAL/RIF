#!/usr/bin/env python3
"""
RIF Orchestrator Integration Module
Issues #55 and #56 Integration

This module provides a unified interface for orchestrator state persistence
and monitoring dashboard functionality, enabling complete orchestration lifecycle management.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from orchestrator_state_persistence import OrchestratorStatePersistence
from orchestrator_monitoring_dashboard import OrchestratorMonitoringDashboard

class IntegratedOrchestratorSystem:
    """
    Unified orchestrator system combining state persistence and monitoring.
    Provides complete orchestration lifecycle management.
    """
    
    def __init__(self, db_path: str = 'knowledge/orchestration.duckdb'):
        """
        Initialize the integrated orchestrator system.
        
        Args:
            db_path: Path to the orchestration database
        """
        self.db_path = db_path
        self.persistence = OrchestratorStatePersistence(db_path)
        self.dashboard = OrchestratorMonitoringDashboard(self.persistence)
        self.active_session_id = None
        
    def start_orchestration_session(self, workflow_type: str = "standard", 
                                  priority: int = 0, context: Optional[Dict] = None) -> str:
        """
        Start a new orchestration session with integrated monitoring.
        
        Args:
            workflow_type: Type of workflow being orchestrated
            priority: Session priority level
            context: Initial context data
            
        Returns:
            session_id: Unique session identifier
        """
        # Start persistence session
        session_id = self.persistence.start_session(workflow_type, priority)
        self.active_session_id = session_id
        
        # Initialize with context if provided
        if context:
            initial_state = {
                'current_state': 'initialized',
                'context': context,
                'history': [{
                    'state': 'initialized',
                    'timestamp': datetime.now().isoformat(),
                    'reason': f'Session started with workflow type: {workflow_type}',
                    'context': context
                }],
                'agent_assignments': {}
            }
            self.persistence.save_state(initial_state)
        
        # Track session start event
        self.dashboard.track_metrics('session_start', {
            'session_id': session_id,
            'workflow_type': workflow_type,
            'priority': priority
        })
        
        return session_id
    
    def transition_state(self, from_state: str, to_state: str, reason: str,
                        agents_selected: List[str], context_updates: Optional[Dict] = None,
                        confidence_score: float = 1.0) -> bool:
        """
        Perform a state transition with full persistence and monitoring.
        
        Args:
            from_state: Starting state
            to_state: Target state  
            reason: Reason for transition
            agents_selected: Agents selected for execution
            context_updates: Updates to apply to context
            confidence_score: Confidence in the decision
            
        Returns:
            Success status
        """
        if not self.active_session_id:
            raise RuntimeError("No active session - call start_orchestration_session() first")
        
        start_time = time.time()
        
        try:
            # Record the decision
            decision_id = self.persistence.record_decision(
                from_state, to_state, reason, agents_selected, confidence_score,
                execution_time_ms=0, success=True
            )
            
            # Get current state for context merging
            current_state = self.persistence.recover_state(self.active_session_id)
            if not current_state:
                raise RuntimeError("Could not recover current state")
            
            # Update context with new data
            updated_context = current_state.get('context', {})
            if context_updates:
                updated_context.update(context_updates)
            
            # Update history
            updated_history = current_state.get('history', [])
            updated_history.append({
                'state': to_state,
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'decision_id': decision_id,
                'agents_selected': agents_selected,
                'confidence_score': confidence_score
            })
            
            # Update agent assignments
            updated_assignments = current_state.get('agent_assignments', {})
            for agent in agents_selected:
                updated_assignments[agent] = 'active'
            
            # Save updated state
            new_state = {
                'current_state': to_state,
                'context': updated_context,
                'history': updated_history,
                'agent_assignments': updated_assignments
            }
            
            success = self.persistence.save_state(new_state)
            
            if success:
                # Calculate execution time
                execution_time = int((time.time() - start_time) * 1000)
                
                # Update decision with actual execution time
                self.persistence.db.execute(
                    "UPDATE orchestration_decisions SET execution_time_ms = ? WHERE decision_id = ?",
                    [execution_time, decision_id]
                )
                
                # Track transition in dashboard
                self.dashboard.track_metrics('state_transition', {
                    'from': from_state,
                    'to': to_state,
                    'duration': execution_time,
                    'agents': agents_selected,
                    'confidence': confidence_score
                })
                
                return True
            else:
                # Mark decision as failed
                self.persistence.db.execute(
                    "UPDATE orchestration_decisions SET success = FALSE, error_details = ? WHERE decision_id = ?",
                    ["Failed to save state", decision_id]
                )
                return False
                
        except Exception as e:
            # Record failure
            execution_time = int((time.time() - start_time) * 1000)
            self.persistence.record_decision(
                from_state, to_state, reason, agents_selected, confidence_score,
                execution_time, False, str(e)
            )
            
            # Track error
            self.dashboard.track_metrics('transition_error', {
                'from': from_state,
                'to': to_state,
                'error': str(e)
            })
            
            raise
    
    def launch_agents(self, agents: List[str], task_description: str) -> Dict[str, str]:
        """
        Launch agents and track their assignment.
        
        Args:
            agents: List of agent names to launch
            task_description: Description of the task
            
        Returns:
            Dictionary mapping agent names to their assignment IDs
        """
        if not self.active_session_id:
            raise RuntimeError("No active session")
        
        assignments = {}
        
        for agent in agents:
            assignment_id = f"assignment_{int(time.time())}_{agent}"
            assignments[agent] = assignment_id
            
            # Track agent launch
            self.dashboard.track_metrics('agent_launch', {
                'agent': agent,
                'session_id': self.active_session_id,
                'task': task_description,
                'assignment_id': assignment_id
            })
        
        # Record metric
        self.persistence.record_metric(
            'agent_launch', 'agents_launched', len(agents),
            {'agents': agents, 'task': task_description}
        )
        
        return assignments
    
    def complete_orchestration_session(self, outcome: str = "completed", 
                                     final_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete the current orchestration session with summary.
        
        Args:
            outcome: Final outcome (completed, failed, cancelled)
            final_context: Final context data
            
        Returns:
            Session summary and statistics
        """
        if not self.active_session_id:
            raise RuntimeError("No active session to complete")
        
        try:
            # Get current state
            current_state = self.persistence.recover_state(self.active_session_id)
            if not current_state:
                raise RuntimeError("Could not recover session state")
            
            # Update final state
            updated_context = current_state.get('context', {})
            if final_context:
                updated_context.update(final_context)
            
            updated_history = current_state.get('history', [])
            updated_history.append({
                'state': outcome,
                'timestamp': datetime.now().isoformat(),
                'reason': f'Session completed with outcome: {outcome}',
                'final_context': final_context
            })
            
            final_state = {
                'current_state': outcome,
                'context': updated_context,
                'history': updated_history,
                'agent_assignments': {}  # Clear active assignments
            }
            
            self.persistence.save_state(final_state)
            
            # Generate session summary
            summary = self.generate_session_summary(self.active_session_id)
            
            # Track completion
            self.dashboard.track_metrics('session_complete', {
                'session_id': self.active_session_id,
                'outcome': outcome,
                'total_duration': summary.get('total_duration_ms', 0),
                'total_transitions': summary.get('total_decisions', 0)
            })
            
            # Clear active session
            self.active_session_id = None
            
            return summary
            
        except Exception as e:
            # Track error
            self.dashboard.track_metrics('completion_error', {
                'session_id': self.active_session_id,
                'error': str(e)
            })
            raise
    
    def generate_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive session summary with statistics.
        
        Args:
            session_id: Session to summarize, or active session if None
            
        Returns:
            Complete session summary
        """
        target_session = session_id or self.active_session_id
        if not target_session:
            raise RuntimeError("No session specified and no active session")
        
        # Get session state
        session_state = self.persistence.recover_state(target_session)
        if not session_state:
            raise RuntimeError(f"Session {target_session} not found")
        
        # Get decision history
        decisions = self.persistence.get_decision_history(target_session)
        
        # Get performance stats
        performance = self.persistence.get_performance_stats(target_session)
        
        # Calculate session metrics
        history = session_state.get('history', [])
        
        if len(history) >= 2:
            start_time = datetime.fromisoformat(history[0]['timestamp'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(history[-1]['timestamp'].replace('Z', '+00:00'))
            total_duration = (end_time - start_time).total_seconds() * 1000
        else:
            total_duration = 0
        
        summary = {
            'session_id': target_session,
            'workflow_type': session_state.get('workflow_type', 'unknown'),
            'current_state': session_state['current_state'],
            'priority': session_state.get('priority', 0),
            'created_at': session_state.get('created_at'),
            'updated_at': session_state.get('updated_at'),
            'total_duration_ms': total_duration,
            'total_decisions': len(decisions),
            'total_states_visited': len(history),
            'success_rate': performance.get('success_rate', 0),
            'avg_execution_time_ms': performance.get('avg_execution_time_ms', 0),
            'avg_confidence_score': performance.get('avg_confidence_score', 0),
            'context_summary': {
                'github_issues': len(session_state.get('context', {}).get('github_issues', [])),
                'active_agents': len([a for a in session_state.get('agent_assignments', {}).values() if a == 'active']),
                'total_agents_used': len(session_state.get('agent_assignments', {}))
            },
            'state_progression': [step.get('state', 'unknown') for step in history],
            'decision_summary': self._summarize_decisions(decisions)
        }
        
        return summary
    
    def get_live_dashboard(self) -> Dict[str, Any]:
        """
        Get live dashboard data with session context.
        
        Returns:
            Live dashboard data
        """
        dashboard_data = self.dashboard.get_dashboard_data()
        
        # Add integration-specific data
        dashboard_data['integration'] = {
            'active_session_id': self.active_session_id,
            'system_version': 'RIF-Orchestrator-v1.0',
            'database_path': self.db_path,
            'total_sessions': len(self.persistence.db.execute("SELECT session_id FROM orchestration_state").fetchall())
        }
        
        return dashboard_data
    
    def recover_session(self, session_id: str) -> bool:
        """
        Recover and resume a previous orchestration session.
        
        Args:
            session_id: Session ID to recover
            
        Returns:
            Recovery success status
        """
        try:
            # Attempt to recover state
            recovered_state = self.persistence.recover_state(session_id)
            if not recovered_state:
                return False
            
            # Set as active session
            self.active_session_id = session_id
            
            # Track recovery
            self.dashboard.track_metrics('session_recovery', {
                'session_id': session_id,
                'recovered_state': recovered_state['current_state']
            })
            
            return True
            
        except Exception as e:
            self.dashboard.track_metrics('recovery_error', {
                'session_id': session_id,
                'error': str(e)
            })
            return False
    
    def _summarize_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize decision history for the session summary."""
        if not decisions:
            return {'total': 0}
        
        successful_decisions = [d for d in decisions if d['success']]
        failed_decisions = [d for d in decisions if not d['success']]
        
        return {
            'total': len(decisions),
            'successful': len(successful_decisions),
            'failed': len(failed_decisions),
            'avg_confidence': sum(d['confidence_score'] for d in decisions) / len(decisions),
            'most_common_transition': self._find_most_common_transition(decisions),
            'longest_execution_ms': max((d['execution_time_ms'] for d in decisions), default=0),
            'shortest_execution_ms': min((d['execution_time_ms'] for d in decisions), default=0)
        }
    
    def _find_most_common_transition(self, decisions: List[Dict[str, Any]]) -> str:
        """Find the most common state transition."""
        if not decisions:
            return "none"
        
        transitions = {}
        for decision in decisions:
            transition = f"{decision['from_state']} → {decision['to_state']}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        if not transitions:
            return "none"
        
        return max(transitions, key=transitions.get)
    
    def close(self):
        """Close the integrated system and database connections."""
        if self.persistence:
            self.persistence.close()

# Convenience functions
def create_orchestrator_system(db_path: str = 'knowledge/orchestration.duckdb') -> IntegratedOrchestratorSystem:
    """Create a new integrated orchestrator system."""
    return IntegratedOrchestratorSystem(db_path)

def demo_orchestration_workflow():
    """Demonstrate complete orchestration workflow."""
    print("🚀 Starting RIF Orchestrator Demo")
    
    # Create system
    orchestrator = create_orchestrator_system(':memory:')  # Use memory DB for demo
    
    try:
        # Start session
        print("1. Starting orchestration session...")
        session_id = orchestrator.start_orchestration_session(
            workflow_type="demo_workflow",
            priority=1,
            context={
                'github_issues': [1, 2, 3],
                'project_type': 'python',
                'complexity': 'medium'
            }
        )
        print(f"   ✅ Session started: {session_id}")
        
        # Simulate workflow progression
        workflow_steps = [
            ('initialized', 'analyzing', 'Starting issue analysis', ['RIF-Analyst']),
            ('analyzing', 'planning', 'Analysis complete', ['RIF-Planner']),
            ('planning', 'implementing', 'Plan approved', ['RIF-Implementer']),
            ('implementing', 'validating', 'Implementation complete', ['RIF-Validator']),
            ('validating', 'completed', 'All tests passed', ['RIF-Learner'])
        ]
        
        for from_state, to_state, reason, agents in workflow_steps:
            print(f"2. Transitioning {from_state} → {to_state}")
            
            # Launch agents
            assignments = orchestrator.launch_agents(agents, f"Execute {to_state} phase")
            print(f"   ✅ Launched agents: {list(assignments.keys())}")
            
            # Transition state
            success = orchestrator.transition_state(
                from_state, to_state, reason, agents,
                context_updates={'last_step': to_state},
                confidence_score=0.9
            )
            print(f"   ✅ Transition successful: {success}")
        
        # Complete session
        print("3. Completing orchestration session...")
        summary = orchestrator.complete_orchestration_session(
            outcome="completed",
            final_context={'demo_completed': True, 'success': True}
        )
        print(f"   ✅ Session completed")
        
        # Generate dashboard data
        print("4. Generating dashboard data...")
        dashboard_data = orchestrator.get_live_dashboard()
        print(f"   ✅ Dashboard generated with {len(dashboard_data)} sections")
        
        # Print summary
        print("\n📊 WORKFLOW SUMMARY")
        print(f"   Session ID: {summary['session_id']}")
        print(f"   Workflow Type: {summary['workflow_type']}")
        print(f"   Total Duration: {summary['total_duration_ms']:.0f}ms")
        print(f"   Total Decisions: {summary['total_decisions']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   States Visited: {summary['total_states_visited']}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        orchestrator.close()

if __name__ == "__main__":
    demo_orchestration_workflow()