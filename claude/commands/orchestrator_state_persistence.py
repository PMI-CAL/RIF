#!/usr/bin/env python3
"""
RIF Orchestrator State Persistence System
Issue #55: Implement orchestrator state persistence

This module provides comprehensive state persistence for the RIF orchestrator,
enabling recovery from interruptions and maintaining complete decision history.
"""

import json
import uuid
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import defaultdict

try:
    import duckdb
except ImportError:
    print("DuckDB not available - install with: pip install duckdb")
    duckdb = None

class OrchestratorStatePersistence:
    """
    Manages orchestrator state persistence with DuckDB backend.
    Provides state serialization, decision history, and recovery mechanisms.
    """
    
    def __init__(self, db_path: str = 'knowledge/orchestration.duckdb'):
        """Initialize the persistence system with database connection."""
        self.db_path = db_path
        self.db = None
        self.session_id = None
        self._connect_db()
        self._create_schema()
    
    def _connect_db(self):
        """Establish database connection with error handling."""
        if duckdb is None:
            raise RuntimeError("DuckDB not available - cannot initialize persistence")
        
        try:
            # Ensure the directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = duckdb.connect(self.db_path)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {e}")
    
    def _create_schema(self):
        """Create the database schema for orchestration persistence."""
        try:
            # Orchestration state table - stores current state snapshots
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS orchestration_state (
                    session_id VARCHAR PRIMARY KEY,
                    current_state VARCHAR NOT NULL,
                    context JSON NOT NULL,
                    history JSON NOT NULL,
                    agent_assignments JSON,
                    workflow_type VARCHAR,
                    priority INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Orchestration decisions table - tracks all decision points
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS orchestration_decisions (
                    decision_id VARCHAR PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    from_state VARCHAR NOT NULL,
                    to_state VARCHAR NOT NULL,
                    decision_reason TEXT,
                    agents_selected JSON,
                    confidence_score FLOAT DEFAULT 0.0,
                    execution_time_ms INTEGER,
                    success BOOLEAN DEFAULT TRUE,
                    error_details TEXT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (session_id) REFERENCES orchestration_state(session_id)
                )
            """)
            
            # Orchestration metrics table - performance and usage metrics
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS orchestration_metrics (
                    metric_id VARCHAR PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    metric_type VARCHAR NOT NULL,
                    metric_name VARCHAR NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metadata JSON,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (session_id) REFERENCES orchestration_state(session_id)
                )
            """)
            
            # Create indexes for performance
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_state_session ON orchestration_state(session_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_session ON orchestration_decisions(session_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON orchestration_decisions(timestamp)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON orchestration_metrics(session_id)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create database schema: {e}")
    
    def start_session(self, workflow_type: str = "standard", priority: int = 0) -> str:
        """
        Start a new orchestration session.
        
        Args:
            workflow_type: Type of workflow being orchestrated
            priority: Session priority level
            
        Returns:
            session_id: Unique identifier for this orchestration session
        """
        self.session_id = str(uuid.uuid4())
        
        initial_state = {
            'session_id': self.session_id,
            'current_state': 'initialized',
            'context': json.dumps({
                'workflow_type': workflow_type,
                'start_time': datetime.now().isoformat(),
                'github_issues': [],
                'active_agents': []
            }),
            'history': json.dumps([{
                'state': 'initialized',
                'timestamp': datetime.now().isoformat(),
                'reason': f'Session started with workflow type: {workflow_type}'
            }]),
            'agent_assignments': json.dumps({}),
            'workflow_type': workflow_type,
            'priority': priority
        }
        
        try:
            self.db.execute(
                """INSERT INTO orchestration_state 
                   (session_id, current_state, context, history, agent_assignments, workflow_type, priority, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NOW(), NOW())""",
                [
                    initial_state['session_id'],
                    initial_state['current_state'],
                    initial_state['context'],
                    initial_state['history'],
                    initial_state['agent_assignments'],
                    initial_state['workflow_type'],
                    initial_state['priority']
                ]
            )
            return self.session_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to start session: {e}")
    
    def save_state(self, orchestrator_state: Dict[str, Any]) -> bool:
        """
        Persist the current orchestrator state.
        
        Args:
            orchestrator_state: Complete orchestrator state dictionary
            
        Returns:
            bool: Success status
        """
        if not self.session_id:
            raise RuntimeError("No active session - call start_session() first")
        
        try:
            # Validate required fields
            required_fields = ['current_state', 'context', 'history']
            for field in required_fields:
                if field not in orchestrator_state:
                    raise ValueError(f"Missing required field: {field}")
            
            # Serialize complex objects
            context_json = json.dumps(orchestrator_state['context'])
            history_json = json.dumps(orchestrator_state['history'])
            agent_assignments_json = json.dumps(orchestrator_state.get('agent_assignments', {}))
            
            # Update the database
            self.db.execute(
                """UPDATE orchestration_state 
                   SET current_state = ?, context = ?, history = ?, 
                       agent_assignments = ?, updated_at = NOW()
                   WHERE session_id = ?""",
                [
                    orchestrator_state['current_state'],
                    context_json,
                    history_json,
                    agent_assignments_json,
                    self.session_id
                ]
            )
            
            return True
            
        except Exception as e:
            print(f"Error saving orchestrator state: {e}")
            return False
    
    def record_decision(self, from_state: str, to_state: str, reason: str, 
                       agents_selected: List[str], confidence_score: float = 1.0,
                       execution_time_ms: int = 0, success: bool = True, 
                       error_details: Optional[str] = None) -> str:
        """
        Record an orchestration decision in the history.
        
        Args:
            from_state: Starting state
            to_state: Target state
            reason: Reason for the decision
            agents_selected: List of agents selected for execution
            confidence_score: Confidence in the decision (0.0-1.0)
            execution_time_ms: Execution time in milliseconds
            success: Whether the decision was successful
            error_details: Error details if unsuccessful
            
        Returns:
            decision_id: Unique identifier for this decision
        """
        decision_id = str(uuid.uuid4())
        
        try:
            self.db.execute(
                """INSERT INTO orchestration_decisions 
                   (decision_id, session_id, from_state, to_state, decision_reason, 
                    agents_selected, confidence_score, execution_time_ms, success, 
                    error_details, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())""",
                [
                    decision_id,
                    self.session_id,
                    from_state,
                    to_state,
                    reason,
                    json.dumps(agents_selected),
                    confidence_score,
                    execution_time_ms,
                    success,
                    error_details
                ]
            )
            
            return decision_id
            
        except Exception as e:
            print(f"Error recording decision: {e}")
            return ""
    
    def record_metric(self, metric_type: str, metric_name: str, value: float, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a performance or usage metric.
        
        Args:
            metric_type: Type of metric (performance, usage, quality, etc.)
            metric_name: Name of the specific metric
            value: Metric value
            metadata: Additional metric metadata
            
        Returns:
            metric_id: Unique identifier for this metric
        """
        metric_id = str(uuid.uuid4())
        
        try:
            self.db.execute(
                """INSERT INTO orchestration_metrics 
                   (metric_id, session_id, metric_type, metric_name, metric_value, 
                    metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, NOW())""",
                [
                    metric_id,
                    self.session_id,
                    metric_type,
                    metric_name,
                    value,
                    json.dumps(metadata or {})
                ]
            )
            
            return metric_id
            
        except Exception as e:
            print(f"Error recording metric: {e}")
            return ""
    
    def recover_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Recover orchestrator state from persistence.
        
        Args:
            session_id: Session to recover
            
        Returns:
            Orchestrator state dictionary or None if not found
        """
        try:
            result = self.db.execute(
                """SELECT session_id, current_state, context, history, 
                          agent_assignments, workflow_type, priority,
                          created_at, updated_at
                   FROM orchestration_state 
                   WHERE session_id = ?""",
                [session_id]
            ).fetchone()
            
            if not result:
                return None
            
            return self._reconstruct_orchestrator_state(result)
            
        except Exception as e:
            print(f"Error recovering state: {e}")
            return None
    
    def _reconstruct_orchestrator_state(self, db_result) -> Dict[str, Any]:
        """
        Reconstruct orchestrator state from database result.
        
        Args:
            db_result: Database query result tuple
            
        Returns:
            Reconstructed orchestrator state
        """
        (session_id, current_state, context_json, history_json, 
         agent_assignments_json, workflow_type, priority, created_at, updated_at) = db_result
        
        return {
            'session_id': session_id,
            'current_state': current_state,
            'context': json.loads(context_json) if context_json else {},
            'history': json.loads(history_json) if history_json else [],
            'agent_assignments': json.loads(agent_assignments_json) if agent_assignments_json else {},
            'workflow_type': workflow_type,
            'priority': priority,
            'created_at': str(created_at),
            'updated_at': str(updated_at)
        }
    
    def get_decision_history(self, session_id: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get decision history for analysis.
        
        Args:
            session_id: Specific session ID, or None for current session
            limit: Maximum number of decisions to return
            
        Returns:
            List of decision records
        """
        target_session = session_id or self.session_id
        if not target_session:
            return []
        
        try:
            results = self.db.execute(
                """SELECT decision_id, session_id, from_state, to_state, 
                          decision_reason, agents_selected, confidence_score,
                          execution_time_ms, success, error_details, timestamp
                   FROM orchestration_decisions 
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                [target_session, limit]
            ).fetchall()
            
            return [
                {
                    'decision_id': row[0],
                    'session_id': row[1],
                    'from_state': row[2],
                    'to_state': row[3],
                    'reason': row[4],
                    'agents_selected': json.loads(row[5]) if row[5] else [],
                    'confidence_score': row[6],
                    'execution_time_ms': row[7],
                    'success': row[8],
                    'error_details': row[9],
                    'timestamp': str(row[10])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"Error retrieving decision history: {e}")
            return []
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of active orchestration sessions.
        
        Returns:
            List of active session information
        """
        try:
            results = self.db.execute(
                """SELECT session_id, current_state, workflow_type, priority,
                          created_at, updated_at
                   FROM orchestration_state
                   WHERE current_state NOT IN ('completed', 'failed', 'cancelled')
                   ORDER BY priority DESC, created_at ASC"""
            ).fetchall()
            
            return [
                {
                    'session_id': row[0],
                    'current_state': row[1],
                    'workflow_type': row[2],
                    'priority': row[3],
                    'created_at': str(row[4]),
                    'updated_at': str(row[5])
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"Error retrieving active sessions: {e}")
            return []
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up old completed sessions to manage database size.
        
        Args:
            days_old: Age threshold for cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            # Delete old completed sessions and their related data
            result = self.db.execute(
                """DELETE FROM orchestration_state 
                   WHERE current_state IN ('completed', 'failed', 'cancelled')
                   AND updated_at < (NOW() - INTERVAL ? DAYS)""",
                [days_old]
            )
            
            return result.rowcount if hasattr(result, 'rowcount') else 0
            
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0
    
    def validate_state_integrity(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate the integrity of stored state data.
        
        Args:
            session_id: Specific session to validate, or None for all
            
        Returns:
            Validation report
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sessions_checked': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            query = "SELECT session_id, context, history, agent_assignments FROM orchestration_state"
            params = []
            
            if session_id:
                query += " WHERE session_id = ?"
                params.append(session_id)
            
            results = self.db.execute(query, params).fetchall()
            validation_report['sessions_checked'] = len(results)
            
            for row in results:
                session_id, context, history, agent_assignments = row
                
                # Validate JSON parsing
                try:
                    json.loads(context) if context else {}
                except json.JSONDecodeError as e:
                    validation_report['valid'] = False
                    validation_report['errors'].append(f"Invalid context JSON in session {session_id}: {e}")
                
                try:
                    json.loads(history) if history else []
                except json.JSONDecodeError as e:
                    validation_report['valid'] = False
                    validation_report['errors'].append(f"Invalid history JSON in session {session_id}: {e}")
                
                try:
                    json.loads(agent_assignments) if agent_assignments else {}
                except json.JSONDecodeError as e:
                    validation_report['valid'] = False
                    validation_report['errors'].append(f"Invalid agent_assignments JSON in session {session_id}: {e}")
            
            return validation_report
            
        except Exception as e:
            validation_report['valid'] = False
            validation_report['errors'].append(f"Validation error: {e}")
            return validation_report
    
    def get_performance_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for orchestration sessions.
        
        Args:
            session_id: Specific session, or None for all sessions
            
        Returns:
            Performance statistics
        """
        try:
            base_query = """
                SELECT 
                    COUNT(*) as total_decisions,
                    AVG(execution_time_ms) as avg_execution_time,
                    MAX(execution_time_ms) as max_execution_time,
                    MIN(execution_time_ms) as min_execution_time,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_decisions,
                    AVG(confidence_score) as avg_confidence
                FROM orchestration_decisions
            """
            
            if session_id:
                base_query += " WHERE session_id = ?"
                params = [session_id]
            else:
                params = []
            
            result = self.db.execute(base_query, params).fetchone()
            
            if not result or result[0] == 0:
                return {'no_data': True}
            
            total, avg_time, max_time, min_time, successful, avg_confidence = result
            
            return {
                'total_decisions': total,
                'success_rate': successful / total if total > 0 else 0,
                'avg_execution_time_ms': float(avg_time) if avg_time else 0,
                'max_execution_time_ms': max_time,
                'min_execution_time_ms': min_time,
                'avg_confidence_score': float(avg_confidence) if avg_confidence else 0,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()
            self.db = None

# Utility functions for integration
def create_persistence_system(db_path: str = 'knowledge/orchestration.duckdb') -> OrchestratorStatePersistence:
    """Create and initialize a persistence system instance."""
    return OrchestratorStatePersistence(db_path)

def test_persistence_system() -> bool:
    """Test the persistence system functionality."""
    try:
        # Create test persistence system
        persistence = OrchestratorStatePersistence(':memory:')  # Use in-memory DB for testing
        
        # Test session creation
        session_id = persistence.start_session('test_workflow', priority=1)
        assert session_id is not None
        
        # Test state saving
        test_state = {
            'current_state': 'testing',
            'context': {'test': 'data'},
            'history': [{'action': 'test', 'timestamp': datetime.now().isoformat()}],
            'agent_assignments': {'test_agent': 'active'}
        }
        success = persistence.save_state(test_state)
        assert success
        
        # Test decision recording
        decision_id = persistence.record_decision(
            'initialized', 'testing', 'Test decision', ['test_agent'], 0.9
        )
        assert decision_id != ""
        
        # Test state recovery
        recovered_state = persistence.recover_state(session_id)
        assert recovered_state is not None
        assert recovered_state['current_state'] == 'testing'
        
        # Test decision history
        history = persistence.get_decision_history(session_id)
        assert len(history) == 1
        assert history[0]['from_state'] == 'initialized'
        
        persistence.close()
        print("✅ Persistence system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Persistence system test failed: {e}")
        return False

if __name__ == "__main__":
    # Run basic tests
    test_persistence_system()