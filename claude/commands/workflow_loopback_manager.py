#!/usr/bin/env python3
"""
RIF Workflow Loop-back Manager
Issue #53: Create workflow loop-back mechanism

This module provides intelligent loop-back detection, context preservation,
and state rollback capabilities for the RIF orchestration system.
"""

import json
import uuid
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from pathlib import Path

try:
    from orchestrator_state_persistence import OrchestratorStatePersistence
except ImportError:
    # Fallback for testing
    OrchestratorStatePersistence = None


class ValidationResult:
    """
    Standardized validation result structure for loop-back decision making.
    """
    
    def __init__(self, success: bool = True, details: str = ""):
        self.success = success
        self.details = details
        self.missing_requirements = []
        self.architectural_flaws = []
        self.implementation_errors = []
        self.test_failures = []
        self.quality_gate_failures = []
        self.timestamp = datetime.now().isoformat()
    
    def add_missing_requirement(self, requirement: str, severity: str = "medium"):
        """Add a missing requirement to the validation result."""
        self.missing_requirements.append({
            'requirement': requirement,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        self.success = False
    
    def add_architectural_flaw(self, flaw: str, impact: str = "medium"):
        """Add an architectural flaw to the validation result."""
        self.architectural_flaws.append({
            'flaw': flaw,
            'impact': impact,
            'timestamp': datetime.now().isoformat()
        })
        self.success = False
    
    def add_implementation_error(self, error: str, location: str = ""):
        """Add an implementation error to the validation result."""
        self.implementation_errors.append({
            'error': error,
            'location': location,
            'timestamp': datetime.now().isoformat()
        })
        self.success = False
    
    def add_test_failure(self, test_name: str, error_message: str):
        """Add a test failure to the validation result."""
        self.test_failures.append({
            'test_name': test_name,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        })
        self.success = False
    
    def add_quality_gate_failure(self, gate: str, expected: str, actual: str):
        """Add a quality gate failure to the validation result."""
        self.quality_gate_failures.append({
            'gate': gate,
            'expected': expected,
            'actual': actual,
            'timestamp': datetime.now().isoformat()
        })
        self.success = False
    
    def get_loop_back_priority(self) -> List[Tuple[str, str]]:
        """
        Determine loop-back priority based on validation failures.
        
        Returns:
            List of (target_state, reason) tuples in priority order
        """
        loop_backs = []
        
        # Priority 1: Missing requirements need analysis
        if self.missing_requirements:
            loop_backs.append(('analyzing', f'Missing requirements: {len(self.missing_requirements)} items'))
        
        # Priority 2: Architectural flaws need architecture phase
        if self.architectural_flaws:
            loop_backs.append(('architecting', f'Architectural flaws: {len(self.architectural_flaws)} issues'))
        
        # Priority 3: Implementation errors need implementation phase
        if self.implementation_errors or self.test_failures:
            error_count = len(self.implementation_errors) + len(self.test_failures)
            loop_backs.append(('implementing', f'Implementation issues: {error_count} errors'))
        
        # Priority 4: Quality gates might need various phases
        if self.quality_gate_failures:
            loop_backs.append(('implementing', f'Quality gate failures: {len(self.quality_gate_failures)} gates'))
        
        return loop_backs


class MaxLoopsExceeded(Exception):
    """Exception raised when maximum loop limit is exceeded."""
    pass


class WorkflowLoopbackManager:
    """
    Intelligent workflow loop-back detection and execution system.
    
    Provides context-aware loop-back detection, state rollback capabilities,
    and infinite loop prevention with configurable limits.
    """
    
    def __init__(self, max_loops: int = 3, persistence_system: Optional[OrchestratorStatePersistence] = None):
        """
        Initialize the loop-back manager.
        
        Args:
            max_loops: Maximum number of loops allowed per transition
            persistence_system: Optional persistence system for state management
        """
        self.max_loops = max_loops
        self.loop_counts = defaultdict(int)
        self.context_snapshots = {}
        self.loop_history = []
        self.persistence = persistence_system
        
        # Performance tracking
        self.metrics = {
            'total_loop_backs': 0,
            'successful_recoveries': 0,
            'max_loops_exceeded': 0,
            'context_preservation_failures': 0,
            'avg_rollback_time_ms': 0.0
        }
    
    def should_loop_back(self, validation_results: ValidationResult, current_state: str) -> Optional[Tuple[str, str]]:
        """
        Determine if workflow should loop back based on validation results.
        
        Args:
            validation_results: Results from validation phase
            current_state: Current workflow state
            
        Returns:
            (target_state, reason) tuple if loop-back needed, None otherwise
        """
        if validation_results.success:
            return None
        
        # Get prioritized loop-back options
        loop_back_priorities = validation_results.get_loop_back_priority()
        if not loop_back_priorities:
            return None
        
        # Select the highest priority loop-back that doesn't exceed limits
        for target_state, reason in loop_back_priorities:
            loop_key = f"{current_state}->{target_state}"
            
            if self.loop_counts[loop_key] < self.max_loops:
                return target_state, reason
        
        # All potential loop-backs would exceed limits
        return None
    
    def execute_loopback(self, current_state: str, target_state: str, context: Dict[str, Any], 
                        reason: str = "Validation failure") -> Dict[str, Any]:
        """
        Execute loop-back with context preservation and integrity validation.
        
        Args:
            current_state: Current workflow state
            target_state: Target state for loop-back
            context: Current workflow context
            reason: Reason for the loop-back
            
        Returns:
            Updated context with loop-back information
            
        Raises:
            MaxLoopsExceeded: If maximum loop limit would be exceeded
        """
        start_time = time.time()
        loop_key = f"{current_state}->{target_state}"
        
        # Check loop limit
        if self.loop_counts[loop_key] >= self.max_loops:
            self.metrics['max_loops_exceeded'] += 1
            raise MaxLoopsExceeded(
                f"Maximum loops ({self.max_loops}) exceeded for transition {loop_key}. "
                f"Current count: {self.loop_counts[loop_key]}"
            )
        
        try:
            # Increment loop count
            self.loop_counts[loop_key] += 1
            
            # Create context snapshot for preservation
            snapshot_id = self._create_context_snapshot(context, current_state, target_state)
            
            # Preserve important context information
            preserved_context = self._preserve_context(context, current_state, target_state, reason)
            
            # Execute state rollback with integrity validation
            rollback_result = self._rollback_to_state(target_state, preserved_context)
            
            # Record loop-back in history
            loop_record = {
                'loop_id': str(uuid.uuid4()),
                'from_state': current_state,
                'to_state': target_state,
                'reason': reason,
                'loop_count': self.loop_counts[loop_key],
                'snapshot_id': snapshot_id,
                'timestamp': datetime.now().isoformat(),
                'execution_time_ms': int((time.time() - start_time) * 1000)
            }
            
            self.loop_history.append(loop_record)
            
            # Update metrics
            self.metrics['total_loop_backs'] += 1
            if rollback_result.get('success', False):
                self.metrics['successful_recoveries'] += 1
            
            # Update average rollback time
            total_time = (self.metrics['avg_rollback_time_ms'] * (self.metrics['total_loop_backs'] - 1) + 
                         loop_record['execution_time_ms'])
            self.metrics['avg_rollback_time_ms'] = total_time / self.metrics['total_loop_backs']
            
            # Persist loop-back information if persistence system available
            if self.persistence:
                self.persistence.record_decision(
                    current_state, target_state, f"Loop-back: {reason}",
                    [], confidence_score=0.8, 
                    execution_time_ms=loop_record['execution_time_ms'],
                    success=rollback_result.get('success', True)
                )
            
            return rollback_result['context']
            
        except Exception as e:
            self.metrics['context_preservation_failures'] += 1
            raise RuntimeError(f"Loop-back execution failed: {str(e)}")
    
    def _create_context_snapshot(self, context: Dict[str, Any], current_state: str, 
                                target_state: str) -> str:
        """
        Create a snapshot of the current context for recovery purposes.
        
        Args:
            context: Current workflow context
            current_state: Current workflow state
            target_state: Target state for loop-back
            
        Returns:
            Snapshot ID for later retrieval
        """
        snapshot_id = str(uuid.uuid4())
        
        # Create complete context snapshot with integrity validation
        # Clean context to make it JSON serializable
        serializable_context = self._make_context_serializable(context.copy())
        
        snapshot = {
            'snapshot_id': snapshot_id,
            'timestamp': datetime.now().isoformat(),
            'from_state': current_state,
            'to_state': target_state,
            'context': serializable_context,
            'checksum': self._calculate_context_checksum(serializable_context)
        }
        
        self.context_snapshots[snapshot_id] = snapshot
        
        # Limit snapshot storage to prevent memory growth
        if len(self.context_snapshots) > 100:
            # Remove oldest snapshots
            oldest_snapshots = sorted(self.context_snapshots.keys())[:20]
            for old_id in oldest_snapshots:
                del self.context_snapshots[old_id]
        
        return snapshot_id
    
    def _preserve_context(self, context: Dict[str, Any], current_state: str, 
                         target_state: str, reason: str) -> Dict[str, Any]:
        """
        Preserve essential context information during loop-back.
        
        Args:
            context: Original workflow context
            current_state: Current workflow state
            target_state: Target state for loop-back
            reason: Reason for loop-back
            
        Returns:
            Enhanced context with preserved information
        """
        preserved_context = context.copy()
        
        # Add loop-back metadata
        loop_metadata = {
            'loop_back_reason': reason,
            'loop_from_state': current_state,
            'previous_attempts': self.loop_counts.get(f"{current_state}->{target_state}", 0),
            'loop_timestamp': datetime.now().isoformat()
        }
        
        # Preserve validation feedback if present (convert to serializable format)
        if 'validation_results' in context:
            validation_results = context['validation_results']
            if hasattr(validation_results, '__dict__'):
                # Convert ValidationResult object to dictionary
                loop_metadata['validation_feedback'] = {
                    'success': validation_results.success,
                    'details': validation_results.details,
                    'missing_requirements': validation_results.missing_requirements,
                    'architectural_flaws': validation_results.architectural_flaws,
                    'implementation_errors': validation_results.implementation_errors,
                    'test_failures': getattr(validation_results, 'test_failures', []),
                    'quality_gate_failures': getattr(validation_results, 'quality_gate_failures', []),
                    'timestamp': getattr(validation_results, 'timestamp', datetime.now().isoformat())
                }
            else:
                loop_metadata['validation_feedback'] = validation_results
        
        # Preserve accumulated learning and patterns
        learning_keys = ['patterns_discovered', 'lessons_learned', 'performance_metrics', 
                        'quality_insights', 'agent_performance']
        for key in learning_keys:
            if key in context:
                loop_metadata[f'accumulated_{key}'] = context[key]
        
        # Add loop-back metadata to context
        if 'loop_back_metadata' not in preserved_context:
            preserved_context['loop_back_metadata'] = []
        preserved_context['loop_back_metadata'].append(loop_metadata)
        
        # Update retry count for compatibility with existing systems
        preserved_context['retry_count'] = preserved_context.get('retry_count', 0) + 1
        
        return preserved_context
    
    def _rollback_to_state(self, target_state: str, preserved_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute state rollback with integrity validation.
        
        Args:
            target_state: Target state to roll back to
            preserved_context: Context with preserved information
            
        Returns:
            Rollback result with success status and updated context
        """
        try:
            # Validate context integrity
            if not self._validate_context_integrity(preserved_context):
                return {
                    'success': False,
                    'error': 'Context integrity validation failed',
                    'context': preserved_context
                }
            
            # Clean up state-specific context that should not persist
            cleaned_context = self._clean_state_specific_context(preserved_context, target_state)
            
            # Add rollback tracking
            cleaned_context['rollback_history'] = cleaned_context.get('rollback_history', [])
            cleaned_context['rollback_history'].append({
                'target_state': target_state,
                'timestamp': datetime.now().isoformat(),
                'context_checksum': self._calculate_context_checksum(cleaned_context)
            })
            
            return {
                'success': True,
                'target_state': target_state,
                'context': cleaned_context,
                'integrity_validated': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'context': preserved_context
            }
    
    def _clean_state_specific_context(self, context: Dict[str, Any], target_state: str) -> Dict[str, Any]:
        """
        Clean context of state-specific information that shouldn't persist in rollback.
        
        Args:
            context: Context to clean
            target_state: Target state being rolled back to
            
        Returns:
            Cleaned context appropriate for target state
        """
        cleaned_context = context.copy()
        
        # Define state-specific keys that should be removed during rollback
        state_specific_keys = {
            'analyzing': ['implementation_plan', 'architecture_design', 'code_artifacts'],
            'planning': ['architecture_design', 'code_artifacts', 'test_results'],
            'architecting': ['code_artifacts', 'test_results', 'validation_results'],
            'implementing': ['test_results', 'validation_results'],
            'validating': []  # Keep most context for validation retry
        }
        
        keys_to_remove = state_specific_keys.get(target_state, [])
        for key in keys_to_remove:
            cleaned_context.pop(key, None)
        
        return cleaned_context
    
    def _make_context_serializable(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert context to JSON serializable format.
        
        Args:
            context: Context dictionary that may contain non-serializable objects
            
        Returns:
            JSON serializable context dictionary
        """
        serializable_context = {}
        
        for key, value in context.items():
            try:
                if hasattr(value, '__dict__') and hasattr(value, 'success'):
                    # Likely a ValidationResult object
                    serializable_context[key] = {
                        'success': value.success,
                        'details': getattr(value, 'details', ''),
                        'missing_requirements': getattr(value, 'missing_requirements', []),
                        'architectural_flaws': getattr(value, 'architectural_flaws', []),
                        'implementation_errors': getattr(value, 'implementation_errors', []),
                        'test_failures': getattr(value, 'test_failures', []),
                        'quality_gate_failures': getattr(value, 'quality_gate_failures', []),
                        'timestamp': getattr(value, 'timestamp', datetime.now().isoformat())
                    }
                else:
                    # Try to serialize directly
                    json.dumps(value)
                    serializable_context[key] = value
            except (TypeError, ValueError):
                # If not serializable, convert to string representation
                serializable_context[key] = str(value)
        
        return serializable_context
    
    def _calculate_context_checksum(self, context: Dict[str, Any]) -> str:
        """
        Calculate integrity checksum for context data.
        
        Args:
            context: Context dictionary to checksum
            
        Returns:
            SHA-256 checksum hex string
        """
        # Create deterministic JSON representation for checksumming
        context_json = json.dumps(context, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(context_json.encode()).hexdigest()
    
    def _validate_context_integrity(self, context: Dict[str, Any]) -> bool:
        """
        Validate context integrity using checksums if available.
        
        Args:
            context: Context to validate
            
        Returns:
            True if integrity is valid
        """
        # For now, perform basic structure validation
        # In production, this could check against stored checksums
        required_keys = ['workflow_type']
        
        for key in required_keys:
            if key not in context:
                return False
        
        # Validate JSON serializable
        try:
            json.dumps(context)
            return True
        except (TypeError, ValueError):
            return False
    
    def get_loop_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive loop-back statistics and metrics.
        
        Returns:
            Dictionary with loop-back statistics
        """
        # Calculate loop patterns
        loop_patterns = defaultdict(int)
        for loop_key, count in self.loop_counts.items():
            from_state, to_state = loop_key.split('->')
            loop_patterns[f"{from_state}_to_{to_state}"] = count
        
        # Calculate success rates
        total_loops = self.metrics['total_loop_backs']
        success_rate = (self.metrics['successful_recoveries'] / total_loops) if total_loops > 0 else 0.0
        
        return {
            'total_loop_backs': total_loops,
            'successful_recoveries': self.metrics['successful_recoveries'],
            'success_rate': success_rate,
            'max_loops_exceeded_count': self.metrics['max_loops_exceeded'],
            'context_preservation_failures': self.metrics['context_preservation_failures'],
            'avg_rollback_time_ms': self.metrics['avg_rollback_time_ms'],
            'loop_patterns': dict(loop_patterns),
            'current_loop_counts': dict(self.loop_counts),
            'active_snapshots': len(self.context_snapshots),
            'loop_history_count': len(self.loop_history),
            'max_loops_limit': self.max_loops
        }
    
    def reset_loop_counts(self, loop_key: Optional[str] = None):
        """
        Reset loop counts (useful for testing or administrative reset).
        
        Args:
            loop_key: Specific loop key to reset, or None for all
        """
        if loop_key:
            self.loop_counts.pop(loop_key, None)
        else:
            self.loop_counts.clear()
    
    def get_context_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a context snapshot by ID.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            Snapshot dictionary or None if not found
        """
        return self.context_snapshots.get(snapshot_id)
    
    def cleanup_old_snapshots(self, max_age_hours: int = 24):
        """
        Clean up old context snapshots to manage memory usage.
        
        Args:
            max_age_hours: Maximum age in hours for snapshot retention
        """
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (max_age_hours * 3600)
        
        snapshot_ids_to_remove = []
        for snapshot_id, snapshot in self.context_snapshots.items():
            snapshot_time = datetime.fromisoformat(snapshot['timestamp']).timestamp()
            if snapshot_time < cutoff_time:
                snapshot_ids_to_remove.append(snapshot_id)
        
        for snapshot_id in snapshot_ids_to_remove:
            del self.context_snapshots[snapshot_id]


# Utility functions and test helpers
def create_test_validation_result() -> ValidationResult:
    """Create a test validation result with various failure types."""
    result = ValidationResult(success=False)
    result.add_missing_requirement("User authentication specification", "high")
    result.add_architectural_flaw("Database connection pooling not implemented", "medium")
    result.add_implementation_error("Null pointer exception in user service", "src/user_service.py:123")
    result.add_test_failure("test_user_login", "Expected 200, got 500")
    result.add_quality_gate_failure("code_coverage", "80%", "65%")
    return result


def demo_workflow_loopback():
    """Demonstrate workflow loop-back functionality."""
    print("🔄 Starting Workflow Loop-back Demo")
    
    # Create loop-back manager
    loop_manager = WorkflowLoopbackManager(max_loops=3)
    
    print("1. Testing loop-back detection...")
    
    # Create test validation result
    validation_result = create_test_validation_result()
    
    # Test loop-back decision
    loop_back_decision = loop_manager.should_loop_back(validation_result, 'validating')
    if loop_back_decision:
        target_state, reason = loop_back_decision
        print(f"   Loop-back needed: {target_state} (Reason: {reason})")
    else:
        print("   No loop-back needed")
    
    print("2. Testing context preservation and rollback...")
    
    # Test context with various data
    test_context = {
        'workflow_type': 'demo',
        'github_issues': [53],
        'complexity': 'medium',
        'agent_performance': {'RIF-Validator': 0.85},
        'validation_results': validation_result,
        'patterns_discovered': ['retry_pattern', 'validation_pattern']
    }
    
    if loop_back_decision:
        try:
            # Execute loop-back
            updated_context = loop_manager.execute_loopback(
                'validating', target_state, test_context,
                f"Demo loop-back: {reason}"
            )
            
            print(f"   ✅ Loop-back successful to state: {target_state}")
            print(f"   Context preserved: {len(updated_context)} keys")
            
            # Check preserved metadata
            if 'loop_back_metadata' in updated_context:
                metadata_count = len(updated_context['loop_back_metadata'])
                print(f"   Loop-back metadata: {metadata_count} entries")
        
        except MaxLoopsExceeded as e:
            print(f"   ⚠️ Max loops exceeded: {e}")
        
        except Exception as e:
            print(f"   ❌ Loop-back failed: {e}")
    
    print("3. Testing loop limits...")
    
    # Test multiple loop-backs to trigger limit
    for i in range(5):
        try:
            updated_context = loop_manager.execute_loopback(
                'implementing', 'analyzing', test_context,
                f"Test loop {i+1}"
            )
            print(f"   Loop {i+1}: Success")
        except MaxLoopsExceeded as e:
            print(f"   Loop {i+1}: Max loops exceeded (expected)")
            break
        except Exception as e:
            print(f"   Loop {i+1}: Error - {e}")
            break
    
    print("4. Performance statistics...")
    stats = loop_manager.get_loop_statistics()
    print(f"   Total loop-backs: {stats['total_loop_backs']}")
    print(f"   Success rate: {stats['success_rate']:.2f}")
    print(f"   Avg rollback time: {stats['avg_rollback_time_ms']:.1f}ms")
    print(f"   Active snapshots: {stats['active_snapshots']}")
    
    print("✅ Workflow Loop-back Demo completed successfully!")


if __name__ == "__main__":
    demo_workflow_loopback()