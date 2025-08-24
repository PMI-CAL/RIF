#!/usr/bin/env python3
"""
RIF Shadow Integration Bridge - Emergency Fix for Issue #147
Connects RIF workflow states to shadow creation triggers.
Phase 1 Implementation - RIF Orchestration Integration
"""

import json
import subprocess
import sys
import yaml
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

class RIFShadowIntegration:
    """
    Bridges RIF orchestration workflow states with shadow quality tracking system.
    Automatically creates shadows when issues reach critical states or meet trigger conditions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the RIF shadow integration bridge."""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/shadow-quality-tracking.yaml"
        self.rif_workflow_path = "/Users/cal/DEV/RIF/config/rif-workflow.yaml"
        self.shadow_monitor_path = "/Users/cal/DEV/RIF/systems/shadow_issue_monitor.py"
        
        # Load configurations
        self.shadow_config = self._load_shadow_config()
        self.rif_config = self._load_rif_workflow_config()
        
        # State tracking
        self.issue_states: Dict[int, str] = {}
        self.last_check_time = None
        
    def _load_shadow_config(self) -> Dict[str, Any]:
        """Load shadow quality tracking configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load shadow config: {e}")
            return {}
    
    def _load_rif_workflow_config(self) -> Dict[str, Any]:
        """Load RIF workflow configuration."""
        try:
            with open(self.rif_workflow_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load RIF workflow config: {e}")
            return {}
    
    def get_current_issue_states(self) -> Dict[int, Dict[str, Any]]:
        """Get current states of all open issues."""
        try:
            cmd = [
                'gh', 'issue', 'list',
                '--state', 'open',
                '--json', 'number,title,labels,createdAt,updatedAt',
                '--limit', '100'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Error getting GitHub issues: {result.stderr}")
                return {}
            
            issues = json.loads(result.stdout)
            current_states = {}
            
            for issue in issues:
                issue_number = issue['number']
                labels = [label['name'] for label in issue.get('labels', [])]
                
                # Extract state, priority, and complexity
                current_state = None
                priority = None
                complexity = None
                
                for label in labels:
                    if label.startswith('state:'):
                        current_state = label.replace('state:', '')
                    elif label.startswith('priority:'):
                        priority = label.replace('priority:', '')
                    elif label.startswith('complexity:'):
                        complexity = label.replace('complexity:', '')
                
                current_states[issue_number] = {
                    'title': issue.get('title', ''),
                    'state': current_state,
                    'priority': priority,
                    'complexity': complexity,
                    'labels': labels,
                    'updated_at': issue.get('updatedAt', ''),
                    'created_at': issue.get('createdAt', '')
                }
            
            return current_states
            
        except Exception as e:
            print(f"Error getting issue states: {e}")
            return {}
    
    def detect_state_transitions(self, current_states: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect state transitions that should trigger shadow creation."""
        transitions = []
        
        for issue_number, current_info in current_states.items():
            current_state = current_info.get('state')
            previous_state = self.issue_states.get(issue_number)
            
            # Track state change
            if previous_state != current_state:
                transition = {
                    'issue_number': issue_number,
                    'title': current_info.get('title', ''),
                    'previous_state': previous_state,
                    'current_state': current_state,
                    'priority': current_info.get('priority'),
                    'complexity': current_info.get('complexity'),
                    'timestamp': datetime.now().isoformat(),
                    'should_create_shadow': False,
                    'reason': ''
                }
                
                # Check if this transition should trigger shadow creation
                shadow_triggers = self._check_shadow_triggers(current_info, transition)
                transition.update(shadow_triggers)
                
                transitions.append(transition)
                
                # Update tracked state
                self.issue_states[issue_number] = current_state
        
        return transitions
    
    def _check_shadow_triggers(self, issue_info: Dict[str, Any], transition: Dict[str, Any]) -> Dict[str, Any]:
        """Check if issue meets shadow creation trigger conditions."""
        reasons = []
        should_create = False
        
        current_state = issue_info.get('state')
        priority = issue_info.get('priority', '')
        complexity = issue_info.get('complexity', '')
        labels = issue_info.get('labels', [])
        
        # 1. State-based triggers
        shadow_trigger_states = ['validating', 'implementing']
        if current_state in shadow_trigger_states:
            reasons.append(f"Reached state: {current_state}")
            should_create = True
        
        # 2. Priority-based triggers
        if priority == 'critical':
            reasons.append("Critical priority")
            should_create = True
        
        # 3. Complexity-based triggers for medium+ priority
        high_complexity = complexity in ['high', 'very-high']
        if high_complexity and priority in ['high', 'medium']:
            reasons.append(f"High complexity ({complexity}) with {priority} priority")
            should_create = True
        
        # 4. Security/Agent-related triggers
        security_labels = [label for label in labels if 'security' in label.lower()]
        agent_labels = [label for label in labels if label.startswith('agent:')]
        
        if security_labels:
            reasons.append("Security-related issue")
            should_create = True
        
        if agent_labels:
            reasons.append("Agent system issue")
            should_create = True
        
        # 5. Special RIF system issues
        rif_keywords = ['rif-', 'shadow', 'orchestration', 'quality', 'dpibs']
        title_lower = issue_info.get('title', '').lower()
        if any(keyword in title_lower for keyword in rif_keywords):
            reasons.append("RIF system issue")
            should_create = True
        
        return {
            'should_create_shadow': should_create,
            'trigger_reasons': reasons,
            'reason': f"Triggers: {', '.join(reasons)}" if reasons else "No triggers met"
        }
    
    def create_shadow_for_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        """Create shadow issue for a state transition."""
        issue_number = transition['issue_number']
        
        try:
            # Use shadow monitor to create the shadow
            cmd = ['python3', self.shadow_monitor_path, 'backfill', str(issue_number)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Shadow creation failed: {result.stderr}",
                    'stdout': result.stdout
                }
            
            # Parse result
            try:
                results = json.loads(result.stdout)
                if results and len(results) > 0:
                    shadow_result = results[0]
                    return {
                        'success': shadow_result.get('shadow_creation', {}).get('success', False),
                        'shadow_issue_number': shadow_result.get('shadow_creation', {}).get('shadow_issue_number'),
                        'message': f"Shadow created for issue #{issue_number}",
                        'transition': transition
                    }
                else:
                    return {
                        'success': False,
                        'error': "No results returned from shadow creation"
                    }
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': f"Could not parse shadow creation result: {result.stdout}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Exception creating shadow: {e}"
            }
    
    def log_transition_to_main_issue(self, issue_number: int, transition: Dict[str, Any], shadow_result: Dict[str, Any]):
        """Log the state transition and shadow creation to the main issue."""
        try:
            if shadow_result.get('success', False):
                shadow_number = shadow_result.get('shadow_issue_number', 'unknown')
                message = f"""🔄 **RIF State Transition Detected**

**Issue**: #{issue_number}
**Transition**: {transition.get('previous_state', 'unknown')} → {transition.get('current_state', 'unknown')}
**Shadow Created**: #{shadow_number}
**Triggers**: {transition.get('reason', 'Unknown')}

Shadow quality tracking is now active for continuous verification throughout the remaining workflow."""
            else:
                message = f"""🔄 **RIF State Transition Detected**

**Issue**: #{issue_number}
**Transition**: {transition.get('previous_state', 'unknown')} → {transition.get('current_state', 'unknown')}
**Shadow Creation**: Failed - {shadow_result.get('error', 'Unknown error')}

Manual shadow creation may be required for quality tracking."""

            # Post comment to main issue
            escaped_message = message.replace('"', '\\"').replace('`', '\\`')
            cmd = ['gh', 'issue', 'comment', str(issue_number), '--body', escaped_message]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Warning: Could not log transition to issue #{issue_number}: {result.stderr}")
                
        except Exception as e:
            print(f"Warning: Error logging transition to issue #{issue_number}: {e}")
    
    def process_workflow_integration(self) -> Dict[str, Any]:
        """Process RIF workflow integration and create shadows as needed."""
        integration_start = datetime.now()
        print(f"Starting RIF shadow integration check at {integration_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current issue states
        current_states = self.get_current_issue_states()
        print(f"Retrieved {len(current_states)} open issues")
        
        if not current_states:
            return {
                'integration_time': integration_start.isoformat(),
                'issues_processed': 0,
                'transitions_detected': 0,
                'shadows_created': 0,
                'errors': ['No issues retrieved']
            }
        
        # Detect state transitions
        transitions = self.detect_state_transitions(current_states)
        print(f"Detected {len(transitions)} state transitions")
        
        # Process transitions that should create shadows
        shadows_created = 0
        errors = []
        processed_transitions = []
        
        for transition in transitions:
            if transition.get('should_create_shadow', False):
                issue_number = transition['issue_number']
                print(f"Processing shadow creation for issue #{issue_number}: {transition.get('reason', 'Unknown')}")
                
                # Create shadow issue
                shadow_result = self.create_shadow_for_transition(transition)
                transition['shadow_result'] = shadow_result
                
                if shadow_result.get('success', False):
                    shadows_created += 1
                    print(f"✅ Shadow created for issue #{issue_number}")
                    
                    # Log to main issue
                    self.log_transition_to_main_issue(issue_number, transition, shadow_result)
                else:
                    error_msg = f"Issue #{issue_number}: {shadow_result.get('error', 'Unknown error')}"
                    errors.append(error_msg)
                    print(f"❌ {error_msg}")
            else:
                print(f"⏭️  No shadow needed for issue #{transition['issue_number']}: {transition.get('reason', 'Unknown')}")
            
            processed_transitions.append(transition)
        
        # Save state for next check
        self.last_check_time = integration_start
        
        integration_end = datetime.now()
        integration_duration = (integration_end - integration_start).total_seconds()
        
        summary = {
            'integration_time': integration_start.isoformat(),
            'integration_duration_seconds': integration_duration,
            'issues_processed': len(current_states),
            'transitions_detected': len(transitions),
            'shadows_created': shadows_created,
            'errors': errors,
            'transitions': processed_transitions
        }
        
        print(f"RIF integration completed in {integration_duration:.1f}s: {shadows_created} shadows created, {len(errors)} errors")
        return summary
    
    def get_shadow_creation_candidates(self) -> List[Dict[str, Any]]:
        """Get list of issues that are candidates for shadow creation."""
        current_states = self.get_current_issue_states()
        candidates = []
        
        for issue_number, issue_info in current_states.items():
            # Create a mock transition to check triggers
            mock_transition = {
                'issue_number': issue_number,
                'current_state': issue_info.get('state'),
                'previous_state': None
            }
            
            shadow_triggers = self._check_shadow_triggers(issue_info, mock_transition)
            
            if shadow_triggers.get('should_create_shadow', False):
                candidate = {
                    'issue_number': issue_number,
                    'title': issue_info.get('title', ''),
                    'state': issue_info.get('state'),
                    'priority': issue_info.get('priority'),
                    'complexity': issue_info.get('complexity'),
                    'trigger_reasons': shadow_triggers.get('trigger_reasons', []),
                    'reason': shadow_triggers.get('reason', '')
                }
                candidates.append(candidate)
        
        return candidates


def main():
    """Command-line interface for RIF shadow integration."""
    if len(sys.argv) < 2:
        print("Usage: python rif_shadow_integration.py <command> [args]")
        print("Commands:")
        print("  check                    - Check current RIF workflow integration")
        print("  process                  - Process workflow integration and create shadows")
        print("  candidates               - List shadow creation candidates")
        print("  transition <issue_number> <new_state>  - Manually trigger state transition check")
        return 1
    
    command = sys.argv[1]
    integration = RIFShadowIntegration()
    
    if command == "check":
        current_states = integration.get_current_issue_states()
        status = {
            'total_issues': len(current_states),
            'states': {},
            'priorities': {},
            'complexities': {}
        }
        
        for info in current_states.values():
            state = info.get('state', 'unknown')
            priority = info.get('priority', 'unknown') 
            complexity = info.get('complexity', 'unknown')
            
            status['states'][state] = status['states'].get(state, 0) + 1
            status['priorities'][priority] = status['priorities'].get(priority, 0) + 1
            status['complexities'][complexity] = status['complexities'].get(complexity, 0) + 1
        
        print(json.dumps(status, indent=2))
        
    elif command == "process":
        result = integration.process_workflow_integration()
        print(json.dumps(result, indent=2))
        
    elif command == "candidates":
        candidates = integration.get_shadow_creation_candidates()
        print(json.dumps(candidates, indent=2))
        
    elif command == "transition" and len(sys.argv) >= 4:
        issue_number = int(sys.argv[2])
        new_state = sys.argv[3]
        
        print(f"Simulating state transition for issue #{issue_number} to state '{new_state}'")
        
        # Get current issue info
        current_states = integration.get_current_issue_states()
        if issue_number not in current_states:
            print(f"Error: Issue #{issue_number} not found in open issues")
            return 1
        
        issue_info = current_states[issue_number]
        old_state = issue_info.get('state')
        issue_info['state'] = new_state  # Simulate state change
        
        # Check if this would trigger shadow creation
        mock_transition = {
            'issue_number': issue_number,
            'previous_state': old_state,
            'current_state': new_state
        }
        
        triggers = integration._check_shadow_triggers(issue_info, mock_transition)
        mock_transition.update(triggers)
        
        if triggers.get('should_create_shadow', False):
            print(f"✅ Would create shadow: {triggers.get('reason', 'Unknown')}")
            shadow_result = integration.create_shadow_for_transition(mock_transition)
            print(f"Shadow creation result: {json.dumps(shadow_result, indent=2)}")
        else:
            print(f"⏭️  Would not create shadow: {triggers.get('reason', 'Unknown')}")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())