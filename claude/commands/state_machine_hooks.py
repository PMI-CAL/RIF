#!/usr/bin/env python3
"""
State Machine Hooks - Integration point for automated PR creation
Provides hooks for RIF workflow state transitions to trigger automated actions.

This module integrates with the existing GitHubStateManager to add automated
PR creation when specific state transitions occur.
"""

import logging
from typing import Dict, Optional, Any, Callable
from .pr_creation_service import PRCreationService
from .github_state_manager import GitHubStateManager

class StateMachineHooks:
    """
    Provides hooks for automated actions on state transitions.
    """
    
    def __init__(self, config_path: str = "config/rif-workflow.yaml"):
        """
        Initialize state machine hooks.
        
        Args:
            config_path: Path to workflow configuration
        """
        self.config_path = config_path
        self.pr_service = PRCreationService(config_path)
        self.github_manager = GitHubStateManager(config_path)
        self.setup_logging()
        
        # Register built-in hooks
        self.hooks = {
            'pr_creation': self.pr_service.handle_state_transition,
            # Additional hooks can be registered here
        }
        
    def setup_logging(self):
        """Setup logging for state machine hooks."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - StateMachineHooks - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def register_hook(self, hook_name: str, hook_function: Callable) -> bool:
        """
        Register a custom hook function.
        
        Args:
            hook_name: Name identifier for the hook
            hook_function: Function to call (signature: issue_number, from_state, to_state, context)
            
        Returns:
            True if successfully registered
        """
        try:
            self.hooks[hook_name] = hook_function
            self.logger.info(f"Registered hook: {hook_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register hook {hook_name}: {e}")
            return False
    
    def execute_hooks(self, issue_number: int, from_state: str, to_state: str,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute all registered hooks for a state transition.
        
        Args:
            issue_number: GitHub issue number
            from_state: Current state
            to_state: Target state
            context: Additional context data
            
        Returns:
            Results from all hook executions
        """
        results = {
            'issue_number': issue_number,
            'transition': f"{from_state} → {to_state}",
            'hook_results': {},
            'overall_success': True,
            'actions_taken': []
        }
        
        self.logger.info(f"Executing hooks for issue #{issue_number}: {from_state} → {to_state}")
        
        for hook_name, hook_function in self.hooks.items():
            try:
                self.logger.debug(f"Executing hook: {hook_name}")
                
                hook_result = hook_function(issue_number, from_state, to_state, context)
                results['hook_results'][hook_name] = hook_result
                
                # Track successful actions
                if hook_result.get('success') and hook_result.get('action') != 'no_action':
                    action_description = f"{hook_name}: {hook_result.get('action', 'executed')}"
                    results['actions_taken'].append(action_description)
                    
                # Track overall success
                if not hook_result.get('success', True):
                    results['overall_success'] = False
                    
            except Exception as e:
                error_msg = f"Hook {hook_name} failed: {e}"
                self.logger.error(error_msg)
                
                results['hook_results'][hook_name] = {
                    'success': False,
                    'error': error_msg
                }
                results['overall_success'] = False
        
        self.logger.info(f"Hook execution completed for issue #{issue_number}. Actions: {len(results['actions_taken'])}")
        return results
    
    def enhanced_transition_state(self, issue_number: int, new_state: str,
                                 reason: str = "", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced state transition that executes hooks before/after state change.
        
        This method wraps the standard GitHubStateManager.transition_state to add
        hook execution for automated actions like PR creation.
        
        Args:
            issue_number: GitHub issue number
            new_state: Target state
            reason: Reason for transition
            context: Additional context for hooks
            
        Returns:
            Combined results from state transition and hook execution
        """
        try:
            # Get current state before transition
            current_state = self.github_manager.get_current_state(issue_number)
            
            self.logger.info(f"Enhanced state transition for issue #{issue_number}: {current_state} → {new_state}")
            
            # Execute pre-transition hooks if needed (currently none, but extensible)
            # pre_hook_results = self.execute_hooks(issue_number, current_state, f"pre_{new_state}", context)
            
            # Execute the actual state transition
            transition_success, transition_message = self.github_manager.transition_state(
                issue_number, new_state, reason
            )
            
            # Execute post-transition hooks
            hook_results = self.execute_hooks(issue_number, current_state, new_state, context)
            
            return {
                'success': transition_success,
                'message': transition_message,
                'from_state': current_state,
                'to_state': new_state,
                'hooks_executed': len(hook_results['hook_results']),
                'actions_taken': hook_results['actions_taken'],
                'hook_results': hook_results,
                'enhanced_transition': True
            }
            
        except Exception as e:
            error_msg = f"Enhanced state transition failed: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'enhanced_transition': True
            }


# Convenience function for external integration
def create_enhanced_github_manager(config_path: str = "config/rif-workflow.yaml") -> StateMachineHooks:
    """
    Create an enhanced GitHub state manager with automated PR creation hooks.
    
    Args:
        config_path: Path to workflow configuration
        
    Returns:
        StateMachineHooks instance with PR creation enabled
    """
    return StateMachineHooks(config_path)


def demo_state_machine_hooks():
    """Demonstrate state machine hooks functionality."""
    print("🔧 State Machine Hooks Demo")
    
    hooks = StateMachineHooks()
    
    print("1. Registered hooks:")
    for hook_name in hooks.hooks.keys():
        print(f"   - {hook_name}")
    
    print("2. Testing enhanced state transition...")
    result = hooks.enhanced_transition_state(
        205, 
        'pr_creating', 
        "Testing automated PR creation",
        {'test_mode': True}
    )
    
    print(f"   Transition success: {result.get('success')}")
    print(f"   Actions taken: {len(result.get('actions_taken', []))}")
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    demo_state_machine_hooks()