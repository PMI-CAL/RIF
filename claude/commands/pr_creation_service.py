#!/usr/bin/env python3
"""
PR Creation Service - Phase 3 of Issue #205
Integrates with RIF workflow state machine to automatically create PRs on state transitions.

This service monitors state transitions and triggers automated PR creation when
implementation is complete and validation passes quality gates.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .github_state_manager import GitHubStateManager

class PRCreationService:
    """
    Service for automated PR creation integrated with RIF state machine.
    """
    
    def __init__(self, config_path: str = "config/rif-workflow.yaml"):
        """
        Initialize the PR creation service.
        
        Args:
            config_path: Path to workflow configuration
        """
        self.config_path = config_path
        self.github_manager = GitHubStateManager(config_path)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for PR creation service."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PRCreationService - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def should_create_pr(self, issue_number: int, current_state: str, new_state: str) -> bool:
        """
        Determine if a PR should be created based on state transition.
        
        Args:
            issue_number: GitHub issue number
            current_state: Current workflow state
            new_state: Target workflow state
            
        Returns:
            True if PR should be created
        """
        # PR creation triggers
        pr_creation_triggers = [
            # Primary trigger: implementation complete, moving to validation
            (current_state == 'implementing' and new_state == 'validating'),
            # Secondary trigger: explicit pr_creating state
            (new_state == 'pr_creating'),
            # Alternative trigger: validation complete, ready for PR
            (current_state == 'validating' and new_state == 'documenting')
        ]
        
        should_create = any(pr_creation_triggers)
        
        if should_create:
            self.logger.info(f"PR creation triggered for issue #{issue_number}: {current_state} → {new_state}")
        
        return should_create
    
    def check_quality_gates(self, issue_number: int) -> Dict[str, Any]:
        """
        Check quality gates status for PR creation readiness.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Quality gates status and results
        """
        try:
            # Look for quality evidence in knowledge system
            evidence_path = Path(f"knowledge/evidence/issue-{issue_number}-implementation-evidence.json")
            
            quality_results = {
                'overall_status': 'pending',
                'gates': {},
                'ready_for_pr': False,
                'draft_pr': True  # Default to draft until all gates pass
            }
            
            if evidence_path.exists():
                with open(evidence_path, 'r') as f:
                    evidence_data = json.load(f)
                    
                # Extract quality gate results
                quality_results['gates'] = evidence_data.get('evidence', {})
                
                # Assess overall status
                gates_status = quality_results['gates']
                
                # Check individual gates
                code_quality_ok = self._check_code_quality_gate(gates_status.get('quality', {}))
                tests_ok = self._check_testing_gate(gates_status.get('tests', {}))
                security_ok = self._check_security_gate(gates_status.get('quality', {}))
                
                quality_results['gates_status'] = {
                    'code_quality': code_quality_ok,
                    'testing': tests_ok,
                    'security': security_ok
                }
                
                # Overall readiness assessment
                if code_quality_ok and tests_ok and security_ok:
                    quality_results['overall_status'] = 'ready'
                    quality_results['ready_for_pr'] = True
                    quality_results['draft_pr'] = False
                elif tests_ok:  # At least tests pass
                    quality_results['overall_status'] = 'partial'
                    quality_results['ready_for_pr'] = True
                    quality_results['draft_pr'] = True
                else:
                    quality_results['overall_status'] = 'failing'
                    quality_results['ready_for_pr'] = False
                    quality_results['draft_pr'] = True
                    
            self.logger.info(f"Quality gates status for issue #{issue_number}: {quality_results['overall_status']}")
            return quality_results
            
        except Exception as e:
            self.logger.warning(f"Failed to check quality gates for issue #{issue_number}: {e}")
            return {
                'overall_status': 'unknown',
                'gates': {},
                'ready_for_pr': True,  # Allow PR creation even if gates unclear
                'draft_pr': True,  # But make it draft for safety
                'error': str(e)
            }
    
    def _check_code_quality_gate(self, quality_data: Dict[str, Any]) -> bool:
        """Check if code quality gate passes."""
        if not quality_data:
            return False
            
        linting = quality_data.get('linting', {})
        type_check = quality_data.get('type_check', {})
        
        linting_ok = linting.get('errors', 1) == 0
        typing_ok = type_check.get('passing', False)
        
        return linting_ok and typing_ok
    
    def _check_testing_gate(self, test_data: Dict[str, Any]) -> bool:
        """Check if testing gate passes."""
        if not test_data:
            return False
            
        unit_tests = test_data.get('unit', {})
        coverage = test_data.get('coverage', 0)
        
        tests_passing = unit_tests.get('passing', 0) > 0
        coverage_ok = coverage >= 80  # 80% threshold
        
        return tests_passing and coverage_ok
    
    def _check_security_gate(self, quality_data: Dict[str, Any]) -> bool:
        """Check if security gate passes."""
        if not quality_data:
            return True  # No security scan data, assume OK
            
        security = quality_data.get('security', {})
        vulnerabilities = security.get('vulnerabilities', 0)
        
        return vulnerabilities == 0
    
    def determine_pr_strategy(self, issue_number: int, quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine PR creation strategy based on quality gates and context.
        
        Args:
            issue_number: GitHub issue number
            quality_results: Quality gates assessment
            
        Returns:
            PR creation strategy configuration
        """
        strategy = {
            'create_pr': True,
            'draft': True,  # Default to draft for safety
            'title_prefix': 'Draft:',
            'labels': ['rif-managed', 'automated-pr', 'work-in-progress'],
            'reviewers': [],
            'assignees': [],
            'merge_strategy': 'manual'
        }
        
        # Adjust strategy based on quality results
        if quality_results.get('overall_status') == 'ready':
            strategy.update({
                'draft': False,
                'title_prefix': 'Ready:',
                'labels': ['rif-managed', 'automated-pr', 'ready-for-review'],
                'reviewers': ['@me'],  # Self-assign for review
                'merge_strategy': 'auto_merge_on_approval'
            })
        elif quality_results.get('overall_status') == 'partial':
            strategy.update({
                'draft': True,
                'title_prefix': 'WIP:',
                'labels': ['rif-managed', 'automated-pr', 'needs-quality-fixes'],
                'reviewers': ['@me']
            })
        elif quality_results.get('overall_status') == 'failing':
            strategy.update({
                'create_pr': False,  # Don't create PR if quality is failing
                'reason': 'Quality gates failing, delaying PR creation'
            })
            
        self.logger.info(f"PR strategy for issue #{issue_number}: {strategy}")
        return strategy
    
    def create_automated_pr(self, issue_number: int, 
                           trigger_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an automated PR for an issue with full context integration.
        
        Args:
            issue_number: GitHub issue number
            trigger_context: Additional context from trigger (optional)
            
        Returns:
            PR creation results
        """
        try:
            self.logger.info(f"Starting automated PR creation for issue #{issue_number}")
            
            # Check quality gates
            quality_results = self.check_quality_gates(issue_number)
            
            # Determine PR creation strategy
            strategy = self.determine_pr_strategy(issue_number, quality_results)
            
            if not strategy.get('create_pr', True):
                return {
                    'success': False,
                    'skipped': True,
                    'reason': strategy.get('reason', 'PR creation not recommended'),
                    'quality_status': quality_results.get('overall_status')
                }
            
            # Get issue metadata for title generation
            issue_metadata = self.github_manager.get_issue_metadata(issue_number)
            issue_title = issue_metadata.get('title', f'Issue #{issue_number}')
            
            # Generate PR title with appropriate prefix
            title_prefix = strategy.get('title_prefix', 'Auto:')
            pr_title = f"{title_prefix} {issue_title}"
            
            # Build PR data
            pr_data = {
                'title': pr_title,
                'draft': strategy.get('draft', True),
                'labels': strategy.get('labels', []),
                'reviewers': strategy.get('reviewers', []),
                'assignees': strategy.get('assignees', []),
                'quality_results': quality_results  # Pass to template aggregator
            }
            
            # Create the pull request
            pr_result = self.github_manager.create_pull_request(issue_number, pr_data)
            
            if pr_result.get('success'):
                # Log successful creation
                self.logger.info(f"Successfully created PR #{pr_result.get('pr_number')} for issue #{issue_number}")
                
                # Post status comment to issue
                self._post_pr_creation_comment(issue_number, pr_result, strategy, quality_results)
                
                # Transition issue state if needed
                if strategy.get('draft'):
                    new_state = 'pr_validating'
                else:
                    new_state = 'pr_validating'
                    
                self.github_manager.transition_state(
                    issue_number, 
                    new_state,
                    f"PR #{pr_result.get('pr_number')} created successfully"
                )
                
                return {
                    'success': True,
                    'pr_number': pr_result.get('pr_number'),
                    'pr_url': pr_result.get('pr_url'),
                    'strategy': strategy,
                    'quality_status': quality_results.get('overall_status')
                }
            else:
                return {
                    'success': False,
                    'error': pr_result.get('error'),
                    'quality_status': quality_results.get('overall_status')
                }
                
        except Exception as e:
            error_msg = f"Failed to create automated PR for issue #{issue_number}: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _post_pr_creation_comment(self, issue_number: int, pr_result: Dict[str, Any],
                                 strategy: Dict[str, Any], quality_results: Dict[str, Any]):
        """Post a status comment about PR creation to the issue."""
        try:
            pr_number = pr_result.get('pr_number')
            pr_url = pr_result.get('pr_url')
            draft_status = "Draft PR" if strategy.get('draft') else "Ready PR"
            quality_status = quality_results.get('overall_status', 'unknown')
            
            comment = f"""## 🚀 Automated PR Created

**PR #{pr_number}** has been automatically created: {pr_url}

**Status**: {draft_status}  
**Quality Gates**: {quality_status.title()}

### Quality Assessment
"""
            
            # Add quality gate details
            gates_status = quality_results.get('gates_status', {})
            for gate_name, passed in gates_status.items():
                status_emoji = "✅" if passed else "❌"
                comment += f"- {status_emoji} **{gate_name.replace('_', ' ').title()}**: {'Passed' if passed else 'Failed'}\n"
            
            comment += f"""
### Next Steps
{('- PR is ready for review and merge' if not strategy.get('draft') else '- Address quality gate issues and convert from draft')}
- GitHub Actions will validate the changes
- Manual review recommended for critical changes

**Note**: This PR was automatically created by RIF-PR-Manager based on implementation completion.
"""
            
            self.github_manager.post_comment_to_issue(issue_number, comment)
            
        except Exception as e:
            self.logger.warning(f"Failed to post PR creation comment: {e}")
    
    def handle_state_transition(self, issue_number: int, from_state: str, 
                               to_state: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a state transition and create PR if appropriate.
        
        Args:
            issue_number: GitHub issue number
            from_state: Current state
            to_state: Target state
            context: Additional transition context
            
        Returns:
            Action results
        """
        try:
            self.logger.info(f"Handling state transition for issue #{issue_number}: {from_state} → {to_state}")
            
            if self.should_create_pr(issue_number, from_state, to_state):
                return self.create_automated_pr(issue_number, context)
            else:
                return {
                    'success': True,
                    'action': 'no_action',
                    'reason': f'State transition {from_state} → {to_state} does not trigger PR creation'
                }
                
        except Exception as e:
            error_msg = f"Failed to handle state transition: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}


def demo_pr_creation_service():
    """Demonstrate PR creation service functionality."""
    print("🔧 PR Creation Service Demo")
    
    service = PRCreationService()
    
    print("1. Checking quality gates...")
    quality_results = service.check_quality_gates(205)
    print(f"   Quality status: {quality_results.get('overall_status')}")
    
    print("2. Determining PR strategy...")
    strategy = service.determine_pr_strategy(205, quality_results)
    print(f"   Draft PR: {strategy.get('draft')}")
    print(f"   Title prefix: {strategy.get('title_prefix')}")
    
    print("3. Testing state transition...")
    result = service.handle_state_transition(205, 'implementing', 'validating')
    print(f"   Action: {result.get('action', 'create_pr')}")
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    demo_pr_creation_service()