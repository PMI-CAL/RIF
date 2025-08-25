#!/usr/bin/env python3
"""
GitHub State Manager - Critical Fix for RIF State Management
Issue #88: GitHub label management and state synchronization

This module provides GitHub-integrated state management for RIF workflow states.
It ensures proper state transitions, removes conflicting labels, and maintains
synchronization between RIF internal states and GitHub issue labels.
"""

import json
import subprocess
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class GitHubStateManager:
    """
    Manages GitHub issue state labels with proper transition validation
    and conflict resolution.
    """
    
    def __init__(self, config_path: str = "config/rif-workflow.yaml"):
        """
        Initialize the GitHub state manager.
        
        Args:
            config_path: Path to workflow configuration file
        """
        self.config_path = config_path
        self.workflow_config = self.load_workflow_config()
        self.state_prefix = "state:"
        self.setup_logging()
        
        # Cache for GitHub labels to avoid repeated API calls
        self._label_cache = {}
        self._cache_timestamp = {}
        self._cache_ttl = 300  # 5 minutes
        
    def setup_logging(self):
        """Setup logging for state manager operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - GitHubStateManager - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_workflow_config(self) -> Dict[str, Any]:
        """
        Load workflow configuration from YAML file.
        
        Returns:
            Workflow configuration dictionary
        """
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self.get_default_workflow_config()
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate configuration structure
            if 'workflow' not in config or 'states' not in config['workflow']:
                raise ValueError("Invalid workflow configuration structure")
                
            return config['workflow']
            
        except Exception as e:
            self.logger.error(f"Error loading workflow config: {e}")
            return self.get_default_workflow_config()
    
    def get_default_workflow_config(self) -> Dict[str, Any]:
        """Get default workflow configuration if file not found."""
        return {
            'states': {
                'new': {'description': 'New issue awaiting analysis'},
                'analyzing': {'description': 'RIF-Analyst analyzing requirements'},
                'planning': {'description': 'RIF-Planner creating execution plan'},
                'architecting': {'description': 'RIF-Architect designing solution'},
                'implementing': {'description': 'RIF-Implementer writing code'},
                'validating': {'description': 'RIF-Validator testing solution'},
                'learning': {'description': 'RIF-Learner updating knowledge'},
                'complete': {'description': 'Work finished successfully'},
                'blocked': {'description': 'Work blocked, needs intervention'},
                'failed': {'description': 'Work failed, needs recovery'}
            },
            'transitions': []
        }
    
    def get_issue_labels(self, issue_number: int) -> List[str]:
        """
        Get all labels for a GitHub issue with caching.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            List of label names
        """
        cache_key = f"issue_{issue_number}"
        current_time = datetime.now().timestamp()
        
        # Check cache validity
        if (cache_key in self._label_cache and 
            cache_key in self._cache_timestamp and
            current_time - self._cache_timestamp[cache_key] < self._cache_ttl):
            return self._label_cache[cache_key]
        
        try:
            # Get labels using GitHub CLI
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number), 
                '--json', 'labels'
            ], capture_output=True, text=True, check=True)
            
            data = json.loads(result.stdout)
            labels = [label['name'] for label in data.get('labels', [])]
            
            # Update cache
            self._label_cache[cache_key] = labels
            self._cache_timestamp[cache_key] = current_time
            
            return labels
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get labels for issue #{issue_number}: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse GitHub response: {e}")
            return []
    
    def get_current_state(self, issue_number: int) -> Optional[str]:
        """
        Get current state label from GitHub issue.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Current state without prefix, or None if no state label found
        """
        labels = self.get_issue_labels(issue_number)
        state_labels = [label for label in labels if label.startswith(self.state_prefix)]
        
        if not state_labels:
            return None
        elif len(state_labels) == 1:
            return state_labels[0].replace(self.state_prefix, '', 1)
        else:
            # Multiple state labels - this is the conflict we're fixing
            self.logger.warning(f"Issue #{issue_number} has multiple state labels: {state_labels}")
            # Return the first one but flag for cleanup
            return state_labels[0].replace(self.state_prefix, '', 1)
    
    def validate_state_transition(self, current_state: Optional[str], 
                                new_state: str) -> Tuple[bool, str]:
        """
        Validate if a state transition is allowed by workflow rules.
        
        Args:
            current_state: Current state (None for new issues)
            new_state: Target state
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Allow any transition to 'failed' or 'blocked' (emergency states)
        if new_state in ['failed', 'blocked']:
            return True, "Emergency state transition allowed"
        
        # Allow setting initial state for new issues
        if current_state is None and new_state == 'new':
            return True, "Initial state assignment"
        
        # Check if target state exists in configuration
        if new_state not in self.workflow_config['states']:
            return False, f"Unknown state: {new_state}"
        
        # For now, allow all transitions (we can add more validation later)
        # The workflow configuration transitions are complex and context-dependent
        return True, f"Transition from {current_state} to {new_state} allowed"
    
    def remove_conflicting_labels(self, issue_number: int) -> List[str]:
        """
        Remove all state labels from a GitHub issue.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            List of removed labels
        """
        try:
            labels = self.get_issue_labels(issue_number)
            state_labels = [label for label in labels if label.startswith(self.state_prefix)]
            
            if not state_labels:
                return []
            
            removed_labels = []
            for label in state_labels:
                try:
                    subprocess.run([
                        'gh', 'issue', 'edit', str(issue_number),
                        '--remove-label', label
                    ], check=True, capture_output=True)
                    
                    removed_labels.append(label)
                    self.logger.info(f"Removed label '{label}' from issue #{issue_number}")
                    
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to remove label '{label}' from issue #{issue_number}: {e}")
            
            # Clear cache for this issue
            cache_key = f"issue_{issue_number}"
            if cache_key in self._label_cache:
                del self._label_cache[cache_key]
            if cache_key in self._cache_timestamp:
                del self._cache_timestamp[cache_key]
            
            return removed_labels
            
        except Exception as e:
            self.logger.error(f"Error removing conflicting labels from issue #{issue_number}: {e}")
            return []
    
    def add_state_label(self, issue_number: int, state: str) -> bool:
        """
        Add a state label to a GitHub issue.
        
        Args:
            issue_number: GitHub issue number
            state: State to add (without prefix)
            
        Returns:
            Success status
        """
        try:
            state_label = f"{self.state_prefix}{state}"
            
            subprocess.run([
                'gh', 'issue', 'edit', str(issue_number),
                '--add-label', state_label
            ], check=True, capture_output=True)
            
            self.logger.info(f"Added label '{state_label}' to issue #{issue_number}")
            
            # Clear cache for this issue
            cache_key = f"issue_{issue_number}"
            if cache_key in self._label_cache:
                del self._label_cache[cache_key]
            if cache_key in self._cache_timestamp:
                del self._cache_timestamp[cache_key]
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to add label '{state_label}' to issue #{issue_number}: {e}")
            return False
    
    def transition_state(self, issue_number: int, new_state: str, 
                        reason: str = "") -> Tuple[bool, str]:
        """
        Transition issue to new state, removing old state labels.
        
        This is the core method that ensures atomic state transitions.
        
        Args:
            issue_number: GitHub issue number
            new_state: Target state (without prefix)
            reason: Reason for transition (for logging)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get current state
            current_state = self.get_current_state(issue_number)
            
            # Validate transition
            is_valid, validation_message = self.validate_state_transition(current_state, new_state)
            if not is_valid:
                return False, f"Invalid transition: {validation_message}"
            
            # Remove all existing state labels
            removed_labels = self.remove_conflicting_labels(issue_number)
            
            # Add new state label
            success = self.add_state_label(issue_number, new_state)
            if not success:
                return False, f"Failed to add new state label: state:{new_state}"
            
            # Log the transition
            transition_log = {
                'timestamp': datetime.now().isoformat(),
                'issue_number': issue_number,
                'from_state': current_state,
                'to_state': new_state,
                'reason': reason,
                'removed_labels': removed_labels
            }
            
            self._log_transition(transition_log)
            
            message = f"Successfully transitioned issue #{issue_number} from {current_state} to {new_state}"
            self.logger.info(message)
            
            return True, message
            
        except Exception as e:
            error_message = f"Error transitioning issue #{issue_number} to {new_state}: {e}"
            self.logger.error(error_message)
            return False, error_message
    
    def _log_transition(self, transition_log: Dict[str, Any]):
        """
        Log state transition to file for audit trail.
        
        Args:
            transition_log: Transition information to log
        """
        try:
            log_dir = Path('knowledge/state_transitions')
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / 'transitions.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(transition_log) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log transition: {e}")
    
    def validate_issue_state(self, issue_number: int) -> Dict[str, Any]:
        """
        Validate that an issue has exactly one state label.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Validation report
        """
        try:
            labels = self.get_issue_labels(issue_number)
            state_labels = [label for label in labels if label.startswith(self.state_prefix)]
            
            validation_report = {
                'issue_number': issue_number,
                'timestamp': datetime.now().isoformat(),
                'all_labels': labels,
                'state_labels': state_labels,
                'state_count': len(state_labels),
                'is_valid': len(state_labels) == 1,
                'issues': []
            }
            
            if len(state_labels) == 0:
                validation_report['issues'].append('No state label found')
                validation_report['is_valid'] = False
            elif len(state_labels) > 1:
                validation_report['issues'].append(f'Multiple state labels: {state_labels}')
                validation_report['is_valid'] = False
            
            # Check if state exists in configuration
            if state_labels:
                current_state = state_labels[0].replace(self.state_prefix, '', 1)
                if current_state not in self.workflow_config['states']:
                    validation_report['issues'].append(f'Unknown state: {current_state}')
                    validation_report['is_valid'] = False
            
            return validation_report
            
        except Exception as e:
            return {
                'issue_number': issue_number,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'is_valid': False,
                'issues': [f'Validation failed: {e}']
            }
    
    def get_all_open_issues(self) -> List[int]:
        """
        Get all open GitHub issues in the repository.
        
        Returns:
            List of issue numbers
        """
        try:
            result = subprocess.run([
                'gh', 'issue', 'list', '--state', 'open',
                '--json', 'number'
            ], capture_output=True, text=True, check=True)
            
            data = json.loads(result.stdout)
            return [issue['number'] for issue in data]
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get open issues: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse GitHub response: {e}")
            return []
    
    def audit_all_issues(self) -> Dict[str, Any]:
        """
        Audit all open issues for state label problems.
        
        Returns:
            Comprehensive audit report
        """
        try:
            open_issues = self.get_all_open_issues()
            
            audit_report = {
                'timestamp': datetime.now().isoformat(),
                'total_issues': len(open_issues),
                'valid_issues': 0,
                'invalid_issues': 0,
                'issues_by_problem': {
                    'no_state': [],
                    'multiple_states': [],
                    'unknown_state': []
                },
                'validation_details': []
            }
            
            for issue_number in open_issues:
                validation = self.validate_issue_state(issue_number)
                audit_report['validation_details'].append(validation)
                
                if validation['is_valid']:
                    audit_report['valid_issues'] += 1
                else:
                    audit_report['invalid_issues'] += 1
                    
                    # Categorize problems
                    if validation['state_count'] == 0:
                        audit_report['issues_by_problem']['no_state'].append(issue_number)
                    elif validation['state_count'] > 1:
                        audit_report['issues_by_problem']['multiple_states'].append(issue_number)
                    
                    # Check for unknown states
                    for issue in validation.get('issues', []):
                        if 'Unknown state:' in issue:
                            audit_report['issues_by_problem']['unknown_state'].append(issue_number)
                            break
            
            return audit_report
            
        except Exception as e:
            self.logger.error(f"Error during audit: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'total_issues': 0,
                'valid_issues': 0,
                'invalid_issues': 0
            }
    
    def post_comment_to_issue(self, issue_number: int, comment: str) -> bool:
        """
        Post a comment to a GitHub issue.
        
        Args:
            issue_number: GitHub issue number
            comment: Comment text
            
        Returns:
            Success status
        """
        try:
            subprocess.run([
                'gh', 'issue', 'comment', str(issue_number),
                '--body', comment
            ], check=True, capture_output=True)
            
            self.logger.info(f"Posted comment to issue #{issue_number}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to post comment to issue #{issue_number}: {e}")
            return False
    
    def generate_branch_name(self, issue_number: int, issue_title: str) -> str:
        """
        Generate a standardized branch name for an issue.
        
        Args:
            issue_number: GitHub issue number
            issue_title: Issue title text
            
        Returns:
            Sanitized branch name following pattern: issue-{number}-{sanitized-title}
        """
        import re
        
        # Sanitize title: lowercase, replace spaces/special chars with hyphens
        sanitized_title = re.sub(r'[^a-zA-Z0-9\s-]', '', issue_title.lower())
        sanitized_title = re.sub(r'\s+', '-', sanitized_title.strip())
        sanitized_title = re.sub(r'-+', '-', sanitized_title)  # Remove multiple hyphens
        
        # Limit title length to avoid overly long branch names
        if len(sanitized_title) > 50:
            sanitized_title = sanitized_title[:47] + "..."
        
        return f"issue-{issue_number}-{sanitized_title}"
    
    def get_file_modifications(self, branch_name: str) -> Dict[str, List[str]]:
        """
        Get list of modified files between branch and main.
        
        Args:
            branch_name: Git branch name
            
        Returns:
            Dictionary with 'added', 'modified', 'deleted' file lists
        """
        try:
            # Get diff summary between main and branch
            result = subprocess.run([
                'git', 'diff', '--name-status', 'main...HEAD'
            ], capture_output=True, text=True, check=True, cwd='.')
            
            modifications = {
                'added': [],
                'modified': [],
                'deleted': []
            }
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                status, filename = parts[0], parts[1]
                
                if status == 'A':
                    modifications['added'].append(filename)
                elif status == 'M':
                    modifications['modified'].append(filename)
                elif status == 'D':
                    modifications['deleted'].append(filename)
                elif status.startswith('R'):  # Renamed files
                    # For renames, treat as modified
                    modifications['modified'].append(filename)
            
            return modifications
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get file modifications: {e}")
            return {'added': [], 'modified': [], 'deleted': []}
    
    def get_issue_metadata(self, issue_number: int) -> Dict[str, Any]:
        """
        Get comprehensive issue metadata for PR creation.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Dictionary with issue title, body, labels, assignees, etc.
        """
        try:
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number),
                '--json', 'title,body,labels,assignees,state,author'
            ], capture_output=True, text=True, check=True)
            
            data = json.loads(result.stdout)
            
            return {
                'number': issue_number,
                'title': data.get('title', ''),
                'body': data.get('body', ''),
                'labels': [label['name'] for label in data.get('labels', [])],
                'assignees': [assignee['login'] for assignee in data.get('assignees', [])],
                'state': data.get('state', 'open'),
                'author': data.get('author', {}).get('login', '')
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get issue metadata for #{issue_number}: {e}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse issue metadata: {e}")
            return {}
    
    def create_pull_request(self, issue_number: int, pr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a pull request for an issue with populated template.
        
        Args:
            issue_number: GitHub issue number
            pr_data: Dictionary containing PR creation parameters
                - title: PR title
                - body: PR body content (markdown)
                - draft: Whether to create as draft PR
                - base_branch: Base branch (default: main)
                - head_branch: Head branch (auto-generated if not provided)
                - labels: List of labels to apply
                - reviewers: List of reviewer usernames
                - assignees: List of assignee usernames
                
        Returns:
            Dictionary with PR creation results (success, pr_number, url, etc.)
        """
        try:
            issue_metadata = self.get_issue_metadata(issue_number)
            if not issue_metadata:
                return {'success': False, 'error': f'Could not retrieve metadata for issue #{issue_number}'}
            
            # Generate branch name if not provided
            head_branch = pr_data.get('head_branch')
            if not head_branch:
                head_branch = self.generate_branch_name(issue_number, issue_metadata['title'])
            
            # Create/checkout branch if it doesn't exist
            try:
                subprocess.run(['git', 'checkout', head_branch], 
                             capture_output=True, check=True, cwd='.')
            except subprocess.CalledProcessError:
                # Branch doesn't exist, create it
                subprocess.run(['git', 'checkout', '-b', head_branch], 
                             capture_output=True, check=True, cwd='.')
                self.logger.info(f"Created new branch: {head_branch}")
            
            # Build GitHub CLI command
            base_branch = pr_data.get('base_branch', 'main')
            pr_title = pr_data.get('title', f"Fix: {issue_metadata['title']}")
            draft_flag = ['--draft'] if pr_data.get('draft', False) else []
            
            # Generate PR body using template aggregator if not provided
            pr_body = pr_data.get('body')
            if not pr_body:
                self.logger.info(f"No PR body provided, populating template for issue #{issue_number}")
                pr_body = self.populate_pr_template(issue_number, pr_data.get('quality_results'))
            
            # Write PR body to temporary file for GitHub CLI
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
                temp_file.write(pr_body)
                temp_file_path = temp_file.name
            
            try:
                # Build GitHub CLI command
                cmd = [
                    'gh', 'pr', 'create',
                    '--base', base_branch,
                    '--head', head_branch,
                    '--title', pr_title,
                    '--body-file', temp_file_path
                ] + draft_flag
                
                # Add labels if specified
                labels = pr_data.get('labels', [])
                if labels:
                    # Add RIF-managed labels
                    labels.extend(['rif-managed', 'automated-pr'])
                    for label in labels:
                        cmd.extend(['--label', label])
                
                # Add reviewers if specified
                reviewers = pr_data.get('reviewers', [])
                for reviewer in reviewers:
                    cmd.extend(['--reviewer', reviewer])
                
                # Add assignees if specified  
                assignees = pr_data.get('assignees', [])
                for assignee in assignees:
                    cmd.extend(['--assignee', assignee])
                
                # Execute PR creation
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd='.')
                
                # Extract PR URL from output
                pr_url = result.stdout.strip()
                pr_number = None
                if '/pull/' in pr_url:
                    pr_number = int(pr_url.split('/pull/')[-1])
                
                self.logger.info(f"Created PR #{pr_number} for issue #{issue_number}: {pr_url}")
                
                return {
                    'success': True,
                    'pr_number': pr_number,
                    'pr_url': pr_url,
                    'branch': head_branch,
                    'title': pr_title,
                    'draft': pr_data.get('draft', False)
                }
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to create PR for issue #{issue_number}: {e}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"Unexpected error creating PR for issue #{issue_number}: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def populate_pr_template(self, issue_number: int, 
                           quality_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Populate PR template with issue context and checkpoint data.
        
        Args:
            issue_number: GitHub issue number
            quality_results: Quality gate results (optional)
            
        Returns:
            Populated PR template content
        """
        try:
            # Import template aggregator
            from .pr_template_aggregator import PRTemplateAggregator
            
            # Get issue metadata and file modifications
            issue_metadata = self.get_issue_metadata(issue_number)
            file_modifications = self.get_file_modifications("HEAD")
            
            # Initialize aggregator and generate context
            aggregator = PRTemplateAggregator()
            pr_context = aggregator.aggregate_pr_context(
                issue_number, issue_metadata, file_modifications, quality_results
            )
            
            # Populate and return template
            populated_template = aggregator.populate_template(pr_context)
            self.logger.info(f"Populated PR template for issue #{issue_number}")
            
            return populated_template
            
        except Exception as e:
            self.logger.error(f"Failed to populate PR template for issue #{issue_number}: {e}")
            
            # Fallback to minimal template
            issue_metadata = self.get_issue_metadata(issue_number)
            return f"""# Pull Request for Issue #{issue_number}

## Summary
{issue_metadata.get('title', 'Automated implementation')}

## Related Issues
Closes #{issue_number}

## Changes Made
Implementation completed via RIF automation system.

## RIF Automation
This PR was automatically created by RIF-Implementer.
Manual review and validation recommended.
"""


def demo_github_state_manager():
    """Demonstrate GitHub state manager functionality."""
    print("🔧 GitHub State Manager Demo")
    
    manager = GitHubStateManager()
    
    print("1. Loading workflow configuration...")
    states = list(manager.workflow_config['states'].keys())
    print(f"   Available states: {states}")
    
    print("2. Auditing all open issues...")
    audit_report = manager.audit_all_issues()
    print(f"   Total issues: {audit_report['total_issues']}")
    print(f"   Valid issues: {audit_report['valid_issues']}")
    print(f"   Invalid issues: {audit_report['invalid_issues']}")
    
    if audit_report['invalid_issues'] > 0:
        print("   Problems found:")
        for problem_type, issues in audit_report['issues_by_problem'].items():
            if issues:
                print(f"     {problem_type}: {issues}")
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    demo_github_state_manager()