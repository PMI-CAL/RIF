#!/usr/bin/env python3
"""
Branch Workflow Enforcer
Ensures all work happens on proper issue-specific branches.

This system addresses the critical failure identified in issue #232 where
work was happening on the main branch instead of dedicated issue branches,
despite the branch management system being implemented.

Key Features:
1. Automatic branch creation for new issues
2. Branch enforcement before any work begins
3. Integration with RIF workflow states
4. Automatic branch switching and tracking
5. Prevention of work on main/master branches
"""

import subprocess
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

class BranchWorkflowEnforcer:
    """
    Enforces proper branch workflow for RIF issue management.
    
    Ensures that:
    1. All issues have dedicated branches
    2. Work happens on the correct branch
    3. Branch creation is automatic
    4. Main branch is protected from direct work
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.enforcement_log = self.repo_path / "knowledge" / "branch_enforcement_log.json"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for branch enforcement tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - BranchWorkflowEnforcer - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def enforce_issue_branch(self, issue_number: int, agent_name: str = "system", 
                           force_create: bool = False) -> Dict[str, Any]:
        """
        Enforce that work for an issue happens on its dedicated branch.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of agent requesting enforcement
            force_create: Force creation of new branch if needed
            
        Returns:
            Dict with enforcement results
        """
        try:
            self.logger.info(f"Enforcing branch for issue #{issue_number}")
            
            # Get issue details
            issue_details = self._get_issue_details(issue_number)
            if not issue_details:
                return {
                    "status": "error",
                    "message": f"Could not retrieve details for issue #{issue_number}",
                    "issue_number": issue_number
                }
            
            # Check current branch
            current_branch = self._get_current_branch()
            expected_branch_prefix = f"issue-{issue_number}-"
            
            # Check if we're already on the correct branch
            if current_branch.startswith(expected_branch_prefix):
                return {
                    "status": "already_enforced",
                    "current_branch": current_branch,
                    "message": f"Already on correct branch for issue #{issue_number}",
                    "issue_number": issue_number
                }
            
            # Check if we're on a protected branch
            if self._is_protected_branch(current_branch):
                self.logger.warning(f"Work attempted on protected branch: {current_branch}")
                
                # Create or switch to issue branch
                branch_result = self._ensure_issue_branch_exists(issue_number, issue_details)
                
                if branch_result["status"] in ["created", "exists"]:
                    # Switch to issue branch
                    switch_result = self._switch_to_branch(branch_result["branch_name"])
                    
                    if switch_result["status"] == "success":
                        self._log_enforcement_event("branch_enforced", {
                            "issue_number": issue_number,
                            "agent_name": agent_name,
                            "from_branch": current_branch,
                            "to_branch": branch_result["branch_name"],
                            "reason": "protected_branch_violation"
                        })
                        
                        return {
                            "status": "enforced",
                            "previous_branch": current_branch,
                            "current_branch": branch_result["branch_name"],
                            "message": f"Switched from protected branch {current_branch} to issue branch {branch_result['branch_name']}",
                            "branch_created": branch_result["status"] == "created",
                            "issue_number": issue_number
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Failed to switch to issue branch: {switch_result['message']}",
                            "issue_number": issue_number
                        }
                else:
                    return {
                        "status": "error", 
                        "message": f"Failed to create issue branch: {branch_result['message']}",
                        "issue_number": issue_number
                    }
            
            # If on a different issue branch, that might be intentional
            if current_branch.startswith("issue-"):
                return {
                    "status": "on_different_issue_branch",
                    "current_branch": current_branch,
                    "message": f"Currently on different issue branch: {current_branch}",
                    "issue_number": issue_number,
                    "warning": "Make sure this is intentional"
                }
            
            # For any other branch, enforce issue branch
            branch_result = self._ensure_issue_branch_exists(issue_number, issue_details)
            
            if branch_result["status"] in ["created", "exists"]:
                switch_result = self._switch_to_branch(branch_result["branch_name"])
                
                if switch_result["status"] == "success":
                    self._log_enforcement_event("branch_enforced", {
                        "issue_number": issue_number,
                        "agent_name": agent_name,
                        "from_branch": current_branch,
                        "to_branch": branch_result["branch_name"],
                        "reason": "workflow_enforcement"
                    })
                    
                    return {
                        "status": "enforced",
                        "previous_branch": current_branch,
                        "current_branch": branch_result["branch_name"],
                        "message": f"Enforced issue branch for #{issue_number}",
                        "branch_created": branch_result["status"] == "created",
                        "issue_number": issue_number
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to switch to issue branch: {switch_result['message']}",
                        "issue_number": issue_number
                    }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to ensure issue branch exists: {branch_result['message']}",
                    "issue_number": issue_number
                }
                
        except Exception as e:
            self.logger.error(f"Error enforcing branch for issue #{issue_number}: {e}")
            return {
                "status": "error",
                "message": f"Branch enforcement failed: {str(e)}",
                "issue_number": issue_number
            }
    
    def check_branch_compliance(self, issue_number: int) -> Dict[str, Any]:
        """
        Check if current branch is compliant for working on an issue.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Dict with compliance status
        """
        try:
            current_branch = self._get_current_branch()
            expected_branch_prefix = f"issue-{issue_number}-"
            
            compliance_report = {
                "issue_number": issue_number,
                "current_branch": current_branch,
                "expected_prefix": expected_branch_prefix,
                "is_compliant": False,
                "violations": [],
                "recommendations": []
            }
            
            # Check if on correct issue branch
            if current_branch.startswith(expected_branch_prefix):
                compliance_report["is_compliant"] = True
                compliance_report["status"] = "compliant"
            else:
                compliance_report["is_compliant"] = False
                
                # Check specific violations
                if self._is_protected_branch(current_branch):
                    compliance_report["violations"].append(
                        f"Working on protected branch '{current_branch}' is not allowed"
                    )
                    compliance_report["recommendations"].append(
                        f"Switch to issue-{issue_number}-* branch"
                    )
                elif current_branch.startswith("issue-"):
                    other_issue = self._extract_issue_number(current_branch)
                    compliance_report["violations"].append(
                        f"Working on branch for different issue #{other_issue}"
                    )
                    compliance_report["recommendations"].append(
                        f"Switch to correct issue branch or continue on issue-{other_issue} if intentional"
                    )
                else:
                    compliance_report["violations"].append(
                        f"Branch '{current_branch}' doesn't follow issue naming convention"
                    )
                    compliance_report["recommendations"].append(
                        f"Create and switch to issue-{issue_number}-* branch"
                    )
                
                compliance_report["status"] = "non_compliant"
            
            return compliance_report
            
        except Exception as e:
            self.logger.error(f"Error checking branch compliance for issue #{issue_number}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "issue_number": issue_number,
                "is_compliant": False
            }
    
    def prevent_protected_branch_work(self, issue_number: int, agent_name: str) -> Dict[str, Any]:
        """
        Prevent work from happening on protected branches.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of agent attempting work
            
        Returns:
            Dict with prevention results
        """
        try:
            current_branch = self._get_current_branch()
            
            if self._is_protected_branch(current_branch):
                self.logger.warning(f"Preventing {agent_name} from working on protected branch: {current_branch}")
                
                # Enforce issue branch
                enforcement_result = self.enforce_issue_branch(issue_number, agent_name)
                
                if enforcement_result["status"] == "enforced":
                    # Post enforcement comment to issue
                    enforcement_comment = f"""**🔒 BRANCH PROTECTION ENFORCED**

Agent `{agent_name}` attempted to work on protected branch `{current_branch}`.

**Action Taken**: Automatically switched to issue branch `{enforcement_result['current_branch']}`

**Protection Rule**: Direct work on `{current_branch}` is not allowed. All issue work must happen on dedicated issue branches.

**Branch Created**: {'Yes' if enforcement_result.get('branch_created', False) else 'No'}

---
*This protection was implemented to address issue #232 and ensure proper branch workflow.*"""
                    
                    subprocess.run([
                        'gh', 'issue', 'comment', str(issue_number),
                        '--body', enforcement_comment
                    ], check=True)
                    
                    return {
                        "status": "protection_enforced",
                        "message": f"Protected branch work prevented, switched to {enforcement_result['current_branch']}",
                        "protected_branch": current_branch,
                        "enforced_branch": enforcement_result["current_branch"],
                        "issue_number": issue_number
                    }
                else:
                    return {
                        "status": "enforcement_failed",
                        "message": f"Failed to enforce branch protection: {enforcement_result.get('message', 'Unknown error')}",
                        "protected_branch": current_branch,
                        "issue_number": issue_number
                    }
            else:
                return {
                    "status": "not_protected",
                    "message": f"Current branch {current_branch} is not protected",
                    "current_branch": current_branch
                }
                
        except Exception as e:
            self.logger.error(f"Error preventing protected branch work: {e}")
            return {
                "status": "error",
                "message": str(e),
                "issue_number": issue_number
            }
    
    def auto_create_issue_branch(self, issue_number: int, from_state: str = None, to_state: str = None) -> Dict[str, Any]:
        """
        Automatically create branch for issue when transitioning to implementing state.
        
        Args:
            issue_number: GitHub issue number
            from_state: Previous workflow state
            to_state: New workflow state
            
        Returns:
            Dict with branch creation results
        """
        try:
            # Check if this transition requires branch creation
            if to_state == "implementing" and from_state in ["planning", "architecting", None]:
                issue_details = self._get_issue_details(issue_number)
                if not issue_details:
                    return {
                        "status": "error",
                        "message": f"Could not retrieve issue details for #{issue_number}"
                    }
                
                # Create issue branch
                result = self._create_issue_branch(
                    issue_number,
                    issue_details.get("title", f"Issue {issue_number}")
                )
                
                if result["status"] == "created":
                    self.logger.info(f"Auto-created branch for issue #{issue_number} on state transition")
                    
                    self._log_enforcement_event("auto_branch_created", {
                        "issue_number": issue_number,
                        "from_state": from_state,
                        "to_state": to_state,
                        "branch_name": result["branch_name"]
                    })
                
                return result
            else:
                return {
                    "status": "no_action",
                    "message": f"No branch creation needed for {from_state}->{to_state} transition"
                }
                
        except Exception as e:
            self.logger.error(f"Error auto-creating branch for issue #{issue_number}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def validate_workflow_branch_setup(self) -> Dict[str, Any]:
        """
        Validate that branch workflow is properly configured.
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "branch_manager_configured": False,
            "git_hooks_installed": False,
            "protected_branches": [],
            "current_branch": None,
            "issues_missing_branches": [],
            "recommendations": []
        }
        
        try:
            # Check branch manager setup
            validation_results["branch_manager_configured"] = True
            validation_results["current_branch"] = self._get_current_branch()
            
            # Check git hooks
            hooks_status = self._check_hooks_installed()
            validation_results["git_hooks_installed"] = all(hooks_status.values())
            
            # Check for protected branches
            protected_patterns = ["main", "master", "develop", "staging", "production"]
            validation_results["protected_branches"] = [
                pattern for pattern in protected_patterns 
                if self._branch_exists(pattern)
            ]
            
            # Check open issues for missing branches
            open_issues = self._get_open_implementing_issues()
            for issue_number in open_issues:
                expected_branch = f"issue-{issue_number}-"
                if not any(branch.startswith(expected_branch) for branch in self._list_all_branches()):
                    validation_results["issues_missing_branches"].append(issue_number)
            
            # Generate recommendations
            if validation_results["issues_missing_branches"]:
                validation_results["recommendations"].append(
                    f"Create branches for issues: {validation_results['issues_missing_branches']}"
                )
            
            if not validation_results["git_hooks_installed"]:
                validation_results["recommendations"].append("Install git hooks for branch protection")
            
            validation_results["status"] = "completed"
            
        except Exception as e:
            validation_results["status"] = "error"
            validation_results["error"] = str(e)
        
        return validation_results
    
    # Private helper methods
    
    def _get_issue_details(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Get issue details from GitHub."""
        try:
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number),
                '--json', 'title,body,state,labels'
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout)
        except:
            return None
    
    def _ensure_issue_branch_exists(self, issue_number: int, issue_details: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure issue branch exists, create if necessary."""
        issue_title = issue_details.get("title", f"Issue {issue_number}")
        
        return self._create_issue_branch(issue_number, issue_title)
    
    def _switch_to_branch(self, branch_name: str) -> Dict[str, Any]:
        """Switch to specified branch."""
        try:
            result = subprocess.run([
                'git', 'checkout', branch_name
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)
            
            return {
                "status": "success",
                "branch_name": branch_name,
                "message": f"Switched to branch {branch_name}"
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Failed to switch to branch {branch_name}: {e.stderr}"
            }
    
    def _is_protected_branch(self, branch_name: str) -> bool:
        """Check if branch is protected."""
        protected_branches = ["main", "master", "develop", "staging", "production"]
        return branch_name in protected_branches
    
    def _extract_issue_number(self, branch_name: str) -> Optional[int]:
        """Extract issue number from branch name."""
        if branch_name.startswith("issue-"):
            try:
                return int(branch_name.split("-")[1])
            except (IndexError, ValueError):
                return None
        return None
    
    def _branch_exists(self, branch_name: str) -> bool:
        """Check if branch exists locally or remotely."""
        try:
            # Check local
            subprocess.run([
                'git', 'show-ref', '--verify', '--quiet', f'refs/heads/{branch_name}'
            ], cwd=self.repo_path, check=True)
            return True
        except subprocess.CalledProcessError:
            pass
        
        try:
            # Check remote
            subprocess.run([
                'git', 'show-ref', '--verify', '--quiet', f'refs/remotes/origin/{branch_name}'
            ], cwd=self.repo_path, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _list_all_branches(self) -> List[str]:
        """List all branches in repository."""
        try:
            result = subprocess.run([
                'git', 'branch', '-a'
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)
            
            branches = []
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and not line.startswith('*'):
                    # Clean up branch names
                    branch = line.replace('remotes/origin/', '').strip()
                    if branch and branch != 'HEAD':
                        branches.append(branch)
            
            return branches
        except:
            return []
    
    def _get_open_implementing_issues(self) -> List[int]:
        """Get list of open issues in implementing state."""
        try:
            result = subprocess.run([
                'gh', 'issue', 'list', '--state', 'open', 
                '--label', 'state:implementing',
                '--json', 'number'
            ], capture_output=True, text=True, check=True)
            
            issues = json.loads(result.stdout)
            return [issue['number'] for issue in issues]
        except:
            return []
    
    def _get_current_branch(self) -> str:
        """Get current branch name."""
        try:
            result = subprocess.run([
                'git', 'symbolic-ref', '--short', 'HEAD'
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "detached"
    
    def _create_issue_branch(self, issue_number: int, issue_title: str, from_branch: str = "main") -> Dict[str, str]:
        """Create a new branch for an issue."""
        try:
            # Sanitize title for branch name
            sanitized_title = self._sanitize_title(issue_title)
            branch_name = f"issue-{issue_number}-{sanitized_title}"
            
            # Validate branch name length
            if len(branch_name) > 250:
                sanitized_title = sanitized_title[:200]
                branch_name = f"issue-{issue_number}-{sanitized_title}"
            
            # Check if branch already exists
            if self._branch_exists(branch_name):
                return {
                    "status": "exists",
                    "branch_name": branch_name,
                    "message": f"Branch {branch_name} already exists",
                    "issue_number": issue_number
                }
            
            # Create and checkout new branch
            subprocess.run([
                'git', 'checkout', '-b', branch_name, from_branch
            ], cwd=self.repo_path, check=True)
            
            return {
                "status": "created",
                "branch_name": branch_name,
                "message": f"Successfully created branch {branch_name}",
                "issue_number": issue_number,
                "from_branch": from_branch
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "branch_name": branch_name if 'branch_name' in locals() else "unknown",
                "message": f"Git command failed: {e.stderr.decode() if e.stderr else str(e)}",
                "issue_number": issue_number
            }
        except Exception as e:
            return {
                "status": "error", 
                "branch_name": "unknown",
                "message": f"Branch creation failed: {str(e)}",
                "issue_number": issue_number
            }
    
    def _sanitize_title(self, title: str) -> str:
        """Convert issue title to branch-safe string."""
        import re
        # Remove special characters, replace spaces with hyphens
        sanitized = re.sub(r'[^\w\s-]', '', title)
        sanitized = re.sub(r'[-\s]+', '-', sanitized)
        return sanitized.lower().strip('-')[:50]  # Limit length
    
    def _check_hooks_installed(self) -> Dict[str, bool]:
        """Check if git hooks are properly installed."""
        hooks = ["pre-commit", "pre-push"]
        status = {}
        git_dir = self.repo_path / ".git"
        
        for hook in hooks:
            hook_path = git_dir / "hooks" / hook
            status[hook] = hook_path.exists() and os.access(hook_path, os.X_OK)
        
        return status
    
    def _log_enforcement_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log branch enforcement events."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }
        
        self.enforcement_log.parent.mkdir(exist_ok=True)
        
        try:
            with open(self.enforcement_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log enforcement event: {e}")


def main():
    """CLI interface for branch workflow enforcer."""
    if len(sys.argv) < 3:
        print("Usage: python branch_workflow_enforcer.py <command> <issue_number> [args...]")
        print("Commands:")
        print("  enforce <issue_number> [agent_name]        - Enforce issue branch")
        print("  check <issue_number>                       - Check branch compliance")
        print("  prevent <issue_number> <agent_name>        - Prevent protected branch work")
        print("  auto-create <issue_number> <to_state>      - Auto-create branch on state transition")
        print("  validate                                    - Validate branch workflow setup")
        return 1
    
    command = sys.argv[1]
    
    enforcer = BranchWorkflowEnforcer()
    
    if command == "enforce":
        if len(sys.argv) < 3:
            print("Usage: enforce <issue_number> [agent_name]")
            return 1
        issue_number = int(sys.argv[2])
        agent_name = sys.argv[3] if len(sys.argv) > 3 else "system"
        result = enforcer.enforce_issue_branch(issue_number, agent_name)
        print(json.dumps(result, indent=2))
        
    elif command == "check":
        if len(sys.argv) < 3:
            print("Usage: check <issue_number>")
            return 1
        issue_number = int(sys.argv[2])
        result = enforcer.check_branch_compliance(issue_number)
        print(json.dumps(result, indent=2))
        
    elif command == "prevent":
        if len(sys.argv) < 4:
            print("Usage: prevent <issue_number> <agent_name>")
            return 1
        issue_number = int(sys.argv[2])
        agent_name = sys.argv[3]
        result = enforcer.prevent_protected_branch_work(issue_number, agent_name)
        print(json.dumps(result, indent=2))
        
    elif command == "auto-create":
        if len(sys.argv) < 4:
            print("Usage: auto-create <issue_number> <to_state>")
            return 1
        issue_number = int(sys.argv[2])
        to_state = sys.argv[3]
        from_state = sys.argv[4] if len(sys.argv) > 4 else None
        result = enforcer.auto_create_issue_branch(issue_number, from_state, to_state)
        print(json.dumps(result, indent=2))
        
    elif command == "validate":
        result = enforcer.validate_workflow_branch_setup()
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())