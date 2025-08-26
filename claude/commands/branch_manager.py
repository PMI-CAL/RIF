"""
RIF Branch Management System
Automated branch creation, naming, and cleanup for issue-based workflow
"""

import subprocess
import re
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class BranchManager:
    """Core branch management functionality for RIF workflow integration"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.git_dir = self.repo_path / ".git"
        self.emergency_log = self.git_dir / "emergency-overrides.log"
        
    def create_issue_branch(self, issue_number: int, title: str, from_branch: str = "main") -> Dict[str, str]:
        """
        Create a new branch for an issue following RIF naming conventions
        
        Args:
            issue_number: GitHub issue number
            title: Issue title for branch naming
            from_branch: Source branch to create from (default: main)
            
        Returns:
            Dict with branch creation results
        """
        try:
            # Sanitize title for branch name
            sanitized_title = self._sanitize_title(title)
            branch_name = f"issue-{issue_number}-{sanitized_title}"
            
            # Validate branch name length (git has limits)
            if len(branch_name) > 250:
                # Truncate but keep issue number
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
            
            # Ensure we're up to date with remote
            self._run_git(["fetch", "origin", from_branch])
            
            # Create and checkout new branch
            self._run_git(["checkout", "-b", branch_name, f"origin/{from_branch}"])
            
            # Push new branch to remote with upstream tracking
            self._run_git(["push", "-u", "origin", branch_name])
            
            # Log branch creation
            self._log_branch_operation("create", branch_name, issue_number)
            
            return {
                "status": "created",
                "branch_name": branch_name,
                "message": f"Successfully created and pushed branch {branch_name}",
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
    
    def handle_emergency_branch(self, incident_id: str, description: str) -> Dict[str, str]:
        """
        Create an emergency branch with special naming and logging
        
        Args:
            incident_id: Unique incident identifier
            description: Brief description of emergency
            
        Returns:
            Dict with emergency branch creation results
        """
        try:
            sanitized_desc = self._sanitize_title(description)
            branch_name = f"emergency-{incident_id}-{sanitized_desc}"
            
            # Emergency branches are created from main and immediately tracked
            self._run_git(["fetch", "origin", "main"])
            self._run_git(["checkout", "-b", branch_name, "origin/main"])
            self._run_git(["push", "-u", "origin", branch_name])
            
            # Log emergency branch creation with special handling
            self._log_emergency_branch(incident_id, branch_name, description)
            
            return {
                "status": "emergency_created",
                "branch_name": branch_name,
                "message": f"Emergency branch {branch_name} created for incident {incident_id}",
                "incident_id": incident_id,
                "compliance_required": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "branch_name": "unknown", 
                "message": f"Emergency branch creation failed: {str(e)}",
                "incident_id": incident_id
            }
    
    def validate_branch_naming(self, branch_name: str) -> Dict[str, any]:
        """
        Validate if branch follows RIF naming conventions
        
        Args:
            branch_name: Branch name to validate
            
        Returns:
            Dict with validation results
        """
        patterns = {
            "issue": re.compile(r'^issue-(\d+)-.+'),
            "emergency": re.compile(r'^emergency-([^-]+)-.+'),
            "main": re.compile(r'^main$')
        }
        
        for pattern_type, pattern in patterns.items():
            match = pattern.match(branch_name)
            if match:
                if pattern_type == "issue":
                    return {
                        "valid": True,
                        "type": "issue",
                        "issue_number": int(match.group(1)),
                        "message": f"Valid issue branch for issue #{match.group(1)}"
                    }
                elif pattern_type == "emergency":
                    return {
                        "valid": True,
                        "type": "emergency", 
                        "incident_id": match.group(1),
                        "message": f"Valid emergency branch for incident {match.group(1)}"
                    }
                elif pattern_type == "main":
                    return {
                        "valid": False,
                        "type": "main",
                        "message": "Direct work on main branch not allowed"
                    }
        
        return {
            "valid": False,
            "type": "unknown",
            "message": f"Branch name '{branch_name}' doesn't follow RIF conventions"
        }
    
    def cleanup_merged_branches(self, exclude_patterns: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Clean up merged branches following retention policies
        
        Args:
            exclude_patterns: Patterns to exclude from cleanup
            
        Returns:
            Dict with cleanup results
        """
        exclude_patterns = exclude_patterns or ["main", "develop", "staging"]
        
        try:
            # Get merged branches
            merged_branches = self._get_merged_branches()
            
            # Filter out excluded patterns
            cleanup_candidates = []
            for branch in merged_branches:
                should_exclude = False
                for pattern in exclude_patterns:
                    if re.search(pattern, branch):
                        should_exclude = True
                        break
                
                if not should_exclude:
                    cleanup_candidates.append(branch)
            
            # Check age and clean up old branches
            cleaned_branches = []
            errors = []
            
            for branch in cleanup_candidates:
                try:
                    # Get last commit date for branch
                    last_commit_date = self._get_branch_last_commit_date(branch)
                    age_days = (datetime.now() - last_commit_date).days
                    
                    # Clean up branches older than 7 days
                    if age_days > 7:
                        # Delete local branch
                        try:
                            self._run_git(["branch", "-d", branch])
                            cleaned_branches.append(f"local/{branch}")
                        except subprocess.CalledProcessError:
                            # Force delete if needed
                            self._run_git(["branch", "-D", branch])
                            cleaned_branches.append(f"local/{branch} (forced)")
                        
                        # Delete remote branch
                        try:
                            self._run_git(["push", "origin", "--delete", branch])
                            cleaned_branches.append(f"remote/{branch}")
                        except subprocess.CalledProcessError as e:
                            if "remote ref does not exist" not in str(e):
                                errors.append(f"Failed to delete remote/{branch}: {str(e)}")
                    
                except Exception as e:
                    errors.append(f"Failed to process branch {branch}: {str(e)}")
            
            self._log_branch_operation("cleanup", cleaned_branches)
            
            return {
                "status": "completed",
                "cleaned_branches": cleaned_branches,
                "errors": errors,
                "candidates_checked": len(cleanup_candidates),
                "branches_cleaned": len(cleaned_branches)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cleanup failed: {str(e)}",
                "cleaned_branches": [],
                "errors": [str(e)]
            }
    
    def enforce_branch_protection(self, branch_patterns: List[str]) -> Dict[str, any]:
        """
        Enforce branch protection rules (mainly through git hooks)
        
        Args:
            branch_patterns: Patterns of branches to protect
            
        Returns:
            Dict with protection status
        """
        current_branch = self._get_current_branch()
        
        protection_status = {
            "current_branch": current_branch,
            "protected_branches": branch_patterns,
            "enforcement_active": True,
            "hooks_installed": self._check_hooks_installed()
        }
        
        # Check if current branch is protected
        for pattern in branch_patterns:
            if re.search(pattern, current_branch):
                protection_status["current_branch_protected"] = True
                protection_status["protection_message"] = f"Branch {current_branch} is protected by pattern: {pattern}"
                break
        else:
            protection_status["current_branch_protected"] = False
        
        return protection_status
    
    def integrate_with_workflow(self, workflow_config: Dict) -> Dict[str, any]:
        """
        Integration point with RIF workflow system
        
        Args:
            workflow_config: RIF workflow configuration
            
        Returns:
            Integration status and configuration
        """
        integration_config = {
            "branch_creation_triggers": [
                "state:planning->implementing", 
                "state:architecting->implementing"
            ],
            "auto_pr_creation": workflow_config.get("github", {}).get("auto_pr", {}).get("enabled", True),
            "branch_protection_patterns": ["main", "master", "develop"],
            "naming_conventions": {
                "issue": "issue-{number}-{sanitized-title}",
                "emergency": "emergency-{incident-id}-{sanitized-description}"
            }
        }
        
        return {
            "status": "integrated",
            "configuration": integration_config,
            "hooks_active": self._check_hooks_installed(),
            "workflow_version": workflow_config.get("version", "unknown")
        }
    
    # Private helper methods
    
    def _sanitize_title(self, title: str) -> str:
        """Convert issue title to branch-safe string"""
        # Remove special characters, replace spaces with hyphens
        sanitized = re.sub(r'[^\w\s-]', '', title)
        sanitized = re.sub(r'[-\s]+', '-', sanitized)
        return sanitized.lower().strip('-')[:50]  # Limit length
    
    def _branch_exists(self, branch_name: str) -> bool:
        """Check if branch exists locally or remotely"""
        try:
            # Check local
            self._run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"])
            return True
        except subprocess.CalledProcessError:
            pass
        
        try:
            # Check remote
            self._run_git(["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch_name}"])
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute git command in repo directory"""
        return subprocess.run(
            ["git"] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    def _get_current_branch(self) -> str:
        """Get current branch name"""
        try:
            result = self._run_git(["symbolic-ref", "--short", "HEAD"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "detached"
    
    def _get_merged_branches(self) -> List[str]:
        """Get list of merged branches"""
        try:
            result = self._run_git(["branch", "--merged", "main"])
            branches = [line.strip().lstrip('* ') for line in result.stdout.split('\n') if line.strip()]
            # Remove main itself
            return [b for b in branches if b != "main"]
        except subprocess.CalledProcessError:
            return []
    
    def _get_branch_last_commit_date(self, branch_name: str) -> datetime:
        """Get last commit date for a branch"""
        try:
            result = self._run_git(["log", "-1", "--format=%ci", branch_name])
            date_str = result.stdout.strip()
            return datetime.fromisoformat(date_str.replace(' +', '+').replace(' -', '-'))
        except (subprocess.CalledProcessError, ValueError):
            # If we can't get date, assume recent
            return datetime.now()
    
    def _check_hooks_installed(self) -> Dict[str, bool]:
        """Check if git hooks are properly installed"""
        hooks = ["pre-commit", "pre-push"]
        status = {}
        
        for hook in hooks:
            hook_path = self.git_dir / "hooks" / hook
            status[hook] = hook_path.exists() and os.access(hook_path, os.X_OK)
        
        return status
    
    def _log_branch_operation(self, operation: str, branch_info, issue_number: int = None):
        """Log branch operations for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "branch_info": branch_info,
            "issue_number": issue_number
        }
        
        log_file = self.repo_path / "knowledge" / "branch_operations.log"
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _log_emergency_branch(self, incident_id: str, branch_name: str, description: str):
        """Log emergency branch creation with special handling"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "emergency_branch",
            "incident_id": incident_id,
            "branch_name": branch_name,
            "description": description,
            "compliance_required": True,
            "created_by": "branch_manager"
        }
        
        emergency_log_file = self.repo_path / "knowledge" / "emergency_branches.log"
        emergency_log_file.parent.mkdir(exist_ok=True)
        
        with open(emergency_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class WorkflowBranchIntegration:
    """Integration layer between BranchManager and RIF workflow"""
    
    def __init__(self, branch_manager: BranchManager):
        self.branch_manager = branch_manager
    
    def on_state_transition(self, from_state: str, to_state: str, issue_data: Dict) -> Dict[str, any]:
        """Handle branch operations on workflow state transitions"""
        
        # Create branch when transitioning to implementing
        if to_state == "implementing" and from_state in ["planning", "architecting"]:
            issue_number = issue_data.get("number")
            issue_title = issue_data.get("title", f"Issue {issue_number}")
            
            if issue_number:
                return self.branch_manager.create_issue_branch(issue_number, issue_title)
        
        return {"status": "no_action", "message": f"No branch action for {from_state}->{to_state}"}
    
    def validate_implementation_ready(self, issue_data: Dict) -> Dict[str, any]:
        """Validate that proper branch exists before allowing implementation"""
        issue_number = issue_data.get("number")
        
        if not issue_number:
            return {"valid": False, "message": "Issue number required for branch validation"}
        
        expected_branch = f"issue-{issue_number}-"
        current_branch = self.branch_manager._get_current_branch()
        
        if current_branch.startswith(expected_branch):
            return {
                "valid": True,
                "current_branch": current_branch,
                "message": f"Implementation ready on proper branch: {current_branch}"
            }
        else:
            return {
                "valid": False, 
                "current_branch": current_branch,
                "expected_prefix": expected_branch,
                "message": f"Must be on issue branch to implement issue #{issue_number}"
            }