"""
Branch Workflow Enforcement System
Integrates branch management with RIF workflow state transitions
"""

import subprocess
import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from claude.commands.branch_manager import BranchManager, WorkflowBranchIntegration


class BranchWorkflowEnforcer:
    """Enforces branch creation and validation during RIF workflow transitions"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.branch_manager = BranchManager(repo_path)
        self.workflow_integration = WorkflowBranchIntegration(self.branch_manager)
        
    def enforce_pre_implementation_branch(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce branch creation before implementation work begins
        Called when transitioning to state:implementing
        """
        issue_number = issue_data.get("number")
        issue_title = issue_data.get("title", f"Issue {issue_number}")
        
        if not issue_number:
            return {
                "status": "error",
                "message": "Issue number required for branch enforcement",
                "blocking": True
            }
        
        # Check if we're already on the correct branch
        validation_result = self.workflow_integration.validate_implementation_ready(issue_data)
        
        if validation_result["valid"]:
            return {
                "status": "validated",
                "message": f"Already on correct branch: {validation_result['current_branch']}",
                "branch_name": validation_result['current_branch'],
                "blocking": False
            }
        
        # Create the issue branch
        creation_result = self.branch_manager.create_issue_branch(issue_number, issue_title)
        
        if creation_result["status"] in ["created", "exists"]:
            # Switch to the branch if it was just created
            if creation_result["status"] == "created":
                try:
                    self._run_git(["checkout", creation_result["branch_name"]])
                except subprocess.CalledProcessError as e:
                    return {
                        "status": "error", 
                        "message": f"Failed to switch to branch {creation_result['branch_name']}: {str(e)}",
                        "blocking": True
                    }
            
            return {
                "status": "enforced",
                "message": f"Branch created and ready: {creation_result['branch_name']}",
                "branch_name": creation_result["branch_name"],
                "blocking": False
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to create branch: {creation_result['message']}",
                "blocking": True
            }
    
    def validate_branch_compliance(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that current branch complies with RIF standards
        Can be called at any time during implementation
        """
        current_branch = self.branch_manager._get_current_branch()
        validation_result = self.branch_manager.validate_branch_naming(current_branch)
        
        if not validation_result["valid"]:
            return {
                "compliant": False,
                "current_branch": current_branch,
                "message": validation_result["message"],
                "required_action": "Switch to proper issue branch or create one"
            }
        
        # Check if branch matches the issue
        issue_number = issue_data.get("number")
        if validation_result["type"] == "issue" and issue_number:
            if validation_result["issue_number"] != issue_number:
                return {
                    "compliant": False,
                    "current_branch": current_branch,
                    "message": f"Branch is for issue #{validation_result['issue_number']} but working on issue #{issue_number}",
                    "required_action": f"Switch to issue-{issue_number} branch"
                }
        
        return {
            "compliant": True,
            "current_branch": current_branch,
            "message": f"Branch compliance verified: {current_branch}",
            "branch_type": validation_result["type"]
        }
    
    def prevent_main_branch_work(self) -> Dict[str, Any]:
        """
        Prevent direct work on main/master branches
        Should be called before any implementation work
        """
        current_branch = self.branch_manager._get_current_branch()
        
        if current_branch in ["main", "master", "develop"]:
            return {
                "blocked": True,
                "current_branch": current_branch,
                "message": f"Direct work on {current_branch} branch is prohibited",
                "required_action": "Create feature branch for this issue"
            }
        
        return {
            "blocked": False,
            "current_branch": current_branch,
            "message": f"Work on {current_branch} is allowed"
        }
    
    def ensure_user_validation_state(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure issue transitions to awaiting-user-validation instead of complete
        Prevents autonomous issue closure
        """
        issue_number = issue_data.get("number")
        
        if not issue_number:
            return {"status": "error", "message": "Issue number required"}
        
        # Check current issue state
        try:
            result = self._run_gh(["issue", "view", str(issue_number), "--json", "state,labels"])
            issue_info = json.loads(result.stdout)
            
            current_labels = [label["name"] for label in issue_info.get("labels", [])]
            
            # Remove any state:complete labels and add state:awaiting-user-validation
            for label in current_labels:
                if label == "state:complete":
                    self._run_gh(["issue", "edit", str(issue_number), "--remove-label", "state:complete"])
            
            # Add awaiting-user-validation label if not present
            if "state:awaiting-user-validation" not in current_labels:
                self._run_gh(["issue", "edit", str(issue_number), "--add-label", "state:awaiting-user-validation"])
            
            return {
                "status": "enforced",
                "message": "Issue state set to awaiting-user-validation",
                "issue_number": issue_number
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Failed to update issue state: {str(e)}"
            }
    
    def post_user_validation_request(self, issue_data: Dict[str, Any], implementation_summary: str) -> Dict[str, Any]:
        """
        Post user validation request comment to issue
        """
        issue_number = issue_data.get("number")
        
        if not issue_number:
            return {"status": "error", "message": "Issue number required"}
        
        validation_comment = f"""## 🚨 IMPLEMENTATION COMPLETE - USER VALIDATION REQUIRED 🚨

**Implementation Summary**: {implementation_summary}

### Please Validate This Implementation

⚠️ **This issue requires YOUR confirmation before it can be closed.** 

Please test the implementation and respond with:
- ✅ **"Confirmed: Implementation works as expected"** (to proceed with closure)
- ❌ **"Issues found: [describe problems]"** (to return for fixes)

### What to Test
1. Verify the functionality works as described in the issue
2. Test edge cases and error scenarios
3. Confirm the solution meets your requirements
4. Check that no regressions were introduced

**IMPORTANT**: Only you can confirm when this issue is truly resolved. Agents cannot close issues without your explicit approval.

---
*This is an automated validation request from RIF. The issue will remain open until you provide confirmation.*"""

        try:
            self._run_gh(["issue", "comment", str(issue_number), "--body", validation_comment])
            
            return {
                "status": "posted",
                "message": "User validation request posted to issue",
                "issue_number": issue_number
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error", 
                "message": f"Failed to post validation request: {str(e)}"
            }
    
    def check_user_validation_response(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if user has responded to validation request
        """
        issue_number = issue_data.get("number")
        
        if not issue_number:
            return {"status": "error", "message": "Issue number required"}
        
        try:
            result = self._run_gh(["issue", "view", str(issue_number), "--comments", "--json", "comments"])
            issue_info = json.loads(result.stdout)
            
            comments = issue_info.get("comments", [])
            
            # Look for recent user validation comments
            for comment in reversed(comments[-10:]):  # Check last 10 comments
                body = comment.get("body", "").lower()
                author = comment.get("author", {}).get("login", "")
                
                # Skip bot/agent comments
                if author in ["github-actions", "rif-bot"] or "rif-" in author:
                    continue
                    
                if "confirmed:" in body and "implementation works" in body:
                    return {
                        "status": "confirmed",
                        "message": "User confirmed implementation works",
                        "can_proceed_to_close": True,
                        "comment_author": author
                    }
                elif "issues found:" in body:
                    return {
                        "status": "issues_found", 
                        "message": "User reported issues with implementation",
                        "can_proceed_to_close": False,
                        "needs_fixes": True,
                        "comment_author": author
                    }
            
            return {
                "status": "awaiting",
                "message": "Still awaiting user validation response",
                "can_proceed_to_close": False
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Failed to check user response: {str(e)}"
            }
    
    # Helper methods
    
    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute git command"""
        return subprocess.run(
            ["git"] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    def _run_gh(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute GitHub CLI command"""
        return subprocess.run(
            ["gh"] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )


# Integration functions for use by agents

def enforce_branch_for_implementation(issue_number: int, issue_title: str) -> Dict[str, Any]:
    """
    Convenience function for agents to enforce branch creation before implementation
    """
    enforcer = BranchWorkflowEnforcer()
    issue_data = {"number": issue_number, "title": issue_title}
    return enforcer.enforce_pre_implementation_branch(issue_data)


def validate_current_branch_compliance(issue_number: int) -> Dict[str, Any]:
    """
    Convenience function for agents to validate branch compliance
    """
    enforcer = BranchWorkflowEnforcer()
    issue_data = {"number": issue_number}
    return enforcer.validate_branch_compliance(issue_data)


def request_user_validation(issue_number: int, implementation_summary: str) -> Dict[str, Any]:
    """
    Convenience function for agents to request user validation
    """
    enforcer = BranchWorkflowEnforcer()
    issue_data = {"number": issue_number}
    
    # First ensure proper state
    state_result = enforcer.ensure_user_validation_state(issue_data)
    if state_result["status"] == "error":
        return state_result
    
    # Then post validation request
    return enforcer.post_user_validation_request(issue_data, implementation_summary)


def check_user_validation_status(issue_number: int) -> Dict[str, Any]:
    """
    Convenience function to check if user has validated implementation
    """
    enforcer = BranchWorkflowEnforcer()
    issue_data = {"number": issue_number}
    return enforcer.check_user_validation_response(issue_data)