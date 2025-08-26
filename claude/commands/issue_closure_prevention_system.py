#!/usr/bin/env python3
"""
Issue Closure Prevention System
Comprehensive system to prevent the critical failures identified in issue #232.

This system integrates:
1. User validation gates
2. Branch workflow enforcement  
3. Agent behavior monitoring
4. Mandatory validation checkpoints

Critical Rules Enforced:
- ONLY users can close their issues (agents cannot)
- All work must happen on proper issue branches
- User validation is mandatory before closure
- Agents must request validation, never assume completion
"""

import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Import our validation systems
try:
    from .user_validation_gate import UserValidationGate
    from .branch_workflow_enforcer import BranchWorkflowEnforcer
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from user_validation_gate import UserValidationGate
    from branch_workflow_enforcer import BranchWorkflowEnforcer


class IssueClosurePreventionSystem:
    """
    Comprehensive system to prevent improper issue closure and branch violations.
    
    This addresses the specific failures from issue #232:
    1. Issue #225 closed without user validation
    2. Work happening on wrong branches
    3. Agents claiming success without user confirmation
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.user_validation_gate = UserValidationGate(str(repo_path))
        self.branch_enforcer = BranchWorkflowEnforcer(str(repo_path))
        self.prevention_log = self.repo_path / "knowledge" / "issue_closure_prevention_log.json"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for prevention system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - IssueClosurePrevention - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_agent_work_start(self, issue_number: int, agent_name: str) -> Dict[str, Any]:
        """
        Validate that an agent can start work on an issue.
        
        This enforces proper branch setup before any work begins.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of agent requesting to start work
            
        Returns:
            Dict with validation results and required actions
        """
        try:
            self.logger.info(f"Validating work start for agent {agent_name} on issue #{issue_number}")
            
            validation_results = {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "can_start_work": False,
                "validations": {},
                "required_actions": [],
                "warnings": []
            }
            
            # 1. Validate branch compliance
            branch_compliance = self.branch_enforcer.check_branch_compliance(issue_number)
            validation_results["validations"]["branch_compliance"] = branch_compliance
            
            if not branch_compliance["is_compliant"]:
                validation_results["required_actions"].extend([
                    f"Switch to proper issue branch for #{issue_number}",
                    "Run branch enforcement to create/switch to correct branch"
                ])
                
                # Automatically enforce branch if possible
                enforcement_result = self.branch_enforcer.enforce_issue_branch(issue_number, agent_name)
                validation_results["validations"]["branch_enforcement"] = enforcement_result
                
                if enforcement_result["status"] == "enforced":
                    validation_results["warnings"].append("Automatically switched to correct branch")
                    validation_results["can_start_work"] = True
                else:
                    validation_results["can_start_work"] = False
                    validation_results["required_actions"].append("Manual branch fix required")
            else:
                validation_results["can_start_work"] = True
            
            # 2. Check for existing validation requests
            existing_validation = self.user_validation_gate.check_user_validation_status(issue_number)
            validation_results["validations"]["existing_validation"] = existing_validation
            
            if existing_validation["status"] == "pending_user_validation":
                validation_results["warnings"].append("Issue has pending user validation - ensure work doesn't conflict")
            
            # Log validation
            self._log_prevention_event("agent_work_validation", validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating agent work start: {e}")
            return {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "can_start_work": False,
                "error": str(e),
                "required_actions": ["Fix validation system error before proceeding"]
            }
    
    def prevent_improper_closure(self, issue_number: int, agent_name: str, 
                                completion_claim: str) -> Dict[str, Any]:
        """
        Prevent agents from improperly closing issues.
        
        This is the core prevention mechanism that addresses issue #232.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of agent attempting closure
            completion_claim: Agent's claim about what was completed
            
        Returns:
            Dict with prevention results and actions taken
        """
        try:
            self.logger.warning(f"Preventing improper closure attempt by {agent_name} on issue #{issue_number}")
            
            # Check if closure is allowed
            closure_permission = self.user_validation_gate.can_close_issue(issue_number, "agent")
            
            prevention_results = {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "completion_claim": completion_claim,
                "closure_prevented": False,
                "prevention_reason": "",
                "actions_taken": [],
                "user_validation_required": True
            }
            
            if not closure_permission["can_close"]:
                # PREVENT CLOSURE
                prevention_results["closure_prevented"] = True
                prevention_results["prevention_reason"] = closure_permission["reason"]
                
                # Block the closure attempt
                block_result = self.user_validation_gate.prevent_agent_closure(issue_number, agent_name)
                prevention_results["actions_taken"].append("Posted closure blocking comment")
                
                # Request user validation instead
                validation_request = self.user_validation_gate.request_user_validation(
                    issue_number, 
                    agent_name, 
                    completion_claim
                )
                prevention_results["actions_taken"].append("Requested user validation")
                prevention_results["validation_request"] = validation_request
                
                # Update issue state to await validation
                try:
                    subprocess.run([
                        'gh', 'issue', 'edit', str(issue_number),
                        '--add-label', 'state:awaiting-user-validation'
                    ], check=True)
                    prevention_results["actions_taken"].append("Added awaiting-user-validation label")
                except:
                    prevention_results["warnings"] = ["Failed to add validation label"]
                
                # Log critical prevention event
                self._log_prevention_event("closure_prevented", {
                    "issue_number": issue_number,
                    "agent_name": agent_name,
                    "completion_claim": completion_claim,
                    "prevention_reason": prevention_results["prevention_reason"],
                    "validation_requested": True
                })
                
                self.logger.warning(f"🚫 PREVENTED: Agent {agent_name} attempted to close issue #{issue_number} without user validation")
                
            else:
                # Closure is allowed (user has validated)
                prevention_results["closure_prevented"] = False
                prevention_results["prevention_reason"] = "User validation confirmed"
                self.logger.info(f"✅ ALLOWED: Issue #{issue_number} closure authorized by user validation")
            
            return prevention_results
            
        except Exception as e:
            self.logger.error(f"Error preventing improper closure: {e}")
            return {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "closure_prevented": True,
                "prevention_reason": f"System error: {str(e)}",
                "error": True
            }
    
    def enforce_validation_workflow(self, issue_number: int, agent_name: str, 
                                  workflow_state: str) -> Dict[str, Any]:
        """
        Enforce proper validation workflow for issue state transitions.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of agent requesting state change
            workflow_state: Target workflow state
            
        Returns:
            Dict with enforcement results
        """
        try:
            enforcement_results = {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "target_state": workflow_state,
                "enforcement_applied": False,
                "validations": {},
                "actions_taken": []
            }
            
            # Enforce branch requirements for implementing state
            if workflow_state == "implementing":
                branch_validation = self.validate_agent_work_start(issue_number, agent_name)
                enforcement_results["validations"]["branch_setup"] = branch_validation
                
                if branch_validation["can_start_work"]:
                    enforcement_results["actions_taken"].append("Branch validation passed")
                else:
                    enforcement_results["enforcement_applied"] = True
                    enforcement_results["actions_taken"].append("Branch enforcement applied")
            
            # Prevent direct transition to complete without validation
            elif workflow_state in ["complete", "closed"]:
                closure_prevention = self.prevent_improper_closure(
                    issue_number, 
                    agent_name, 
                    f"Agent attempting to transition to {workflow_state}"
                )
                enforcement_results["validations"]["closure_prevention"] = closure_prevention
                
                if closure_prevention["closure_prevented"]:
                    enforcement_results["enforcement_applied"] = True
                    enforcement_results["actions_taken"].append("Closure prevented - user validation required")
            
            # Log enforcement
            self._log_prevention_event("workflow_enforcement", enforcement_results)
            
            return enforcement_results
            
        except Exception as e:
            self.logger.error(f"Error enforcing validation workflow: {e}")
            return {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "enforcement_applied": True,
                "error": str(e)
            }
    
    def audit_recent_closures(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Audit recent issue closures to detect validation failures.
        
        Args:
            days_back: Number of days to audit
            
        Returns:
            Dict with audit results
        """
        try:
            self.logger.info(f"Auditing issue closures from last {days_back} days")
            
            # Get recently closed issues
            result = subprocess.run([
                'gh', 'issue', 'list', '--state', 'closed',
                '--json', 'number,title,closedAt,comments',
                '--limit', '50'
            ], capture_output=True, text=True, check=True)
            
            closed_issues = json.loads(result.stdout)
            
            audit_results = {
                "audit_period_days": days_back,
                "total_closed_issues": len(closed_issues),
                "validation_violations": [],
                "properly_validated": [],
                "suspicious_closures": []
            }
            
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_back)
            
            for issue in closed_issues:
                closed_at = datetime.fromisoformat(issue['closedAt'].replace('Z', '+00:00'))
                
                if closed_at >= cutoff_date:
                    issue_number = issue['number']
                    
                    # Check for user validation
                    validation_status = self.user_validation_gate.check_user_validation_status(issue_number)
                    
                    if validation_status["status"] == "validated":
                        audit_results["properly_validated"].append({
                            "issue": issue_number,
                            "title": issue['title'],
                            "closed_at": issue['closedAt'],
                            "validation_confirmed": True
                        })
                    elif validation_status["status"] == "no_validation_pending":
                        # Check if this looks suspicious (closed without validation request)
                        audit_results["suspicious_closures"].append({
                            "issue": issue_number,
                            "title": issue['title'],
                            "closed_at": issue['closedAt'],
                            "reason": "No validation request found"
                        })
                    else:
                        audit_results["validation_violations"].append({
                            "issue": issue_number,
                            "title": issue['title'], 
                            "closed_at": issue['closedAt'],
                            "validation_status": validation_status["status"]
                        })
            
            # Log audit results
            self._log_prevention_event("closure_audit", audit_results)
            
            return audit_results
            
        except Exception as e:
            self.logger.error(f"Error auditing recent closures: {e}")
            return {
                "audit_period_days": days_back,
                "error": str(e),
                "total_closed_issues": 0
            }
    
    def install_prevention_hooks(self) -> Dict[str, Any]:
        """
        Install git hooks and GitHub workflows for prevention system.
        
        Returns:
            Dict with installation results
        """
        installation_results = {
            "git_hooks_installed": False,
            "github_workflows_created": False,
            "prevention_active": False,
            "installation_details": []
        }
        
        try:
            # Create pre-commit hook for branch enforcement
            git_dir = self.repo_path / ".git"
            hooks_dir = git_dir / "hooks"
            hooks_dir.mkdir(exist_ok=True)
            
            pre_commit_hook = hooks_dir / "pre-commit"
            hook_content = """#!/bin/bash
# RIF Issue Closure Prevention System - Pre-commit Hook

# Check if we're on a protected branch
current_branch=$(git symbolic-ref --short HEAD)
if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
    echo "🚫 BLOCKED: Direct commits to $current_branch are not allowed"
    echo "Please work on an issue-specific branch"
    exit 1
fi

# Check if branch follows issue convention
if [[ ! "$current_branch" =~ ^issue-[0-9]+ ]]; then
    echo "⚠️  WARNING: Branch $current_branch doesn't follow issue naming convention"
    echo "Expected format: issue-<number>-<description>"
    echo "Continuing anyway, but consider renaming for clarity"
fi
"""
            
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)
            
            pre_commit_hook.chmod(0o755)
            installation_results["git_hooks_installed"] = True
            installation_results["installation_details"].append("Pre-commit hook installed")
            
            # Create validation reminder template
            templates_dir = self.repo_path / ".github" / "ISSUE_TEMPLATE"
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            validation_reminder = templates_dir / "validation_reminder.md"
            reminder_content = """---
name: Validation Reminder
about: Reminder about user validation requirements
title: 'Remember: Only YOU can close your issues'
labels: ['validation-reminder']
assignees: ''
---

## 🔍 User Validation Required

**Important**: This issue can only be closed by YOU, the issue creator.

When agents complete work on your issue, they will:
1. Request your validation
2. Wait for your confirmation
3. NOT close the issue themselves

**Your Response Options:**
- ✅ "Validated" - Issue is resolved and can be closed
- ❌ "Rejected" - Issue is not resolved, needs more work
- 🔍 "Needs Review" - Need more information before validating

This system prevents issues from being closed prematurely without your confirmation.
"""
            
            with open(validation_reminder, 'w') as f:
                f.write(reminder_content)
            
            installation_results["github_workflows_created"] = True
            installation_results["installation_details"].append("GitHub templates created")
            
            installation_results["prevention_active"] = True
            self.logger.info("Issue closure prevention system installed successfully")
            
        except Exception as e:
            installation_results["error"] = str(e)
            self.logger.error(f"Error installing prevention hooks: {e}")
        
        return installation_results
    
    def _log_prevention_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log prevention events for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }
        
        self.prevention_log.parent.mkdir(exist_ok=True)
        
        try:
            with open(self.prevention_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log prevention event: {e}")


def main():
    """CLI interface for issue closure prevention system."""
    if len(sys.argv) < 2:
        print("Usage: python issue_closure_prevention_system.py <command> [args...]")
        print("Commands:")
        print("  validate-start <issue_number> <agent_name>     - Validate agent can start work")
        print("  prevent-closure <issue_number> <agent_name> <claim> - Prevent improper closure")
        print("  enforce-workflow <issue_number> <agent_name> <state> - Enforce workflow validation")
        print("  audit-closures [days_back]                      - Audit recent closures")
        print("  install-hooks                                   - Install prevention system")
        return 1
    
    command = sys.argv[1]
    
    prevention_system = IssueClosurePreventionSystem()
    
    if command == "validate-start":
        if len(sys.argv) < 4:
            print("Usage: validate-start <issue_number> <agent_name>")
            return 1
        issue_number = int(sys.argv[2])
        agent_name = sys.argv[3]
        result = prevention_system.validate_agent_work_start(issue_number, agent_name)
        print(json.dumps(result, indent=2))
        
    elif command == "prevent-closure":
        if len(sys.argv) < 5:
            print("Usage: prevent-closure <issue_number> <agent_name> <completion_claim>")
            return 1
        issue_number = int(sys.argv[2])
        agent_name = sys.argv[3]
        completion_claim = sys.argv[4]
        result = prevention_system.prevent_improper_closure(issue_number, agent_name, completion_claim)
        print(json.dumps(result, indent=2))
        
    elif command == "enforce-workflow":
        if len(sys.argv) < 5:
            print("Usage: enforce-workflow <issue_number> <agent_name> <workflow_state>")
            return 1
        issue_number = int(sys.argv[2])
        agent_name = sys.argv[3]
        workflow_state = sys.argv[4]
        result = prevention_system.enforce_validation_workflow(issue_number, agent_name, workflow_state)
        print(json.dumps(result, indent=2))
        
    elif command == "audit-closures":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        result = prevention_system.audit_recent_closures(days_back)
        print(json.dumps(result, indent=2))
        
    elif command == "install-hooks":
        result = prevention_system.install_prevention_hooks()
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())