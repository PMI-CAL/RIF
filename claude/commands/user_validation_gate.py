#!/usr/bin/env python3
"""
User Validation Gate System
Prevents issues from being closed without explicit user validation.

This system addresses the critical failure identified in issue #232 where 
issue #225 was closed by an agent without user confirmation despite the user
stating the issue was still broken.

Key Features:
1. Mandatory user confirmation before issue closure
2. User validation state tracking
3. Automatic prevention of agent-initiated closures
4. Validation checkpoint system
5. User notification for validation requests
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging


class UserValidationGate:
    """
    System for enforcing user validation before issue closure.
    
    Prevents the critical failure where agents close issues without
    user confirmation, ensuring only users can validate their issues.
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.validation_log = self.repo_path / "knowledge" / "user_validation_log.json"
        self.pending_validations = self.repo_path / "knowledge" / "pending_user_validations.json"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for validation tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - UserValidationGate - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def request_user_validation(self, issue_number: int, agent_name: str, 
                              completion_description: str, validation_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Request user validation for issue completion.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of requesting agent
            completion_description: Description of what was completed
            validation_details: Additional validation context
            
        Returns:
            Dict with validation request results
        """
        try:
            self.logger.info(f"Requesting user validation for issue #{issue_number}")
            
            # Create validation request
            validation_request = {
                "issue_number": issue_number,
                "agent_name": agent_name,
                "completion_description": completion_description,
                "validation_details": validation_details or {},
                "request_timestamp": datetime.now().isoformat(),
                "status": "pending_user_validation",
                "user_response": None,
                "user_response_timestamp": None,
                "validation_id": f"validation_{issue_number}_{int(datetime.now().timestamp())}"
            }
            
            # Store pending validation
            self._store_pending_validation(validation_request)
            
            # Update issue with validation request comment
            validation_comment = self._create_validation_request_comment(
                agent_name, completion_description, validation_request["validation_id"]
            )
            
            # Post comment requesting user validation
            result = subprocess.run([
                'gh', 'issue', 'comment', str(issue_number),
                '--body', validation_comment
            ], capture_output=True, text=True, check=True)
            
            # Add validation state label
            self._add_validation_state_label(issue_number)
            
            # Log validation request
            self._log_validation_event("validation_requested", validation_request)
            
            return {
                "status": "validation_requested",
                "validation_id": validation_request["validation_id"],
                "message": f"User validation requested for issue #{issue_number}",
                "comment_posted": True,
                "issue_number": issue_number
            }
            
        except Exception as e:
            self.logger.error(f"Failed to request user validation for issue #{issue_number}: {e}")
            return {
                "status": "error",
                "message": f"Failed to request validation: {str(e)}",
                "issue_number": issue_number
            }
    
    def check_user_validation_status(self, issue_number: int) -> Dict[str, Any]:
        """
        Check if user has provided validation for an issue.
        
        Args:
            issue_number: GitHub issue number
            
        Returns:
            Dict with validation status information
        """
        try:
            pending_validations = self._load_pending_validations()
            
            # Find validation request for this issue
            for validation_id, validation in pending_validations.items():
                if validation["issue_number"] == issue_number and validation["status"] == "pending_user_validation":
                    
                    # Check for user response in issue comments
                    user_response = self._check_for_user_response(issue_number, validation["request_timestamp"])
                    
                    if user_response:
                        # Update validation with user response
                        validation["user_response"] = user_response["response"]
                        validation["user_response_timestamp"] = user_response["timestamp"]
                        validation["status"] = user_response["validation_result"]
                        
                        # Update stored validation
                        self._update_pending_validation(validation_id, validation)
                        
                        # Log validation completion
                        self._log_validation_event("user_response_received", validation)
                        
                        return {
                            "status": validation["status"],
                            "user_response": user_response["response"],
                            "validation_result": user_response["validation_result"],
                            "can_close": user_response["validation_result"] == "validated",
                            "timestamp": user_response["timestamp"]
                        }
                    else:
                        return {
                            "status": "pending_user_validation",
                            "user_response": None,
                            "can_close": False,
                            "pending_since": validation["request_timestamp"]
                        }
            
            # No pending validation found
            return {
                "status": "no_validation_pending",
                "can_close": False,
                "message": "No user validation request found for this issue"
            }
            
        except Exception as e:
            self.logger.error(f"Error checking validation status for issue #{issue_number}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "can_close": False
            }
    
    def can_close_issue(self, issue_number: int, requesting_entity: str = "system") -> Dict[str, Any]:
        """
        Check if an issue can be closed based on validation requirements.
        
        Args:
            issue_number: GitHub issue number
            requesting_entity: Who is requesting closure (user/agent/system)
            
        Returns:
            Dict with closure permission and reasoning
        """
        try:
            self.logger.info(f"Checking closure permission for issue #{issue_number}")
            
            # Always allow user to close their own issues
            if requesting_entity == "user":
                return {
                    "can_close": True,
                    "reason": "User has authority to close their own issues",
                    "validation_required": False
                }
            
            # Check validation status for non-user requests
            validation_status = self.check_user_validation_status(issue_number)
            
            if validation_status["status"] == "validated":
                return {
                    "can_close": True,
                    "reason": "User has validated the issue resolution",
                    "validation_required": True,
                    "validation_details": validation_status
                }
            elif validation_status["status"] == "rejected":
                return {
                    "can_close": False,
                    "reason": "User has rejected the proposed resolution",
                    "validation_required": True,
                    "validation_details": validation_status
                }
            elif validation_status["status"] == "pending_user_validation":
                return {
                    "can_close": False,
                    "reason": "User validation is pending - issue cannot be closed until user confirms",
                    "validation_required": True,
                    "validation_details": validation_status
                }
            else:
                return {
                    "can_close": False,
                    "reason": "No user validation found - agents cannot close issues without user confirmation",
                    "validation_required": True,
                    "validation_details": validation_status
                }
                
        except Exception as e:
            self.logger.error(f"Error checking closure permission for issue #{issue_number}: {e}")
            return {
                "can_close": False,
                "reason": f"Error checking validation status: {str(e)}",
                "validation_required": True
            }
    
    def prevent_agent_closure(self, issue_number: int, agent_name: str) -> Dict[str, Any]:
        """
        Prevent an agent from closing an issue without user validation.
        
        Args:
            issue_number: GitHub issue number
            agent_name: Name of agent attempting closure
            
        Returns:
            Dict with prevention results
        """
        try:
            closure_permission = self.can_close_issue(issue_number, "agent")
            
            if not closure_permission["can_close"]:
                self.logger.warning(f"Preventing agent {agent_name} from closing issue #{issue_number}")
                
                # Post blocking comment
                blocking_comment = f"""**🚫 CLOSURE BLOCKED BY USER VALIDATION GATE**

Agent `{agent_name}` attempted to close this issue, but closure is blocked pending user validation.

**Reason**: {closure_permission["reason"]}

**User Action Required**: Please review the proposed resolution and respond with:
- ✅ **"Validated"** - if the issue is truly resolved and can be closed
- ❌ **"Rejected"** - if the issue is not resolved and needs more work
- 🔍 **"Needs Review"** - if you need more information before validating

**Critical Rule**: Only users can authorize issue closure to prevent premature closure without proper validation.

---
*This protection was implemented in response to issue #232 where issues were being closed without user confirmation.*"""
                
                subprocess.run([
                    'gh', 'issue', 'comment', str(issue_number),
                    '--body', blocking_comment
                ], check=True)
                
                # Log prevention
                self._log_validation_event("closure_blocked", {
                    "issue_number": issue_number,
                    "agent_name": agent_name,
                    "reason": closure_permission["reason"]
                })
                
                return {
                    "status": "closure_blocked",
                    "message": "Agent closure prevented - user validation required",
                    "issue_number": issue_number,
                    "agent_name": agent_name,
                    "reason": closure_permission["reason"]
                }
            else:
                return {
                    "status": "closure_allowed",
                    "message": "Issue can be closed - user validation confirmed",
                    "issue_number": issue_number
                }
                
        except Exception as e:
            self.logger.error(f"Error preventing agent closure for issue #{issue_number}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "issue_number": issue_number
            }
    
    def _create_validation_request_comment(self, agent_name: str, completion_description: str, validation_id: str) -> str:
        """Create validation request comment for GitHub issue."""
        return f"""**🔍 USER VALIDATION REQUESTED**

Agent `{agent_name}` has completed work on this issue and is requesting user validation before closure.

## What Was Completed
{completion_description}

## User Action Required
Please test the implementation and respond with your validation:

- ✅ **"Validated"** - Issue is resolved and can be closed
- ❌ **"Rejected"** - Issue is not resolved, needs more work  
- 🔍 **"Needs Review"** - Need more information/clarification

## Important
- Only YOU can authorize closure of this issue
- This prevents issues from being closed prematurely
- Your explicit confirmation is required before closure

**Validation ID**: `{validation_id}`

---
*This validation system was implemented to address issue #232 and ensure proper user confirmation.*"""
    
    def _check_for_user_response(self, issue_number: int, since_timestamp: str) -> Optional[Dict[str, Any]]:
        """Check for user validation response in issue comments."""
        try:
            # Get issue comments since validation request
            result = subprocess.run([
                'gh', 'issue', 'view', str(issue_number),
                '--json', 'comments'
            ], capture_output=True, text=True, check=True)
            
            issue_data = json.loads(result.stdout)
            comments = issue_data.get('comments', [])
            
            since_dt = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
            
            # Look for user validation responses
            validation_keywords = {
                'validated': ['validated', 'approve', 'confirmed', 'working', 'resolved', 'fixed'],
                'rejected': ['rejected', 'not working', 'still broken', 'not fixed', 'failed'],
                'needs_review': ['needs review', 'unclear', 'more info', 'clarify']
            }
            
            for comment in comments:
                comment_dt = datetime.fromisoformat(comment['createdAt'].replace('Z', '+00:00'))
                
                if comment_dt > since_dt:
                    comment_body = comment['body'].lower()
                    
                    # Check for validation keywords
                    for validation_type, keywords in validation_keywords.items():
                        if any(keyword in comment_body for keyword in keywords):
                            return {
                                'response': comment['body'],
                                'timestamp': comment['createdAt'],
                                'validation_result': validation_type,
                                'comment_author': comment['author']['login']
                            }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking for user response on issue #{issue_number}: {e}")
            return None
    
    def _store_pending_validation(self, validation_request: Dict[str, Any]):
        """Store pending validation request."""
        pending_validations = self._load_pending_validations()
        pending_validations[validation_request["validation_id"]] = validation_request
        
        self.pending_validations.parent.mkdir(exist_ok=True)
        with open(self.pending_validations, 'w') as f:
            json.dump(pending_validations, f, indent=2)
    
    def _load_pending_validations(self) -> Dict[str, Any]:
        """Load pending validations from storage."""
        if not self.pending_validations.exists():
            return {}
        
        try:
            with open(self.pending_validations, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _update_pending_validation(self, validation_id: str, validation_data: Dict[str, Any]):
        """Update pending validation with new data."""
        pending_validations = self._load_pending_validations()
        pending_validations[validation_id] = validation_data
        
        with open(self.pending_validations, 'w') as f:
            json.dump(pending_validations, f, indent=2)
    
    def _add_validation_state_label(self, issue_number: int):
        """Add awaiting user validation label to issue."""
        try:
            subprocess.run([
                'gh', 'issue', 'edit', str(issue_number),
                '--add-label', 'state:awaiting-user-validation'
            ], check=True)
        except Exception as e:
            self.logger.warning(f"Failed to add validation label to issue #{issue_number}: {e}")
    
    def _log_validation_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log validation events for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }
        
        self.validation_log.parent.mkdir(exist_ok=True)
        
        # Append to log file
        try:
            with open(self.validation_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log validation event: {e}")


def main():
    """CLI interface for user validation gate system."""
    if len(sys.argv) < 3:
        print("Usage: python user_validation_gate.py <command> <issue_number> [args...]")
        print("Commands:")
        print("  request <issue_number> <agent_name> <description>  - Request user validation")
        print("  check <issue_number>                               - Check validation status")
        print("  can-close <issue_number> [requesting_entity]       - Check if issue can be closed")
        print("  prevent <issue_number> <agent_name>                - Prevent agent closure")
        return 1
    
    command = sys.argv[1]
    issue_number = int(sys.argv[2])
    
    gate = UserValidationGate()
    
    if command == "request":
        if len(sys.argv) < 5:
            print("Usage: request <issue_number> <agent_name> <description>")
            return 1
        agent_name = sys.argv[3]
        description = sys.argv[4]
        result = gate.request_user_validation(issue_number, agent_name, description)
        print(json.dumps(result, indent=2))
        
    elif command == "check":
        result = gate.check_user_validation_status(issue_number)
        print(json.dumps(result, indent=2))
        
    elif command == "can-close":
        requesting_entity = sys.argv[3] if len(sys.argv) > 3 else "system"
        result = gate.can_close_issue(issue_number, requesting_entity)
        print(json.dumps(result, indent=2))
        
    elif command == "prevent":
        if len(sys.argv) < 4:
            print("Usage: prevent <issue_number> <agent_name>")
            return 1
        agent_name = sys.argv[3]
        result = gate.prevent_agent_closure(issue_number, agent_name)
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())