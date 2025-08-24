#!/usr/bin/env python3
"""
GitHub Branch Protection Rollback Script
Issue #203: Configure GitHub branch protection rules for main branch

Emergency script to remove all branch protection rules when immediate access to main branch is required.
This script provides comprehensive audit logging and safety checks for emergency situations.
"""

import json
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add the project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from claude.commands.github_api_client import get_api_client, APICallResult
except ImportError:
    # Fallback for testing - create mock client
    logger.warning("Could not import GitHubAPIClient - using mock for testing")
    
    class MockAPICallResult:
        def __init__(self, success=True, data=None, error_message=None):
            self.success = success
            self.data = data
            self.error_message = error_message
            self.attempt_count = 1
            self.context_id = "mock_context"
            self.rate_limit_remaining = None
    
    class MockGitHubAPIClient:
        def get_branch_protection(self, branch):
            return MockAPICallResult(False, None, "No protection found (mock)")
        
        def remove_branch_protection(self, branch, audit_reason):
            return MockAPICallResult(True, {"message": "Mock protection removed"})
    
    def get_api_client():
        return MockGitHubAPIClient()
    
    APICallResult = MockAPICallResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergencyRollbackManager:
    """Manages emergency rollback of GitHub branch protection rules."""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.emergency_log_path = Path(".github/branch-protection-emergency-log.json")
        self.audit_log: List[Dict[str, Any]] = []
        self._ensure_log_directory()
        
    def _ensure_log_directory(self):
        """Ensure the log directory exists."""
        self.emergency_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def validate_emergency_authorization(
        self,
        reason: str,
        emergency_contact: Optional[str] = None,
        approval_timestamp: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that emergency override is properly authorized."""
        
        # Valid emergency reasons
        valid_reasons = [
            "critical_security_patch",
            "emergency_hotfix", 
            "system_outage_recovery",
            "deployment_blocker",
            "quarterly_test",  # For testing procedures
            "manual_rollback"   # For manual administrative actions
        ]
        
        if reason not in valid_reasons:
            return False, f"Invalid emergency reason. Must be one of: {', '.join(valid_reasons)}"
        
        # For critical emergencies, require additional validation
        critical_reasons = ["critical_security_patch", "emergency_hotfix", "system_outage_recovery"]
        if reason in critical_reasons:
            if not emergency_contact:
                return False, f"Emergency contact required for {reason}"
            if not approval_timestamp:
                return False, f"Approval timestamp required for {reason}"
        
        return True, None
    
    def get_current_protection(self, branch: str = "main") -> Dict[str, Any]:
        """Get current branch protection configuration before removal."""
        logger.info(f"Retrieving current protection configuration for branch '{branch}'")
        
        result = self.api_client.get_branch_protection(branch)
        
        if result.success:
            logger.info("Current branch protection configuration retrieved")
            return {
                "exists": True,
                "config": result.data,
                "retrieved_at": datetime.now().isoformat()
            }
        else:
            logger.info("No branch protection currently configured")
            return {
                "exists": False,
                "error": result.error_message,
                "retrieved_at": datetime.now().isoformat()
            }
    
    def create_restoration_backup(
        self, 
        branch: str, 
        current_protection: Dict[str, Any]
    ) -> str:
        """Create backup of current configuration for later restoration."""
        backup_filename = f"branch-protection-backup-{branch}-{int(time.time())}.json"
        backup_path = Path(".github") / backup_filename
        
        backup_data = {
            "branch": branch,
            "backup_timestamp": datetime.now().isoformat(),
            "protection_config": current_protection,
            "restoration_notes": "Use scripts/setup_branch_protection.py to restore"
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Protection configuration backed up to {backup_path}")
        return str(backup_path)
    
    def remove_branch_protection(
        self, 
        branch: str, 
        audit_reason: str,
        emergency_contact: Optional[str] = None,
        approval_timestamp: Optional[str] = None,
        confirm: bool = False
    ) -> bool:
        """Remove branch protection with full audit logging."""
        
        # Get current protection before removal
        current_protection = self.get_current_protection(branch)
        
        if not current_protection["exists"]:
            logger.info(f"No branch protection found on '{branch}' - nothing to remove")
            return True
        
        # Create restoration backup
        backup_path = self.create_restoration_backup(branch, current_protection)
        
        # Confirmation check (unless overridden)
        if not confirm:
            print("\n" + "="*60)
            print("🚨 EMERGENCY BRANCH PROTECTION REMOVAL 🚨")
            print("="*60)
            print(f"Branch: {branch}")
            print(f"Reason: {audit_reason}")
            print(f"Emergency Contact: {emergency_contact or 'Not specified'}")
            print(f"Approval Timestamp: {approval_timestamp or 'Not specified'}")
            print(f"Backup Created: {backup_path}")
            print("\nThis will remove ALL protection rules from the specified branch.")
            print("The branch will allow direct pushes, bypassing reviews and status checks.")
            print("\n⚠️  ENSURE YOU RESTORE PROTECTION AFTER EMERGENCY WORK IS COMPLETE ⚠️")
            print("="*60)
            
            response = input("\nType 'EMERGENCY' to confirm removal: ")
            if response != "EMERGENCY":
                print("Rollback cancelled.")
                return False
        
        # Execute rollback
        logger.warning(f"🚨 REMOVING BRANCH PROTECTION for '{branch}' - Emergency: {audit_reason}")
        
        start_time = time.time()
        result = self.api_client.remove_branch_protection(branch, audit_reason)
        duration = time.time() - start_time
        
        # Create comprehensive audit log entry
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "emergency_rollback",
            "branch": branch,
            "reason": audit_reason,
            "emergency_contact": emergency_contact,
            "approval_timestamp": approval_timestamp,
            "success": result.success,
            "duration_seconds": round(duration, 2),
            "error": result.error_message,
            "previous_config_backup": backup_path,
            "api_call_details": {
                "attempt_count": result.attempt_count,
                "context_id": result.context_id,
                "rate_limit_remaining": result.rate_limit_remaining
            },
            "restoration_command": f"python3 scripts/setup_branch_protection.py --branch {branch}",
            "emergency_override_id": f"emergency_{branch}_{int(time.time())}"
        }
        
        self.audit_log.append(audit_entry)
        
        if result.success:
            logger.info("✅ Branch protection removed successfully")
            print("\n" + "="*60)
            print("✅ EMERGENCY ROLLBACK COMPLETED")
            print("="*60)
            print(f"Branch '{branch}' protection has been removed")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Backup saved to: {backup_path}")
            print(f"Emergency ID: {audit_entry['emergency_override_id']}")
            print("\n🔄 TO RESTORE PROTECTION AFTER EMERGENCY WORK:")
            print(f"   python3 scripts/setup_branch_protection.py --branch {branch}")
            print("\n📝 REMEMBER TO:")
            print("   1. Complete emergency work as quickly as possible")
            print("   2. Restore protection immediately after")
            print("   3. Create post-emergency review PR")
            print("   4. Document incident in issue tracker")
            print("="*60)
            return True
        else:
            logger.error(f"❌ Branch protection removal failed: {result.error_message}")
            print(f"\n❌ EMERGENCY ROLLBACK FAILED: {result.error_message}")
            return False
    
    def save_emergency_log(self) -> None:
        """Save emergency audit log to persistent file."""
        existing_log = []
        
        # Load existing log if it exists
        if self.emergency_log_path.exists():
            try:
                with open(self.emergency_log_path, 'r') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Existing emergency log is corrupted, starting fresh")
        
        # Append new entries
        all_entries = existing_log + self.audit_log
        
        # Save combined log
        with open(self.emergency_log_path, 'w') as f:
            json.dump(all_entries, f, indent=2, default=str)
        
        logger.info(f"Emergency audit log updated: {self.emergency_log_path}")
    
    def show_recent_emergencies(self, limit: int = 10) -> None:
        """Display recent emergency overrides for context."""
        if not self.emergency_log_path.exists():
            print("No emergency override history found.")
            return
        
        try:
            with open(self.emergency_log_path, 'r') as f:
                log_entries = json.load(f)
        except json.JSONDecodeError:
            print("Emergency log file is corrupted.")
            return
        
        if not log_entries:
            print("No emergency overrides recorded.")
            return
        
        recent_entries = log_entries[-limit:]
        
        print(f"\n📊 RECENT EMERGENCY OVERRIDES (last {len(recent_entries)})")
        print("="*80)
        
        for entry in recent_entries:
            status = "✅ SUCCESS" if entry.get("success") else "❌ FAILED"
            timestamp = entry.get("timestamp", "Unknown")
            reason = entry.get("reason", "No reason")
            branch = entry.get("branch", "Unknown")
            contact = entry.get("emergency_contact", "Not specified")
            
            print(f"{status} | {timestamp}")
            print(f"   Branch: {branch} | Reason: {reason}")
            print(f"   Contact: {contact}")
            if entry.get("error"):
                print(f"   Error: {entry['error']}")
            print()
        
        print("="*80)

def main():
    """Main entry point for the rollback script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Emergency Branch Protection Rollback",
        epilog="⚠️  This script removes ALL branch protection rules. Use only in genuine emergencies!"
    )
    parser.add_argument("--branch", default="main",
                       help="Branch to remove protection from (default: main)")
    parser.add_argument("--reason", required=False,
                       choices=["critical_security_patch", "emergency_hotfix", 
                               "system_outage_recovery", "deployment_blocker",
                               "quarterly_test", "manual_rollback"],
                       help="Reason for emergency override (required for audit)")
    parser.add_argument("--emergency-contact",
                       help="Name/ID of person authorizing the override")
    parser.add_argument("--approval-timestamp",
                       help="Timestamp of authorization (ISO format)")
    parser.add_argument("--confirm", action="store_true",
                       help="Skip confirmation prompt (for automated use)")
    parser.add_argument("--show-recent", type=int, metavar="N",
                       help="Show N recent emergency overrides and exit")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = EmergencyRollbackManager()
    
    # Handle show recent history
    if args.show_recent:
        manager.show_recent_emergencies(args.show_recent)
        sys.exit(0)
    
    # Validate that reason is provided for actual rollback operations
    if not args.reason:
        print("❌ ERROR: --reason is required for rollback operations")
        print("Use --show-recent to view emergency history without performing rollback")
        sys.exit(1)
    
    try:
        # Validate emergency authorization
        is_valid, error = manager.validate_emergency_authorization(
            args.reason, args.emergency_contact, args.approval_timestamp
        )
        
        if not is_valid:
            logger.error(f"Emergency authorization validation failed: {error}")
            print(f"\n❌ AUTHORIZATION FAILED: {error}")
            sys.exit(1)
        
        # Execute emergency rollback
        success = manager.remove_branch_protection(
            branch=args.branch,
            audit_reason=args.reason,
            emergency_contact=args.emergency_contact,
            approval_timestamp=args.approval_timestamp,
            confirm=args.confirm
        )
        
        # Save audit log
        manager.save_emergency_log()
        
        if success:
            print(f"\n🔔 NEXT STEPS:")
            print(f"   1. Perform your emergency work on branch '{args.branch}'")
            print(f"   2. Test your changes thoroughly")
            print(f"   3. Restore protection: python3 scripts/setup_branch_protection.py --branch {args.branch}")
            print(f"   4. Create post-emergency review PR")
            print(f"   5. Document incident if required")
            
            # Show recent history for context
            print(f"\n📈 EMERGENCY OVERRIDE FREQUENCY:")
            manager.show_recent_emergencies(5)
            
            sys.exit(0)
        else:
            print(f"\n❌ Emergency rollback failed. Check logs for details.")
            print(f"   Alternative: Use GitHub web interface to manually remove protection")
            print(f"   Contact: GitHub Support if all methods fail")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Emergency rollback interrupted by user")
        print(f"   Branch protection status may be unchanged")
        print(f"   Check status: gh api /repos/{{owner}}/{{repo}}/branches/{args.branch}/protection")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Emergency rollback failed with exception: {str(e)}")
        print(f"\n💥 UNEXPECTED ERROR: {str(e)}")
        print(f"   This may indicate a system problem requiring manual intervention")
        print(f"   Contact repository administrators immediately")
        sys.exit(1)

if __name__ == "__main__":
    main()