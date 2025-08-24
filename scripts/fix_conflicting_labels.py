#!/usr/bin/env python3
"""
Fix Conflicting State Labels Script
Issue #88: Critical fix for RIF state management system

This script fixes all existing GitHub issues with conflicting state labels,
ensuring each issue has exactly one state label that matches the RIF workflow.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add the parent directory to sys.path so we can import from claude/commands
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude.commands.github_state_manager import GitHubStateManager

class ConflictingLabelsFixer:
    """
    Fixes all issues with conflicting or missing state labels.
    """
    
    def __init__(self):
        """Initialize the conflict fixer."""
        self.manager = GitHubStateManager()
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'issues_processed': 0,
            'issues_fixed': 0,
            'issues_failed': 0,
            'fixes_applied': [],
            'errors': []
        }
    
    def run_full_cleanup(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run complete cleanup of all open issues.
        
        Args:
            dry_run: If True, only report what would be fixed without making changes
            
        Returns:
            Cleanup report
        """
        print(f"🔧 Starting {'DRY RUN' if dry_run else 'LIVE'} cleanup of conflicting state labels...")
        
        # Step 1: Audit all issues
        print("1. Auditing all open issues...")
        audit_report = self.manager.audit_all_issues()
        
        print(f"   Found {audit_report['total_issues']} open issues")
        print(f"   Valid: {audit_report['valid_issues']}, Invalid: {audit_report['invalid_issues']}")
        
        if audit_report['invalid_issues'] == 0:
            print("✅ All issues already have correct state labels!")
            return self.report
        
        # Step 2: Fix issues by category
        print("\n2. Fixing issues by category...")
        
        # Fix issues with multiple state labels
        multiple_states = audit_report['issues_by_problem']['multiple_states']
        if multiple_states:
            print(f"   Fixing {len(multiple_states)} issues with multiple state labels...")
            for issue_number in multiple_states:
                self._fix_multiple_states(issue_number, dry_run)
        
        # Fix issues with no state labels
        no_state = audit_report['issues_by_problem']['no_state']
        if no_state:
            print(f"   Fixing {len(no_state)} issues with no state labels...")
            for issue_number in no_state:
                self._fix_missing_state(issue_number, dry_run)
        
        # Fix issues with unknown state labels
        unknown_state = audit_report['issues_by_problem']['unknown_state']
        if unknown_state:
            print(f"   Fixing {len(unknown_state)} issues with unknown state labels...")
            for issue_number in unknown_state:
                self._fix_unknown_state(issue_number, dry_run)
        
        # Step 3: Final validation
        print("\n3. Running final validation...")
        final_audit = self.manager.audit_all_issues()
        
        self.report['final_audit'] = final_audit
        self.report['improvement'] = {
            'before_invalid': audit_report['invalid_issues'],
            'after_invalid': final_audit['invalid_issues'],
            'issues_fixed': audit_report['invalid_issues'] - final_audit['invalid_issues']
        }
        
        print(f"   Issues remaining invalid: {final_audit['invalid_issues']}")
        print(f"   Total issues fixed: {self.report['improvement']['issues_fixed']}")
        
        # Step 4: Save report
        self._save_report()
        
        if dry_run:
            print(f"\n✅ DRY RUN completed. {self.report['issues_processed']} issues would be processed.")
        else:
            print(f"\n✅ Cleanup completed! Fixed {self.report['issues_fixed']} out of {self.report['issues_processed']} issues.")
            if self.report['issues_failed'] > 0:
                print(f"⚠️  {self.report['issues_failed']} issues failed to fix.")
        
        return self.report
    
    def _fix_multiple_states(self, issue_number: int, dry_run: bool = False):
        """
        Fix an issue with multiple state labels.
        
        Args:
            issue_number: GitHub issue number
            dry_run: If True, only report what would be done
        """
        try:
            self.report['issues_processed'] += 1
            
            if dry_run:
                validation = self.manager.validate_issue_state(issue_number)
                fix_action = f"Would remove {len(validation['state_labels'])} state labels and set to 'analyzing'"
                self.report['fixes_applied'].append({
                    'issue_number': issue_number,
                    'problem': 'multiple_states',
                    'action': fix_action,
                    'dry_run': True
                })
                return
            
            # Remove all state labels first
            removed_labels = self.manager.remove_conflicting_labels(issue_number)
            
            # Determine appropriate state for the issue
            # For issues with multiple conflicting states, default to 'analyzing'
            target_state = 'analyzing'
            
            # Try to transition to the determined state
            success, message = self.manager.transition_state(
                issue_number, target_state, 
                "Automatic cleanup: resolved conflicting state labels"
            )
            
            if success:
                self.report['issues_fixed'] += 1
                self.report['fixes_applied'].append({
                    'issue_number': issue_number,
                    'problem': 'multiple_states',
                    'removed_labels': removed_labels,
                    'new_state': target_state,
                    'action': 'fixed'
                })
                print(f"      ✅ Issue #{issue_number}: Fixed multiple states -> {target_state}")
                
                # Post comment explaining the fix
                comment = f"""🔧 **RIF State Management Fix Applied**

**Issue**: This issue had multiple conflicting state labels: {removed_labels}

**Resolution**: Removed all conflicting labels and set state to `{target_state}` to restart proper workflow.

**Next Steps**: The RIF system will now process this issue according to the correct workflow state machine.

*This fix was applied automatically by the RIF state management system to resolve Issue #88.*"""

                self.manager.post_comment_to_issue(issue_number, comment)
                
            else:
                self.report['issues_failed'] += 1
                self.report['errors'].append({
                    'issue_number': issue_number,
                    'problem': 'multiple_states',
                    'error': message
                })
                print(f"      ❌ Issue #{issue_number}: Failed to fix - {message}")
        
        except Exception as e:
            self.report['issues_failed'] += 1
            self.report['errors'].append({
                'issue_number': issue_number,
                'problem': 'multiple_states',
                'error': str(e)
            })
            print(f"      ❌ Issue #{issue_number}: Exception - {e}")
    
    def _fix_missing_state(self, issue_number: int, dry_run: bool = False):
        """
        Fix an issue with no state labels.
        
        Args:
            issue_number: GitHub issue number  
            dry_run: If True, only report what would be done
        """
        try:
            self.report['issues_processed'] += 1
            
            if dry_run:
                fix_action = "Would add state:new label"
                self.report['fixes_applied'].append({
                    'issue_number': issue_number,
                    'problem': 'no_state',
                    'action': fix_action,
                    'dry_run': True
                })
                return
            
            # For new issues without any state, set to 'new'
            target_state = 'new'
            
            success, message = self.manager.transition_state(
                issue_number, target_state,
                "Automatic cleanup: added missing state label"
            )
            
            if success:
                self.report['issues_fixed'] += 1
                self.report['fixes_applied'].append({
                    'issue_number': issue_number,
                    'problem': 'no_state',
                    'new_state': target_state,
                    'action': 'fixed'
                })
                print(f"      ✅ Issue #{issue_number}: Added missing state -> {target_state}")
                
                # Post comment explaining the fix
                comment = f"""🔧 **RIF State Management Fix Applied**

**Issue**: This issue was missing a state label, preventing it from entering the RIF workflow.

**Resolution**: Added `state:{target_state}` label to enable automatic processing.

**Next Steps**: The RIF system will now analyze this issue and progress it through the workflow as appropriate.

*This fix was applied automatically by the RIF state management system to resolve Issue #88.*"""

                self.manager.post_comment_to_issue(issue_number, comment)
                
            else:
                self.report['issues_failed'] += 1
                self.report['errors'].append({
                    'issue_number': issue_number,
                    'problem': 'no_state',
                    'error': message
                })
                print(f"      ❌ Issue #{issue_number}: Failed to fix - {message}")
                
        except Exception as e:
            self.report['issues_failed'] += 1
            self.report['errors'].append({
                'issue_number': issue_number,
                'problem': 'no_state',
                'error': str(e)
            })
            print(f"      ❌ Issue #{issue_number}: Exception - {e}")
    
    def _fix_unknown_state(self, issue_number: int, dry_run: bool = False):
        """
        Fix an issue with unknown/invalid state labels.
        
        Args:
            issue_number: GitHub issue number
            dry_run: If True, only report what would be done
        """
        try:
            self.report['issues_processed'] += 1
            
            if dry_run:
                validation = self.manager.validate_issue_state(issue_number)
                fix_action = f"Would remove unknown state labels and set to 'analyzing'"
                self.report['fixes_applied'].append({
                    'issue_number': issue_number,
                    'problem': 'unknown_state',
                    'action': fix_action,
                    'dry_run': True
                })
                return
            
            # Remove invalid state labels
            removed_labels = self.manager.remove_conflicting_labels(issue_number)
            
            # Set to analyzing state to restart workflow
            target_state = 'analyzing'
            
            success, message = self.manager.transition_state(
                issue_number, target_state,
                "Automatic cleanup: replaced unknown state label"
            )
            
            if success:
                self.report['issues_fixed'] += 1
                self.report['fixes_applied'].append({
                    'issue_number': issue_number,
                    'problem': 'unknown_state',
                    'removed_labels': removed_labels,
                    'new_state': target_state,
                    'action': 'fixed'
                })
                print(f"      ✅ Issue #{issue_number}: Fixed unknown state -> {target_state}")
                
                # Post comment explaining the fix
                comment = f"""🔧 **RIF State Management Fix Applied**

**Issue**: This issue had unknown/invalid state labels: {removed_labels}

**Resolution**: Removed invalid labels and set state to `{target_state}` to restart workflow.

**Next Steps**: The RIF system will now properly analyze this issue and progress it through the correct workflow states.

*This fix was applied automatically by the RIF state management system to resolve Issue #88.*"""

                self.manager.post_comment_to_issue(issue_number, comment)
                
            else:
                self.report['issues_failed'] += 1
                self.report['errors'].append({
                    'issue_number': issue_number,
                    'problem': 'unknown_state',
                    'error': message
                })
                print(f"      ❌ Issue #{issue_number}: Failed to fix - {message}")
                
        except Exception as e:
            self.report['issues_failed'] += 1
            self.report['errors'].append({
                'issue_number': issue_number,
                'problem': 'unknown_state',
                'error': str(e)
            })
            print(f"      ❌ Issue #{issue_number}: Exception - {e}")
    
    def _save_report(self):
        """Save the cleanup report to file."""
        try:
            report_dir = Path('knowledge/state_cleanup')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f'cleanup_report_{timestamp}.json'
            
            with open(report_file, 'w') as f:
                json.dump(self.report, f, indent=2)
            
            print(f"   📄 Report saved to: {report_file}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save report: {e}")
    
    def fix_specific_issues(self, issue_numbers: List[int], dry_run: bool = False) -> Dict[str, Any]:
        """
        Fix specific issues by number.
        
        Args:
            issue_numbers: List of issue numbers to fix
            dry_run: If True, only report what would be done
            
        Returns:
            Fix report
        """
        print(f"🔧 Fixing specific issues: {issue_numbers} ({'DRY RUN' if dry_run else 'LIVE'})")
        
        for issue_number in issue_numbers:
            print(f"\n   Processing issue #{issue_number}...")
            validation = self.manager.validate_issue_state(issue_number)
            
            if validation['is_valid']:
                print(f"      ✅ Issue #{issue_number} is already valid")
                continue
            
            # Determine fix needed based on validation issues
            if validation['state_count'] == 0:
                self._fix_missing_state(issue_number, dry_run)
            elif validation['state_count'] > 1:
                self._fix_multiple_states(issue_number, dry_run)
            else:
                # Check for unknown state
                has_unknown = any('Unknown state:' in issue for issue in validation.get('issues', []))
                if has_unknown:
                    self._fix_unknown_state(issue_number, dry_run)
        
        self._save_report()
        return self.report


def main():
    """Main entry point for the cleanup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix conflicting GitHub state labels')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be fixed without making changes')
    parser.add_argument('--issues', nargs='+', type=int,
                       help='Fix specific issue numbers only')
    
    args = parser.parse_args()
    
    fixer = ConflictingLabelsFixer()
    
    try:
        if args.issues:
            report = fixer.fix_specific_issues(args.issues, args.dry_run)
        else:
            report = fixer.run_full_cleanup(args.dry_run)
        
        print(f"\n📊 Final Report:")
        print(f"   Issues processed: {report['issues_processed']}")
        print(f"   Issues fixed: {report['issues_fixed']}")
        print(f"   Issues failed: {report['issues_failed']}")
        
        if report['errors']:
            print(f"   Errors encountered: {len(report['errors'])}")
            for error in report['errors']:
                print(f"      Issue #{error['issue_number']}: {error['error']}")
        
        return 0 if report['issues_failed'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Cleanup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())