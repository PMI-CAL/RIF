#!/usr/bin/env python3
"""
Shadow Quality Tracking System
Creates and manages parallel quality tracking issues for continuous verification.
Part of Issue #16 - Adversarial Testing Enhancement
"""

import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

class ShadowQualityTracker:
    """
    Manages shadow quality tracking issues for parallel verification.
    """
    
    def __init__(self):
        self.shadow_prefix = "Quality Tracking:"
        self.quality_labels = ["quality:shadow", "state:quality-tracking"]
    
    def create_shadow_quality_issue(self, main_issue_number: int) -> Dict[str, Any]:
        """
        Creates a shadow issue for quality tracking of the main issue.
        
        Args:
            main_issue_number: The main issue number to track
            
        Returns:
            Dict containing shadow issue details
        """
        try:
            # Get main issue details
            main_issue = self._get_issue_details(main_issue_number)
            if not main_issue:
                raise ValueError(f"Main issue #{main_issue_number} not found")
            
            shadow_title = f"{self.shadow_prefix} Issue #{main_issue_number}"
            shadow_body = self._generate_shadow_body(main_issue_number, main_issue)
            
            # Create shadow issue
            labels = " ".join([f"--label '{label}'" for label in self.quality_labels])
            gh_command = f'gh issue create --title "{shadow_title}" --body "{shadow_body}" {labels}'
            
            result = subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create shadow issue: {result.stderr}")
            
            # Extract issue number from URL
            shadow_url = result.stdout.strip()
            shadow_number = int(shadow_url.split('/')[-1])
            
            # Log the creation
            self._log_shadow_creation(main_issue_number, shadow_number)
            
            return {
                "shadow_issue_number": shadow_number,
                "shadow_url": shadow_url,
                "main_issue": main_issue_number,
                "created_at": datetime.now().isoformat()
            }
            
        except ValueError as ve:
            if "invalid literal for int()" in str(ve):
                # Handle case where result.stdout contains issue details instead of URL
                print(f"Error parsing issue number: {ve}")
                return {"error": f"Failed to parse issue number from GitHub response: {ve}"}
            print(f"Error creating shadow issue: {ve}")
            return {"error": str(ve)}
        except Exception as e:
            print(f"Error creating shadow issue: {e}")
            return {"error": str(e)}
    
    def _generate_shadow_body(self, main_issue_number: int, main_issue: Dict) -> str:
        """Generate the body content for the shadow quality issue."""
        complexity = self._extract_complexity(main_issue.get('labels', []))
        risk_level = self._assess_initial_risk_level(main_issue)
        
        return f'''# Shadow Quality Issue for #{main_issue_number}

This issue tracks quality verification activities for the main issue throughout its development lifecycle.

**Main Issue**: #{main_issue_number} - {main_issue.get('title', 'Unknown')}  
**Complexity**: {complexity}  
**Initial Risk Level**: {risk_level}  
**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Verification Checkpoints
- [ ] **Initial Analysis**: Requirements and complexity assessed
- [ ] **Architecture Review**: System design verified (if applicable)
- [ ] **Implementation Verification**: Code quality and testing assessed
- [ ] **Evidence Collection**: All required proof gathered
- [ ] **Security Assessment**: Risk-based security verification
- [ ] **Performance Validation**: Performance requirements met
- [ ] **Final Quality Gate**: All criteria satisfied

## Current Quality Metrics
- **Quality Score**: TBD (will be updated during validation)
- **Risk Level**: {risk_level} (dynamic, updated based on changes)
- **Evidence Completeness**: 0% (updated as evidence is collected)
- **Verification Depth**: TBD (shallow/standard/deep/intensive)

## Quality Standards Tracking
- **Code Coverage**: Target 80%+ (TBD)
- **Security Scan**: No critical vulnerabilities (TBD)
- **Test Pass Rate**: 100% (TBD)
- **Performance**: Meet baseline requirements (TBD)

## Audit Trail
Quality verification activities will be logged below as they occur.

---

*This is an automated shadow issue created by the RIF Adversarial Testing System.*  
*It runs in parallel with the main development workflow for continuous quality monitoring.*

### Instructions for Validators
1. **Log all verification activities** as comments on this issue
2. **Update metrics** in the issue description as they change
3. **Document evidence gaps** that need to be addressed
4. **Escalate concerns** immediately if critical issues found
5. **Close this issue** only when main issue is complete and validated

### Quality Escalation Triggers
Auto-escalation will occur if any of these conditions are met:
- Security files modified in main issue
- No tests added with implementation
- Large diff size (>500 lines)
- Previous validation failures
- Critical vulnerabilities detected
        '''
    
    def log_quality_activity(self, shadow_issue_number: int, activity: Dict[str, Any]) -> bool:
        """
        Logs a verification activity to the shadow issue.
        
        Args:
            shadow_issue_number: Shadow issue to log to
            activity: Activity details to log
            
        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            activity_type = activity.get('type', 'Unknown Activity')
            agent = activity.get('agent', 'Unknown Agent')
            action = activity.get('action', 'No action specified')
            result = activity.get('result', 'No result specified')
            evidence = activity.get('evidence', 'No evidence provided')
            notes = activity.get('notes', '')
            
            log_entry = f'''### {timestamp} - {activity_type}

**Agent**: {agent}  
**Action**: {action}  
**Result**: {result}  
**Evidence**: {evidence}

{notes}

---'''
            
            # Escape quotes for shell command
            escaped_entry = log_entry.replace('"', '\\"').replace('`', '\\`')
            gh_command = f'gh issue comment {shadow_issue_number} --body "{escaped_entry}"'
            
            result = subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: Failed to log activity to shadow issue: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error logging quality activity: {e}")
            return False
    
    def sync_quality_status(self, main_issue_number: int, shadow_issue_number: int) -> bool:
        """
        Synchronizes quality status between main and shadow issues.
        
        Args:
            main_issue_number: Main issue number
            shadow_issue_number: Shadow issue number
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current main issue status
            main_issue = self._get_issue_details(main_issue_number)
            if not main_issue:
                return False
            
            # Calculate current quality metrics
            quality_metrics = self._calculate_quality_metrics(main_issue_number)
            
            # Create status update
            current_state = main_issue.get('state', 'unknown')
            labels = [label['name'] for label in main_issue.get('labels', [])]
            workflow_state = next((label.replace('state:', '') for label in labels if label.startswith('state:')), 'unknown')
            
            status_update = f'''## Quality Status Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

**Main Issue State**: {current_state} (workflow: {workflow_state})  
**Quality Score**: {quality_metrics.get('score', 'TBD')}  
**Evidence Completion**: {quality_metrics.get('evidence_percent', 0)}%  
**Risk Level**: {quality_metrics.get('risk_level', 'TBD')}  
**Verification Depth**: {quality_metrics.get('verification_depth', 'TBD')}

### Recent Changes
{quality_metrics.get('recent_changes', 'No significant changes detected')}

### Quality Concerns
{quality_metrics.get('concerns', 'None identified')}

---'''
            
            # Post update to shadow issue
            escaped_update = status_update.replace('"', '\\"').replace('`', '\\`')
            gh_command = f'gh issue comment {shadow_issue_number} --body "{escaped_update}"'
            
            result = subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error syncing quality status: {e}")
            return False
    
    def close_shadow_issue(self, shadow_issue_number: int, final_metrics: Dict[str, Any]) -> bool:
        """
        Closes a shadow issue with final quality assessment.
        
        Args:
            shadow_issue_number: Shadow issue to close
            final_metrics: Final quality metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            final_comment = f'''## Final Quality Assessment - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

### Final Quality Metrics
- **Final Quality Score**: {final_metrics.get('final_score', 'N/A')}
- **Gate Decision**: {final_metrics.get('gate_decision', 'N/A')}
- **Evidence Completeness**: {final_metrics.get('evidence_percent', 'N/A')}%
- **Risk Level**: {final_metrics.get('final_risk_level', 'N/A')}
- **Issues Found**: {final_metrics.get('issues_found', 0)}
- **Issues Resolved**: {final_metrics.get('issues_resolved', 0)}

### Quality Summary
{final_metrics.get('summary', 'Main issue completed successfully.')}

### Lessons Learned
{final_metrics.get('lessons_learned', 'No specific lessons documented.')}

---

**Shadow quality tracking completed.** Main issue has been validated and closed.
            '''
            
            # Post final comment
            escaped_comment = final_comment.replace('"', '\\"').replace('`', '\\`')
            gh_command = f'gh issue comment {shadow_issue_number} --body "{escaped_comment}"'
            
            comment_result = subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            
            # Close the shadow issue
            close_command = f'gh issue close {shadow_issue_number}'
            close_result = subprocess.run(close_command, shell=True, capture_output=True, text=True)
            
            return comment_result.returncode == 0 and close_result.returncode == 0
            
        except Exception as e:
            print(f"Error closing shadow issue: {e}")
            return False
    
    def generate_audit_trail(self, shadow_issue_number: int) -> Dict[str, Any]:
        """
        Generates a complete audit trail for the shadow issue.
        
        Args:
            shadow_issue_number: Shadow issue to audit
            
        Returns:
            Dict containing audit trail data
        """
        try:
            # Get shadow issue with comments
            gh_command = f'gh issue view {shadow_issue_number} --json title,body,comments,createdAt,closedAt,state'
            result = subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"error": f"Failed to retrieve shadow issue: {result.stderr}"}
            
            issue_data = json.loads(result.stdout)
            
            audit_trail = {
                "shadow_issue_number": shadow_issue_number,
                "title": issue_data.get('title', ''),
                "state": issue_data.get('state', ''),
                "created_at": issue_data.get('createdAt', ''),
                "closed_at": issue_data.get('closedAt', ''),
                "total_activities": len(issue_data.get('comments', [])),
                "activities": [],
                "quality_metrics_history": [],
                "issues_tracked": []
            }
            
            # Parse comments for activities and metrics
            for comment in issue_data.get('comments', []):
                body = comment.get('body', '')
                created_at = comment.get('createdAt', '')
                
                if '### ' in body and ' - ' in body:  # Activity log format
                    audit_trail['activities'].append({
                        "timestamp": created_at,
                        "content": body,
                        "type": self._extract_activity_type(body)
                    })
                elif 'Quality Status Update' in body:  # Status update format
                    audit_trail['quality_metrics_history'].append({
                        "timestamp": created_at,
                        "metrics": self._extract_metrics_from_comment(body)
                    })
            
            return audit_trail
            
        except Exception as e:
            return {"error": f"Error generating audit trail: {e}"}
    
    def _get_issue_details(self, issue_number: int) -> Optional[Dict]:
        """Get issue details from GitHub."""
        try:
            gh_command = f'gh issue view {issue_number} --json title,body,labels,state,createdAt'
            result = subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
            
            return json.loads(result.stdout)
            
        except Exception:
            return None
    
    def _extract_complexity(self, labels: List[Dict]) -> str:
        """Extract complexity from issue labels."""
        for label in labels:
            name = label.get('name', '')
            if name.startswith('complexity:'):
                return name.replace('complexity:', '').title()
        return 'Unknown'
    
    def _assess_initial_risk_level(self, issue: Dict) -> str:
        """Assess initial risk level based on issue content."""
        title = issue.get('title', '').lower()
        body = issue.get('body', '').lower()
        
        high_risk_keywords = ['security', 'authentication', 'payment', 'auth', 'login', 'password', 'crypto']
        medium_risk_keywords = ['database', 'api', 'integration', 'performance', 'migration']
        
        content = f"{title} {body}"
        
        if any(keyword in content for keyword in high_risk_keywords):
            return 'High'
        elif any(keyword in content for keyword in medium_risk_keywords):
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_quality_metrics(self, main_issue_number: int) -> Dict[str, Any]:
        """Calculate current quality metrics for the main issue."""
        # This would integrate with the actual validation system
        # For now, return placeholder metrics
        return {
            'score': 'TBD',
            'evidence_percent': 0,
            'risk_level': 'Medium',
            'verification_depth': 'Standard',
            'recent_changes': 'Issue in active development',
            'concerns': 'Monitoring for evidence collection'
        }
    
    def _log_shadow_creation(self, main_issue_number: int, shadow_number: int):
        """Log shadow issue creation to main issue."""
        try:
            log_message = f'🔍 **Shadow Quality Tracking Issue Created**: #{shadow_number}\\n\\nParallel quality verification will be tracked in issue #{shadow_number} throughout the development lifecycle.'
            
            gh_command = f'gh issue comment {main_issue_number} --body "{log_message}"'
            subprocess.run(gh_command, shell=True, capture_output=True, text=True)
            
        except Exception as e:
            print(f"Warning: Could not log shadow creation to main issue: {e}")
    
    def _extract_activity_type(self, comment_body: str) -> str:
        """Extract activity type from comment body."""
        lines = comment_body.split('\n')
        for line in lines:
            if line.startswith('### ') and ' - ' in line:
                # Extract just the activity type, stop at the first newline or special character
                activity_part = line.split(' - ', 1)[1]
                # Clean up any escaped newlines or extra content
                activity_clean = activity_part.replace('\\n', '').split('\n')[0].split('**')[0].strip()
                return activity_clean
        return 'Unknown Activity'
    
    def _extract_metrics_from_comment(self, comment_body: str) -> Dict[str, str]:
        """Extract metrics from status update comment."""
        metrics = {}
        lines = comment_body.split('\n')
        
        for line in lines:
            if '**' in line and ':' in line:
                # Extract key between first and second **
                parts = line.split('**')
                if len(parts) >= 3:
                    key = parts[1].replace(':', '').strip()
                    # Extract value after the first colon following the key
                    after_key = '**'.join(parts[2:])
                    if ':' in after_key:
                        value = after_key.split(':', 1)[1].strip()
                        metrics[key.lower().replace(' ', '_')] = value
        
        return metrics


def main():
    """Command-line interface for shadow quality tracking."""
    if len(sys.argv) < 2:
        print("Usage: python shadow_quality_tracking.py <command> [args]")
        print("Commands:")
        print("  create-shadow <issue_number>     - Create shadow quality issue")
        print("  log-activity <shadow_number>     - Log activity (requires JSON input)")
        print("  sync-status <main> <shadow>      - Sync quality status")
        print("  close-shadow <shadow_number>     - Close shadow issue")
        print("  audit-trail <shadow_number>      - Generate audit trail")
        return
    
    command = sys.argv[1]
    tracker = ShadowQualityTracker()
    
    if command == "create-shadow" and len(sys.argv) >= 3:
        main_issue = int(sys.argv[2])
        result = tracker.create_shadow_quality_issue(main_issue)
        print(json.dumps(result, indent=2))
        
    elif command == "log-activity" and len(sys.argv) >= 3:
        shadow_number = int(sys.argv[2])
        # Read activity JSON from stdin
        activity = json.loads(sys.stdin.read())
        success = tracker.log_quality_activity(shadow_number, activity)
        print(json.dumps({"success": success}))
        
    elif command == "sync-status" and len(sys.argv) >= 4:
        main_issue = int(sys.argv[2])
        shadow_issue = int(sys.argv[3])
        success = tracker.sync_quality_status(main_issue, shadow_issue)
        print(json.dumps({"success": success}))
        
    elif command == "close-shadow" and len(sys.argv) >= 3:
        shadow_number = int(sys.argv[2])
        # Read final metrics from stdin if provided
        try:
            final_metrics = json.loads(sys.stdin.read())
        except:
            final_metrics = {"summary": "Shadow issue closed manually"}
        success = tracker.close_shadow_issue(shadow_number, final_metrics)
        print(json.dumps({"success": success}))
        
    elif command == "audit-trail" and len(sys.argv) >= 3:
        shadow_number = int(sys.argv[2])
        audit = tracker.generate_audit_trail(shadow_number)
        print(json.dumps(audit, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())