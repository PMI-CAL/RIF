#!/usr/bin/env python3
"""
Shadow Issue Monitor - Emergency Fix for Issue #147
Continuously monitors GitHub issues for shadow trigger conditions.
Phase 1 Implementation - Immediate Trigger Activation
"""

import json
import subprocess
import sys
import time
import os
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

class ShadowIssueMonitor:
    """
    Monitors GitHub issues and automatically creates shadow quality tracking issues
    when trigger conditions are met.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the shadow issue monitor."""
        self.config_path = config_path or "/Users/cal/DEV/RIF/config/shadow-quality-tracking.yaml"
        self.config = self._load_config()
        self.shadow_tracker_path = "/Users/cal/DEV/RIF/claude/commands/shadow_quality_tracking.py"
        
        # Track already processed issues to avoid duplicates
        self.processed_issues: Set[int] = set()
        self.load_processed_issues()
        
        # Monitor state
        self.is_running = False
        self.monitor_interval = 300  # 5 minutes
        
    def _load_config(self) -> Dict[str, Any]:
        """Load shadow quality tracking configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default trigger configuration."""
        return {
            'triggers': {
                'priority': ['critical'],
                'complexity': ['medium', 'high', 'very-high'],
                'state_transitions': ['validating'],
                'security_changes': True,
                'large_changes': '>500 lines'
            },
            'exclusions': {
                'existing_shadows': True,
                'closed_issues': True,
                'low_complexity_non_critical': True
            }
        }
    
    def load_processed_issues(self):
        """Load previously processed issues to avoid duplicates."""
        processed_file = "/Users/cal/DEV/RIF/knowledge/shadow_processed_issues.json"
        try:
            if os.path.exists(processed_file):
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    self.processed_issues = set(data.get('processed_issues', []))
        except Exception as e:
            print(f"Warning: Could not load processed issues: {e}")
            self.processed_issues = set()
    
    def save_processed_issues(self):
        """Save processed issues to avoid future duplicates."""
        processed_file = "/Users/cal/DEV/RIF/knowledge/shadow_processed_issues.json"
        try:
            os.makedirs(os.path.dirname(processed_file), exist_ok=True)
            data = {
                'processed_issues': list(self.processed_issues),
                'last_updated': datetime.now().isoformat()
            }
            with open(processed_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save processed issues: {e}")
    
    def get_github_issues(self) -> List[Dict[str, Any]]:
        """Get all open GitHub issues with metadata."""
        try:
            cmd = [
                'gh', 'issue', 'list',
                '--state', 'open',
                '--json', 'number,title,labels,body,state,createdAt',
                '--limit', '100'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Error getting GitHub issues: {result.stderr}")
                return []
            
            return json.loads(result.stdout)
            
        except subprocess.TimeoutExpired:
            print("Timeout getting GitHub issues")
            return []
        except Exception as e:
            print(f"Error getting GitHub issues: {e}")
            return []
    
    def check_existing_shadow(self, main_issue_number: int) -> bool:
        """Check if a shadow issue already exists for the main issue."""
        try:
            cmd = [
                'gh', 'issue', 'list',
                '--search', f'"Quality Tracking: Issue #{main_issue_number}"',
                '--json', 'number,title',
                '--limit', '10'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                return False  # Assume no shadow exists if search fails
            
            issues = json.loads(result.stdout)
            
            # Check if any issue title matches exactly
            target_title = f"Quality Tracking: Issue #{main_issue_number}"
            for issue in issues:
                if issue.get('title', '') == target_title:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Warning: Could not check for existing shadow for issue #{main_issue_number}: {e}")
            return False  # Assume no shadow exists if check fails
    
    def should_create_shadow(self, issue: Dict[str, Any]) -> tuple[bool, str]:
        """
        Determine if a shadow issue should be created for the given issue.
        
        Returns:
            (should_create: bool, reason: str)
        """
        issue_number = issue['number']
        title = issue.get('title', '')
        body = issue.get('body', '')
        labels = [label['name'] for label in issue.get('labels', [])]
        
        # Skip if already processed
        if issue_number in self.processed_issues:
            return False, "Already processed"
        
        # Skip if shadow already exists
        if self.check_existing_shadow(issue_number):
            self.processed_issues.add(issue_number)
            return False, "Shadow already exists"
        
        # Check trigger conditions
        reasons = []
        
        # 1. Priority-based triggers
        priority_labels = [label for label in labels if label.startswith('priority:')]
        if any('critical' in label for label in priority_labels):
            reasons.append("Critical priority")
        
        # 2. Complexity-based triggers
        complexity_labels = [label for label in labels if label.startswith('complexity:')]
        high_complexity = any(
            complexity in label 
            for complexity in ['high', 'very-high', 'medium'] 
            for label in complexity_labels
        )
        if high_complexity:
            reasons.append("High complexity")
        
        # 3. State-based triggers
        state_labels = [label for label in labels if label.startswith('state:')]
        validating_state = any('validating' in label for label in state_labels)
        if validating_state:
            reasons.append("Validation state")
        
        # 4. Security-related triggers
        security_keywords = ['security', 'authentication', 'auth', 'login', 'password', 'crypto', 'vulnerability']
        content = f"{title} {body}".lower()
        if any(keyword in content for keyword in security_keywords):
            reasons.append("Security-related content")
        
        # 5. Agent-related issues (meta-critical)
        agent_keywords = ['agent', 'rif-', 'orchestration', 'knowledge', 'context']
        if any(keyword in content for keyword in agent_keywords):
            reasons.append("Agent/RIF system issue")
        
        # Decision logic: Create shadow if any significant trigger condition is met
        if reasons:
            return True, f"Triggers: {', '.join(reasons)}"
        
        return False, "No trigger conditions met"
    
    def create_shadow_issue(self, main_issue_number: int) -> Dict[str, Any]:
        """Create a shadow issue for the given main issue."""
        try:
            cmd = [
                'python3', self.shadow_tracker_path,
                'create-shadow', str(main_issue_number)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Command failed: {result.stderr}",
                    "stdout": result.stdout
                }
            
            # Parse result
            try:
                shadow_result = json.loads(result.stdout)
                if 'error' not in shadow_result:
                    shadow_result['success'] = True
                return shadow_result
            except json.JSONDecodeError:
                # If not JSON, treat stdout as success message
                return {
                    "success": True,
                    "message": result.stdout.strip(),
                    "main_issue": main_issue_number
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Timeout creating shadow issue"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception creating shadow issue: {e}"
            }
    
    def process_issue(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single issue for shadow creation."""
        issue_number = issue['number']
        should_create, reason = self.should_create_shadow(issue)
        
        result = {
            'issue_number': issue_number,
            'title': issue.get('title', 'Unknown'),
            'should_create_shadow': should_create,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        if should_create:
            print(f"Creating shadow issue for #{issue_number}: {reason}")
            shadow_result = self.create_shadow_issue(issue_number)
            result['shadow_creation'] = shadow_result
            
            if shadow_result.get('success', False):
                self.processed_issues.add(issue_number)
                result['shadow_issue_number'] = shadow_result.get('shadow_issue_number')
                print(f"✅ Shadow created for issue #{issue_number}")
            else:
                print(f"❌ Failed to create shadow for issue #{issue_number}: {shadow_result.get('error', 'Unknown error')}")
        else:
            print(f"⏭️  Skipping issue #{issue_number}: {reason}")
            # Still mark as processed if it's a valid skip reason
            if reason in ["Already processed", "Shadow already exists", "No trigger conditions met"]:
                self.processed_issues.add(issue_number)
        
        return result
    
    def backfill_critical_issues(self, critical_issue_numbers: List[int]) -> List[Dict[str, Any]]:
        """Create shadow issues for specific critical issues that are missing shadows."""
        results = []
        
        for issue_number in critical_issue_numbers:
            print(f"Processing backfill for critical issue #{issue_number}")
            
            # Remove from processed set to force creation
            if issue_number in self.processed_issues:
                self.processed_issues.remove(issue_number)
            
            # Get issue details
            try:
                cmd = ['gh', 'issue', 'view', str(issue_number), '--json', 'number,title,labels,body,state']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    results.append({
                        'issue_number': issue_number,
                        'success': False,
                        'error': f'Could not retrieve issue details: {result.stderr}'
                    })
                    continue
                
                issue = json.loads(result.stdout)
                process_result = self.process_issue(issue)
                results.append(process_result)
                
            except Exception as e:
                results.append({
                    'issue_number': issue_number,
                    'success': False,
                    'error': f'Exception processing issue: {e}'
                })
        
        return results
    
    def run_single_scan(self) -> Dict[str, Any]:
        """Run a single scan of all GitHub issues."""
        scan_start = datetime.now()
        print(f"Starting shadow issue scan at {scan_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all open issues
        issues = self.get_github_issues()
        print(f"Retrieved {len(issues)} open issues")
        
        if not issues:
            return {
                'scan_time': scan_start.isoformat(),
                'issues_processed': 0,
                'shadows_created': 0,
                'errors': ['No issues retrieved']
            }
        
        # Process each issue
        results = []
        shadows_created = 0
        errors = []
        
        for issue in issues:
            try:
                result = self.process_issue(issue)
                results.append(result)
                
                if result.get('shadow_creation', {}).get('success', False):
                    shadows_created += 1
                elif result.get('shadow_creation', {}).get('error'):
                    errors.append(f"Issue #{result['issue_number']}: {result['shadow_creation']['error']}")
                    
            except Exception as e:
                error_msg = f"Error processing issue #{issue.get('number', 'unknown')}: {e}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        # Save processed issues state
        self.save_processed_issues()
        
        scan_end = datetime.now()
        scan_duration = (scan_end - scan_start).total_seconds()
        
        summary = {
            'scan_time': scan_start.isoformat(),
            'scan_duration_seconds': scan_duration,
            'issues_processed': len(issues),
            'shadows_created': shadows_created,
            'errors': errors,
            'total_processed_issues': len(self.processed_issues)
        }
        
        print(f"Scan completed in {scan_duration:.1f}s: {shadows_created} shadows created, {len(errors)} errors")
        return summary
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop."""
        print(f"Starting continuous shadow issue monitoring (interval: {self.monitor_interval}s)")
        self.is_running = True
        
        while self.is_running:
            try:
                scan_result = self.run_single_scan()
                
                # Log scan result
                log_file = "/Users/cal/DEV/RIF/knowledge/shadow_monitor_log.json"
                try:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    
                    # Load existing log
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_data = json.load(f)
                    else:
                        log_data = {'scans': []}
                    
                    # Add new scan
                    log_data['scans'].append(scan_result)
                    
                    # Keep only last 100 scans
                    log_data['scans'] = log_data['scans'][-100:]
                    
                    # Save log
                    with open(log_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
                        
                except Exception as e:
                    print(f"Warning: Could not log scan result: {e}")
                
                # Wait for next scan
                if self.is_running:
                    time.sleep(self.monitor_interval)
                    
            except KeyboardInterrupt:
                print("Monitoring stopped by user")
                self.is_running = False
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                if self.is_running:
                    time.sleep(self.monitor_interval)
    
    def stop_monitoring(self):
        """Stop the continuous monitoring loop."""
        self.is_running = False


def main():
    """Command-line interface for shadow issue monitor."""
    if len(sys.argv) < 2:
        print("Usage: python shadow_issue_monitor.py <command> [args]")
        print("Commands:")
        print("  scan                           - Run single scan of all issues")
        print("  monitor                        - Start continuous monitoring")
        print("  backfill <issue1> <issue2>...  - Create shadows for specific issues")
        print("  status                         - Show monitor status")
        return 1
    
    command = sys.argv[1]
    monitor = ShadowIssueMonitor()
    
    if command == "scan":
        result = monitor.run_single_scan()
        print(json.dumps(result, indent=2))
        
    elif command == "monitor":
        try:
            monitor.run_continuous_monitoring()
        except KeyboardInterrupt:
            print("Monitoring stopped")
        
    elif command == "backfill" and len(sys.argv) >= 3:
        issue_numbers = []
        for arg in sys.argv[2:]:
            try:
                issue_numbers.append(int(arg))
            except ValueError:
                print(f"Warning: '{arg}' is not a valid issue number")
        
        if issue_numbers:
            print(f"Running backfill for issues: {issue_numbers}")
            results = monitor.backfill_critical_issues(issue_numbers)
            print(json.dumps(results, indent=2))
        else:
            print("No valid issue numbers provided for backfill")
            return 1
        
    elif command == "status":
        status = {
            'processed_issues': len(monitor.processed_issues),
            'config_loaded': monitor.config is not None,
            'shadow_tracker_exists': os.path.exists(monitor.shadow_tracker_path)
        }
        print(json.dumps(status, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())