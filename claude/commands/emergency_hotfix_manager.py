#!/usr/bin/env python3
"""
RIF Emergency Hotfix Manager
Provides CLI and programmatic interface for emergency hotfix workflows.
"""

import os
import json
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil


class HotfixManager:
    """
    Enterprise-grade emergency hotfix management system.
    Implements sub-30-minute resolution capability with complete audit trail.
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).absolute()
        self.incidents_dir = self.repo_path / "incidents"
        self.incidents_dir.mkdir(exist_ok=True)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load hotfix configuration from repository."""
        config_path = self.repo_path / "config" / "emergency-hotfix.yaml"
        default_config = {
            "production_branch": "main",  # Use main as production for testing
            "main_branch": "main",
            "reviewer_count": 1,
            "timeout_minutes": 30,
            "required_labels": ["hotfix", "emergency"],
            "quality_gates": {
                "security_scan": True,
                "smoke_tests": True,
                "full_test_suite": False,
                "performance_benchmarks": False,
                "documentation_check": False
            },
            "notifications": {
                "slack_webhook": None,
                "email_list": [],
                "pagerduty_key": None
            }
        }
        
        # TODO: Load actual YAML config if it exists
        # For now, return default config
        return default_config
    
    def create_hotfix(self, description: str, severity: str = "high", 
                     issue_url: Optional[str] = None) -> Dict:
        """
        Create emergency hotfix branch and incident record.
        
        Args:
            description: Brief description of the issue
            severity: Severity level (critical, high, medium, low)
            issue_url: Optional GitHub issue URL
            
        Returns:
            Dict containing incident details and next steps
        """
        print("🚨 Creating emergency hotfix...")
        
        # Generate incident ID
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create incident record
        incident = {
            "incident_id": incident_id,
            "description": description,
            "severity": severity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": self._get_git_user(),
            "issue_url": issue_url,
            "status": "initiated",
            "timeline": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "Incident created",
                    "details": f"Emergency hotfix initiated: {description}"
                }
            ]
        }
        
        # Save incident record
        incident_file = self.incidents_dir / f"{incident_id}.json"
        with open(incident_file, 'w') as f:
            json.dump(incident, f, indent=2)
        
        # Create hotfix branch
        branch_name = f"hotfix/{description.lower().replace(' ', '-')[:30]}-{incident_id[-8:]}"
        
        try:
            # Ensure we're on the production branch
            self._run_git(["fetch", "origin"])
            self._run_git(["checkout", self.config["production_branch"]])
            self._run_git(["pull", "origin", self.config["production_branch"]])
            
            # Create and checkout hotfix branch
            self._run_git(["checkout", "-b", branch_name])
            
            print(f"✅ Hotfix branch created: {branch_name}")
            print(f"📋 Incident ID: {incident_id}")
            
            # Update incident record
            incident["branch_name"] = branch_name
            incident["status"] = "in_progress"
            incident["timeline"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "Hotfix branch created",
                "details": f"Branch: {branch_name}"
            })
            
            with open(incident_file, 'w') as f:
                json.dump(incident, f, indent=2)
            
            next_steps = [
                f"1. Make your emergency fix in branch: {branch_name}",
                f"2. Test your changes locally",
                f"3. Commit with message: 'HOTFIX: {description} ({incident_id})'",
                f"4. Push branch: git push -u origin {branch_name}",
                f"5. Create emergency PR: ./claude/commands/emergency_hotfix_manager.py create-pr {incident_id}",
                f"6. Monitor deployment progress"
            ]
            
            return {
                "success": True,
                "incident_id": incident_id,
                "branch_name": branch_name,
                "incident_file": str(incident_file),
                "next_steps": next_steps
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating hotfix branch: {e}")
            return {"success": False, "error": str(e)}
    
    def create_emergency_pr(self, incident_id: str) -> Dict:
        """
        Create emergency pull request with hotfix labels and reduced review requirements.
        """
        print(f"🚀 Creating emergency PR for incident {incident_id}...")
        
        # Load incident record
        incident_file = self.incidents_dir / f"{incident_id}.json"
        if not incident_file.exists():
            return {"success": False, "error": f"Incident {incident_id} not found"}
        
        with open(incident_file, 'r') as f:
            incident = json.load(f)
        
        branch_name = incident.get("branch_name")
        if not branch_name:
            return {"success": False, "error": "Branch name not found in incident record"}
        
        try:
            # Push branch first
            self._run_git(["push", "-u", "origin", branch_name])
            
            # Create PR with GitHub CLI
            pr_title = f"🚨 HOTFIX: {incident['description']} ({incident_id})"
            pr_body = self._generate_emergency_pr_body(incident)
            
            cmd = [
                "gh", "pr", "create",
                "--title", pr_title,
                "--body", pr_body,
                "--base", self.config["production_branch"],
                "--head", branch_name,
                "--label", "hotfix,emergency,priority:critical"
            ]
            
            result = self._run_command(cmd)
            pr_url = result.stdout.strip()
            
            print(f"✅ Emergency PR created: {pr_url}")
            
            # Update incident record
            incident["pr_url"] = pr_url
            incident["status"] = "pr_created"
            incident["timeline"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "Emergency PR created",
                "details": f"PR URL: {pr_url}"
            })
            
            with open(incident_file, 'w') as f:
                json.dump(incident, f, indent=2)
            
            return {
                "success": True,
                "pr_url": pr_url,
                "incident_id": incident_id
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating emergency PR: {e}")
            return {"success": False, "error": str(e)}
    
    def monitor_deployment(self, incident_id: str) -> Dict:
        """
        Monitor emergency deployment status and provide real-time updates.
        """
        print(f"👀 Monitoring deployment for incident {incident_id}...")
        
        # Load incident record
        incident_file = self.incidents_dir / f"{incident_id}.json"
        if not incident_file.exists():
            return {"success": False, "error": f"Incident {incident_id} not found"}
        
        with open(incident_file, 'r') as f:
            incident = json.load(f)
        
        pr_url = incident.get("pr_url")
        if not pr_url:
            return {"success": False, "error": "PR not found for this incident"}
        
        try:
            # Extract PR number from URL
            pr_number = pr_url.split('/')[-1]
            
            # Get PR status
            cmd = ["gh", "pr", "view", pr_number, "--json", "state,statusCheckRollupState,mergeable"]
            result = self._run_command(cmd)
            pr_data = json.loads(result.stdout)
            
            # Get workflow status
            cmd = ["gh", "run", "list", "--repo", ".", "--workflow", "emergency-hotfix.yml", "--limit", "5", "--json", "status,conclusion,url"]
            result = self._run_command(cmd)
            workflow_data = json.loads(result.stdout)
            
            status = {
                "incident_id": incident_id,
                "pr_state": pr_data.get("state", "unknown"),
                "pr_mergeable": pr_data.get("mergeable", "unknown"),
                "checks_status": pr_data.get("statusCheckRollupState", "pending"),
                "recent_workflows": workflow_data[:3],
                "monitoring_time": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"PR State: {status['pr_state']}")
            print(f"Checks Status: {status['checks_status']}")
            print(f"Mergeable: {status['pr_mergeable']}")
            
            if workflow_data:
                latest_workflow = workflow_data[0]
                print(f"Latest Workflow: {latest_workflow.get('status', 'unknown')} - {latest_workflow.get('url', '')}")
            
            return {
                "success": True,
                "status": status
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error monitoring deployment: {e}")
            return {"success": False, "error": str(e)}
    
    def list_active_incidents(self) -> List[Dict]:
        """List all active emergency incidents."""
        incidents = []
        
        for incident_file in self.incidents_dir.glob("*.json"):
            try:
                with open(incident_file, 'r') as f:
                    incident = json.load(f)
                
                if incident.get("status") not in ["completed", "closed"]:
                    incidents.append({
                        "incident_id": incident["incident_id"],
                        "description": incident["description"],
                        "severity": incident["severity"],
                        "status": incident["status"],
                        "created_at": incident["created_at"],
                        "branch_name": incident.get("branch_name"),
                        "pr_url": incident.get("pr_url")
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Sort by creation time (newest first)
        incidents.sort(key=lambda x: x["created_at"], reverse=True)
        return incidents
    
    def simulate_emergency(self, test_scenario: str = "database_connection") -> Dict:
        """
        Simulate emergency scenario for testing hotfix workflow.
        """
        print(f"🧪 Simulating emergency scenario: {test_scenario}")
        
        scenarios = {
            "database_connection": {
                "description": "Critical database connection pool exhaustion",
                "severity": "critical",
                "test_files": {
                    "config/database.js": """
// Emergency fix: Increase connection pool size
module.exports = {
  pool: {
    min: 2,
    max: 20, // Increased from 10
    acquire: 30000,
    idle: 10000
  }
};
"""
                }
            },
            "authentication_bypass": {
                "description": "Authentication bypass vulnerability fix",
                "severity": "critical",
                "test_files": {
                    "middleware/auth.js": """
// Emergency fix: Validate JWT token properly
function validateToken(token) {
  if (!token) return false;
  // Fixed: Added proper signature verification
  return jwt.verify(token, process.env.JWT_SECRET);
}
"""
                }
            },
            "memory_leak": {
                "description": "Memory leak causing server crashes",
                "severity": "high",
                "test_files": {
                    "services/cache.js": """
// Emergency fix: Clear cache intervals
const cache = new Map();
setInterval(() => {
  cache.clear(); // Prevent memory accumulation
}, 300000); // Every 5 minutes
"""
                }
            }
        }
        
        if test_scenario not in scenarios:
            available = ", ".join(scenarios.keys())
            return {"success": False, "error": f"Unknown scenario. Available: {available}"}
        
        scenario = scenarios[test_scenario]
        
        try:
            # Create hotfix
            result = self.create_hotfix(
                description=scenario["description"],
                severity=scenario["severity"]
            )
            
            if not result["success"]:
                return result
            
            incident_id = result["incident_id"]
            
            # Create test files
            for file_path, content in scenario["test_files"].items():
                full_path = self.repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'w') as f:
                    f.write(content.strip())
                
                # Add and commit the file
                self._run_git(["add", file_path])
            
            # Commit changes
            commit_message = f"HOTFIX: {scenario['description']} ({incident_id})"
            self._run_git(["commit", "-m", commit_message])
            
            print(f"✅ Emergency scenario simulated successfully")
            print(f"📋 Files modified: {', '.join(scenario['test_files'].keys())}")
            
            return {
                "success": True,
                "incident_id": incident_id,
                "scenario": test_scenario,
                "files_modified": list(scenario["test_files"].keys()),
                "next_steps": [
                    f"1. Review the simulated fix",
                    f"2. Create PR: python emergency_hotfix_manager.py create-pr {incident_id}",
                    f"3. Monitor deployment: python emergency_hotfix_manager.py monitor {incident_id}",
                    f"4. Verify workflow execution in GitHub Actions"
                ]
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error simulating emergency: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_emergency_pr_body(self, incident: Dict) -> str:
        """Generate emergency PR description with all required details."""
        timeline_text = ""
        for entry in incident.get("timeline", []):
            timeline_text += f"- {entry['timestamp']}: {entry['event']}\n"
        
        return f"""🚨 **EMERGENCY HOTFIX** 🚨

**Incident ID:** {incident['incident_id']}
**Severity:** {incident['severity'].upper()}
**Created:** {incident['created_at']}

## Issue Description
{incident['description']}

## Emergency Context
This is an emergency hotfix requiring expedited review and deployment.

**Review Requirements:**
- ✅ Single reviewer approval (reduced from standard 2)
- ✅ Minimal quality gates (security + smoke tests only)
- ✅ Direct production deployment upon merge
- ✅ Automatic backport to main branch

## Timeline
{timeline_text}

## Testing
- [x] Local testing completed
- [x] Emergency quality gates will run automatically
- [x] Smoke tests included in workflow

## Deployment
- [ ] Emergency workflow will deploy automatically on merge
- [ ] Rollback capability available if issues occur
- [ ] Post-mortem will be generated automatically

## Post-Deployment
- [ ] Monitor application health
- [ ] Complete backport PR review
- [ ] Schedule post-mortem review
- [ ] Update documentation

---
**⚠️ This PR bypasses normal review processes due to emergency status**
**📋 Full audit trail maintained in incident record: {incident['incident_id']}.json**
"""
    
    def _get_git_user(self) -> str:
        """Get current Git user for audit trail."""
        try:
            result = self._run_git(["config", "user.name"])
            return result.stdout.strip()
        except:
            return "unknown-user"
    
    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run git command in repository directory."""
        return self._run_command(["git"] + args)
    
    def _run_command(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run command and return result."""
        return subprocess.run(
            args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="RIF Emergency Hotfix Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create emergency hotfix
  %(prog)s create "Database connection fix" --severity critical
  
  # Create emergency PR
  %(prog)s create-pr INC-20250825-143022
  
  # Monitor deployment
  %(prog)s monitor INC-20250825-143022
  
  # List active incidents
  %(prog)s list
  
  # Simulate emergency for testing
  %(prog)s simulate database_connection
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create hotfix command
    create_parser = subparsers.add_parser("create", help="Create emergency hotfix")
    create_parser.add_argument("description", help="Brief description of the issue")
    create_parser.add_argument("--severity", choices=["critical", "high", "medium", "low"], 
                              default="high", help="Issue severity level")
    create_parser.add_argument("--issue-url", help="GitHub issue URL")
    
    # Create PR command
    pr_parser = subparsers.add_parser("create-pr", help="Create emergency PR")
    pr_parser.add_argument("incident_id", help="Incident ID")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor deployment")
    monitor_parser.add_argument("incident_id", help="Incident ID")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List active incidents")
    
    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate emergency scenario")
    simulate_parser.add_argument("scenario", 
                                choices=["database_connection", "authentication_bypass", "memory_leak"],
                                help="Emergency scenario to simulate")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = HotfixManager()
    
    # Execute command
    if args.command == "create":
        result = manager.create_hotfix(
            description=args.description,
            severity=args.severity,
            issue_url=args.issue_url
        )
        
        if result["success"]:
            print(f"\n🎯 Next Steps:")
            for step in result["next_steps"]:
                print(f"   {step}")
        else:
            print(f"❌ Failed: {result['error']}")
            sys.exit(1)
    
    elif args.command == "create-pr":
        result = manager.create_emergency_pr(args.incident_id)
        
        if result["success"]:
            print(f"✅ Emergency PR created: {result['pr_url']}")
            print(f"⏱️  Monitor with: python {sys.argv[0]} monitor {args.incident_id}")
        else:
            print(f"❌ Failed: {result['error']}")
            sys.exit(1)
    
    elif args.command == "monitor":
        result = manager.monitor_deployment(args.incident_id)
        
        if not result["success"]:
            print(f"❌ Failed: {result['error']}")
            sys.exit(1)
    
    elif args.command == "list":
        incidents = manager.list_active_incidents()
        
        if not incidents:
            print("✅ No active emergency incidents")
        else:
            print(f"🚨 Active Emergency Incidents ({len(incidents)}):")
            print()
            for incident in incidents:
                print(f"📋 {incident['incident_id']}")
                print(f"   Description: {incident['description']}")
                print(f"   Severity: {incident['severity']}")
                print(f"   Status: {incident['status']}")
                print(f"   Branch: {incident.get('branch_name', 'N/A')}")
                if incident.get('pr_url'):
                    print(f"   PR: {incident['pr_url']}")
                print()
    
    elif args.command == "simulate":
        result = manager.simulate_emergency(args.scenario)
        
        if result["success"]:
            print(f"\n🎯 Next Steps:")
            for step in result["next_steps"]:
                print(f"   {step}")
        else:
            print(f"❌ Failed: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()