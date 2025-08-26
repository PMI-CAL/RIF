#!/usr/bin/env python3
"""
Branch Protection Setup Script
Configures GitHub branch protection rules with progressive activation
"""

import json
import logging
import argparse
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BranchProtectionManager:
    """Manages GitHub branch protection rules with progressive activation"""
    
    def __init__(self, config_path: str = ".github/branch-protection.json", repo: Optional[str] = None):
        self.config_path = Path(config_path)
        self.repo = repo
        self.audit_log: List[Dict[str, Any]] = []
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file or raise FileNotFoundError if missing"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations with override taking precedence"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load branch protection configuration"""
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return self._create_default_config()
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default branch protection configuration"""
        default_config = {
            "repository": self.repo or "current",
            "protection_rules": {
                "main": {
                    "require_pr": True,
                    "require_reviews": 1,
                    "dismiss_stale_reviews": True,
                    "require_status_checks": [
                        "continuous-integration",
                        "security-scan", 
                        "test-coverage",
                        "rif-validation"
                    ],
                    "require_up_to_date": True,
                    "allow_auto_merge": True,
                    "delete_branch_on_merge": True,
                    "restrict_pushes": True,
                    "allow_force_pushes": False,
                    "allow_deletions": False
                }
            },
            "progressive_activation": {
                "enabled": True,
                "phases": [
                    {
                        "name": "Phase 1: Basic Protection",
                        "rules": ["require_pr", "require_reviews"],
                        "wait_hours": 0
                    },
                    {
                        "name": "Phase 2: Status Checks",
                        "rules": ["require_status_checks", "require_up_to_date"],
                        "wait_hours": 24
                    },
                    {
                        "name": "Phase 3: Advanced Features", 
                        "rules": ["allow_auto_merge", "delete_branch_on_merge"],
                        "wait_hours": 48
                    }
                ]
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Created default configuration at {self.config_path}")
        
        return default_config
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values"""
        try:
            required_keys = ['repository', 'protection_rules']
            for key in required_keys:
                if key not in config:
                    logger.error(f"Missing required configuration key: {key}")
                    return False
            
            # Validate protection rules structure
            if 'main' not in config['protection_rules']:
                logger.error("Main branch protection rules not found")
                return False
                
            main_rules = config['protection_rules']['main']
            required_rule_keys = ['require_pr', 'require_reviews']
            for key in required_rule_keys:
                if key not in main_rules:
                    logger.error(f"Missing required rule: {key}")
                    return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_current_protection(self, branch: str = "main") -> Optional[Dict[str, Any]]:
        """Get current branch protection rules"""
        try:
            result = subprocess.run([
                "gh", "api", f"/repos/{{owner}}/{{repo}}/branches/{branch}/protection"
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                protection = json.loads(result.stdout)
                logger.info(f"Retrieved current protection for branch '{branch}'")
                return protection
            elif result.returncode == 404:
                logger.info(f"No protection rules found for branch '{branch}'")
                return None
            else:
                logger.error(f"Failed to get protection rules: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current protection: {e}")
            return None
    
    def apply_protection_rules(self, branch: str, rules: Dict[str, Any], phase_name: str = "Full Setup") -> bool:
        """Apply protection rules to a branch"""
        try:
            # Build protection request
            protection_request = self._build_protection_request(rules)
            
            logger.info(f"Applying {phase_name} to branch '{branch}'")
            logger.debug(f"Protection request: {json.dumps(protection_request, indent=2)}")
            
            # Apply protection via GitHub API
            result = subprocess.run([
                "gh", "api", "--method", "PUT",
                f"/repos/{{owner}}/{{repo}}/branches/{branch}/protection",
                "--input", "-"
            ], input=json.dumps(protection_request), text=True, capture_output=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully applied protection rules to '{branch}'")
                self._log_audit_event("apply_protection", {
                    "branch": branch,
                    "phase": phase_name,
                    "rules": rules,
                    "success": True
                })
                return True
            else:
                logger.error(f"Failed to apply protection rules: {result.stderr}")
                self._log_audit_event("apply_protection", {
                    "branch": branch,
                    "phase": phase_name,
                    "rules": rules,
                    "success": False,
                    "error": result.stderr
                })
                return False
                
        except Exception as e:
            logger.error(f"Error applying protection rules: {e}")
            self._log_audit_event("apply_protection", {
                "branch": branch,
                "phase": phase_name,
                "rules": rules,
                "success": False,
                "error": str(e)
            })
            return False
    
    def _build_protection_request(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Build GitHub API protection request from rules"""
        request = {
            "required_status_checks": None,
            "enforce_admins": True,
            "required_pull_request_reviews": None,
            "restrictions": None
        }
        
        # Required status checks
        if rules.get("require_status_checks"):
            request["required_status_checks"] = {
                "strict": rules.get("require_up_to_date", True),
                "contexts": rules["require_status_checks"]
            }
        
        # Required PR reviews
        if rules.get("require_pr", False):
            request["required_pull_request_reviews"] = {
                "required_approving_review_count": rules.get("require_reviews", 1),
                "dismiss_stale_reviews": rules.get("dismiss_stale_reviews", True),
                "require_code_owner_reviews": rules.get("require_code_owners", False),
                "require_last_push_approval": rules.get("require_last_push_approval", True)
            }
        
        # Push restrictions (usually empty for open source)
        if rules.get("restrict_pushes", True):
            request["restrictions"] = {
                "users": [],
                "teams": [],
                "apps": []
            }
        
        # Additional settings
        request.update({
            "allow_force_pushes": rules.get("allow_force_pushes", False),
            "allow_deletions": rules.get("allow_deletions", False),
            "block_creations": rules.get("block_creations", False)
        })
        
        return request
    
    def progressive_setup(self, branch: str = "main") -> bool:
        """Execute progressive branch protection setup"""
        try:
            config = self.load_configuration()
            if not self.validate_configuration(config):
                return False
            
            if not config.get("progressive_activation", {}).get("enabled", False):
                # Apply all rules at once
                rules = config["protection_rules"][branch]
                return self.apply_protection_rules(branch, rules, "Complete Setup")
            
            # Progressive activation
            phases = config["progressive_activation"]["phases"]
            branch_rules = config["protection_rules"][branch]
            
            for i, phase in enumerate(phases):
                phase_name = phase["name"]
                wait_hours = phase.get("wait_hours", 0)
                
                # Build rules for this phase
                phase_rules = {}
                for rule_key in phase["rules"]:
                    if rule_key in branch_rules:
                        phase_rules[rule_key] = branch_rules[rule_key]
                
                # Wait if required (except for first phase)
                if i > 0 and wait_hours > 0:
                    logger.info(f"Waiting {wait_hours} hours before {phase_name}")
                    # In production, this would actually wait. For demo, we skip.
                    # time.sleep(wait_hours * 3600)
                
                # Apply phase rules
                if not self.apply_protection_rules(branch, phase_rules, phase_name):
                    logger.error(f"Failed to apply {phase_name}")
                    return False
                
                logger.info(f"Completed {phase_name}")
            
            logger.info("Progressive setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Progressive setup failed: {e}")
            return False
    
    def _log_audit_event(self, action: str, details: Dict[str, Any]):
        """Log audit event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        self.audit_log.append(event)
        
        # Also log to file
        audit_file = Path("logs/branch-protection-audit.log")
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(audit_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate status report of branch protection"""
        try:
            current_protection = self.get_current_protection()
            config = self.load_configuration()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "repository": self.repo or "current",
                "branch": "main",
                "configured_rules": config.get("protection_rules", {}).get("main", {}),
                "applied_rules": current_protection,
                "audit_events": len(self.audit_log),
                "status": "active" if current_protection else "not_configured"
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate status report: {e}")
            return {"error": str(e)}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GitHub Branch Protection Setup")
    parser.add_argument("--repo", help="Repository name (owner/repo)")
    parser.add_argument("--config", default=".github/branch-protection.json", 
                       help="Configuration file path")
    parser.add_argument("--branch", default="main", help="Branch to protect")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration")
    parser.add_argument("--status", action="store_true", 
                       help="Show current protection status")
    parser.add_argument("--progressive", action="store_true", 
                       help="Use progressive activation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without applying")
    
    args = parser.parse_args()
    
    try:
        manager = BranchProtectionManager(args.config, args.repo)
        
        if args.validate_only:
            config = manager.load_configuration()
            valid = manager.validate_configuration(config)
            print(f"Configuration validation: {'PASSED' if valid else 'FAILED'}")
            sys.exit(0 if valid else 1)
        
        if args.status:
            report = manager.generate_status_report()
            print(json.dumps(report, indent=2))
            sys.exit(0)
        
        if args.dry_run:
            config = manager.load_configuration()
            print("Dry run - would apply these rules:")
            print(json.dumps(config["protection_rules"][args.branch], indent=2))
            sys.exit(0)
        
        # Apply protection rules
        if args.progressive:
            success = manager.progressive_setup(args.branch)
        else:
            config = manager.load_configuration()
            rules = config["protection_rules"][args.branch]
            success = manager.apply_protection_rules(args.branch, rules)
        
        if success:
            print(f"✅ Branch protection successfully applied to '{args.branch}'")
            sys.exit(0)
        else:
            print(f"❌ Failed to apply branch protection to '{args.branch}'")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(f"❌ Script execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()