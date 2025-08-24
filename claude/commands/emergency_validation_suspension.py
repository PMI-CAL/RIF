#!/usr/bin/env python3
"""
Emergency Validation Suspension Protocol
Addresses Issue #145: Critical context compliance failures

Suspends validation for all issues until context compliance verified.
Prevents further issues from being marked complete without proper evidence.
"""

import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any

class EmergencyValidationSuspension:
    def __init__(self):
        self.suspension_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trigger_issue": 145,
            "trigger_reason": "Critical context compliance failures - 8 critical findings, 24 total failures",
            "suspended_issues": [],
            "protected_states": ["state:complete", "state:validating"],
            "actions_taken": []
        }
    
    def activate_emergency_protocol(self) -> Dict[str, Any]:
        """Activate emergency validation suspension protocol"""
        
        print("🚨 ACTIVATING EMERGENCY VALIDATION SUSPENSION PROTOCOL 🚨")
        print()
        
        # Step 1: Suspend all validating issues
        self._suspend_validating_issues()
        
        # Step 2: Create validation suspension label
        self._create_suspension_labels()
        
        # Step 3: Apply suspension to critical issues
        self._apply_critical_suspensions()
        
        # Step 4: Create validation lockout system
        self._create_validation_lockout()
        
        # Step 5: Document emergency actions
        self._document_emergency_actions()
        
        return self.suspension_record
    
    def _suspend_validating_issues(self) -> None:
        """Suspend all issues currently in validating state"""
        
        print("Step 1: Suspending all validating issues...")
        
        try:
            # Get all validating issues
            cmd = [
                "gh", "issue", "list",
                "--state", "open", 
                "--label", "state:validating",
                "--json", "number,title,labels"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            issues = json.loads(result.stdout)
            
            for issue in issues:
                issue_number = issue["number"]
                issue_title = issue["title"]
                
                # Change state from validating to suspended
                suspend_cmd = [
                    "gh", "issue", "edit", str(issue_number),
                    "--remove-label", "state:validating",
                    "--add-label", "state:suspended"
                ]
                
                try:
                    subprocess.run(suspend_cmd, check=True, capture_output=True)
                    
                    # Add comment explaining suspension
                    comment = f"""🚨 **VALIDATION SUSPENDED - EMERGENCY PROTOCOL** 🚨

**Trigger**: Issue #145 - Critical context compliance failures detected
**Suspension Reason**: Emergency validation suspension due to systematic context compliance failures

**Context Compliance Audit Results**:
- 8 critical findings: Issues with missing literature review requirements
- 24 total compliance failures across issue lifecycle
- Evidence of agents not fully consuming issue context

**Status**: This issue is suspended from validation until:
1. ✅ Context compliance audit review completed
2. ✅ Enhanced context verification system activated  
3. ✅ Knowledge consultation enforcer enabled
4. ✅ Validation protocols hardened

**Next Steps**: Issue will remain suspended until emergency protocol completion."""
                    
                    comment_cmd = [
                        "gh", "issue", "comment", str(issue_number),
                        "--body", comment
                    ]
                    
                    subprocess.run(comment_cmd, check=True)
                    
                    self.suspension_record["suspended_issues"].append({
                        "issue_number": issue_number,
                        "title": issue_title,
                        "previous_state": "validating",
                        "suspension_reason": "emergency_protocol_activation"
                    })
                    
                    print(f"  ✅ Suspended issue #{issue_number}: {issue_title}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Failed to suspend issue #{issue_number}: {e}")
                    self.suspension_record["actions_taken"].append({
                        "action": "suspend_issue",
                        "issue": issue_number,
                        "status": "failed",
                        "error": str(e)
                    })
            
            print(f"Suspended {len(issues)} validating issues")
            print()
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting validating issues: {e}")
    
    def _create_suspension_labels(self) -> None:
        """Create labels for suspension tracking"""
        
        print("Step 2: Creating suspension tracking labels...")
        
        labels_to_create = [
            {
                "name": "state:suspended",
                "description": "Issue suspended due to emergency protocol",
                "color": "8B0000"
            },
            {
                "name": "emergency:context-failure", 
                "description": "Issue flagged for context compliance failure",
                "color": "FF0000"
            },
            {
                "name": "validation:suspended",
                "description": "Validation suspended until emergency resolved",
                "color": "B22222"
            }
        ]
        
        for label in labels_to_create:
            try:
                cmd = [
                    "gh", "label", "create", label["name"],
                    "--description", label["description"],
                    "--color", label["color"]
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✅ Created label: {label['name']}")
                else:
                    # Label might already exist
                    print(f"  ⚠️  Label {label['name']} may already exist")
                    
            except Exception as e:
                print(f"  ❌ Error creating label {label['name']}: {e}")
        
        print()
    
    def _apply_critical_suspensions(self) -> None:
        """Apply suspensions to critical issues identified in audit"""
        
        print("Step 3: Applying suspensions to critical issues...")
        
        # Critical issues from audit 
        critical_issues = [145, 135, 115, 136, 134, 133, 118, 117]
        
        for issue_number in critical_issues:
            try:
                # Get current issue status
                cmd = ["gh", "issue", "view", str(issue_number), "--json", "title,labels,state"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                issue_data = json.loads(result.stdout)
                
                if issue_data["state"] != "OPEN":
                    continue  # Skip closed issues for now
                
                # Apply emergency labels
                label_cmd = [
                    "gh", "issue", "edit", str(issue_number),
                    "--add-label", "emergency:context-failure",
                    "--add-label", "validation:suspended"
                ]
                
                subprocess.run(label_cmd, check=True, capture_output=True)
                
                # Add emergency comment
                comment = f"""🚨 **CRITICAL CONTEXT COMPLIANCE FAILURE** 🚨

**Emergency Audit Finding**: This issue identified as having critical context compliance failure
**Failure Type**: Missing literature review requirement despite explicit requirement in issue body

**Required Immediate Actions**:
1. 🔄 Complete missing literature review with academic citations
2. 🔄 Document research methodology and findings
3. 🔄 Provide evidence of requirement fulfillment
4. 🔄 Submit for enhanced validation review

**Status**: SUSPENDED from further progression until compliance verified
**Emergency Protocol**: Issue #145 - Systematic agent context failures detected"""
                
                comment_cmd = [
                    "gh", "issue", "comment", str(issue_number),
                    "--body", comment
                ]
                
                subprocess.run(comment_cmd, check=True)
                
                print(f"  ✅ Applied emergency suspension to issue #{issue_number}")
                
                self.suspension_record["suspended_issues"].append({
                    "issue_number": issue_number,
                    "title": issue_data["title"],
                    "suspension_type": "critical_context_failure",
                    "failure_type": "missing_literature_review"
                })
                
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to suspend critical issue #{issue_number}: {e}")
        
        print()
    
    def _create_validation_lockout(self) -> None:
        """Create validation lockout system"""
        
        print("Step 4: Creating validation lockout system...")
        
        lockout_config = {
            "lockout_active": True,
            "activation_timestamp": datetime.utcnow().isoformat() + "Z",
            "trigger_issue": 145,
            "lockout_conditions": {
                "no_validation_without_context_audit": True,
                "literature_review_mandatory": True,
                "evidence_requirements_enforced": True,
                "knowledge_consultation_mandatory": True
            },
            "release_conditions": {
                "context_verification_system_active": False,
                "knowledge_consultation_enforcer_active": False,
                "critical_issues_remediated": False,
                "validation_protocols_hardened": False
            },
            "suspended_count": len(self.suspension_record["suspended_issues"])
        }
        
        # Save lockout configuration
        lockout_file = "/Users/cal/DEV/RIF/config/emergency-validation-lockout.json"
        try:
            with open(lockout_file, 'w', encoding='utf-8') as f:
                json.dump(lockout_config, f, indent=2)
            
            print(f"  ✅ Validation lockout configuration saved: {lockout_file}")
            
            self.suspension_record["actions_taken"].append({
                "action": "create_lockout_system",
                "file": lockout_file,
                "status": "success"
            })
            
        except Exception as e:
            print(f"  ❌ Error creating lockout configuration: {e}")
        
        print()
    
    def _document_emergency_actions(self) -> None:
        """Document all emergency actions taken"""
        
        print("Step 5: Documenting emergency actions...")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") 
        record_file = f"/Users/cal/DEV/RIF/knowledge/emergency_validation_suspension_{timestamp}.json"
        
        try:
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(self.suspension_record, f, indent=2)
            
            print(f"  ✅ Emergency actions documented: {record_file}")
            
        except Exception as e:
            print(f"  ❌ Error documenting emergency actions: {e}")
        
        print()


def main():
    """Execute emergency validation suspension protocol"""
    
    suspension_system = EmergencyValidationSuspension()
    results = suspension_system.activate_emergency_protocol()
    
    print("🚨 EMERGENCY VALIDATION SUSPENSION PROTOCOL COMPLETE 🚨")
    print()
    print("SUMMARY:")
    print(f"  Issues suspended: {len(results['suspended_issues'])}")
    print(f"  Actions taken: {len(results['actions_taken'])}")
    print(f"  Emergency lockout: ACTIVE")
    print()
    print("NEXT STEPS:")
    print("  1. Deploy enhanced context verification system")
    print("  2. Activate knowledge consultation enforcer")  
    print("  3. Remediate critical issues")
    print("  4. Harden validation protocols")
    print("  5. Release lockout when conditions met")
    print()
    
    return results


if __name__ == "__main__":
    main()