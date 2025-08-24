#!/usr/bin/env python3
"""
RIF Context Compliance Checkpoint System
Issue #115 & #145: Track context consumption compliance in RIF workflow

This system integrates with the existing RIF checkpoint system to track
context consumption compliance and prevent context failure emergencies.

CHECKPOINT ENHANCEMENTS:
1. Context consumption evidence tracking
2. Requirements compliance verification
3. Research methodology validation checkpoints
4. Emergency prevention measures
5. Recovery mechanisms for context failures
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ContextComplianceCheckpoint:
    """Enhanced checkpoint with context compliance tracking"""
    checkpoint_id: str
    issue_number: int
    agent_type: str
    phase: str
    timestamp: datetime
    status: str
    
    # Context compliance fields
    context_consumption_verified: bool
    requirements_extracted: bool
    research_methodology_compliance: bool
    validation_requirements_understood: bool
    evidence_provided: bool
    compliance_score: float
    
    # Original checkpoint fields
    implementation_summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    next_phase_readiness: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class ContextComplianceCheckpointManager:
    """
    Manages context compliance checkpoints integrated with RIF workflow
    """
    
    def __init__(self, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.knowledge_base_path = knowledge_base_path
        self.checkpoint_dir = os.path.join(knowledge_base_path, "checkpoints")
        self.compliance_dir = os.path.join(knowledge_base_path, "context_compliance")
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.compliance_dir, exist_ok=True)
    
    def create_context_compliance_checkpoint(self, 
                                           checkpoint_id: str,
                                           issue_number: int,
                                           agent_type: str,
                                           phase: str,
                                           context_validation_result: Dict[str, Any],
                                           implementation_summary: Dict[str, Any],
                                           quality_metrics: Dict[str, Any]) -> ContextComplianceCheckpoint:
        """
        Create enhanced checkpoint with context compliance tracking
        """
        checkpoint = ContextComplianceCheckpoint(
            checkpoint_id=checkpoint_id,
            issue_number=issue_number,
            agent_type=agent_type,
            phase=phase,
            timestamp=datetime.now(),
            status="complete" if context_validation_result.get("passed", False) else "context_compliance_failed",
            
            # Context compliance tracking
            context_consumption_verified=context_validation_result.get("passed", False),
            requirements_extracted=context_validation_result.get("requirements_extracted", False),
            research_methodology_compliance=context_validation_result.get("research_methodology_compliance", False),
            validation_requirements_understood=context_validation_result.get("validation_requirements_understood", False),
            evidence_provided=context_validation_result.get("evidence_provided", False),
            compliance_score=context_validation_result.get("compliance_score", 0.0),
            
            # Original checkpoint fields
            implementation_summary=implementation_summary,
            quality_metrics=quality_metrics,
            next_phase_readiness={
                "context_compliance_verified": context_validation_result.get("passed", False),
                "ready_for_next_phase": context_validation_result.get("passed", False),
                "emergency_prevention_active": True
            }
        )
        
        # Save checkpoint
        self._save_checkpoint(checkpoint)
        
        # Create compliance audit entry
        self._create_compliance_audit_entry(checkpoint)
        
        return checkpoint
    
    def _save_checkpoint(self, checkpoint: ContextComplianceCheckpoint):
        """Save checkpoint to file"""
        filename = f"{checkpoint.checkpoint_id}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
    
    def _create_compliance_audit_entry(self, checkpoint: ContextComplianceCheckpoint):
        """Create compliance audit entry"""
        audit_entry = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "issue_number": checkpoint.issue_number,
            "agent_type": checkpoint.agent_type,
            "phase": checkpoint.phase,
            "timestamp": checkpoint.timestamp.isoformat(),
            "compliance_status": "passed" if checkpoint.context_consumption_verified else "failed",
            "compliance_score": checkpoint.compliance_score,
            "context_consumption_verified": checkpoint.context_consumption_verified,
            "requirements_extracted": checkpoint.requirements_extracted,
            "research_methodology_compliance": checkpoint.research_methodology_compliance,
            "validation_requirements_understood": checkpoint.validation_requirements_understood,
            "evidence_provided": checkpoint.evidence_provided,
            "emergency_prevention": {
                "reference_issue": "145",
                "prevention_measures_active": True,
                "context_failure_prevented": checkpoint.context_consumption_verified
            }
        }
        
        # Append to compliance audit log
        audit_file = os.path.join(self.compliance_dir, f"compliance_audit_{datetime.now().strftime('%Y%m%d')}.json")
        
        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                audit_data = json.load(f)
        else:
            audit_data = {"audit_entries": []}
        
        audit_data["audit_entries"].append(audit_entry)
        
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
    
    def validate_checkpoint_compliance(self, checkpoint_id: str) -> Dict[str, Any]:
        """Validate checkpoint compliance status"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        
        if not os.path.exists(checkpoint_file):
            return {"error": "Checkpoint not found", "compliant": False}
        
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        compliance_status = {
            "checkpoint_id": checkpoint_id,
            "compliant": checkpoint_data.get("context_consumption_verified", False),
            "compliance_score": checkpoint_data.get("compliance_score", 0.0),
            "ready_for_next_phase": checkpoint_data.get("next_phase_readiness", {}).get("ready_for_next_phase", False),
            "emergency_prevention_active": checkpoint_data.get("next_phase_readiness", {}).get("emergency_prevention_active", False),
            "validation_timestamp": checkpoint_data.get("timestamp"),
            
            "compliance_details": {
                "requirements_extracted": checkpoint_data.get("requirements_extracted", False),
                "research_methodology_compliance": checkpoint_data.get("research_methodology_compliance", False),
                "validation_requirements_understood": checkpoint_data.get("validation_requirements_understood", False),
                "evidence_provided": checkpoint_data.get("evidence_provided", False)
            }
        }
        
        return compliance_status
    
    def create_emergency_prevention_checkpoint(self, issue_number: int, agent_type: str, 
                                             emergency_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create emergency prevention checkpoint for context failure prevention"""
        checkpoint_id = f"emergency_prevention_issue_{issue_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        emergency_checkpoint = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": "emergency_prevention",
            "issue_number": issue_number,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "emergency_reference": "issue_145_context_failure",
            
            "prevention_measures": {
                "mandatory_context_consumption_verification": True,
                "research_methodology_validation_required": True,
                "validation_requirements_verification_required": True,
                "evidence_documentation_required": True,
                "minimum_compliance_score": 0.8
            },
            
            "emergency_context": emergency_context,
            
            "recovery_mechanism": {
                "context_failure_detected": True,
                "automatic_validation_triggered": True,
                "agent_work_suspended_until_compliance": True,
                "escalation_to_manual_review": emergency_context.get("compliance_score", 0) < 0.5
            },
            
            "compliance_requirements": {
                "full_issue_context_reading": {"required": True, "status": "pending"},
                "requirements_extraction_documentation": {"required": True, "status": "pending"},
                "research_methodology_identification": {"required": True, "status": "pending"},
                "validation_requirements_understanding": {"required": True, "status": "pending"},
                "evidence_of_understanding": {"required": True, "status": "pending"}
            }
        }
        
        # Save emergency checkpoint
        emergency_file = os.path.join(self.compliance_dir, f"{checkpoint_id}.json")
        with open(emergency_file, 'w') as f:
            json.dump(emergency_checkpoint, f, indent=2)
        
        return emergency_checkpoint
    
    def update_issue_115_checkpoint_with_compliance(self):
        """Update the existing issue #115 checkpoint with compliance tracking"""
        existing_checkpoint_file = os.path.join(self.checkpoint_dir, "issue-115-research-implementation-complete.json")
        
        if not os.path.exists(existing_checkpoint_file):
            print("Existing checkpoint not found")
            return
        
        with open(existing_checkpoint_file, 'r') as f:
            existing_data = json.load(f)
        
        # Add compliance tracking to existing checkpoint
        compliance_update = {
            "context_compliance_tracking": {
                "compliance_verification_added": datetime.now().isoformat(),
                "emergency_prevention_measures": {
                    "reference_issue": "145",
                    "context_failure_prevention_active": True,
                    "research_methodology_validation": True,
                    "validation_requirements_verification": True
                },
                "retroactive_compliance_assessment": {
                    "issue_145_analysis": "Research methodology requirements not properly followed",
                    "corrective_measures_implemented": True,
                    "context_integration_system_deployed": True,
                    "validation_enforcer_activated": True
                }
            }
        }
        
        # Merge with existing data
        existing_data.update(compliance_update)
        
        # Save updated checkpoint
        with open(existing_checkpoint_file, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)
        
        print("✓ Updated issue #115 checkpoint with compliance tracking")
    
    def generate_compliance_report(self, issue_number: int) -> Dict[str, Any]:
        """Generate compliance report for issue"""
        compliance_files = [f for f in os.listdir(self.compliance_dir) 
                          if f.startswith(f"compliance_audit_") and f.endswith(".json")]
        
        compliance_entries = []
        for file in compliance_files:
            with open(os.path.join(self.compliance_dir, file), 'r') as f:
                data = json.load(f)
                entries = [entry for entry in data.get("audit_entries", []) 
                          if entry.get("issue_number") == issue_number]
                compliance_entries.extend(entries)
        
        if not compliance_entries:
            return {"error": f"No compliance entries found for issue #{issue_number}"}
        
        # Analyze compliance
        total_entries = len(compliance_entries)
        passed_entries = len([e for e in compliance_entries if e.get("compliance_status") == "passed"])
        average_score = sum(e.get("compliance_score", 0) for e in compliance_entries) / total_entries
        
        report = {
            "issue_number": issue_number,
            "compliance_summary": {
                "total_checkpoints": total_entries,
                "passed_checkpoints": passed_entries,
                "compliance_rate": passed_entries / total_entries if total_entries > 0 else 0,
                "average_compliance_score": average_score
            },
            "emergency_prevention_status": {
                "issue_145_prevention_active": True,
                "context_failure_prevention_measures": "deployed",
                "validation_enforcement": "active"
            },
            "detailed_entries": compliance_entries,
            "generated_at": datetime.now().isoformat()
        }
        
        return report

def integrate_compliance_with_existing_checkpoints():
    """Integration function to enhance existing RIF checkpoints with compliance"""
    manager = ContextComplianceCheckpointManager()
    
    # Update issue #115 checkpoint with compliance tracking
    manager.update_issue_115_checkpoint_with_compliance()
    
    # Create emergency prevention checkpoint for issue #115
    emergency_context = {
        "issue_reference": "145",
        "context_failure_type": "research_methodology_not_followed",
        "compliance_score": 0.3,  # Low score indicating failure
        "corrective_measures": "context_integration_system_deployed"
    }
    
    emergency_checkpoint = manager.create_emergency_prevention_checkpoint(
        115, "rif-implementer", emergency_context
    )
    
    print("✓ Compliance integration complete")
    print(f"✓ Emergency prevention checkpoint created: {emergency_checkpoint['checkpoint_id']}")
    
    return manager

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Context Compliance Checkpoint System")
    parser.add_argument("--integrate", action="store_true", help="Integrate with existing checkpoints")
    parser.add_argument("--report", type=int, help="Generate compliance report for issue number")
    parser.add_argument("--validate", type=str, help="Validate checkpoint compliance")
    
    args = parser.parse_args()
    
    manager = ContextComplianceCheckpointManager()
    
    if args.integrate:
        integrate_compliance_with_existing_checkpoints()
        
    elif args.report:
        report = manager.generate_compliance_report(args.report)
        print(json.dumps(report, indent=2))
        
    elif args.validate:
        validation = manager.validate_checkpoint_compliance(args.validate)
        print(json.dumps(validation, indent=2))
        
    else:
        parser.print_help()