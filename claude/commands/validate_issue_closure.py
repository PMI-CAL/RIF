#!/usr/bin/env python3
"""
Issue Closure Validation Script
Validates that issues meet all requirements before being closed.
"""

import sys
import json
from workflow_validation_system import WorkflowValidationSystem

def validate_issue_closure(issue_number: int) -> Dict[str, Any]:
    """Validate that an issue can be safely closed."""
    validator = WorkflowValidationSystem()
    
    validation_report = {
        'issue_number': issue_number,
        'can_close': True,
        'blocking_reasons': [],
        'warnings': [],
        'quality_score': None,
        'validation_timestamp': datetime.now().isoformat()
    }
    
    # Check state requirements
    current_state = validator.state_manager.get_current_state(issue_number)
    if current_state not in ['complete', 'failed']:
        validation_report['can_close'] = False
        validation_report['blocking_reasons'].append(f"Issue is in state '{current_state}', not complete")
    
    # Check for shadow issue requirements
    if validator._should_have_shadow_issue(issue_number):
        if not validator._has_existing_shadow(issue_number):
            validation_report['warnings'].append("High complexity issue closed without shadow quality tracking")
        else:
            # TODO: Check if shadow issue is also ready for closure
            pass
    
    # Check quality gates
    # TODO: Implement quality gate validation
    
    return validation_report

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_issue_closure.py <issue_number>")
        sys.exit(1)
    
    issue_num = int(sys.argv[1])
    result = validate_issue_closure(issue_num)
    
    print(json.dumps(result, indent=2))
    
    if not result['can_close']:
        sys.exit(1)  # Non-zero exit code indicates validation failure
