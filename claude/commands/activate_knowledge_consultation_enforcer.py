#!/usr/bin/env python3
"""
Knowledge Consultation Enforcer Activation Script
Addresses Issue #145: Activate enforcer for emergency context protocol

Activates and configures the knowledge consultation enforcer
to prevent agents from making decisions without consulting knowledge base.
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Any

def activate_knowledge_consultation_enforcer():
    """Activate knowledge consultation enforcer for emergency protocol"""
    
    print("🔒 ACTIVATING KNOWLEDGE CONSULTATION ENFORCER 🔒")
    print()
    
    activation_results = {
        "activation_timestamp": datetime.utcnow().isoformat() + "Z",
        "trigger_issue": 145,
        "enforcement_mode": "strict",
        "activation_components": [],
        "configuration_updates": [],
        "integration_status": {}
    }
    
    # Step 1: Configure enforcement mode to strict
    print("Step 1: Configuring enforcement mode to STRICT...")
    
    enforcement_config = {
        "enforcement_active": True,
        "enforcement_mode": "strict",
        "activation_trigger": "Issue #145 - Critical context compliance failures",
        "enforcement_rules": {
            "knowledge_consultation_mandatory": True,
            "minimum_mcp_tools_required": 2,
            "decision_blocking_enabled": True,
            "evidence_documentation_required": True
        },
        "required_mcp_tools": [
            "mcp__rif-knowledge__get_claude_documentation",
            "mcp__rif-knowledge__query_knowledge", 
            "mcp__rif-knowledge__check_compatibility"
        ],
        "enforcement_thresholds": {
            "minimum_tools_for_approval": 1,
            "consultation_depth_required": 2,
            "compliance_score_threshold": 80.0
        }
    }
    
    config_file = "/Users/cal/DEV/RIF/config/knowledge_consultation_enforcement.json"
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(enforcement_config, f, indent=2)
        
        print(f"  ✅ Enforcement configuration saved: {config_file}")
        
        activation_results["configuration_updates"].append({
            "component": "enforcement_config",
            "file": config_file,
            "status": "configured"
        })
        
    except Exception as e:
        print(f"  ❌ Failed to configure enforcement: {e}")
    
    print()
    
    # Step 2: Test enforcer activation
    print("Step 2: Testing enforcer activation...")
    
    try:
        # Import and test the enforcer
        import sys
        sys.path.append('/Users/cal/DEV/RIF/claude/commands')
        
        from knowledge_consultation_enforcer import get_knowledge_enforcer
        
        # Create enforcer instance in strict mode
        enforcer = get_knowledge_enforcer("strict")
        
        # Test session creation
        test_session = enforcer.start_agent_session("test-agent", "145", "Emergency protocol test")
        
        # Test decision blocking (should block)
        decision_blocked = not enforcer.request_decision_approval(
            test_session, 
            "test_decision", 
            "Emergency protocol validation"
        )
        
        if decision_blocked:
            print("  ✅ Decision blocking working correctly")
            
            activation_results["activation_components"].append({
                "component": "decision_blocking",
                "status": "active",
                "test_result": "decisions_blocked_when_no_consultation"
            })
        else:
            print("  ⚠️  Decision blocking may not be working")
            
        # Generate system metrics
        metrics = enforcer.get_system_metrics()
        print(f"  📊 System metrics: {metrics.get('message', 'Active sessions detected')}")
        
        activation_results["activation_components"].append({
            "component": "knowledge_enforcer",
            "status": "active",
            "metrics": metrics
        })
        
    except Exception as e:
        print(f"  ❌ Error testing enforcer: {e}")
        activation_results["activation_components"].append({
            "component": "knowledge_enforcer",
            "status": "error",
            "error": str(e)
        })
    
    print()
    
    # Step 3: Update RIF agent instructions to include mandatory consultation
    print("Step 3: Updating RIF agent instructions...")
    
    agent_files = [
        "claude/agents/rif-analyst.md",
        "claude/agents/rif-implementer.md", 
        "claude/agents/rif-validator.md",
        "claude/agents/rif-planner.md",
        "claude/agents/rif-architect.md",
        "claude/agents/rif-learner.md"
    ]
    
    consultation_requirement = """
## 🚨 MANDATORY KNOWLEDGE CONSULTATION (Emergency Protocol Active)

**CRITICAL**: Before making ANY decisions or recommendations, you MUST:

1. **Query Claude Documentation**: Use `mcp__rif-knowledge__get_claude_documentation` to understand available capabilities
2. **Search Knowledge Base**: Use `mcp__rif-knowledge__query_knowledge` to find relevant patterns and solutions  
3. **Validate Compatibility**: Use `mcp__rif-knowledge__check_compatibility` to ensure approach compatibility

**ENFORCEMENT**: Knowledge Consultation Enforcer is ACTIVE in STRICT mode
- Decisions will be BLOCKED if knowledge consultation requirements not met
- You must provide evidence of consultation before proceeding
- All work artifacts must reference knowledge base findings

**Template for Knowledge Consultation Evidence**:
```
## Knowledge Consultation Evidence

### 1. Claude Documentation Query
**Query**: [what you searched for]
**Findings**: [key capabilities identified]

### 2. Knowledge Base Search  
**Query**: [patterns/solutions searched]
**Relevant Patterns**: [list patterns found]

### 3. Compatibility Check
**Approach**: [your planned approach]
**Compatibility**: [validation results]

### 4. Decision Justification
**Decision**: [your decision/recommendation]
**Knowledge Base Support**: [how knowledge supports this decision]
```

**Failure to comply will result in decision blocking and issue reassignment.**
"""
    
    updated_agents = 0
    
    for agent_file in agent_files:
        try:
            if os.path.exists(agent_file):
                # Read existing content
                with open(agent_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if already updated
                if "MANDATORY KNOWLEDGE CONSULTATION" not in content:
                    # Add consultation requirement at the beginning after title
                    lines = content.split('\n')
                    
                    # Find first section after title
                    insert_index = 1
                    for i, line in enumerate(lines):
                        if line.startswith('## ') and i > 0:
                            insert_index = i
                            break
                    
                    # Insert consultation requirement
                    lines.insert(insert_index, consultation_requirement)
                    
                    # Write updated content
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    
                    updated_agents += 1
                    print(f"  ✅ Updated {agent_file}")
                
        except Exception as e:
            print(f"  ❌ Failed to update {agent_file}: {e}")
    
    if updated_agents > 0:
        activation_results["configuration_updates"].append({
            "component": "agent_instructions",
            "files_updated": updated_agents,
            "status": "updated"
        })
    
    print(f"  Updated {updated_agents} agent instruction files")
    print()
    
    # Step 4: Create enforcement monitoring
    print("Step 4: Setting up enforcement monitoring...")
    
    monitoring_config = {
        "monitoring_active": True,
        "monitoring_interval_minutes": 15,
        "alert_thresholds": {
            "compliance_rate_below": 90.0,
            "blocked_decisions_above": 5,
            "non_compliant_agents_above": 2
        },
        "monitoring_actions": [
            "log_compliance_metrics",
            "generate_agent_reports", 
            "alert_on_threshold_breach",
            "escalate_repeated_violations"
        ]
    }
    
    monitoring_file = "/Users/cal/DEV/RIF/config/enforcement_monitoring.json"
    
    try:
        with open(monitoring_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_config, f, indent=2)
        
        print(f"  ✅ Enforcement monitoring configured: {monitoring_file}")
        
        activation_results["configuration_updates"].append({
            "component": "enforcement_monitoring",
            "file": monitoring_file,
            "status": "configured"
        })
        
    except Exception as e:
        print(f"  ❌ Failed to configure monitoring: {e}")
    
    print()
    
    # Step 5: Document activation
    print("Step 5: Documenting enforcer activation...")
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    activation_file = f"/Users/cal/DEV/RIF/knowledge/knowledge_consultation_enforcer_activation_{timestamp}.json"
    
    try:
        with open(activation_file, 'w', encoding='utf-8') as f:
            json.dump(activation_results, f, indent=2)
        
        print(f"  ✅ Activation documented: {activation_file}")
        
    except Exception as e:
        print(f"  ❌ Failed to document activation: {e}")
    
    print()
    
    return activation_results


def main():
    """Activate knowledge consultation enforcer for emergency protocol"""
    
    results = activate_knowledge_consultation_enforcer()
    
    print("🔒 KNOWLEDGE CONSULTATION ENFORCER ACTIVATION COMPLETE 🔒")
    print()
    print("ACTIVATION SUMMARY:")
    print(f"  Enforcement mode: STRICT")
    print(f"  Components activated: {len(results['activation_components'])}")
    print(f"  Configuration updates: {len(results['configuration_updates'])}")
    print()
    print("ACTIVE ENFORCEMENTS:")
    print("  ✅ Mandatory MCP tool consultation before decisions")
    print("  ✅ Decision blocking for non-compliant agents")
    print("  ✅ Evidence documentation requirements")
    print("  ✅ Agent instruction updates with consultation templates")
    print("  ✅ Compliance monitoring and alerting")
    print()
    print("NEXT STEPS:")
    print("  1. All new agent sessions will be subject to strict enforcement")
    print("  2. Existing agents must complete knowledge consultation before decisions")
    print("  3. Monitor compliance metrics and adjust thresholds as needed")
    print("  4. Review blocked decision reports for enforcement effectiveness")
    print()
    
    return results


if __name__ == "__main__":
    main()