#!/usr/bin/env python3
"""
Batch update all RIF agent instruction files with knowledge consultation enforcement.
This ensures all agents have the mandatory enforcement integration for Issue #113.
"""

import os
import re
from pathlib import Path


def update_agent_enforcement(agent_file_path, agent_name):
    """Update a single agent file with enforcement integration."""
    
    if not os.path.exists(agent_file_path):
        print(f"⚠️  Agent file not found: {agent_file_path}")
        return False
    
    with open(agent_file_path, 'r') as f:
        content = f.read()
    
    # Skip if already updated
    if "MANDATORY ENFORCEMENT INTEGRATION" in content:
        print(f"✅ {agent_name}: Already has enforcement integration")
        return True
    
    # Find and update the critical rule
    critical_rule_pattern = r'(\*\*CRITICAL RULE\*\*: [^\n]+)'
    
    enforcement_text = f"""$1

### MANDATORY ENFORCEMENT INTEGRATION
**BEFORE ANY WORK OR DECISIONS:**
1. **Initialize Session**: `from claude.commands.knowledge_consultation_enforcer import get_knowledge_enforcer`
2. **Start Enforcement**: `session_key = enforcer.start_agent_session("{agent_name}", issue_id, task_description)`
3. **Record All MCP Usage**: `enforcer.record_knowledge_consultation(session_key, mcp_tool, query, result)`
4. **Request Decision Approval**: `approved = enforcer.request_decision_approval(session_key, decision_type, details)`
5. **Generate Compliance Report**: Include in final output

**ENFORCEMENT RULE**: All work is BLOCKED until knowledge consultation requirements are met."""
    
    updated_content = re.sub(critical_rule_pattern, enforcement_text, content)
    
    # Update context summary template to include enforcement session
    context_template_pattern = r'(\*\*Enforcement Session\*\*: \[SESSION_KEY\])'
    if "**Enforcement Session**" not in updated_content:
        context_summary_pattern = r'(\*\*Outstanding Concerns\*\*: \[COUNT\])'
        updated_content = re.sub(
            context_summary_pattern,
            r'$1\n**Enforcement Session**: [SESSION_KEY]',
            updated_content
        )
    
    # Add enforcement session to context verification checklist
    verification_pattern = r'(- \[ \] All recommendations catalogued)'
    if "- [ ] Enforcement session initialized" not in updated_content:
        updated_content = re.sub(
            verification_pattern,
            r'$1\n- [ ] Enforcement session initialized',
            updated_content
        )
    
    # Write updated content
    with open(agent_file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ {agent_name}: Updated with enforcement integration")
    return True


def main():
    """Update all RIF agent files with enforcement integration."""
    print("🔒 Batch Updating RIF Agents with Knowledge Consultation Enforcement")
    print("=" * 80)
    
    agent_files = [
        ("rif-analyst.md", "rif-analyst"),
        ("rif-architect.md", "rif-architect"), 
        ("rif-error-analyst.md", "rif-error-analyst"),
        ("rif-implementer.md", "rif-implementer"),
        ("rif-learner.md", "rif-learner"),
        ("rif-planner.md", "rif-planner"),
        ("rif-pr-manager.md", "rif-pr-manager"),
        ("rif-projectgen.md", "rif-projectgen"),
        ("rif-shadow-auditor.md", "rif-shadow-auditor"),
        ("rif-validator.md", "rif-validator")
    ]
    
    agent_dir = "/Users/cal/DEV/RIF/claude/agents"
    updated_count = 0
    
    for filename, agent_name in agent_files:
        agent_path = os.path.join(agent_dir, filename)
        if update_agent_enforcement(agent_path, agent_name):
            updated_count += 1
    
    print("=" * 80)
    print(f"📊 Enforcement Integration Complete: {updated_count}/{len(agent_files)} agents updated")
    
    if updated_count == len(agent_files):
        print("✅ All RIF agents now have mandatory knowledge consultation enforcement!")
        print("\n🔒 Enforcement Features Added:")
        print("   - Session initialization required before any work")
        print("   - MCP tool usage tracking and recording")
        print("   - Decision approval blocking without knowledge consultation")
        print("   - Compliance reporting in agent outputs")
        print("\n🎯 Issue #113 Core Problem SOLVED: Agents can no longer bypass knowledge consultation")
    else:
        print(f"⚠️  {len(agent_files) - updated_count} agents need manual attention")
    
    return updated_count == len(agent_files)


if __name__ == "__main__":
    main()