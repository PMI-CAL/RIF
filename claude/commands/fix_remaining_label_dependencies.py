#!/usr/bin/env python3
"""
Script to fix remaining current_state_label dependencies for Issue #273
This script systematically replaces label dependency with content analysis
"""

import re
import sys
from pathlib import Path

def fix_enhanced_orchestration_intelligence():
    """Fix all remaining current_state_label references in enhanced_orchestration_intelligence.py"""
    
    file_path = Path("claude/commands/enhanced_orchestration_intelligence.py")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define the replacement patterns
    replacements = [
        # Pattern 1: Simple state access in loops
        (
            r'(\s+)current_state = context_model\.issue_context\.current_state_label',
            r'\1# ISSUE #273 FIX: Replace label dependency with content analysis\n\1current_state = context_model.issue_context.current_state_from_content'
        ),
        
        # Pattern 2: In group processing
        (
            r'(\s+)current_state = context_model\.issue_context\.current_state_label',
            r'\1# ISSUE #273 FIX: Replace label dependency with content analysis\n\1current_state = context_model.issue_context.current_state_from_content'
        ),
        
        # Pattern 3: For context_model in group
        (
            r'(\s+for context_model in group:\s+)(\s+)current_state = context_model\.issue_context\.current_state_label',
            r'\1\2# ISSUE #273 FIX: Replace label dependency with content analysis\n\1\2current_state = context_model.issue_context.current_state_from_content'
        ),
        
        # Pattern 4: In agent recommendations
        (
            r'(\s+def get_agent_recommendations.*?\n.*?)current_state = context_model\.issue_context\.current_state_label',
            r'\1# ISSUE #273 FIX: Replace label dependency with content analysis\n        current_state = context_model.issue_context.current_state_from_content',
            re.DOTALL
        ),
        
        # Pattern 5: In validation methods
        (
            r'(\s+)current_state = context_model\.issue_context\.current_state_label(\s+if)',
            r'\1# ISSUE #273 FIX: Replace label dependency with content analysis\n\1current_state = context_model.issue_context.current_state_from_content\2'
        ),
        
        # Pattern 6: In resource constraint checks  
        (
            r'(\s+)current_state = cm\.issue_context\.current_state_label',
            r'\1# ISSUE #273 FIX: Replace label dependency with content analysis\n\1current_state = cm.issue_context.current_state_from_content'
        ),
        
        # Pattern 7: In batch processing with condition checks
        (
            r'if bcm\.issue_context\.current_state_label and',
            r'# ISSUE #273 FIX: Replace label dependency with content analysis\n                    if bcm.issue_context.current_state_from_content and'
        ),
        
        # Pattern 8: String checks in conditions
        (
            r'\'implementing\' in bcm\.issue_context\.current_state_label',
            r'\'implementing\' in (bcm.issue_context.current_state_from_content or \'\')'
        ),
        
        # Pattern 9: Issue context direct access
        (
            r'(\s+)issue_context = self\.context_analyzer\.analyze_issue\(issue_num\)\s+current_state = issue_context\.current_state_label',
            r'\1issue_context = self.context_analyzer.analyze_issue(issue_num)\n\1# ISSUE #273 FIX: Replace label dependency with content analysis\n\1current_state = issue_context.current_state_from_content'
        ),
        
        # Pattern 10: In prompt generation
        (
            r'Current state: \{cm\.issue_context\.current_state_label\}',
            r'Current state: {cm.issue_context.current_state_from_content or "unknown"}'
        ),
    ]
    
    # Apply replacements
    modified = False
    for pattern, replacement, *flags in replacements:
        regex_flags = flags[0] if flags else 0
        new_content = re.sub(pattern, replacement, content, flags=regex_flags)
        if new_content != content:
            content = new_content
            modified = True
            print(f"Applied replacement for pattern: {pattern[:50]}...")
    
    # Write back the modified content
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Successfully updated {file_path}")
        return True
    else:
        print("No modifications needed")
        return False

def fix_orchestration_utilities():
    """Fix orchestration_utilities.py references"""
    
    file_path = Path("claude/commands/orchestration_utilities.py")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define the replacement patterns for utility functions
    replacements = [
        # Context analyzer current_state usage
        (
            r'(\s+)current_state = context\.current_state_label',
            r'\1# ISSUE #273 FIX: Replace label dependency with content analysis\n\1current_state = context.current_state_from_content'
        ),
    ]
    
    # Apply replacements
    modified = False
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            modified = True
            print(f"Applied orchestration utilities replacement for: {pattern[:50]}...")
    
    # Write back the modified content
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Successfully updated {file_path}")
        return True
    else:
        print("No modifications needed in orchestration_utilities.py")
        return False

def main():
    """Main function to run all fixes"""
    print("Starting Issue #273 fix: Replace current_state_label dependencies with content analysis")
    print("=" * 80)
    
    success_count = 0
    
    # Fix enhanced_orchestration_intelligence.py
    if fix_enhanced_orchestration_intelligence():
        success_count += 1
    
    # Fix orchestration_utilities.py  
    if fix_orchestration_utilities():
        success_count += 1
    
    print("=" * 80)
    print(f"Fix completed. {success_count} files modified.")
    
    if success_count > 0:
        print("\n✅ CRITICAL FIX APPLIED: Label dependencies replaced with content analysis")
        print("   Enhanced orchestration intelligence now uses content-driven state determination")
        print("   This addresses the core issue in #273 line 668 and all related dependencies")
    else:
        print("\n⚠️  No files required modification - all dependencies may already be fixed")

if __name__ == "__main__":
    main()