#!/usr/bin/env python3
"""
Test script to validate the knowledge system is working correctly
"""

import duckdb
import json
from pathlib import Path

def test_knowledge_system():
    """Test the complete knowledge system"""
    
    db_path = "/Users/cal/DEV/RIF/knowledge/knowledge.duckdb"
    
    print("=" * 60)
    print("KNOWLEDGE SYSTEM VALIDATION TEST")
    print("=" * 60)
    
    # Connect to database
    conn = duckdb.connect(db_path, read_only=True)
    
    # Test 1: Check entity counts
    print("\n1. ENTITY COUNTS:")
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    patterns = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'pattern'").fetchone()[0]
    issues = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'issue'").fetchone()[0]
    decisions = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'decision'").fetchone()[0]
    tools = conn.execute("SELECT COUNT(*) FROM entities WHERE type = 'tool'").fetchone()[0]
    relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    
    print(f"  Total entities: {total:,}")
    print(f"  - Patterns: {patterns}")
    print(f"  - Issues: {issues}")
    print(f"  - Decisions: {decisions}")
    print(f"  - Claude Tools: {tools}")
    print(f"  Relationships: {relationships}")
    
    # Test 2: Claude Code documentation
    print("\n2. CLAUDE CODE DOCUMENTATION:")
    claude_tools = conn.execute("""
        SELECT name FROM entities WHERE type = 'tool' ORDER BY name
    """).fetchall()
    
    expected_tools = [
        'Claude Tool: Bash', 'Claude Tool: BashOutput', 'Claude Tool: Edit',
        'Claude Tool: ExitPlanMode', 'Claude Tool: Glob', 'Claude Tool: Grep',
        'Claude Tool: KillBash', 'Claude Tool: LS', 'Claude Tool: MultiEdit',
        'Claude Tool: NotebookEdit', 'Claude Tool: Read', 'Claude Tool: Task',
        'Claude Tool: TodoWrite', 'Claude Tool: WebFetch', 'Claude Tool: WebSearch',
        'Claude Tool: Write'
    ]
    
    actual_tools = [t[0] for t in claude_tools]
    missing = set(expected_tools) - set(actual_tools)
    
    if missing:
        print(f"  ❌ Missing tools: {missing}")
    else:
        print(f"  ✅ All {len(expected_tools)} Claude Code tools documented")
    
    # Test 3: Verify 500+ entities requirement
    print("\n3. ENTITY REQUIREMENT:")
    if total >= 500:
        print(f"  ✅ {total:,} entities (exceeds 500 requirement)")
    else:
        print(f"  ❌ {total:,} entities (below 500 requirement)")
    
    # Test 4: Test search functionality
    print("\n4. SEARCH FUNCTIONALITY:")
    
    # Search for patterns
    pattern_results = conn.execute("""
        SELECT name, description 
        FROM entities 
        WHERE type = 'pattern' AND 
              (name LIKE '%database%' OR description LIKE '%database%')
        LIMIT 3
    """).fetchall()
    
    print(f"  Database patterns found: {len(pattern_results)}")
    for name, desc in pattern_results:
        print(f"    • {name[:50]}")
    
    # Search for Claude documentation
    claude_results = conn.execute("""
        SELECT name, type, description
        FROM entities
        WHERE name LIKE '%Claude%' OR description LIKE '%Claude Code%'
        LIMIT 5
    """).fetchall()
    
    print(f"\n  Claude-related entities found: {len(claude_results)}")
    for name, entity_type, desc in claude_results:
        print(f"    • [{entity_type}] {name[:40]}")
    
    # Test 5: Relationship integrity
    print("\n5. RELATIONSHIP INTEGRITY:")
    orphan_rels = conn.execute("""
        SELECT COUNT(*) 
        FROM relationships r
        WHERE NOT EXISTS (SELECT 1 FROM entities WHERE id = r.source_id)
           OR NOT EXISTS (SELECT 1 FROM entities WHERE id = r.target_id)
    """).fetchone()[0]
    
    if orphan_rels == 0:
        print(f"  ✅ No orphaned relationships")
    else:
        print(f"  ❌ {orphan_rels} orphaned relationships found")
    
    # Test 6: Recent additions
    print("\n6. RECENT ADDITIONS:")
    recent = conn.execute("""
        SELECT type, name, created_at
        FROM entities
        WHERE created_at > CURRENT_TIMESTAMP - INTERVAL 1 HOUR
        ORDER BY created_at DESC
        LIMIT 5
    """).fetchall()
    
    print(f"  Recent entities (last hour): {len(recent)}")
    for entity_type, name, created in recent:
        print(f"    • [{entity_type}] {name[:40]} - {created}")
    
    # Overall status
    print("\n" + "=" * 60)
    print("OVERALL STATUS:")
    
    all_good = (
        total >= 500 and
        tools == 16 and
        len(missing) == 0 and
        orphan_rels == 0 and
        relationships > 0
    )
    
    if all_good:
        print("✅ KNOWLEDGE SYSTEM IS FULLY OPERATIONAL")
        print(f"   - {total:,} entities catalogued")
        print(f"   - All Claude Code documentation present")
        print(f"   - {relationships} relationships mapped")
        print(f"   - Database integrity verified")
    else:
        print("⚠️  KNOWLEDGE SYSTEM NEEDS ATTENTION")
        if total < 500:
            print(f"   - Only {total} entities (need 500+)")
        if tools != 16:
            print(f"   - Only {tools} Claude tools (need 16)")
        if len(missing) > 0:
            print(f"   - Missing tools: {missing}")
        if relationships == 0:
            print("   - No relationships found")
        if orphan_rels > 0:
            print(f"   - {orphan_rels} orphaned relationships")
    
    print("=" * 60)
    
    conn.close()

if __name__ == "__main__":
    test_knowledge_system()