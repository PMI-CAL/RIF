#!/usr/bin/env python3
"""
Test that all Claude Code facts returned by the MCP server are correct
"""

import json
import subprocess

def query_server(tool, arguments):
    """Query the MCP server using echo approach"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": arguments
        }
    }
    
    cmd = f'echo \'{json.dumps(request)}\' | python3 /Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py'
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd='/Users/cal/DEV/RIF'
    )
    
    stdout, stderr = process.communicate(timeout=15)
    
    try:
        response = json.loads(stdout.strip())
        if 'result' in response:
            return response['result']['content'][0]['text']
        else:
            return f"ERROR: {response}"
    except Exception as e:
        return f"FAILED: {stdout[:100]} | {str(e)}"

def test_claude_facts():
    """Test comprehensive Claude Code facts"""
    print("=" * 70)
    print("COMPREHENSIVE CLAUDE CODE FACT CHECKING")
    print("=" * 70)
    
    # Define fact tests with expected answers
    fact_tests = [
        {
            "fact": "Task.parallel() is pseudocode, not a real function",
            "query": ("check_compatibility", {"approach": "Using Task.parallel()"}),
            "expected_keywords": ["pseudocode", "incompatible", "not", "real"],
            "wrong_keywords": ["available", "use task.parallel"]
        },
        {
            "fact": "Agents cannot communicate directly or share memory",
            "query": ("check_compatibility", {"approach": "agents sharing memory"}),
            "expected_keywords": ["incompatible", "cannot", "memory"],
            "wrong_keywords": ["can share", "direct communication"]
        },
        {
            "fact": "No persistent background processes",
            "query": ("get_claude_documentation", {"topic": "persistent processes"}),
            "expected_keywords": ["no persistent", "background", "limitation"],
            "wrong_keywords": ["can run background", "persistent processes available"]
        },
        {
            "fact": "Session-scoped memory only",
            "query": ("get_claude_documentation", {"topic": "session memory"}),
            "expected_keywords": ["session", "limitation", "scoped"],
            "wrong_keywords": ["permanent memory", "persists across sessions"]
        },
        {
            "fact": "Claude Code IS the orchestrator, not orchestrated",
            "query": ("check_compatibility", {"approach": "external orchestration"}),
            "expected_keywords": ["claude code is", "orchestrator", "cannot be externally"],
            "wrong_keywords": ["can be orchestrated", "external system controls"]
        },
        {
            "fact": "MCP servers provide external integrations",
            "query": ("get_claude_documentation", {"topic": "MCP integration"}),
            "expected_keywords": ["mcp", "integration", "external"],
            "wrong_keywords": ["no external", "mcp not available"]
        },
        {
            "fact": "Hooks are event-triggered, not continuous",
            "query": ("get_claude_documentation", {"topic": "hooks automation"}),
            "expected_keywords": ["event", "triggered", "not continuous"],
            "wrong_keywords": ["continuous background", "always running"]
        },
        {
            "fact": "Subagents are contextual specialists, not independent processes",
            "query": ("get_claude_documentation", {"topic": "subagents"}),
            "expected_keywords": ["contextual", "specialist", "session"],
            "wrong_keywords": ["independent processes", "separate systems"]
        },
        {
            "fact": "run_in_background allows session-scoped background tasks",
            "query": ("check_compatibility", {"approach": "session background tasks"}),
            "expected_keywords": ["run_in_background", "session", "scoped"],
            "wrong_keywords": ["no background", "impossible"]
        },
        {
            "fact": "Multiple Task tools in one response enables parallel execution",
            "query": ("check_compatibility", {"approach": "parallel task execution"}),
            "expected_keywords": ["multiple", "task", "response", "parallel"],
            "wrong_keywords": ["task.parallel()", "single task only"]
        },
        {
            "fact": "Files and GitHub issues enable agent coordination",
            "query": ("check_compatibility", {"approach": "agent coordination"}),
            "expected_keywords": ["files", "github", "coordination"],
            "wrong_keywords": ["direct communication", "shared memory"]
        },
        {
            "fact": "Claude Code has file operation tools (Read, Write, Edit)",
            "query": ("get_claude_documentation", {"topic": "file operations"}),
            "expected_keywords": ["read", "write", "edit", "file"],
            "wrong_keywords": ["no file access", "cannot edit files"]
        },
        {
            "fact": "Claude Code has command execution via Bash tool",
            "query": ("get_claude_documentation", {"topic": "command execution"}),
            "expected_keywords": ["bash", "command", "execution"],
            "wrong_keywords": ["no command", "cannot execute"]
        },
        {
            "fact": "Claude Code has web access via WebSearch and WebFetch",
            "query": ("get_claude_documentation", {"topic": "web access"}),
            "expected_keywords": ["web", "search", "fetch"],
            "wrong_keywords": ["no web access", "offline only"]
        },
        {
            "fact": "Claude Code supports code analysis with Grep and Glob",
            "query": ("get_claude_documentation", {"topic": "code analysis"}),
            "expected_keywords": ["grep", "glob", "analysis", "search"],
            "wrong_keywords": ["no code analysis", "cannot search"]
        }
    ]
    
    correct_facts = 0
    total_facts = len(fact_tests)
    
    for i, test in enumerate(fact_tests, 1):
        print(f"\nFact Test {i}/{total_facts}: {test['fact']}")
        print("-" * 70)
        
        # Query the server
        tool, args = test['query']
        result = query_server(tool, args)
        
        # Check for expected keywords
        expected_found = 0
        for keyword in test['expected_keywords']:
            if keyword.lower() in result.lower():
                expected_found += 1
        
        # Check for wrong keywords (these should NOT appear)
        wrong_found = 0
        for keyword in test['wrong_keywords']:
            if keyword.lower() in result.lower():
                wrong_found += 1
        
        # Determine if fact is correct
        expected_ratio = expected_found / len(test['expected_keywords'])
        has_wrong = wrong_found > 0
        
        if expected_ratio >= 0.5 and not has_wrong:
            print(f"✅ CORRECT FACT ({expected_found}/{len(test['expected_keywords'])} expected keywords)")
            correct_facts += 1
        elif expected_ratio >= 0.5 and has_wrong:
            print(f"⚠️ MOSTLY CORRECT but contains {wrong_found} incorrect statements")
            correct_facts += 0.5
        else:
            print(f"❌ INCORRECT FACT ({expected_found}/{len(test['expected_keywords'])} expected keywords, {wrong_found} wrong)")
            
        # Show key evidence
        if expected_found > 0:
            found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in result.lower()]
            print(f"   Evidence: {', '.join(found_keywords[:3])}")
        
        if wrong_found > 0:
            wrong_keywords = [kw for kw in test['wrong_keywords'] if kw.lower() in result.lower()]
            print(f"   Wrong info: {', '.join(wrong_keywords[:2])}")
    
    print(f"\n{'='*70}")
    print("CLAUDE CODE FACT ACCURACY RESULTS")
    print(f"{'='*70}")
    
    accuracy = (correct_facts / total_facts) * 100
    print(f"\n📊 Results:")
    print(f"   Total facts tested: {total_facts}")
    print(f"   Correct facts: {correct_facts}")
    print(f"   Fact accuracy: {accuracy:.1f}%")
    
    if accuracy >= 95:
        print(f"\n✅ EXCELLENT: {accuracy:.1f}% fact accuracy - Claude facts are highly reliable")
        return True
    elif accuracy >= 90:
        print(f"\n✅ GOOD: {accuracy:.1f}% fact accuracy - Claude facts are mostly reliable")
        return True
    elif accuracy >= 80:
        print(f"\n⚠️ ACCEPTABLE: {accuracy:.1f}% fact accuracy - Some fact issues need fixing")
        return False
    else:
        print(f"\n❌ POOR: {accuracy:.1f}% fact accuracy - Major fact issues need addressing")
        return False

if __name__ == '__main__':
    import sys
    success = test_claude_facts()
    sys.exit(0 if success else 1)