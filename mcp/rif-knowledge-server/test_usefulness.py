#!/usr/bin/env python3
"""
Test if the MCP server actually provides USEFUL answers to real questions
"""

import json
import subprocess
import sys

def query_server(tool, arguments):
    """Query the MCP server"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": arguments
        }
    }
    
    process = subprocess.Popen(
        ["python3", "/Users/cal/DEV/RIF/mcp/rif-knowledge-server/rif_knowledge_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=json.dumps(request), timeout=5)
    
    try:
        response = json.loads(stdout.strip())
        if 'result' in response:
            return response['result']['content'][0]['text']
        else:
            return f"ERROR: {response}"
    except:
        return f"FAILED: {stdout[:100]}"

def main():
    print("=" * 70)
    print("TESTING REAL-WORLD USEFULNESS OF RIF KNOWLEDGE MCP SERVER")
    print("=" * 70)
    
    # Real questions someone working with Claude Code would ask
    test_cases = [
        {
            "question": "How do I launch multiple agents to work in parallel?",
            "tool": "get_claude_documentation",
            "args": {"topic": "Task tool parallel execution"},
            "expected_keywords": ["Task", "parallel", "multiple", "response"],
            "should_find": True
        },
        {
            "question": "Can agents communicate directly with each other?",
            "tool": "check_compatibility",
            "args": {"approach": "agents communicating directly via shared memory"},
            "expected_keywords": ["INCOMPATIBLE", "cannot", "files", "GitHub"],
            "should_find": True
        },
        {
            "question": "What's the correct way to use MCP servers?",
            "tool": "query_knowledge",
            "args": {"query": "MCP server configuration", "entity_types": ["pattern"]},
            "expected_keywords": ["MCP", "claude mcp add", "configuration"],
            "should_find": True
        },
        {
            "question": "How do I handle errors in RIF agents?",
            "tool": "query_knowledge",
            "args": {"query": "error handling agent", "entity_types": ["pattern", "decision"]},
            "expected_keywords": ["error", "handling", "pattern"],
            "should_find": True
        },
        {
            "question": "Can Claude Code run background processes?",
            "tool": "get_claude_documentation",
            "args": {"topic": "background processes limitations"},
            "expected_keywords": ["limitation", "background", "session", "run_in_background"],
            "should_find": True
        },
        {
            "question": "What patterns exist for GitHub integration?",
            "tool": "query_knowledge",
            "args": {"query": "GitHub gh CLI", "entity_types": ["pattern"]},
            "expected_keywords": ["GitHub", "gh", "pattern"],
            "should_find": True
        },
        {
            "question": "Is it possible to monitor external systems continuously?",
            "tool": "check_compatibility",
            "args": {"approach": "continuous external system monitoring"},
            "expected_keywords": ["INCOMPATIBLE", "cannot", "session"],
            "should_find": True
        },
        {
            "question": "What orchestration patterns work with Claude Code?",
            "tool": "query_knowledge",
            "args": {"query": "orchestration Claude Code", "entity_types": ["pattern", "decision"]},
            "expected_keywords": ["orchestrat", "pattern", "Claude"],
            "should_find": True
        }
    ]
    
    useful_count = 0
    total_score = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {test['question']}")
        print("-" * 70)
        
        result = query_server(test['tool'], test['args'])
        
        # Check if response contains expected keywords
        found_keywords = []
        missing_keywords = []
        
        for keyword in test['expected_keywords']:
            if keyword.lower() in result.lower():
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Calculate usefulness score
        score = len(found_keywords) / len(test['expected_keywords'])
        total_score += score
        
        # Check if result is actually useful
        is_useful = (
            len(result) > 100 and  # Not too short
            "No results found" not in result and  # Actually found something
            score >= 0.5  # At least half the keywords present
        )
        
        if is_useful:
            useful_count += 1
            print(f"✅ USEFUL ANSWER (score: {score:.0%})")
            print(f"   Found keywords: {', '.join(found_keywords)}")
            if missing_keywords:
                print(f"   Missing: {', '.join(missing_keywords)}")
        else:
            print(f"❌ NOT USEFUL (score: {score:.0%})")
            if missing_keywords:
                print(f"   Missing keywords: {', '.join(missing_keywords)}")
            if "No results" in result:
                print("   Problem: No results found")
            elif len(result) < 100:
                print(f"   Problem: Response too short ({len(result)} chars)")
        
        # Show preview of response
        print(f"\nResponse preview:")
        lines = result.split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"   {line[:100]}")
    
    print("\n" + "=" * 70)
    print("FINAL USEFULNESS ASSESSMENT")
    print("=" * 70)
    
    print(f"\n📊 Results:")
    print(f"   Useful answers: {useful_count}/{len(test_cases)}")
    print(f"   Average keyword match: {(total_score/len(test_cases)):.0%}")
    print(f"   Overall usefulness: {(useful_count/len(test_cases)):.0%}")
    
    if useful_count >= 6:
        print("\n✅ VERDICT: Server provides USEFUL knowledge")
    elif useful_count >= 4:
        print("\n⚠️ VERDICT: Server is PARTIALLY useful but needs improvement")
    else:
        print("\n❌ VERDICT: Server is NOT USEFUL - provides poor quality answers")
    
    # Test specific Claude Code facts
    print("\n" + "=" * 70)
    print("FACT CHECKING")
    print("=" * 70)
    
    facts_correct = 0
    
    # Check Task.parallel() fact
    result = query_server("check_compatibility", {"approach": "Using Task.parallel()"})
    if "pseudocode" in result.lower() or "incompatible" in result.upper():
        print("✅ Correctly identifies Task.parallel() as pseudocode")
        facts_correct += 1
    else:
        print("❌ WRONG about Task.parallel()")
    
    # Check agent communication fact
    result = query_server("check_compatibility", {"approach": "agents sharing memory"})
    if "incompatible" in result.upper() or "cannot" in result.lower():
        print("✅ Correctly identifies agents can't share memory")
        facts_correct += 1
    else:
        print("❌ WRONG about agent memory sharing")
    
    # Check persistence fact
    result = query_server("get_claude_documentation", {"topic": "persistent processes"})
    if "limitation" in result.lower() or "cannot" in result.lower():
        print("✅ Correctly identifies persistence limitations")
        facts_correct += 1
    else:
        print("❌ WRONG about persistence")
    
    print(f"\nFact accuracy: {facts_correct}/3")
    
    if facts_correct < 2:
        print("❌ CRITICAL: Server provides INCORRECT information!")
    
    return 0 if useful_count >= 4 and facts_correct >= 2 else 1

if __name__ == "__main__":
    sys.exit(main())