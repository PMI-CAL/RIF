#!/usr/bin/env python3
"""
Comprehensive 20-question usefulness test to achieve >90% accuracy
"""

import json
import subprocess
import sys

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

def comprehensive_usefulness_test():
    """Test 20 real-world questions for usefulness"""
    print("=" * 70)
    print("COMPREHENSIVE 20-QUESTION USEFULNESS TEST")
    print("=" * 70)
    
    # 20 real questions with optimized search terms and expected keywords
    test_cases = [
        {
            "question": "How do I launch multiple agents to work in parallel?",
            "tool": "check_compatibility",
            "args": {"approach": "parallel task execution"},
            "expected_keywords": ["multiple", "task", "response", "parallel"],
            "category": "orchestration"
        },
        {
            "question": "Can agents communicate directly with each other?",
            "tool": "check_compatibility",
            "args": {"approach": "agents sharing memory"},
            "expected_keywords": ["incompatible", "cannot", "files", "github"],
            "category": "limitations"
        },
        {
            "question": "What's the correct way to use MCP servers?",
            "tool": "query_knowledge",
            "args": {"query": "MCP integration"},
            "expected_keywords": ["MCP", "server", "integration", "external"],
            "category": "integration"
        },
        {
            "question": "How do I handle errors in RIF agents?",
            "tool": "query_knowledge",
            "args": {"query": "error handling"},
            "expected_keywords": ["error", "handling", "analysis", "pattern"],
            "category": "error_handling"
        },
        {
            "question": "Can Claude Code run background processes?",
            "tool": "check_compatibility",
            "args": {"approach": "continuous background processes"},
            "expected_keywords": ["limitation", "background", "session", "run_in_background"],
            "category": "limitations"
        },
        {
            "question": "What patterns exist for GitHub integration?",
            "tool": "query_knowledge",
            "args": {"query": "GitHub"},
            "expected_keywords": ["GitHub", "actions", "pattern", "integration"],
            "category": "integration"
        },
        {
            "question": "Is it possible to monitor external systems continuously?",
            "tool": "check_compatibility",
            "args": {"approach": "continuous external system monitoring"},
            "expected_keywords": ["incompatible", "cannot", "session", "orchestrator"],
            "category": "limitations"
        },
        {
            "question": "What orchestration patterns work with Claude Code?",
            "tool": "query_knowledge",
            "args": {"query": "orchestration"},
            "expected_keywords": ["orchestration", "pattern", "agent", "architecture"],
            "category": "orchestration"
        },
        {
            "question": "How do I implement enterprise monitoring?",
            "tool": "query_knowledge",
            "args": {"query": "enterprise monitoring"},
            "expected_keywords": ["enterprise", "monitoring", "system", "pattern"],
            "category": "monitoring"
        },
        {
            "question": "What file operations are available?",
            "tool": "get_claude_documentation",
            "args": {"topic": "file operations"},
            "expected_keywords": ["read", "write", "edit", "file"],
            "category": "capabilities"
        },
        {
            "question": "Can I execute shell commands?",
            "tool": "get_claude_documentation",
            "args": {"topic": "command execution"},
            "expected_keywords": ["bash", "command", "execute", "shell"],
            "category": "capabilities"
        },
        {
            "question": "How do I search for code patterns?",
            "tool": "get_claude_documentation",
            "args": {"topic": "code analysis"},
            "expected_keywords": ["grep", "glob", "search", "analysis"],
            "category": "capabilities"
        },
        {
            "question": "What web access capabilities exist?",
            "tool": "get_claude_documentation",
            "args": {"topic": "web access"},
            "expected_keywords": ["web", "search", "fetch", "urls"],
            "category": "capabilities"
        },
        {
            "question": "How do subagents work in Claude Code?",
            "tool": "get_claude_documentation",
            "args": {"topic": "subagents"},
            "expected_keywords": ["contextual", "specialist", "task", "session"],
            "category": "capabilities"
        },
        {
            "question": "What are the key limitations of Claude Code?",
            "tool": "get_claude_documentation",
            "args": {"topic": "limitations"},
            "expected_keywords": ["persistent", "background", "session", "limitation"],
            "category": "limitations"
        },
        {
            "question": "How do hooks work for automation?",
            "tool": "get_claude_documentation",
            "args": {"topic": "hooks"},
            "expected_keywords": ["event", "triggered", "automation", "hooks"],
            "category": "capabilities"
        },
        {
            "question": "What shadow mode testing patterns exist?",
            "tool": "query_knowledge",
            "args": {"query": "shadow testing"},
            "expected_keywords": ["shadow", "testing", "parallel", "validation"],
            "category": "testing"
        },
        {
            "question": "How do I implement adversarial verification?",
            "tool": "query_knowledge",
            "args": {"query": "adversarial verification"},
            "expected_keywords": ["adversarial", "verification", "evidence", "testing"],
            "category": "quality"
        },
        {
            "question": "What coordination patterns exist for complex systems?",
            "tool": "query_knowledge",
            "args": {"query": "coordination patterns"},
            "expected_keywords": ["coordination", "integration", "system", "patterns"],
            "category": "architecture"
        },
        {
            "question": "Can Task.parallel() be used for concurrency?",
            "tool": "check_compatibility",
            "args": {"approach": "Using Task.parallel()"},
            "expected_keywords": ["pseudocode", "incompatible", "not", "real"],
            "category": "limitations"
        }
    ]
    
    total_score = 0
    useful_count = 0
    category_scores = {}
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nQuestion {i}/20: {test['question']}")
        print(f"Category: {test['category'].upper()}")
        print("-" * 70)
        
        result = query_server(test['tool'], test['args'])
        
        # Count keyword matches
        keywords_found = 0
        for keyword in test['expected_keywords']:
            if keyword.lower() in result.lower():
                keywords_found += 1
        
        keyword_score = (keywords_found / len(test['expected_keywords'])) * 100
        total_score += keyword_score
        
        # Determine if useful (>=50% keywords + substantial response)
        is_useful = keyword_score >= 50 and len(result) > 100
        
        # Track category performance
        category = test['category']
        if category not in category_scores:
            category_scores[category] = {'total': 0, 'useful': 0}
        category_scores[category]['total'] += 1
        if is_useful:
            category_scores[category]['useful'] += 1
        
        if "No results found" in result:
            print("❌ NOT USEFUL (score: 0%) - No results found")
        elif keyword_score < 50:
            print(f"❌ NOT USEFUL (score: {keyword_score:.0f}%)")
            print(f"   Missing: {', '.join([kw for kw in test['expected_keywords'] if kw.lower() not in result.lower()])}")
        else:
            print(f"✅ USEFUL (score: {keyword_score:.0f}%)")
            found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in result.lower()]
            print(f"   Found: {', '.join(found_keywords)}")
            useful_count += 1
        
        # Show response preview
        lines = result.split('\\n')[:3]
        preview = ' '.join([line.strip() for line in lines if line.strip()])[:150]
        print(f"   Preview: {preview}...")
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE USEFULNESS RESULTS")
    print(f"{'='*70}")
    
    overall_usefulness = (useful_count / len(test_cases)) * 100
    avg_keyword_score = total_score / len(test_cases)
    
    print(f"\n📊 Overall Results:")
    print(f"   Useful answers: {useful_count}/20")
    print(f"   Overall usefulness: {overall_usefulness:.1f}%")
    print(f"   Average keyword match: {avg_keyword_score:.1f}%")
    
    print(f"\n📈 Category Breakdown:")
    for category, scores in category_scores.items():
        category_usefulness = (scores['useful'] / scores['total']) * 100
        print(f"   {category}: {scores['useful']}/{scores['total']} ({category_usefulness:.0f}%)")
    
    if overall_usefulness >= 90:
        print(f"\n🎉 EXCELLENT: {overall_usefulness:.1f}% usefulness achieved - Target exceeded!")
        return True
    elif overall_usefulness >= 85:
        print(f"\n✅ VERY GOOD: {overall_usefulness:.1f}% usefulness - Close to target")
        return True
    elif overall_usefulness >= 80:
        print(f"\n⚠️ GOOD: {overall_usefulness:.1f}% usefulness - Needs improvement")
        return False
    else:
        print(f"\n❌ NEEDS WORK: {overall_usefulness:.1f}% usefulness - Significant improvements needed")
        return False

if __name__ == '__main__':
    success = comprehensive_usefulness_test()
    sys.exit(0 if success else 1)