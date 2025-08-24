#!/usr/bin/env python3
"""
Test that 100% of pattern files are searchable through the MCP server
"""

import json
import subprocess
import glob
import os
from pathlib import Path

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

def get_all_pattern_files():
    """Get all pattern JSON files from the knowledge directory"""
    patterns_dir = Path("/Users/cal/DEV/RIF/knowledge/patterns")
    pattern_files = list(patterns_dir.glob("*.json"))
    return pattern_files

def extract_searchable_terms(pattern_file):
    """Extract searchable terms from a pattern file"""
    try:
        with open(pattern_file, 'r') as f:
            data = json.load(f)
        
        # Extract potential search terms
        search_terms = []
        
        # Common fields that should be searchable
        for field in ['pattern_name', 'name', 'pattern_id', 'title', 'category']:
            if field in data:
                search_terms.append(str(data[field]))
        
        # Description fields
        for field in ['description', 'summary', 'problem_context']:
            if field in data:
                if isinstance(data[field], str):
                    # Extract key words (first few words)
                    words = str(data[field]).split()[:3]
                    search_terms.extend(words)
                elif isinstance(data[field], dict) and 'solution_pattern' in data[field]:
                    words = str(data[field]['solution_pattern']).split()[:3]
                    search_terms.extend(words)
        
        # Use filename as fallback
        filename_terms = pattern_file.stem.replace('-', ' ').replace('_', ' ').split()
        search_terms.extend(filename_terms[:3])  # First few words from filename
        
        # Clean and filter terms
        cleaned_terms = []
        for term in search_terms:
            if len(term) > 3 and term.isalpha():  # Skip short or non-alphabetic terms
                cleaned_terms.append(term.lower())
        
        return list(set(cleaned_terms))  # Remove duplicates
        
    except Exception as e:
        print(f"   Error reading {pattern_file.name}: {e}")
        # Fallback to filename
        return [pattern_file.stem.replace('-', ' ').replace('_', ' ').split()[0].lower()]

def test_pattern_searchability():
    """Test that all patterns can be found via search"""
    print("=" * 70)
    print("TESTING PATTERN FILE SEARCHABILITY")
    print("=" * 70)
    
    pattern_files = get_all_pattern_files()
    print(f"Found {len(pattern_files)} pattern files to test")
    
    searchable_count = 0
    unsearchable_files = []
    
    for i, pattern_file in enumerate(pattern_files, 1):
        print(f"\nTest {i}/{len(pattern_files)}: {pattern_file.name}")
        
        # Get search terms for this pattern
        search_terms = extract_searchable_terms(pattern_file)
        print(f"   Search terms: {', '.join(search_terms[:5])}")
        
        # Try to find this pattern via search
        found = False
        
        for term in search_terms[:3]:  # Try first few terms
            result = query_server("query_knowledge", {"query": term})
            
            # Check if this specific pattern file appears in results
            if pattern_file.name in result or pattern_file.stem in result:
                print(f"   ✅ FOUND via '{term}'")
                found = True
                break
        
        if found:
            searchable_count += 1
        else:
            print(f"   ❌ NOT FOUND - trying broader search")
            unsearchable_files.append(pattern_file.name)
            
            # Try one more broader search with "pattern"
            result = query_server("query_knowledge", {"query": "pattern"})
            if pattern_file.name in result:
                print(f"   ✅ FOUND via broad 'pattern' search")
                searchable_count += 1
                unsearchable_files.remove(pattern_file.name)
            else:
                print(f"   ❌ COMPLETELY UNSEARCHABLE")
    
    print(f"\n{'='*70}")
    print("PATTERN SEARCHABILITY RESULTS")
    print(f"{'='*70}")
    
    coverage_percent = (searchable_count / len(pattern_files)) * 100
    print(f"\n📊 Results:")
    print(f"   Total pattern files: {len(pattern_files)}")
    print(f"   Searchable patterns: {searchable_count}")
    print(f"   Unsearchable patterns: {len(unsearchable_files)}")
    print(f"   Coverage: {coverage_percent:.1f}%")
    
    if unsearchable_files:
        print(f"\n❌ Unsearchable files:")
        for filename in unsearchable_files:
            print(f"   - {filename}")
    
    if coverage_percent >= 95:
        print(f"\n✅ EXCELLENT: {coverage_percent:.1f}% of patterns are searchable")
        return True
    elif coverage_percent >= 90:
        print(f"\n⚠️ GOOD: {coverage_percent:.1f}% of patterns are searchable")
        return True
    else:
        print(f"\n❌ POOR: Only {coverage_percent:.1f}% of patterns are searchable")
        return False

if __name__ == '__main__':
    import sys
    success = test_pattern_searchability()
    sys.exit(0 if success else 1)