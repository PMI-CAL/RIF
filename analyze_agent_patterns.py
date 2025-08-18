#!/usr/bin/env python3
"""
Analyze RIF agent definitions for patterns that might create .md files instead of using LightRAG.
This script is part of Phase 3 implementation for GitHub issue #10.
"""

import os
import re
from pathlib import Path


def analyze_agent_file(file_path: str) -> dict:
    """Analyze a single agent file for learning/knowledge patterns."""
    analysis = {
        "file": file_path,
        "issues": [],
        "learning_patterns": [],
        "knowledge_patterns": [],
        "good_patterns": []
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for patterns that might indicate .md file creation
        md_creation_patterns = [
            r'Write.*\.md',
            r'create.*\.md',
            r'save.*\.md', 
            r'knowledge.*\.md',
            r'learning.*\.md',
            r'document.*\.md'
        ]
        
        for pattern in md_creation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                analysis["issues"].extend(matches)
        
        # Check for learning/knowledge documentation patterns
        learning_patterns = [
            r'learning',
            r'knowledge.*capture',
            r'document.*learning',
            r'store.*learning',
            r'save.*pattern',
            r'record.*decision'
        ]
        
        for pattern in learning_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                analysis["learning_patterns"].extend(matches)
        
        # Check for proper LightRAG usage
        lightrag_patterns = [
            r'LightRAG',
            r'store_pattern',
            r'store_decision',
            r'vector.*database',
            r'chromadb',
            r'knowledge.*base'
        ]
        
        for pattern in lightrag_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                analysis["good_patterns"].extend(matches)
                
    except Exception as e:
        analysis["issues"].append(f"Error reading file: {e}")
    
    return analysis


def analyze_all_rif_agents(agents_dir: str) -> dict:
    """Analyze all RIF agent files for learning patterns."""
    results = {
        "total_agents": 0,
        "agents_with_issues": 0,
        "agents_with_learning": 0,
        "agents_with_lightrag": 0,
        "detailed_analysis": []
    }
    
    agents_path = Path(agents_dir)
    rif_agent_files = list(agents_path.glob("rif-*.md"))
    
    results["total_agents"] = len(rif_agent_files)
    
    for agent_file in rif_agent_files:
        analysis = analyze_agent_file(str(agent_file))
        results["detailed_analysis"].append(analysis)
        
        if analysis["issues"]:
            results["agents_with_issues"] += 1
        
        if analysis["learning_patterns"]:
            results["agents_with_learning"] += 1
            
        if analysis["good_patterns"]:
            results["agents_with_lightrag"] += 1
    
    return results


def generate_recommendations(results: dict) -> list:
    """Generate recommendations for updating agents."""
    recommendations = []
    
    for analysis in results["detailed_analysis"]:
        agent_name = Path(analysis["file"]).stem
        
        if analysis["issues"]:
            recommendations.append({
                "agent": agent_name,
                "priority": "high", 
                "action": "Fix file creation patterns",
                "details": f"Found potential .md file creation: {analysis['issues']}"
            })
        
        if analysis["learning_patterns"] and not analysis["good_patterns"]:
            recommendations.append({
                "agent": agent_name,
                "priority": "medium",
                "action": "Add LightRAG integration",
                "details": f"Has learning patterns but no LightRAG usage: {analysis['learning_patterns'][:3]}"
            })
        
        if not analysis["learning_patterns"] and not analysis["good_patterns"]:
            recommendations.append({
                "agent": agent_name,
                "priority": "low",
                "action": "Consider learning integration",
                "details": "No learning or LightRAG patterns found"
            })
    
    return recommendations


def main():
    """Main analysis function."""
    print("🔍 Analyzing RIF Agent Definitions for Learning Patterns")
    print("=" * 60)
    
    agents_dir = os.path.join(os.path.dirname(__file__), "claude", "agents")
    
    if not os.path.exists(agents_dir):
        print(f"❌ Agents directory not found: {agents_dir}")
        return 1
    
    # Analyze all RIF agents
    results = analyze_all_rif_agents(agents_dir)
    
    # Print summary
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📄 Total RIF agents analyzed: {results['total_agents']}")
    print(f"⚠️  Agents with potential issues: {results['agents_with_issues']}")
    print(f"🧠 Agents with learning patterns: {results['agents_with_learning']}")
    print(f"✅ Agents with LightRAG usage: {results['agents_with_lightrag']}")
    
    # Print detailed analysis
    print("\n📋 DETAILED ANALYSIS")
    print("=" * 60)
    
    for analysis in results["detailed_analysis"]:
        agent_name = Path(analysis["file"]).stem
        print(f"\n📄 {agent_name.upper()}")
        
        if analysis["issues"]:
            print(f"  ⚠️  Issues: {len(analysis['issues'])}")
            for issue in analysis["issues"][:3]:  # Show first 3
                print(f"     • {issue}")
        else:
            print("  ✅ No file creation issues found")
        
        if analysis["learning_patterns"]:
            print(f"  🧠 Learning patterns: {len(analysis['learning_patterns'])}")
        
        if analysis["good_patterns"]:
            print(f"  ✅ LightRAG patterns: {len(analysis['good_patterns'])}")
    
    # Generate and print recommendations
    recommendations = generate_recommendations(results)
    
    if recommendations:
        print("\n🔧 RECOMMENDATIONS")
        print("=" * 60)
        
        priority_order = ["high", "medium", "low"]
        for priority in priority_order:
            priority_recs = [r for r in recommendations if r["priority"] == priority]
            if priority_recs:
                print(f"\n🚨 {priority.upper()} PRIORITY:")
                for rec in priority_recs:
                    print(f"  📄 {rec['agent']}: {rec['action']}")
                    print(f"     → {rec['details']}")
    else:
        print("\n✅ No recommendations needed - all agents properly configured!")
    
    # Return status
    if results["agents_with_issues"] > 0:
        print(f"\n⚠️  Analysis complete with {results['agents_with_issues']} agents needing updates")
        return 1
    else:
        print("\n✅ Analysis complete - all agents properly configured!")
        return 0


if __name__ == "__main__":
    exit(main())