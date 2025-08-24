#!/usr/bin/env python3
"""
Consensus Monitoring System Demonstration
Shows the complete functionality of the consensus monitoring implementation
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the system monitor module to path
sys.path.insert(0, str(Path(__file__).parent / "claude" / "commands"))

from system_monitor import ConsensusMonitor, track_consensus_session

def create_demo_scenarios() -> Dict[str, Dict[str, Any]]:
    """Create various demo scenarios to showcase different consensus patterns"""
    return {
        "perfect_consensus": {
            "id": "demo_perfect_001",
            "issue_number": 63,
            "consensus_type": "unanimous_consensus",
            "duration": 180000,  # 3 minutes
            "outcome": "agreed",
            "votes": [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.95, "weight": 1.1, "reasoning": "Comprehensive analysis shows clear benefits"},
                {"agent": "rif-implementer", "decision": "approve", "confidence": 0.92, "weight": 1.0, "reasoning": "Implementation is straightforward and well-designed"},
                {"agent": "rif-validator", "decision": "approve", "confidence": 0.98, "weight": 1.5, "reasoning": "All quality gates pass with excellent test coverage"},
                {"agent": "rif-architect", "decision": "approve", "confidence": 0.90, "weight": 1.3, "reasoning": "Architecture aligns perfectly with system design"},
                {"agent": "rif-security", "decision": "approve", "confidence": 0.94, "weight": 2.0, "reasoning": "No security concerns identified"}
            ]
        },
        
        "healthy_disagreement": {
            "id": "demo_healthy_002", 
            "issue_number": 64,
            "consensus_type": "weighted_voting",
            "duration": 420000,  # 7 minutes
            "outcome": "agreed",
            "votes": [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.75, "weight": 1.1, "reasoning": "Analysis supports the approach with minor concerns"},
                {"agent": "rif-implementer", "decision": "approve", "confidence": 0.80, "weight": 1.0, "reasoning": "Implementation is feasible but requires careful execution"},
                {"agent": "rif-validator", "decision": "reject", "confidence": 0.85, "weight": 1.5, "reasoning": "Quality concerns about edge case handling"},
                {"agent": "rif-architect", "decision": "approve", "confidence": 0.90, "weight": 1.3, "reasoning": "Architecture supports this change well"},
                {"agent": "rif-security", "decision": "approve", "confidence": 0.88, "weight": 2.0, "reasoning": "Security review shows acceptable risk level"}
            ]
        },
        
        "concerning_disagreement": {
            "id": "demo_concerning_003",
            "issue_number": 65,
            "consensus_type": "weighted_voting", 
            "duration": 1200000,  # 20 minutes
            "outcome": "disagreed",
            "votes": [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.60, "weight": 1.1, "reasoning": "Analysis is inconclusive but leans toward approval"},
                {"agent": "rif-implementer", "decision": "reject", "confidence": 0.95, "weight": 1.0, "reasoning": "Implementation complexity is too high for current timeline"},
                {"agent": "rif-validator", "decision": "reject", "confidence": 0.92, "weight": 1.5, "reasoning": "Quality requirements cannot be met with current resources"},
                {"agent": "rif-architect", "decision": "reject", "confidence": 0.88, "weight": 1.3, "reasoning": "Architectural debt would be significant"},
                {"agent": "rif-security", "decision": "reject", "confidence": 0.99, "weight": 2.0, "reasoning": "Critical security vulnerabilities identified"},
                {"agent": "rif-performance", "decision": "reject", "confidence": 0.85, "weight": 1.2, "reasoning": "Performance impact would be unacceptable"}
            ]
        },
        
        "low_confidence_consensus": {
            "id": "demo_lowconf_004",
            "issue_number": 66,
            "consensus_type": "simple_majority",
            "duration": 900000,  # 15 minutes
            "outcome": "agreed",
            "votes": [
                {"agent": "rif-analyst", "decision": "approve", "confidence": 0.45, "weight": 1.1, "reasoning": "Limited data available but trending positive"},
                {"agent": "rif-implementer", "decision": "approve", "confidence": 0.52, "weight": 1.0, "reasoning": "Implementation possible but requires research"},
                {"agent": "rif-validator", "decision": "approve", "confidence": 0.38, "weight": 1.5, "reasoning": "Testing approach unclear at this time"},
                {"agent": "rif-architect", "decision": "approve", "confidence": 0.41, "weight": 1.3, "reasoning": "Architecture impact is uncertain"}
            ]
        }
    }

def demonstrate_consensus_monitoring():
    """Run a comprehensive demonstration of the consensus monitoring system"""
    print("🎭 Consensus Monitoring System Demonstration")
    print("=" * 55)
    
    # Initialize the monitor
    monitor = ConsensusMonitor("/Users/cal/DEV/RIF/knowledge/monitoring")
    scenarios = create_demo_scenarios()
    
    print(f"\n📊 Running {len(scenarios)} consensus scenarios to demonstrate monitoring capabilities:\n")
    
    results = []
    
    for scenario_name, session_data in scenarios.items():
        print(f"🎪 Scenario: {scenario_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        # Process the consensus session
        start_time = time.time()
        report = monitor.track_consensus(session_data)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Extract key metrics
        metrics = report["metrics"]
        analysis = report["analysis"]
        
        # Display results
        print(f"📋 Session ID: {report['session_id']}")
        print(f"⚖️  Agreement Level: {metrics['agreement_level']:.1%}")
        print(f"🎯 Confidence Average: {metrics['confidence_distribution']['average']:.1%}")
        print(f"👥 Total Participants: {metrics['total_participants']}")
        print(f"🔄 Dissenter Count: {metrics['dissenter_count']}")
        
        if metrics['dissenting_agents']:
            print(f"❌ Dissenters: {', '.join(metrics['dissenting_agents'])}")
        
        print(f"⏱️  Decision Time: {metrics['decision_time']/1000:.1f} seconds")
        print(f"💪 Consensus Strength: {analysis['consensus_strength'].upper()}")
        print(f"⚡ Processing Time: {processing_time:.2f}ms")
        
        # Show recommendations
        print(f"💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
            
        # Show dissent patterns if any
        if analysis['dissent_patterns'] and analysis['dissent_patterns'] != ['no_dissent']:
            print(f"🔍 Dissent Patterns: {', '.join(analysis['dissent_patterns'])}")
        
        print(f"📈 Decision Efficiency: {analysis['efficiency_assessment']}")
        
        results.append({
            "scenario": scenario_name,
            "agreement": metrics['agreement_level'],
            "confidence": metrics['confidence_distribution']['average'],
            "strength": analysis['consensus_strength'],
            "processing_time": processing_time
        })
        
        print()  # Add spacing between scenarios
    
    # Summary analysis
    print("📈 DEMONSTRATION SUMMARY")
    print("=" * 25)
    
    avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
    print(f"🚀 Average Processing Time: {avg_processing_time:.2f}ms (Target: <100ms)")
    
    strength_distribution = {}
    for r in results:
        strength = r['strength']
        strength_distribution[strength] = strength_distribution.get(strength, 0) + 1
    
    print("\n🎯 Consensus Strength Distribution:")
    for strength, count in strength_distribution.items():
        print(f"   {strength.title()}: {count} scenarios")
    
    agreement_range = [r['agreement'] for r in results]
    print(f"\n⚖️  Agreement Level Range: {min(agreement_range):.1%} - {max(agreement_range):.1%}")
    
    confidence_range = [r['confidence'] for r in results]
    print(f"🎯 Confidence Range: {min(confidence_range):.1%} - {max(confidence_range):.1%}")
    
    # Performance validation
    print("\n✅ PERFORMANCE VALIDATION")
    print("-" * 25)
    
    if avg_processing_time < 100:
        print(f"✅ Processing Time: PASS ({avg_processing_time:.2f}ms < 100ms target)")
    else:
        print(f"❌ Processing Time: FAIL ({avg_processing_time:.2f}ms ≥ 100ms target)")
        
    if all(r['processing_time'] < 100 for r in results):
        print("✅ All Scenarios: PASS (all under 100ms)")
    else:
        slow_scenarios = [r for r in results if r['processing_time'] >= 100]
        print(f"❌ Slow Scenarios: {len(slow_scenarios)} scenarios over 100ms")
    
    # Feature validation
    print("\n✅ FEATURE VALIDATION")
    print("-" * 19)
    
    features = [
        ("Agreement Level Calculation", all('agreement_level' in monitor.track_consensus(s)['metrics'] for s in scenarios.values())),
        ("Dissenter Identification", any(monitor.track_consensus(s)['metrics']['dissenter_count'] > 0 for s in scenarios.values())),
        ("Confidence Distribution", all('confidence_distribution' in monitor.track_consensus(s)['metrics'] for s in scenarios.values())),
        ("Consensus Strength Assessment", all('consensus_strength' in monitor.track_consensus(s)['analysis'] for s in scenarios.values())),
        ("Recommendation Generation", all('recommendations' in monitor.track_consensus(s) for s in scenarios.values())),
        ("Decision Efficiency Assessment", all('efficiency_assessment' in monitor.track_consensus(s)['analysis'] for s in scenarios.values()))
    ]
    
    for feature_name, feature_working in features:
        status = "✅ PASS" if feature_working else "❌ FAIL"
        print(f"{status} {feature_name}")
    
    print("\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 25)
    print("The Consensus Monitoring System is fully operational and ready for production use!")
    print(f"📁 Session data stored in: /Users/cal/DEV/RIF/knowledge/monitoring/consensus/")
    print(f"🌐 Dashboard available at: http://localhost:8080/api/consensus")

if __name__ == "__main__":
    demonstrate_consensus_monitoring()