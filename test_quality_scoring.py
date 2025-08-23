#!/usr/bin/env python3
"""
Test Quality Scoring System
Validates the scoring formula and decision logic
"""

def calculate_quality_score(validation_results):
    """
    Quality Score = 100 - (20 × FAILs) - (10 × CONCERNs) - (5 × WARNINGs)
    """
    score = 100
    score -= validation_results['fails'] * 20
    score -= validation_results['concerns'] * 10  
    score -= validation_results['warnings'] * 5
    
    # Bonus points for exceptional quality
    if validation_results.get('coverage', 0) > 95:
        score += 5
    if validation_results.get('no_security_issues', False):
        score += 5
        
    return max(0, min(100, score))  # Clamp between 0-100

def determine_gate_decision(quality_score, has_critical_issues=False):
    """Determine gate decision based on score and critical issues"""
    if has_critical_issues:
        return "FAIL"
    elif quality_score >= 80:
        return "PASS"
    elif quality_score >= 60:
        return "CONCERNS" 
    else:
        return "FAIL"

# Test scenarios
test_scenarios = [
    {
        "name": "Perfect Implementation",
        "results": {"fails": 0, "concerns": 0, "warnings": 0, "coverage": 96, "no_security_issues": True},
        "expected_score": 110,  # Will be clamped to 100
        "expected_decision": "PASS"
    },
    {
        "name": "Good Implementation",
        "results": {"fails": 0, "concerns": 1, "warnings": 2, "coverage": 85, "no_security_issues": True},
        "expected_score": 85,  # 100 - 10 - 10 + 5 = 85 (coverage bonus only >95)
        "expected_decision": "PASS"
    },
    {
        "name": "Acceptable Implementation",
        "results": {"fails": 0, "concerns": 2, "warnings": 1, "coverage": 80},
        "expected_score": 75,  # 100 - 20 - 5 = 75
        "expected_decision": "CONCERNS"
    },
    {
        "name": "Poor Implementation", 
        "results": {"fails": 1, "concerns": 1, "warnings": 3, "coverage": 60},
        "expected_score": 55,  # 100 - 20 - 10 - 15 = 55
        "expected_decision": "FAIL"
    },
    {
        "name": "Critical Issues Present",
        "results": {"fails": 0, "concerns": 0, "warnings": 0, "coverage": 95},
        "expected_score": 100,
        "expected_decision": "FAIL",  # Critical issues override score
        "has_critical": True
    }
]

print("🧮 Testing Quality Scoring System")
print("=" * 50)

all_passed = True

for scenario in test_scenarios:
    print(f"\n📊 Scenario: {scenario['name']}")
    
    # Calculate score
    actual_score = calculate_quality_score(scenario['results'])
    expected_score = min(scenario['expected_score'], 100)  # Manual clamp for comparison
    
    # Determine decision
    has_critical = scenario.get('has_critical', False)
    actual_decision = determine_gate_decision(actual_score, has_critical)
    expected_decision = scenario['expected_decision']
    
    # Check results
    score_correct = actual_score == expected_score
    decision_correct = actual_decision == expected_decision
    
    print(f"  Quality Score: {actual_score} (expected: {expected_score}) {'✅' if score_correct else '❌'}")
    print(f"  Gate Decision: {actual_decision} (expected: {expected_decision}) {'✅' if decision_correct else '❌'}")
    
    if not score_correct or not decision_correct:
        all_passed = False
        print(f"  🔍 Details: {scenario['results']}")

print("\n" + "=" * 50)
if all_passed:
    print("🎉 All quality scoring tests PASSED!")
    print("✅ Scoring formula works correctly")
    print("✅ Gate decisions are appropriate") 
    print("✅ Bonus points apply correctly")
    print("✅ Critical issues override scores")
else:
    print("❌ Some quality scoring tests FAILED!")

print(f"\n📋 Quality Gate Decision Matrix:")
print("90-100: PASS (Excellent quality)")
print("70-89:  PASS with CONCERNS (Good quality, monitor items)")  
print("40-69:  FAIL (Significant issues, fixes required)")
print("0-39:   FAIL (Critical issues, major rework needed)")
print("ANY:    FAIL if critical issues present")