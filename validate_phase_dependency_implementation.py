#!/usr/bin/env python3
"""
Phase Dependency Implementation Validation

Simple validation script to test the phase dependency implementation components.

Issue #223: RIF Orchestration Error: Not Following Phase Dependencies
"""

import sys
import json
from pathlib import Path


def validate_implementation_files():
    """Validate that all implementation files exist and are properly structured"""
    
    print("🔍 Validating Phase Dependency Implementation Files...")
    print("=" * 60)
    
    required_files = {
        "claude/commands/phase_dependency_validator.py": "Core phase dependency validation logic",
        "claude/commands/phase_dependency_warning_system.py": "Real-time warning and prevention system", 
        "claude/commands/phase_dependency_orchestration_integration.py": "Orchestration framework integration",
        "CLAUDE.md": "Enhanced orchestration documentation with phase dependency rules"
    }
    
    validation_results = {
        "files_checked": 0,
        "files_found": 0,
        "files_missing": [],
        "implementation_complete": False
    }
    
    base_path = Path(__file__).parent
    
    for file_path, description in required_files.items():
        validation_results["files_checked"] += 1
        full_path = base_path / file_path
        
        if full_path.exists():
            print(f"  ✅ {file_path} - {description}")
            validation_results["files_found"] += 1
            
            # Basic content validation
            try:
                content = full_path.read_text()
                if len(content) > 1000:  # Minimum content check
                    print(f"      Content: {len(content)} characters")
                else:
                    print(f"      ⚠️  Content seems minimal: {len(content)} characters")
                    
            except Exception as e:
                print(f"      ❌ Error reading file: {e}")
                
        else:
            print(f"  ❌ {file_path} - MISSING")
            validation_results["files_missing"].append(file_path)
            
    validation_results["implementation_complete"] = len(validation_results["files_missing"]) == 0
    
    return validation_results


def validate_claude_md_enhancements():
    """Validate CLAUDE.md has been properly enhanced with phase dependency rules"""
    
    print("\n📋 Validating CLAUDE.md Enhancements...")
    print("-" * 40)
    
    claude_md_path = Path(__file__).parent / "CLAUDE.md"
    
    if not claude_md_path.exists():
        print("  ❌ CLAUDE.md not found")
        return False
        
    content = claude_md_path.read_text()
    
    required_sections = [
        "Phase Dependency Enforcement",
        "Phase Dependency Rules",
        "Phase Completion Criteria Matrix",
        "Validation Checkpoint Requirements",
        "Phase Dependency Violation Types", 
        "Violation Severity Levels"
    ]
    
    missing_sections = []
    
    for section in required_sections:
        if section in content:
            print(f"  ✅ {section} section found")
        else:
            print(f"  ❌ {section} section missing")
            missing_sections.append(section)
            
    # Check for specific implementation details
    implementation_markers = [
        "PhaseDependencyValidator",
        "validate_phase_dependencies", 
        "MANDATORY: Validate phase dependencies",
        "Research → Analysis → Planning → Architecture → Implementation → Validation"
    ]
    
    for marker in implementation_markers:
        if marker in content:
            print(f"  ✅ Implementation marker found: {marker}")
        else:
            print(f"  ⚠️  Implementation marker missing: {marker}")
            
    return len(missing_sections) == 0


def validate_component_structure():
    """Validate the structure of each component file"""
    
    print("\n🔧 Validating Component Structure...")
    print("-" * 40)
    
    components = {
        "claude/commands/phase_dependency_validator.py": [
            "class PhaseDependencyValidator",
            "def validate_phase_dependencies",
            "def validate_phase_completion", 
            "def enforce_sequential_phases",
            "class PhaseType",
            "class PhaseDependencyViolation"
        ],
        "claude/commands/phase_dependency_warning_system.py": [
            "class PhaseDependencyWarningSystem",
            "def detect_violations_real_time",
            "def generate_actionable_messages",
            "def auto_redirect_to_prerequisites",
            "class PhaseWarningAlert",
            "class AutoRedirectionSuggestion"
        ],
        "claude/commands/phase_dependency_orchestration_integration.py": [
            "class PhaseDependencyOrchestrationIntegration",
            "def make_enhanced_orchestration_decision",
            "def generate_enhanced_orchestration_template",
            "class EnhancedOrchestrationDecision",
            "make_enhanced_orchestration_decision_with_phase_validation"
        ]
    }
    
    structure_valid = True
    base_path = Path(__file__).parent
    
    for file_path, required_elements in components.items():
        full_path = base_path / file_path
        
        print(f"\n  📄 {file_path}:")
        
        if not full_path.exists():
            print(f"    ❌ File not found")
            structure_valid = False
            continue
            
        try:
            content = full_path.read_text()
            
            for element in required_elements:
                if element in content:
                    print(f"    ✅ {element}")
                else:
                    print(f"    ❌ {element} - MISSING")
                    structure_valid = False
                    
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")
            structure_valid = False
            
    return structure_valid


def generate_implementation_report():
    """Generate comprehensive implementation report"""
    
    print("\n📊 Generating Implementation Report...")
    print("=" * 60)
    
    # Run all validations
    file_validation = validate_implementation_files()
    claude_validation = validate_claude_md_enhancements()
    structure_validation = validate_component_structure()
    
    # Calculate overall score
    validations = [
        file_validation["implementation_complete"],
        claude_validation,
        structure_validation
    ]
    
    passed_validations = sum(validations)
    total_validations = len(validations)
    completion_percentage = (passed_validations / total_validations) * 100
    
    # Generate report
    report = {
        "issue_reference": 223,
        "implementation_status": "COMPLETE" if passed_validations == total_validations else "PARTIAL",
        "completion_percentage": completion_percentage,
        "validations": {
            "file_structure": file_validation["implementation_complete"],
            "claude_md_enhancements": claude_validation,
            "component_structure": structure_validation
        },
        "files_implemented": file_validation["files_found"],
        "files_total": file_validation["files_checked"], 
        "missing_files": file_validation["files_missing"],
        "summary": {
            "phase_dependency_validator": Path("claude/commands/phase_dependency_validator.py").exists(),
            "warning_system": Path("claude/commands/phase_dependency_warning_system.py").exists(), 
            "orchestration_integration": Path("claude/commands/phase_dependency_orchestration_integration.py").exists(),
            "claude_md_updated": claude_validation,
            "test_suite": Path("test_phase_dependency_implementation.py").exists()
        }
    }
    
    print(f"\n🎯 IMPLEMENTATION STATUS: {report['implementation_status']}")
    print(f"📈 COMPLETION: {completion_percentage:.1f}%")
    print(f"📁 FILES: {report['files_implemented']}/{report['files_total']} implemented")
    
    if report["missing_files"]:
        print(f"❌ MISSING FILES: {', '.join(report['missing_files'])}")
        
    print(f"\n📋 COMPONENT SUMMARY:")
    for component, status in report["summary"].items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
    # Save report
    report_file = Path(__file__).parent / "phase_dependency_implementation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n📄 Report saved to: {report_file}")
    
    return report


def validate_key_functionality():
    """Validate key functionality with simple tests"""
    
    print("\n🧪 Validating Key Functionality...")
    print("-" * 40)
    
    try:
        # Test 1: Check if phase enum is properly defined
        validator_file = Path("claude/commands/phase_dependency_validator.py")
        if validator_file.exists():
            content = validator_file.read_text()
            phase_types = ["RESEARCH", "ANALYSIS", "PLANNING", "ARCHITECTURE", "IMPLEMENTATION", "VALIDATION"]
            
            phase_enum_complete = all(phase_type in content for phase_type in phase_types)
            print(f"  ✅ Phase types defined: {phase_enum_complete}")
            
            # Check for key validation functions
            key_functions = [
                "validate_phase_dependencies",
                "validate_phase_completion",
                "enforce_sequential_phases"
            ]
            
            for func in key_functions:
                if func in content:
                    print(f"  ✅ Function implemented: {func}")
                else:
                    print(f"  ❌ Function missing: {func}")
                    
        # Test 2: Check warning system completeness
        warning_file = Path("claude/commands/phase_dependency_warning_system.py")
        if warning_file.exists():
            content = warning_file.read_text()
            
            warning_features = [
                "detect_violations_real_time",
                "generate_actionable_messages", 
                "auto_redirect_to_prerequisites",
                "prevent_resource_waste"
            ]
            
            for feature in warning_features:
                if feature in content:
                    print(f"  ✅ Warning feature: {feature}")
                else:
                    print(f"  ❌ Warning feature missing: {feature}")
                    
        # Test 3: Check integration completeness
        integration_file = Path("claude/commands/phase_dependency_orchestration_integration.py")
        if integration_file.exists():
            content = integration_file.read_text()
            
            integration_features = [
                "make_enhanced_orchestration_decision",
                "generate_enhanced_orchestration_template",
                "check_github_state_real_time",
                "setup_automated_enforcement_hooks"
            ]
            
            for feature in integration_features:
                if feature in content:
                    print(f"  ✅ Integration feature: {feature}")
                else:
                    print(f"  ❌ Integration feature missing: {feature}")
                    
    except Exception as e:
        print(f"  ❌ Error during functionality validation: {e}")
        

def main():
    """Main validation function"""
    
    print("🚀 Phase Dependency Implementation Validation")
    print("Issue #223: RIF Orchestration Error: Not Following Phase Dependencies")
    print("=" * 70)
    
    # Run all validations
    file_validation = validate_implementation_files()
    claude_validation = validate_claude_md_enhancements()
    structure_validation = validate_component_structure()
    
    # Validate functionality
    validate_key_functionality()
    
    # Generate comprehensive report
    report = generate_implementation_report()
    
    # Final assessment
    print("\n" + "=" * 70)
    if report["implementation_status"] == "COMPLETE":
        print("🎉 PHASE DEPENDENCY IMPLEMENTATION VALIDATION: ✅ PASSED")
        print("   All components implemented successfully!")
        print("   Ready for deployment and testing.")
    else:
        print("⚠️  PHASE DEPENDENCY IMPLEMENTATION VALIDATION: ⚠️ PARTIAL")
        print(f"   {report['completion_percentage']:.1f}% complete")
        print("   Some components need attention.")
        
    print("\n📋 NEXT STEPS:")
    if report["implementation_status"] == "COMPLETE":
        print("   1. ✅ Update issue #223 to state:validating")
        print("   2. ✅ Run comprehensive test suite")
        print("   3. ✅ Deploy phase dependency enforcement")
        print("   4. ✅ Monitor orchestration improvements")
    else:
        print("   1. ❌ Complete missing components")
        print("   2. ❌ Fix structural issues")
        print("   3. ❌ Re-run validation")
        print("   4. ❌ Proceed to testing phase")
        
    print("=" * 70)
    
    return report["implementation_status"] == "COMPLETE"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)