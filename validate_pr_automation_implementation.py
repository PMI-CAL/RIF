#!/usr/bin/env python3
"""
Implementation Validation Script for Automated PR Creation - Issue #205
Validates that all components are correctly implemented and can work together.
"""

import sys
import json
import tempfile
from pathlib import Path

def validate_syntax():
    """Validate Python syntax of all implemented files."""
    print("🔍 Validating Python syntax...")
    
    files_to_check = [
        'claude/commands/github_state_manager.py',
        'claude/commands/pr_template_aggregator.py', 
        'claude/commands/pr_creation_service.py',
        'claude/commands/state_machine_hooks.py'
    ]
    
    syntax_errors = []
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            print(f"   ✅ {file_path}")
        except SyntaxError as e:
            print(f"   ❌ {file_path}: {e}")
            syntax_errors.append((file_path, e))
        except FileNotFoundError:
            print(f"   ⚠️  {file_path}: File not found")
            syntax_errors.append((file_path, "File not found"))
            
    return len(syntax_errors) == 0, syntax_errors

def validate_imports():
    """Validate that all imports work correctly."""
    print("📦 Validating imports...")
    
    import_tests = [
        ('GitHubStateManager extensions', '''
import sys
sys.path.append('.')
from claude.commands.github_state_manager import GitHubStateManager
manager = GitHubStateManager()
# Test new methods exist
assert hasattr(manager, 'generate_branch_name')
assert hasattr(manager, 'get_file_modifications')  
assert hasattr(manager, 'get_issue_metadata')
assert hasattr(manager, 'create_pull_request')
assert hasattr(manager, 'populate_pr_template')
'''),
        ('PR Template Aggregator', '''
import sys
sys.path.append('.')
from claude.commands.pr_template_aggregator import PRTemplateAggregator, PRContext, CheckpointData
aggregator = PRTemplateAggregator()
assert hasattr(aggregator, 'load_template')
assert hasattr(aggregator, 'populate_template')
assert hasattr(aggregator, 'aggregate_pr_context')
'''),
        ('PR Creation Service', '''
import sys  
sys.path.append('.')
from claude.commands.pr_creation_service import PRCreationService
service = PRCreationService()
assert hasattr(service, 'should_create_pr')
assert hasattr(service, 'check_quality_gates')
assert hasattr(service, 'create_automated_pr')
'''),
        ('State Machine Hooks', '''
import sys
sys.path.append('.')
from claude.commands.state_machine_hooks import StateMachineHooks
hooks = StateMachineHooks() 
assert hasattr(hooks, 'execute_hooks')
assert hasattr(hooks, 'enhanced_transition_state')
''')
    ]
    
    import_errors = []
    
    for test_name, test_code in import_tests:
        try:
            exec(test_code)
            print(f"   ✅ {test_name}")
        except Exception as e:
            print(f"   ❌ {test_name}: {e}")
            import_errors.append((test_name, e))
            
    return len(import_errors) == 0, import_errors

def validate_functionality():
    """Validate basic functionality of key components."""
    print("⚙️  Validating functionality...")
    
    functionality_tests = []
    
    # Test GitHubStateManager extensions
    try:
        sys.path.append('.')
        from claude.commands.github_state_manager import GitHubStateManager
        
        manager = GitHubStateManager()
        
        # Test branch name generation
        branch_name = manager.generate_branch_name(205, "Test Implementation")
        assert branch_name == "issue-205-test-implementation"
        
        print("   ✅ GitHubStateManager branch name generation")
    except Exception as e:
        print(f"   ❌ GitHubStateManager functionality: {e}")
        functionality_tests.append(("GitHubStateManager", e))
    
    # Test PR Template Aggregator
    try:
        from claude.commands.pr_template_aggregator import PRTemplateAggregator, CheckpointData
        
        aggregator = PRTemplateAggregator()
        
        # Test checkpoint processing
        test_checkpoints = [
            CheckpointData(
                checkpoint_id="test",
                phase="Test Phase", 
                status="complete",
                description="Test description",
                timestamp="2025-01-24T18:00:00Z",
                components={}
            )
        ]
        
        summary = aggregator.generate_implementation_summary(test_checkpoints)
        assert "Test Phase" in summary
        assert "Complete" in summary or "complete" in summary
        
        print("   ✅ PRTemplateAggregator checkpoint processing")
    except Exception as e:
        print(f"   ❌ PRTemplateAggregator functionality: {e}")
        functionality_tests.append(("PRTemplateAggregator", e))
    
    # Test PR Creation Service
    try:
        from claude.commands.pr_creation_service import PRCreationService
        
        service = PRCreationService()
        
        # Test trigger detection
        should_create = service.should_create_pr(205, 'implementing', 'validating')
        assert should_create == True
        
        should_not_create = service.should_create_pr(205, 'new', 'analyzing')
        assert should_not_create == False
        
        print("   ✅ PRCreationService trigger detection")
    except Exception as e:
        print(f"   ❌ PRCreationService functionality: {e}")
        functionality_tests.append(("PRCreationService", e))
    
    return len(functionality_tests) == 0, functionality_tests

def validate_integration_points():
    """Validate that integration points are correctly configured."""
    print("🔗 Validating integration points...")
    
    integration_issues = []
    
    # Check workflow configuration exists
    workflow_config = Path("config/rif-workflow.yaml")
    if workflow_config.exists():
        print("   ✅ Workflow configuration exists")
    else:
        print("   ❌ Workflow configuration missing")
        integration_issues.append("Workflow configuration missing")
    
    # Check PR template exists
    pr_template = Path(".github/pull_request_template.md")
    if pr_template.exists():
        print("   ✅ PR template exists")
    else:
        print("   ❌ PR template missing")
        integration_issues.append("PR template missing")
    
    # Check knowledge/checkpoints directory
    checkpoint_dir = Path("knowledge/checkpoints")
    if checkpoint_dir.exists():
        print("   ✅ Checkpoint directory exists")
    else:
        print("   ❌ Checkpoint directory missing")
        integration_issues.append("Checkpoint directory missing")
    
    return len(integration_issues) == 0, integration_issues

def validate_checkpoints():
    """Validate that implementation checkpoints are complete."""
    print("📋 Validating implementation checkpoints...")
    
    expected_checkpoints = [
        'issue-205-phase1-pr-engine-complete.json',
        'issue-205-phase2-template-system-complete.json',
        'issue-205-phase3-state-integration-complete.json'
    ]
    
    checkpoint_issues = []
    checkpoint_dir = Path("knowledge/checkpoints")
    
    for checkpoint_file in expected_checkpoints:
        checkpoint_path = checkpoint_dir / checkpoint_file
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    
                if data.get('status') == 'complete':
                    print(f"   ✅ {checkpoint_file}")
                else:
                    print(f"   ⚠️  {checkpoint_file}: Status not complete")
                    checkpoint_issues.append(f"{checkpoint_file}: Incomplete status")
                    
            except Exception as e:
                print(f"   ❌ {checkpoint_file}: {e}")
                checkpoint_issues.append(f"{checkpoint_file}: {e}")
        else:
            print(f"   ❌ {checkpoint_file}: Missing")
            checkpoint_issues.append(f"{checkpoint_file}: Missing")
    
    return len(checkpoint_issues) == 0, checkpoint_issues

def run_validation():
    """Run complete validation of the automated PR creation implementation."""
    print("🚀 Validating Automated PR Creation Implementation - Issue #205")
    print("=" * 70)
    
    validation_results = {}
    
    # Run all validation checks
    syntax_ok, syntax_errors = validate_syntax()
    validation_results['syntax'] = (syntax_ok, syntax_errors)
    
    imports_ok, import_errors = validate_imports()
    validation_results['imports'] = (imports_ok, import_errors)
    
    functionality_ok, functionality_errors = validate_functionality()
    validation_results['functionality'] = (functionality_ok, functionality_errors)
    
    integration_ok, integration_issues = validate_integration_points()
    validation_results['integration'] = (integration_ok, integration_issues)
    
    checkpoints_ok, checkpoint_issues = validate_checkpoints()
    validation_results['checkpoints'] = (checkpoints_ok, checkpoint_issues)
    
    # Calculate overall results
    all_checks = [syntax_ok, imports_ok, functionality_ok, integration_ok, checkpoints_ok]
    overall_success = all(all_checks)
    
    # Print summary
    print("\n📊 Validation Summary")
    print("=" * 30)
    
    check_names = ['Syntax', 'Imports', 'Functionality', 'Integration', 'Checkpoints']
    for i, (check_name, passed) in enumerate(zip(check_names, all_checks)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:12} {status}")
        
        if not passed:
            _, errors = validation_results[check_name.lower()]
            for error in errors[:3]:  # Show first 3 errors
                print(f"             - {error}")
            if len(errors) > 3:
                print(f"             - ... and {len(errors) - 3} more")
    
    print(f"\nOverall Result: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("\n🎉 All validation checks passed!")
        print("   The automated PR creation system is ready for deployment.")
    else:
        print("\n⚠️  Some validation checks failed.")
        print("   Review the errors above before proceeding.")
    
    return overall_success

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)