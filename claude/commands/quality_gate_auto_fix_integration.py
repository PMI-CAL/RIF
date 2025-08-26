#!/usr/bin/env python3
"""
Quality Gate Auto-Fix Integration - Issue #269
Integration layer between quality gate enforcement and auto-fix engine.

This system provides:
1. Seamless integration with existing quality gate enforcement
2. Automatic triggering of fixes when gates fail
3. Validation loop to ensure fixes work
4. Enhanced reporting with fix attempts and results
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    from .quality_gate_enforcement import QualityGateEnforcement
    from .quality_gate_auto_fix_engine import QualityGateAutoFixEngine, FixResult
except ImportError:
    from quality_gate_enforcement import QualityGateEnforcement
    from quality_gate_auto_fix_engine import QualityGateAutoFixEngine, FixResult

class QualityGateAutoFixIntegration:
    """
    Integration layer that enhances quality gate enforcement with auto-fix capabilities.
    """
    
    def __init__(self, config_path: str = "config/rif-workflow.yaml"):
        """Initialize the auto-fix integration."""
        self.config_path = config_path
        self.setup_logging()
        
        # Initialize components
        self.gate_enforcement = QualityGateEnforcement(config_path)
        self.auto_fix_engine = QualityGateAutoFixEngine(config_path=config_path)
        
        # Integration settings
        self.enable_auto_fix = True
        self.max_fix_rounds = 2
        
        self.logger.info("🔧 Quality Gate Auto-Fix Integration initialized")
    
    def setup_logging(self):
        """Setup logging for auto-fix integration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - QualityGateAutoFixIntegration - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_issue_with_auto_fix(self, issue_number: int, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced issue validation with automatic quality gate failure fixes.
        
        This is the main entry point that replaces standard validation with auto-fix capability.
        
        Args:
            issue_number: GitHub issue number
            context: Optional context information
            
        Returns:
            Enhanced validation report with auto-fix results
        """
        self.logger.info(f"🚀 Starting enhanced validation with auto-fix for issue #{issue_number}")
        
        enhanced_report = {
            'issue_number': issue_number,
            'timestamp': datetime.now().isoformat(),
            'auto_fix_enabled': self.enable_auto_fix,
            'original_validation': {},
            'auto_fix_attempts': [],
            'final_validation': {},
            'overall_result': {},
            'fix_summary': {}
        }
        
        try:
            # Phase 1: Initial quality gate validation
            self.logger.info(f"📊 Phase 1: Initial quality gate validation for #{issue_number}")
            initial_validation = self.gate_enforcement.validate_issue_closure_readiness(issue_number)
            enhanced_report['original_validation'] = initial_validation
            
            # Check if issues initially pass
            if initial_validation.get('can_close', False):
                self.logger.info(f"✅ Issue #{issue_number} passes quality gates initially - no auto-fix needed")
                enhanced_report['overall_result'] = {
                    'can_close': True,
                    'decision': 'PASS',
                    'auto_fix_needed': False,
                    'reason': 'Quality gates passed on initial validation'
                }
                return enhanced_report
            
            # Phase 2: Auto-fix attempt (if enabled)
            if not self.enable_auto_fix:
                self.logger.info(f"🚫 Auto-fix disabled - blocking issue #{issue_number}")
                enhanced_report['overall_result'] = initial_validation
                enhanced_report['overall_result']['auto_fix_skipped'] = True
                return enhanced_report
            
            self.logger.info(f"🔧 Phase 2: Attempting auto-fixes for issue #{issue_number}")
            auto_fix_results = self._attempt_comprehensive_auto_fix(
                issue_number, initial_validation, context
            )
            enhanced_report['auto_fix_attempts'] = auto_fix_results
            
            # Phase 3: Final validation after auto-fixes
            self.logger.info(f"🔍 Phase 3: Final validation after auto-fixes for #{issue_number}")
            final_validation = self.gate_enforcement.validate_issue_closure_readiness(issue_number)
            enhanced_report['final_validation'] = final_validation
            
            # Phase 4: Generate overall result
            overall_result = self._generate_overall_result(
                initial_validation, auto_fix_results, final_validation
            )
            enhanced_report['overall_result'] = overall_result
            
            # Phase 5: Generate fix summary
            fix_summary = self._generate_fix_summary(auto_fix_results, overall_result)
            enhanced_report['fix_summary'] = fix_summary
            
            self.logger.info(f"🎯 Enhanced validation complete for #{issue_number}: {overall_result.get('decision', 'UNKNOWN')}")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced validation for #{issue_number}: {e}")
            enhanced_report['overall_result'] = {
                'can_close': False,
                'decision': 'FAIL',
                'error': str(e),
                'reason': f'Enhanced validation failed: {e}'
            }
        
        return enhanced_report
    
    def _attempt_comprehensive_auto_fix(self, issue_number: int, initial_validation: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attempt comprehensive auto-fix across multiple rounds."""
        all_fix_attempts = []
        current_validation = initial_validation
        
        for round_num in range(1, self.max_fix_rounds + 1):
            self.logger.info(f"🔄 Auto-fix round {round_num} for issue #{issue_number}")
            
            # Analyze current failures
            failures = self.auto_fix_engine.analyze_quality_gate_failures(current_validation)
            
            if not failures:
                self.logger.info(f"ℹ️ No auto-fixable failures found in round {round_num}")
                break
            
            # Attempt fixes
            fix_attempts = self.auto_fix_engine.attempt_auto_fix(failures)
            
            # Validate fixes
            validation_report = self.auto_fix_engine.validate_fixes(fix_attempts)
            
            # Commit successful fixes
            commit_successful = False
            if validation_report.get('overall_improvement', False):
                commit_successful = self.auto_fix_engine.commit_fixes(fix_attempts, validation_report)
            
            # Record round results
            round_result = {
                'round': round_num,
                'failures_identified': len(failures),
                'fix_attempts': [self._serialize_fix_attempt(attempt) for attempt in fix_attempts],
                'validation_report': validation_report,
                'commit_successful': commit_successful,
                'improvements_made': validation_report.get('overall_improvement', False)
            }
            
            all_fix_attempts.append(round_result)
            
            # If no improvements made, stop trying
            if not validation_report.get('overall_improvement', False):
                self.logger.info(f"⏹️ No improvements in round {round_num}, stopping auto-fix attempts")
                break
            
            # If all gates now pass, stop trying
            if validation_report.get('ready_for_merge', False):
                self.logger.info(f"✅ All gates passing after round {round_num}, auto-fix successful")
                break
            
            # Re-validate for next round
            current_validation = self.gate_enforcement.validate_issue_closure_readiness(issue_number)
        
        return all_fix_attempts
    
    def _serialize_fix_attempt(self, fix_attempt) -> Dict[str, Any]:
        """Serialize fix attempt object to dictionary."""
        return {
            'failure_type': fix_attempt.failure.failure_type.value,
            'fix_strategy': fix_attempt.fix_strategy,
            'actions_taken': fix_attempt.actions_taken,
            'files_modified': fix_attempt.files_modified,
            'result': fix_attempt.result.value,
            'validation_passed': fix_attempt.validation_passed,
            'error_message': fix_attempt.error_message,
            'improvement_metrics': fix_attempt.improvement_metrics
        }
    
    def _generate_overall_result(self, initial_validation: Dict[str, Any], auto_fix_results: List[Dict[str, Any]], final_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation result combining initial, fixes, and final validation."""
        
        # Determine if auto-fixes were successful
        auto_fix_successful = final_validation.get('can_close', False)
        
        # Count fixes applied
        total_fixes_applied = sum(
            len([attempt for attempt in round_data['fix_attempts'] if attempt['result'] in ['success', 'partial']])
            for round_data in auto_fix_results
        )
        
        # Generate result
        if auto_fix_successful:
            return {
                'can_close': True,
                'decision': 'PASS',
                'auto_fix_attempted': True,
                'auto_fix_successful': True,
                'fixes_applied': total_fixes_applied,
                'reason': f'Quality gate failures auto-fixed successfully ({total_fixes_applied} fixes applied)',
                'merge_blocked': False,
                'ready_for_merge': True
            }
        else:
            remaining_issues = len(final_validation.get('blocking_reasons', []))
            return {
                'can_close': False,
                'decision': 'FAIL',
                'auto_fix_attempted': True,
                'auto_fix_successful': False,
                'fixes_applied': total_fixes_applied,
                'reason': f'Auto-fix attempted ({total_fixes_applied} fixes applied) but {remaining_issues} issues remain',
                'merge_blocked': True,
                'ready_for_merge': False,
                'remaining_issues': final_validation.get('blocking_reasons', [])
            }
    
    def _generate_fix_summary(self, auto_fix_results: List[Dict[str, Any]], overall_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of auto-fix attempts."""
        
        total_rounds = len(auto_fix_results)
        total_failures_identified = sum(round_data['failures_identified'] for round_data in auto_fix_results)
        total_fix_attempts = sum(len(round_data['fix_attempts']) for round_data in auto_fix_results)
        
        successful_fixes = 0
        partial_fixes = 0
        failed_fixes = 0
        
        fix_types_applied = set()
        files_modified = set()
        
        for round_data in auto_fix_results:
            for attempt in round_data['fix_attempts']:
                if attempt['result'] == 'success':
                    successful_fixes += 1
                elif attempt['result'] == 'partial':
                    partial_fixes += 1
                else:
                    failed_fixes += 1
                
                fix_types_applied.add(attempt['failure_type'])
                files_modified.update(attempt['files_modified'])
        
        return {
            'total_rounds': total_rounds,
            'total_failures_identified': total_failures_identified,
            'total_fix_attempts': total_fix_attempts,
            'successful_fixes': successful_fixes,
            'partial_fixes': partial_fixes,
            'failed_fixes': failed_fixes,
            'fix_types_applied': list(fix_types_applied),
            'files_modified': list(files_modified),
            'files_modified_count': len(files_modified),
            'overall_success': overall_result.get('auto_fix_successful', False)
        }
    
    def generate_validation_report_for_github(self, enhanced_report: Dict[str, Any]) -> str:
        """Generate validation report suitable for posting to GitHub issues."""
        issue_number = enhanced_report['issue_number']
        overall_result = enhanced_report['overall_result']
        fix_summary = enhanced_report['fix_summary']
        
        # Header
        lines = [
            "## 🏗️ Enhanced Test Architect Quality Assessment with Auto-Fix",
            "",
            f"**Issue**: #{issue_number}",
            f"**Timestamp**: {enhanced_report['timestamp']}",
            f"**Auto-Fix Enabled**: {'✅' if enhanced_report['auto_fix_enabled'] else '❌'}",
            f"**Final Decision**: **{overall_result.get('decision', 'UNKNOWN')}**",
            f"**Ready for Merge**: {'✅' if overall_result.get('ready_for_merge', False) else '❌'}",
            ""
        ]
        
        # Auto-fix summary
        if enhanced_report['auto_fix_attempts']:
            lines.extend([
                "### 🤖 Auto-Fix Summary",
                f"**Rounds Attempted**: {fix_summary.get('total_rounds', 0)}",
                f"**Failures Identified**: {fix_summary.get('total_failures_identified', 0)}",
                f"**Fixes Applied**: {fix_summary.get('successful_fixes', 0)}",
                f"**Partial Fixes**: {fix_summary.get('partial_fixes', 0)}",
                f"**Failed Fixes**: {fix_summary.get('failed_fixes', 0)}",
                f"**Files Modified**: {fix_summary.get('files_modified_count', 0)}",
                ""
            ])
            
            # Fix types applied
            if fix_summary.get('fix_types_applied'):
                lines.extend([
                    "**Fix Types Applied**:",
                    *[f"- {fix_type.replace('_', ' ').title()}" for fix_type in fix_summary['fix_types_applied']],
                    ""
                ])
            
            # Files modified
            if fix_summary.get('files_modified'):
                lines.extend([
                    "**Files Modified by Auto-Fix**:",
                    *[f"- `{file}`" for file in sorted(fix_summary['files_modified'])],
                    ""
                ])
        else:
            lines.extend([
                "### 🤖 Auto-Fix Summary",
                "**Status**: No auto-fix attempts (quality gates passed initially or auto-fix disabled)",
                ""
            ])
        
        # Final validation results
        final_validation = enhanced_report.get('final_validation', {})
        if final_validation:
            lines.extend([
                "### 📊 Final Quality Gate Status",
                f"**Overall Pass**: {'✅' if final_validation.get('can_close', False) else '❌'}",
                f"**Quality Score**: {final_validation.get('quality_score', {}).get('score', 'N/A')}",
                ""
            ])
            
            # Quality gates details
            quality_gates = final_validation.get('quality_gates', {})
            if quality_gates:
                passed_gates = quality_gates.get('passed_gates', [])
                failed_gates = quality_gates.get('failed_gates', [])
                
                lines.extend([
                    "**Quality Gates Status**:",
                    f"✅ Passed: {', '.join(passed_gates) if passed_gates else 'None'}",
                    f"❌ Failed: {', '.join(failed_gates) if failed_gates else 'None'}",
                    ""
                ])
        
        # Decision rationale
        lines.extend([
            "### 🎯 Validation Decision",
            f"**Decision**: {overall_result.get('decision', 'UNKNOWN')}",
            f"**Reason**: {overall_result.get('reason', 'No reason provided')}",
            ""
        ])
        
        # Next steps
        if overall_result.get('ready_for_merge', False):
            lines.extend([
                "### ✅ Next Steps",
                "- Issue is ready for merge",
                "- All quality gates are passing",
            ])
            if fix_summary.get('successful_fixes', 0) > 0:
                lines.append(f"- Auto-fixes have been applied and committed ({fix_summary['successful_fixes']} fixes)")
        else:
            lines.extend([
                "### ❌ Blocking Issues",
                "The following issues must be resolved before merge:",
            ])
            remaining_issues = overall_result.get('remaining_issues', [])
            if remaining_issues:
                lines.extend([f"- {issue}" for issue in remaining_issues])
            else:
                lines.append("- Check final validation results above")
        
        lines.extend([
            "",
            "---",
            "*This report was generated by the Enhanced RIF-Validator with Auto-Fix capability (Issue #269)*"
        ])
        
        return "\n".join(lines)

def main():
    """Command line interface for quality gate auto-fix integration."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_gate_auto_fix_integration.py <command> [args]")
        print("Commands:")
        print("  validate-with-auto-fix <issue_number>     - Run enhanced validation with auto-fix")
        print("  test-integration                          - Test integration functionality")
        return
    
    command = sys.argv[1]
    integration = QualityGateAutoFixIntegration()
    
    if command == "validate-with-auto-fix" and len(sys.argv) >= 3:
        issue_number = int(sys.argv[2])
        
        # Run enhanced validation
        enhanced_report = integration.validate_issue_with_auto_fix(issue_number)
        
        # Generate GitHub report
        github_report = integration.generate_validation_report_for_github(enhanced_report)
        
        print("🎯 Enhanced Validation Report")
        print("=" * 50)
        print(github_report)
        
        # Save detailed report
        with open(f'enhanced_validation_report_{issue_number}.json', 'w') as f:
            json.dump(enhanced_report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to enhanced_validation_report_{issue_number}.json")
    
    elif command == "test-integration":
        print("🧪 Testing Quality Gate Auto-Fix Integration")
        
        # Test with mock issue
        enhanced_report = integration.validate_issue_with_auto_fix(9999)
        
        print(f"Integration test result: {enhanced_report['overall_result'].get('decision', 'UNKNOWN')}")
        print(f"Auto-fix enabled: {enhanced_report['auto_fix_enabled']}")
        print(f"Fix attempts: {len(enhanced_report['auto_fix_attempts'])}")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())