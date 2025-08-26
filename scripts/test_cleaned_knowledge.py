#!/usr/bin/env python3
"""
Cleaned Knowledge Base Test Suite

This script validates that the cleaned knowledge base:
1. Has correct structure for deployment
2. Contains essential patterns and decisions
3. Databases are properly initialized
4. No development artifacts remain

Usage:
    python scripts/test_cleaned_knowledge.py [--knowledge-dir PATH]
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanedKnowledgeValidator:
    """Validates cleaned knowledge base for deployment readiness"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.validation_results = {
            'structure_check': {'passed': False, 'details': []},
            'content_check': {'passed': False, 'details': []},
            'database_check': {'passed': False, 'details': []},
            'cleanup_check': {'passed': False, 'details': []},
            'size_check': {'passed': False, 'details': []},
            'overall': {'passed': False, 'score': 0}
        }
    
    def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        logger.info("Running full validation of cleaned knowledge base...")
        
        # Run all validation checks
        self._validate_directory_structure()
        self._validate_content_quality()
        self._validate_databases()
        self._validate_cleanup_completeness()
        self._validate_size_requirements()
        
        # Calculate overall score
        self._calculate_overall_score()
        
        return self.validation_results
    
    def _validate_directory_structure(self):
        """Validate knowledge base directory structure"""
        logger.info("Validating directory structure...")
        
        check_results = []
        
        # Required directories for deployment
        required_dirs = [
            'patterns',      # Must have reusable patterns
            'decisions',     # Must have framework decisions
            'conversations', # Framework component
            'context',       # Framework component
            'embeddings',    # Framework component
            'parsing',       # Framework component
            'integration'    # Framework component
        ]
        
        # Should NOT exist after cleanup
        forbidden_dirs = [
            'audits', 'checkpoints', 'issues', 'metrics',
            'enforcement_logs', 'evidence_collection'
        ]
        
        # Check required directories
        missing_required = []
        for dirname in required_dirs:
            dir_path = self.knowledge_dir / dirname
            if not dir_path.exists():
                missing_required.append(dirname)
            else:
                check_results.append(f"✅ {dirname}/ exists")
        
        if missing_required:
            check_results.append(f"❌ Missing required directories: {missing_required}")
        
        # Check forbidden directories  
        remaining_forbidden = []
        for dirname in forbidden_dirs:
            dir_path = self.knowledge_dir / dirname
            if dir_path.exists():
                remaining_forbidden.append(dirname)
        
        if remaining_forbidden:
            check_results.append(f"❌ Development artifacts still present: {remaining_forbidden}")
        else:
            check_results.append("✅ Development artifacts properly removed")
        
        # Check for essential files
        essential_files = [
            'project_metadata.json',
            'MIGRATION_GUIDE.md'
        ]
        
        for filename in essential_files:
            file_path = self.knowledge_dir / filename
            if file_path.exists():
                check_results.append(f"✅ {filename} exists")
            else:
                check_results.append(f"❌ Missing essential file: {filename}")
        
        # Determine pass/fail
        failed_checks = [r for r in check_results if r.startswith('❌')]
        self.validation_results['structure_check'] = {
            'passed': len(failed_checks) == 0,
            'details': check_results
        }
    
    def _validate_content_quality(self):
        """Validate content quality and completeness"""
        logger.info("Validating content quality...")
        
        check_results = []
        
        # Check patterns directory
        patterns_dir = self.knowledge_dir / 'patterns'
        if patterns_dir.exists():
            pattern_files = list(patterns_dir.glob('*.json'))
            if len(pattern_files) >= 5:  # Should have some preserved patterns
                check_results.append(f"✅ Patterns directory has {len(pattern_files)} patterns")
                
                # Check for reusable patterns (not issue-specific)
                reusable_patterns = []
                issue_specific_patterns = []
                
                for pattern_file in pattern_files:
                    if pattern_file.name.startswith('issue-'):
                        issue_specific_patterns.append(pattern_file.name)
                    else:
                        reusable_patterns.append(pattern_file.name)
                
                if issue_specific_patterns:
                    check_results.append(f"❌ Issue-specific patterns still present: {len(issue_specific_patterns)}")
                else:
                    check_results.append("✅ No issue-specific patterns found")
                
                if len(reusable_patterns) >= 3:
                    check_results.append(f"✅ Good selection of reusable patterns: {len(reusable_patterns)}")
                else:
                    check_results.append(f"⚠️ Few reusable patterns: {len(reusable_patterns)}")
            else:
                check_results.append(f"⚠️ Patterns directory has few files: {len(pattern_files)}")
        else:
            check_results.append("❌ Patterns directory missing")
        
        # Check decisions directory
        decisions_dir = self.knowledge_dir / 'decisions'
        if decisions_dir.exists():
            decision_files = list(decisions_dir.glob('*.json'))
            framework_decisions = [f for f in decision_files if 'architecture' in f.name.lower() or 'framework' in f.name.lower()]
            
            if len(framework_decisions) >= 3:
                check_results.append(f"✅ Good framework decisions preserved: {len(framework_decisions)}")
            else:
                check_results.append(f"⚠️ Few framework decisions: {len(framework_decisions)}")
        
        # Check that no development logs remain
        dev_files_found = []
        for pattern in ['*.log', '*.tmp', '*audit*.json', '*checkpoint*.json']:
            found_files = list(self.knowledge_dir.rglob(pattern))
            dev_files_found.extend(found_files)
        
        if dev_files_found:
            check_results.append(f"❌ Development files still present: {len(dev_files_found)}")
        else:
            check_results.append("✅ No development log files found")
        
        # Determine pass/fail
        failed_checks = [r for r in check_results if r.startswith('❌')]
        self.validation_results['content_check'] = {
            'passed': len(failed_checks) == 0,
            'details': check_results
        }
    
    def _validate_databases(self):
        """Validate database initialization"""
        logger.info("Validating database initialization...")
        
        check_results = []
        
        # Check ChromaDB
        chromadb_dir = self.knowledge_dir / 'chromadb'
        if chromadb_dir.exists():
            if chromadb_dir.is_dir() and len(list(chromadb_dir.iterdir())) <= 2:  # Should be empty or minimal
                check_results.append("✅ ChromaDB directory properly reset")
            else:
                check_results.append(f"⚠️ ChromaDB contains files: {len(list(chromadb_dir.iterdir()))}")
        else:
            check_results.append("❌ ChromaDB directory missing")
        
        # Check conversations database
        conv_db = self.knowledge_dir / 'conversations.duckdb'
        if conv_db.exists():
            size_mb = conv_db.stat().st_size / (1024 * 1024)
            if size_mb < 1:  # Should be small/empty
                check_results.append(f"✅ Conversations database reset ({size_mb:.2f}MB)")
            else:
                check_results.append(f"⚠️ Conversations database large ({size_mb:.2f}MB)")
        else:
            check_results.append("ℹ️ Conversations database will be created on first use")
        
        # Check orchestration database
        orch_db = self.knowledge_dir / 'orchestration.duckdb'
        if orch_db.exists():
            size_mb = orch_db.stat().st_size / (1024 * 1024)
            if size_mb < 1:  # Should be small/empty
                check_results.append(f"✅ Orchestration database reset ({size_mb:.2f}MB)")
            else:
                check_results.append(f"⚠️ Orchestration database large ({size_mb:.2f}MB)")
        else:
            check_results.append("ℹ️ Orchestration database will be created on first use")
        
        # Determine pass/fail
        failed_checks = [r for r in check_results if r.startswith('❌')]
        self.validation_results['database_check'] = {
            'passed': len(failed_checks) == 0,
            'details': check_results
        }
    
    def _validate_cleanup_completeness(self):
        """Validate that cleanup was complete"""
        logger.info("Validating cleanup completeness...")
        
        check_results = []
        
        # Check for RIF-specific artifacts that should be gone
        rif_artifacts = [
            'issue_closure_prevention_log.json',
            'pending_user_validations.json', 
            'user_validation_log.json',
            'cutover_config.json',
            'migration_state.json',
            'demo_health_registry.json',
            'shadow-mode.log'
        ]
        
        remaining_artifacts = []
        for artifact in rif_artifacts:
            if (self.knowledge_dir / artifact).exists():
                remaining_artifacts.append(artifact)
        
        if remaining_artifacts:
            check_results.append(f"❌ RIF artifacts still present: {remaining_artifacts}")
        else:
            check_results.append("✅ RIF-specific artifacts properly removed")
        
        # Check for issue-specific files in various directories
        issue_files_found = []
        search_dirs = ['patterns', 'decisions', 'learning', 'analysis']
        
        for dirname in search_dirs:
            dir_path = self.knowledge_dir / dirname
            if dir_path.exists():
                issue_files = list(dir_path.glob('issue-*.json'))
                issue_files_found.extend([f"{dirname}/{f.name}" for f in issue_files])
        
        if issue_files_found:
            check_results.append(f"❌ Issue-specific files remaining: {len(issue_files_found)}")
            if len(issue_files_found) <= 5:  # Show first few
                check_results.extend([f"  - {f}" for f in issue_files_found[:5]])
        else:
            check_results.append("✅ No issue-specific files found")
        
        # Determine pass/fail
        failed_checks = [r for r in check_results if r.startswith('❌')]
        self.validation_results['cleanup_check'] = {
            'passed': len(failed_checks) == 0,
            'details': check_results
        }
    
    def _validate_size_requirements(self):
        """Validate size is appropriate for deployment"""
        logger.info("Validating size requirements...")
        
        check_results = []
        
        # Calculate total size
        total_size = 0
        file_count = 0
        
        for item in self.knowledge_dir.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
        
        total_size_mb = total_size / (1024 * 1024)
        
        # Size requirements for deployment
        if total_size_mb <= 5:
            check_results.append(f"✅ Excellent size for deployment: {total_size_mb:.2f}MB")
        elif total_size_mb <= 10:
            check_results.append(f"✅ Good size for deployment: {total_size_mb:.2f}MB")
        elif total_size_mb <= 20:
            check_results.append(f"⚠️ Acceptable size for deployment: {total_size_mb:.2f}MB")
        else:
            check_results.append(f"❌ Too large for deployment: {total_size_mb:.2f}MB")
        
        check_results.append(f"ℹ️ Total files: {file_count:,}")
        
        # Check for any particularly large files
        large_files = []
        for item in self.knowledge_dir.rglob('*'):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                if size_mb > 2:  # Files over 2MB
                    large_files.append(f"{item.relative_to(self.knowledge_dir)} ({size_mb:.2f}MB)")
        
        if large_files:
            check_results.append(f"⚠️ Large files found: {len(large_files)}")
            for large_file in large_files[:3]:  # Show first 3
                check_results.append(f"  - {large_file}")
        else:
            check_results.append("✅ No unusually large files found")
        
        # Determine pass/fail (pass if under 20MB)
        self.validation_results['size_check'] = {
            'passed': total_size_mb <= 20,
            'details': check_results
        }
    
    def _calculate_overall_score(self):
        """Calculate overall validation score"""
        checks = ['structure_check', 'content_check', 'database_check', 'cleanup_check', 'size_check']
        
        passed_count = sum(1 for check in checks if self.validation_results[check]['passed'])
        total_count = len(checks)
        
        score = (passed_count / total_count) * 100
        overall_passed = score >= 80  # 80% pass rate required
        
        self.validation_results['overall'] = {
            'passed': overall_passed,
            'score': score,
            'passed_checks': passed_count,
            'total_checks': total_count
        }
    
    def generate_validation_report(self) -> str:
        """Generate validation report"""
        results = self.validation_results
        overall = results['overall']
        
        # Header
        status_icon = "✅" if overall['passed'] else "❌"
        report = f"""# Cleaned Knowledge Base Validation Report

## Overall Status: {status_icon} {'PASSED' if overall['passed'] else 'FAILED'} 
**Score**: {overall['score']:.1f}% ({overall['passed_checks']}/{overall['total_checks']} checks passed)

"""
        
        # Individual check results
        check_names = {
            'structure_check': 'Directory Structure',
            'content_check': 'Content Quality', 
            'database_check': 'Database Initialization',
            'cleanup_check': 'Cleanup Completeness',
            'size_check': 'Size Requirements'
        }
        
        for check_key, check_name in check_names.items():
            check_result = results[check_key]
            status_icon = "✅" if check_result['passed'] else "❌"
            
            report += f"## {status_icon} {check_name}\n\n"
            
            for detail in check_result['details']:
                report += f"{detail}\n"
            
            report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if overall['passed']:
            report += "✅ Knowledge base is ready for deployment!\n\n"
            report += "Next steps:\n"
            report += "1. Use `scripts/init_project_knowledge.py` for new projects\n"
            report += "2. Initialize with `./rif-init.sh` in target project\n"
            report += "3. Begin development using GitHub issues\n"
        else:
            report += "❌ Knowledge base requires additional cleanup.\n\n"
            
            # Specific recommendations based on failed checks
            if not results['structure_check']['passed']:
                report += "- Fix directory structure issues\n"
            if not results['content_check']['passed']:
                report += "- Remove remaining development artifacts from content\n"
            if not results['cleanup_check']['passed']:
                report += "- Complete cleanup of RIF-specific files\n"
            if not results['size_check']['passed']:
                report += "- Reduce size by removing large unnecessary files\n"
            
            report += "\nRe-run cleanup script and validation after addressing issues.\n"
        
        return report


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Validate cleaned knowledge base")
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Path to knowledge directory (default: knowledge)"
    )
    parser.add_argument(
        "--output-report",
        help="Save validation report to file"
    )
    parser.add_argument(
        "--json-output",
        help="Save detailed results as JSON"
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = CleanedKnowledgeValidator(args.knowledge_dir)
    results = validator.run_full_validation()
    
    # Generate report
    report = validator.generate_validation_report()
    
    # Save outputs if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to: {args.output_report}")
    
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to: {args.json_output}")
    
    # Print report
    print(report)
    
    # Exit with appropriate code
    exit_code = 0 if results['overall']['passed'] else 1
    exit(exit_code)


if __name__ == "__main__":
    main()