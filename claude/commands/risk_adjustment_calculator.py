#!/usr/bin/env python3
"""
Risk Adjustment Calculator - Issue #93 Phase 1
Analyzes change characteristics to calculate risk multipliers for quality scoring.

This module provides detailed risk assessment based on:
- Change size and scope
- File types and patterns affected
- Historical failure patterns
- Test coverage characteristics
"""

import os
import json
import yaml
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import re

@dataclass
class ChangeAnalysis:
    """Analysis of code changes for risk assessment."""
    files_modified: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    lines_modified: int = 0
    total_changes: int = 0
    change_complexity: str = "low"  # low, medium, high, very_high
    
@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    risk_multiplier: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    risk_factors: Dict[str, Any] = field(default_factory=dict)
    risk_details: Dict[str, float] = field(default_factory=dict)
    mitigation_suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0

class RiskAdjustmentCalculator:
    """
    Calculates risk multipliers for quality scoring based on change characteristics.
    
    Risk factors include:
    - Change size and complexity
    - Security-sensitive file modifications
    - Critical business logic changes
    - Historical failure patterns
    - Test coverage gaps
    - External dependency changes
    """
    
    def __init__(self, config_path: str = "config/quality-dimensions.yaml"):
        """Initialize risk assessment calculator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.risk_config = self.config.get('risk_adjustment', {})
        self.max_risk_multiplier = self.risk_config.get('max_risk_multiplier', 0.3)
        
        # Risk factor configurations
        self.risk_factors_config = self.risk_config.get('risk_factors', {})
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for risk assessment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RiskAdjustmentCalculator - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load risk adjustment configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error(f"Config file {self.config_path} not found")
                return self._get_default_risk_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_risk_config()
    
    def _get_default_risk_config(self) -> Dict[str, Any]:
        """Default risk configuration for fallback."""
        return {
            'risk_adjustment': {
                'max_risk_multiplier': 0.3,
                'risk_factors': {
                    'large_change': {'threshold': 500, 'multiplier': 0.1},
                    'security_files': {'multiplier': 0.15},
                    'critical_paths': {'multiplier': 0.1},
                    'external_dependencies': {'multiplier': 0.05},
                    'previous_failures': {'multiplier': 0.1, 'lookback_days': 30},
                    'no_tests': {'multiplier': 0.2},
                    'low_coverage_area': {'threshold': 50, 'multiplier': 0.1}
                }
            }
        }
    
    def assess_change_risk(
        self, 
        files_modified: List[str],
        issue_number: Optional[int] = None,
        pr_number: Optional[int] = None
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment for code changes.
        
        Args:
            files_modified: List of files modified in the change
            issue_number: GitHub issue number (optional)
            pr_number: GitHub PR number (optional)
            
        Returns:
            Complete risk assessment with multiplier and details
        """
        try:
            # Analyze the changes
            change_analysis = self._analyze_changes(files_modified, pr_number)
            
            # Initialize risk assessment
            risk_assessment = RiskAssessment()
            
            # Calculate individual risk factors
            risk_details = {}
            
            # 1. Large change risk
            large_change_risk = self._assess_large_change_risk(change_analysis)
            risk_details['large_change'] = large_change_risk
            
            # 2. Security files risk
            security_risk = self._assess_security_files_risk(files_modified)
            risk_details['security_files'] = security_risk
            
            # 3. Critical paths risk
            critical_paths_risk = self._assess_critical_paths_risk(files_modified)
            risk_details['critical_paths'] = critical_paths_risk
            
            # 4. External dependencies risk
            dependency_risk = self._assess_dependency_risk(files_modified)
            risk_details['external_dependencies'] = dependency_risk
            
            # 5. Historical failures risk
            history_risk = self._assess_historical_risk(files_modified, issue_number)
            risk_details['previous_failures'] = history_risk
            
            # 6. Test coverage risk
            test_risk = self._assess_test_coverage_risk(files_modified, change_analysis)
            risk_details['no_tests'] = test_risk
            
            # 7. Low coverage area risk
            coverage_risk = self._assess_low_coverage_risk(files_modified)
            risk_details['low_coverage_area'] = coverage_risk
            
            # Calculate total risk multiplier
            total_risk = sum(risk_details.values())
            risk_multiplier = min(total_risk, self.max_risk_multiplier)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_multiplier, risk_details)
            
            # Generate mitigation suggestions
            mitigation_suggestions = self._generate_mitigation_suggestions(risk_details)
            
            # Build final assessment
            risk_assessment.risk_multiplier = risk_multiplier
            risk_assessment.risk_level = risk_level
            risk_assessment.risk_factors = {
                'large_change': large_change_risk > 0,
                'security_files': security_risk > 0,
                'critical_paths': critical_paths_risk > 0,
                'external_dependencies': dependency_risk > 0,
                'previous_failures': history_risk > 0,
                'no_tests': test_risk > 0,
                'low_coverage_area': coverage_risk > 0
            }
            risk_assessment.risk_details = risk_details
            risk_assessment.mitigation_suggestions = mitigation_suggestions
            risk_assessment.confidence = self._calculate_confidence(change_analysis, risk_details)
            
            self.logger.info(f"Risk assessment completed: multiplier={risk_multiplier:.3f}, level={risk_level}")
            return risk_assessment
        
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            # Return minimal risk assessment
            return RiskAssessment(
                risk_multiplier=0.1,  # Conservative fallback
                risk_level="medium",
                risk_factors={'error': True},
                confidence=0.5
            )
    
    def _analyze_changes(self, files_modified: List[str], pr_number: Optional[int]) -> ChangeAnalysis:
        """Analyze the scope and complexity of changes."""
        change_analysis = ChangeAnalysis(files_modified=files_modified)
        
        try:
            # Try to get detailed change statistics from git/GitHub
            if pr_number:
                change_stats = self._get_pr_change_stats(pr_number)
            else:
                change_stats = self._get_git_change_stats(files_modified)
            
            if change_stats:
                change_analysis.lines_added = change_stats.get('additions', 0)
                change_analysis.lines_removed = change_stats.get('deletions', 0)
                change_analysis.total_changes = change_analysis.lines_added + change_analysis.lines_removed
            
            # Determine complexity based on changes
            total_changes = change_analysis.total_changes
            if total_changes < 50:
                change_analysis.change_complexity = "low"
            elif total_changes < 200:
                change_analysis.change_complexity = "medium"
            elif total_changes < 1000:
                change_analysis.change_complexity = "high"
            else:
                change_analysis.change_complexity = "very_high"
        
        except Exception as e:
            self.logger.warning(f"Could not analyze detailed changes: {e}")
            # Fallback: estimate based on file count
            file_count = len(files_modified)
            if file_count < 3:
                change_analysis.change_complexity = "low"
            elif file_count < 10:
                change_analysis.change_complexity = "medium"
            elif file_count < 25:
                change_analysis.change_complexity = "high"
            else:
                change_analysis.change_complexity = "very_high"
        
        return change_analysis
    
    def _get_pr_change_stats(self, pr_number: int) -> Optional[Dict[str, int]]:
        """Get change statistics from GitHub PR."""
        try:
            result = subprocess.run([
                'gh', 'pr', 'view', str(pr_number),
                '--json', 'additions,deletions,changedFiles'
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout)
        except Exception as e:
            self.logger.debug(f"Could not get PR stats: {e}")
            return None
    
    def _get_git_change_stats(self, files_modified: List[str]) -> Optional[Dict[str, int]]:
        """Get change statistics from git diff."""
        try:
            # Get diff stats for modified files
            cmd = ['git', 'diff', '--stat', 'HEAD~1', 'HEAD'] + files_modified
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse git diff --stat output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                summary_line = lines[-1]  # Last line contains summary
                # Parse format: "X files changed, Y insertions(+), Z deletions(-)"
                parts = summary_line.split(',')
                additions = 0
                deletions = 0
                
                for part in parts:
                    if 'insertion' in part:
                        additions = int(re.search(r'(\d+)', part).group(1)) if re.search(r'(\d+)', part) else 0
                    elif 'deletion' in part:
                        deletions = int(re.search(r'(\d+)', part).group(1)) if re.search(r'(\d+)', part) else 0
                
                return {'additions': additions, 'deletions': deletions}
        
        except Exception as e:
            self.logger.debug(f"Could not get git stats: {e}")
            return None
    
    def _assess_large_change_risk(self, change_analysis: ChangeAnalysis) -> float:
        """Assess risk from large changes."""
        config = self.risk_factors_config.get('large_change', {})
        threshold = config.get('threshold', 500)
        multiplier = config.get('multiplier', 0.1)
        
        if change_analysis.total_changes > threshold:
            # Scale risk based on how much larger than threshold
            scale_factor = min(change_analysis.total_changes / threshold, 3.0)  # Cap at 3x
            return multiplier * scale_factor
        
        return 0.0
    
    def _assess_security_files_risk(self, files_modified: List[str]) -> float:
        """Assess risk from security-sensitive file modifications."""
        config = self.risk_factors_config.get('security_files', {})
        patterns = config.get('patterns', [
            '**/auth/**', '**/security/**', '**/login/**', '**/password/**'
        ])
        multiplier = config.get('multiplier', 0.15)
        
        security_files = self._match_file_patterns(files_modified, patterns)
        if security_files:
            # Scale risk based on number of security files
            scale_factor = min(len(security_files), 3) / 3.0  # Cap at 3 files
            return multiplier * (0.5 + 0.5 * scale_factor)  # Minimum 50% of multiplier
        
        return 0.0
    
    def _assess_critical_paths_risk(self, files_modified: List[str]) -> float:
        """Assess risk from critical business logic changes."""
        config = self.risk_factors_config.get('critical_paths', {})
        patterns = config.get('patterns', [
            '**/payment/**', '**/billing/**', '**/api/core/**', '**/database/**'
        ])
        multiplier = config.get('multiplier', 0.1)
        
        critical_files = self._match_file_patterns(files_modified, patterns)
        if critical_files:
            # Scale based on criticality
            scale_factor = min(len(critical_files), 2) / 2.0  # Cap at 2 files
            return multiplier * (0.7 + 0.3 * scale_factor)  # Minimum 70% of multiplier
        
        return 0.0
    
    def _assess_dependency_risk(self, files_modified: List[str]) -> float:
        """Assess risk from external dependency changes."""
        config = self.risk_factors_config.get('external_dependencies', {})
        patterns = config.get('patterns', [
            'package.json', 'requirements.txt', 'go.mod', 'Gemfile'
        ])
        multiplier = config.get('multiplier', 0.05)
        
        dependency_files = self._match_file_patterns(files_modified, patterns)
        if dependency_files:
            return multiplier
        
        return 0.0
    
    def _assess_historical_risk(self, files_modified: List[str], issue_number: Optional[int]) -> float:
        """Assess risk based on historical failures in modified areas."""
        config = self.risk_factors_config.get('previous_failures', {})
        lookback_days = config.get('lookback_days', 30)
        multiplier = config.get('multiplier', 0.1)
        failure_threshold = config.get('failure_count_threshold', 2)
        
        try:
            # Look for recent failures in git history for these files
            since_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            failure_count = 0
            for file_path in files_modified[:10]:  # Limit to avoid performance issues
                # Check for recent reverts or fixes in this file
                cmd = [
                    'git', 'log', '--since', since_date, '--grep', 'fix\\|revert\\|hotfix',
                    '--oneline', '--', file_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    failure_count += len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            if failure_count >= failure_threshold:
                scale_factor = min(failure_count / failure_threshold, 2.0)  # Cap at 2x
                return multiplier * scale_factor
        
        except Exception as e:
            self.logger.debug(f"Could not assess historical risk: {e}")
        
        return 0.0
    
    def _assess_test_coverage_risk(self, files_modified: List[str], change_analysis: ChangeAnalysis) -> float:
        """Assess risk from lack of test additions with code changes."""
        config = self.risk_factors_config.get('no_tests', {})
        multiplier = config.get('multiplier', 0.2)
        
        # Check if any test files were modified
        test_patterns = ['**/test/**', '**/tests/**', '**/*_test.*', '**/*Test.*', '**/spec/**']
        test_files = self._match_file_patterns(files_modified, test_patterns)
        
        # Check if there are significant code changes without test changes
        has_code_changes = any(not self._match_file_patterns([f], test_patterns) for f in files_modified)
        
        if has_code_changes and not test_files:
            # Scale risk based on change size
            if change_analysis.change_complexity in ['high', 'very_high']:
                return multiplier
            elif change_analysis.change_complexity == 'medium':
                return multiplier * 0.7
            else:
                return multiplier * 0.3
        
        return 0.0
    
    def _assess_low_coverage_risk(self, files_modified: List[str]) -> float:
        """Assess risk from changes in low-coverage areas."""
        config = self.risk_factors_config.get('low_coverage_area', {})
        threshold = config.get('threshold', 50)
        multiplier = config.get('multiplier', 0.1)
        
        # This would ideally check actual coverage data
        # For now, we'll use heuristics based on file patterns
        
        # Files that typically have lower coverage
        low_coverage_patterns = [
            '**/config/**', '**/util/**', '**/helper/**', '**/legacy/**'
        ]
        
        low_coverage_files = self._match_file_patterns(files_modified, low_coverage_patterns)
        if low_coverage_files:
            scale_factor = min(len(low_coverage_files) / len(files_modified), 1.0)
            return multiplier * scale_factor
        
        return 0.0
    
    def _match_file_patterns(self, files: List[str], patterns: List[str]) -> List[str]:
        """Match files against glob-like patterns."""
        matched_files = []
        
        for file_path in files:
            for pattern in patterns:
                # Simple pattern matching (could be enhanced with proper glob)
                pattern_regex = pattern.replace('**/', '.*').replace('*', '[^/]*')
                if re.match(pattern_regex, file_path, re.IGNORECASE):
                    matched_files.append(file_path)
                    break
        
        return matched_files
    
    def _determine_risk_level(self, risk_multiplier: float, risk_details: Dict[str, float]) -> str:
        """Determine overall risk level category."""
        if risk_multiplier >= 0.25:
            return "critical"
        elif risk_multiplier >= 0.15:
            return "high"
        elif risk_multiplier >= 0.05:
            return "medium"
        else:
            return "low"
    
    def _generate_mitigation_suggestions(self, risk_details: Dict[str, float]) -> List[str]:
        """Generate risk mitigation suggestions."""
        suggestions = []
        
        if risk_details.get('large_change', 0) > 0:
            suggestions.append("Consider breaking large changes into smaller, more focused PRs")
        
        if risk_details.get('security_files', 0) > 0:
            suggestions.append("Extra security review recommended for auth/security file changes")
            suggestions.append("Consider penetration testing for security-related changes")
        
        if risk_details.get('critical_paths', 0) > 0:
            suggestions.append("Additional testing recommended for critical business logic changes")
            suggestions.append("Consider canary deployment for critical path modifications")
        
        if risk_details.get('external_dependencies', 0) > 0:
            suggestions.append("Verify dependency compatibility and security updates")
            suggestions.append("Check for breaking changes in updated dependencies")
        
        if risk_details.get('previous_failures', 0) > 0:
            suggestions.append("Pay special attention to areas with recent failure history")
            suggestions.append("Consider additional regression testing")
        
        if risk_details.get('no_tests', 0) > 0:
            suggestions.append("Add comprehensive test coverage for new/modified code")
            suggestions.append("Consider property-based testing for complex logic")
        
        if risk_details.get('low_coverage_area', 0) > 0:
            suggestions.append("Increase test coverage in historically under-tested areas")
            suggestions.append("Add integration tests for low-coverage components")
        
        return suggestions
    
    def _calculate_confidence(self, change_analysis: ChangeAnalysis, risk_details: Dict[str, float]) -> float:
        """Calculate confidence level in the risk assessment."""
        confidence = 1.0
        
        # Reduce confidence if we couldn't get detailed change stats
        if change_analysis.total_changes == 0:
            confidence -= 0.2
        
        # Reduce confidence if analysis was limited
        if len(change_analysis.files_modified) > 20:
            confidence -= 0.1  # Large number of files makes analysis less precise
        
        # Increase confidence if multiple risk factors are present
        active_factors = sum(1 for value in risk_details.values() if value > 0)
        if active_factors >= 3:
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))

def main():
    """Command line interface for risk assessment."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Risk Adjustment Calculator')
    parser.add_argument('--config', default='config/quality-dimensions.yaml', help='Configuration file path')
    parser.add_argument('--files', nargs='+', required=True, help='Modified files list')
    parser.add_argument('--issue', type=int, help='GitHub issue number')
    parser.add_argument('--pr', type=int, help='GitHub PR number')
    parser.add_argument('--output', choices=['multiplier', 'level', 'full'], default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = RiskAdjustmentCalculator(config_path=args.config)
    
    # Assess risk
    assessment = calculator.assess_change_risk(
        files_modified=args.files,
        issue_number=args.issue,
        pr_number=args.pr
    )
    
    # Output based on format
    if args.output == 'multiplier':
        print(f"{assessment.risk_multiplier:.3f}")
    elif args.output == 'level':
        print(assessment.risk_level)
    else:
        print(json.dumps({
            'risk_multiplier': assessment.risk_multiplier,
            'risk_level': assessment.risk_level,
            'risk_factors': assessment.risk_factors,
            'risk_details': assessment.risk_details,
            'mitigation_suggestions': assessment.mitigation_suggestions,
            'confidence': assessment.confidence
        }, indent=2))
    
    return 0

if __name__ == "__main__":
    exit(main())