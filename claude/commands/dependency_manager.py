#!/usr/bin/env python3
"""
RIF Dependency Management System

This module provides dependency parsing and validation for GitHub issues,
ensuring that agents are only launched on issues whose dependencies are satisfied.

CRITICAL: This addresses Issue #143 - RIF Orchestration Dependency Management Failure
"""

import json
import re
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Dependency:
    """Represents a single dependency between GitHub issues"""
    issue_number: int
    dependency_type: str  # 'depends_on', 'blocked_by', 'after', 'requires', 'prerequisite'
    source_line: str      # Original text where dependency was found
    confidence: float     # Confidence level of the dependency detection (0.0-1.0)


@dataclass
class DependencyCheckResult:
    """Result of checking whether an issue's dependencies are satisfied"""
    issue_number: int
    can_proceed: bool
    blocking_dependencies: List[Dependency]
    satisfied_dependencies: List[Dependency]
    check_timestamp: str
    reason: Optional[str] = None


class DependencyParser:
    """
    Parses dependency declarations from GitHub issue bodies using configurable patterns.
    Supports 15+ common dependency declaration formats.
    """
    
    def __init__(self, patterns_config_path: Optional[str] = None):
        """
        Initialize the dependency parser.
        
        Args:
            patterns_config_path: Path to YAML config file with regex patterns
        """
        self.patterns_config_path = patterns_config_path or "/Users/cal/DEV/RIF/config/dependency-patterns.yaml"
        self.patterns = self._load_patterns()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load dependency patterns from configuration file"""
        try:
            config_path = Path(self.patterns_config_path)
            if not config_path.exists():
                self.logger.warning(f"Patterns config not found at {config_path}, using defaults")
                return self._get_default_patterns()
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('dependency_patterns', self._get_default_patterns())
        
        except Exception as e:
            self.logger.error(f"Error loading dependency patterns: {e}")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Default dependency patterns if config file is not available"""
        return {
            'depends_on': [
                {'pattern': r'depends on:?\s*#?(\d+)', 'confidence': 0.9},
                {'pattern': r'dependency:?\s*#?(\d+)', 'confidence': 0.9},
                {'pattern': r'needs:?\s*#?(\d+)', 'confidence': 0.8}
            ],
            'blocked_by': [
                {'pattern': r'blocked by:?\s*#?(\d+)', 'confidence': 0.95},
                {'pattern': r'waiting for:?\s*#?(\d+)', 'confidence': 0.8},
                {'pattern': r'cannot proceed until:?\s*#?(\d+)', 'confidence': 0.9}
            ],
            'after': [
                {'pattern': r'after:?\s*#?(\d+)', 'confidence': 0.85},
                {'pattern': r'following:?\s*#?(\d+)', 'confidence': 0.8},
                {'pattern': r'once:?\s*#?(\d+)', 'confidence': 0.7}
            ],
            'requires': [
                {'pattern': r'requires:?\s*#?(\d+)', 'confidence': 0.9},
                {'pattern': r'requirement:?\s*#?(\d+)', 'confidence': 0.85},
                {'pattern': r'must have:?\s*#?(\d+)', 'confidence': 0.8}
            ],
            'prerequisite': [
                {'pattern': r'prerequisite:?\s*#?(\d+)', 'confidence': 0.95},
                {'pattern': r'pre-req:?\s*#?(\d+)', 'confidence': 0.9},
                {'pattern': r'prerequisite:?\s*issue\s*#?(\d+)', 'confidence': 0.95}
            ],
            'implementation_dependencies': [
                {'pattern': r'Implementation Dependencies:.*?Issue #(\d+)', 'confidence': 0.95},
                {'pattern': r'Implementation Dependencies:.*?#(\d+)', 'confidence': 0.9},
                {'pattern': r'Impl\s+Deps?:.*?#(\d+)', 'confidence': 0.8}
            ]
        }
    
    def parse_dependencies(self, issue_body: str, issue_number: int) -> List[Dependency]:
        """
        Parse all dependencies from an issue body.
        
        Args:
            issue_body: The GitHub issue body text
            issue_number: The issue number (for logging)
            
        Returns:
            List of Dependency objects found in the issue body
        """
        dependencies = []
        
        if not issue_body:
            return dependencies
        
        lines = issue_body.split('\n')
        
        for line_num, line in enumerate(lines):
            line_dependencies = self._parse_line_dependencies(line, line_num)
            dependencies.extend(line_dependencies)
        
        # Remove duplicates while preserving highest confidence
        unique_dependencies = {}
        for dep in dependencies:
            key = (dep.issue_number, dep.dependency_type)
            if key not in unique_dependencies or dep.confidence > unique_dependencies[key].confidence:
                unique_dependencies[key] = dep
        
        result = list(unique_dependencies.values())
        
        self.logger.info(f"Found {len(result)} dependencies for issue #{issue_number}")
        return result
    
    def _parse_line_dependencies(self, line: str, line_num: int) -> List[Dependency]:
        """Parse dependencies from a single line of text"""
        dependencies = []
        line_lower = line.lower().strip()
        
        for dep_type, pattern_configs in self.patterns.items():
            for pattern_config in pattern_configs:
                pattern = pattern_config['pattern']
                confidence = pattern_config['confidence']
                
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        issue_num = int(match.group(1))
                        dependency = Dependency(
                            issue_number=issue_num,
                            dependency_type=dep_type,
                            source_line=line.strip(),
                            confidence=confidence
                        )
                        dependencies.append(dependency)
                        
                        self.logger.debug(f"Found {dep_type} dependency #{issue_num} on line {line_num}: {line.strip()}")
                    
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing dependency from line {line_num}: {e}")
        
        return dependencies
    
    def validate_patterns(self) -> Dict[str, bool]:
        """Validate that all regex patterns compile correctly"""
        results = {}
        
        for dep_type, pattern_configs in self.patterns.items():
            for i, pattern_config in enumerate(pattern_configs):
                pattern_key = f"{dep_type}_{i}"
                try:
                    re.compile(pattern_config['pattern'])
                    results[pattern_key] = True
                except re.error as e:
                    self.logger.error(f"Invalid regex pattern {pattern_key}: {e}")
                    results[pattern_key] = False
        
        return results


class DependencyChecker:
    """
    Checks whether issue dependencies are satisfied by querying GitHub API.
    Provides fast dependency resolution with caching.
    """
    
    def __init__(self, cache_timeout_minutes: int = 5):
        """
        Initialize the dependency checker.
        
        Args:
            cache_timeout_minutes: How long to cache GitHub issue data
        """
        self.cache_timeout = cache_timeout_minutes * 60  # Convert to seconds
        self.issue_cache = {}  # Cache for GitHub issue data
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_dependencies(self, issue_number: int, dependencies: List[Dependency]) -> DependencyCheckResult:
        """
        Check if all dependencies for an issue are satisfied.
        
        Args:
            issue_number: The issue number being checked
            dependencies: List of dependencies to validate
            
        Returns:
            DependencyCheckResult with satisfaction status
        """
        start_time = datetime.now()
        
        satisfied = []
        blocking = []
        
        for dependency in dependencies:
            if self._is_dependency_satisfied(dependency):
                satisfied.append(dependency)
            else:
                blocking.append(dependency)
        
        can_proceed = len(blocking) == 0
        
        reason = None
        if not can_proceed:
            blocked_issues = [str(dep.issue_number) for dep in blocking]
            reason = f"Blocked by issues: {', '.join(blocked_issues)}"
        
        end_time = datetime.now()
        check_duration = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
        
        self.logger.info(f"Dependency check for issue #{issue_number} completed in {check_duration:.1f}ms - Can proceed: {can_proceed}")
        
        return DependencyCheckResult(
            issue_number=issue_number,
            can_proceed=can_proceed,
            blocking_dependencies=blocking,
            satisfied_dependencies=satisfied,
            check_timestamp=datetime.now().isoformat(),
            reason=reason
        )
    
    def _is_dependency_satisfied(self, dependency: Dependency) -> bool:
        """
        Check if a single dependency is satisfied.
        
        Args:
            dependency: The dependency to check
            
        Returns:
            True if dependency is satisfied, False otherwise
        """
        try:
            issue_data = self._get_issue_data(dependency.issue_number)
            
            if not issue_data:
                self.logger.warning(f"Could not fetch data for dependency issue #{dependency.issue_number}")
                return False
            
            # Check if issue is closed
            if issue_data.get('state') == 'CLOSED':
                return True
            
            # Check if issue has state:complete label
            labels = [label.get('name', '') for label in issue_data.get('labels', [])]
            if 'state:complete' in labels:
                return True
            
            # Issue is open and not complete
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking dependency #{dependency.issue_number}: {e}")
            return False
    
    def _get_issue_data(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """
        Get issue data from GitHub API with caching.
        
        Args:
            issue_number: Issue number to fetch
            
        Returns:
            Issue data dict or None if fetch fails
        """
        # Check cache first
        now = datetime.now().timestamp()
        cache_key = issue_number
        
        if cache_key in self.issue_cache:
            cached_data, timestamp = self.issue_cache[cache_key]
            if now - timestamp < self.cache_timeout:
                return cached_data
        
        # Fetch fresh data from GitHub
        try:
            cmd = f"gh issue view {issue_number} --json number,title,state,labels,closedAt"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.logger.warning(f"Failed to fetch issue #{issue_number}: {result.stderr}")
                return None
            
            issue_data = json.loads(result.stdout)
            
            # Cache the data
            self.issue_cache[cache_key] = (issue_data, now)
            
            return issue_data
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout fetching issue #{issue_number}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON for issue #{issue_number}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching issue #{issue_number}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the issue data cache"""
        self.issue_cache.clear()
        self.logger.info("Dependency checker cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            'cache_size': len(self.issue_cache),
            'cache_timeout_seconds': self.cache_timeout
        }


class DependencyManager:
    """
    High-level dependency management interface that combines parsing and checking.
    This is the main class that orchestration utilities should use.
    """
    
    def __init__(self, patterns_config_path: Optional[str] = None):
        """
        Initialize the dependency manager.
        
        Args:
            patterns_config_path: Path to dependency patterns configuration
        """
        self.parser = DependencyParser(patterns_config_path)
        self.checker = DependencyChecker()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def can_work_on_issue(self, issue_number: int) -> Tuple[bool, Optional[str], DependencyCheckResult]:
        """
        Check if an issue can be worked on (all dependencies satisfied).
        
        Args:
            issue_number: GitHub issue number to check
            
        Returns:
            Tuple of (can_proceed, reason_if_blocked, detailed_check_result)
        """
        try:
            # Get issue data
            issue_data = self._get_issue_data_for_parsing(issue_number)
            if not issue_data:
                return False, f"Could not fetch issue #{issue_number}", None
            
            # Parse dependencies from issue body
            dependencies = self.parser.parse_dependencies(issue_data.get('body', ''), issue_number)
            
            # If no dependencies, can proceed
            if not dependencies:
                result = DependencyCheckResult(
                    issue_number=issue_number,
                    can_proceed=True,
                    blocking_dependencies=[],
                    satisfied_dependencies=[],
                    check_timestamp=datetime.now().isoformat(),
                    reason="No dependencies found"
                )
                return True, None, result
            
            # Check dependencies
            result = self.checker.check_dependencies(issue_number, dependencies)
            
            return result.can_proceed, result.reason, result
            
        except Exception as e:
            self.logger.error(f"Error checking if can work on issue #{issue_number}: {e}")
            return False, f"Error checking dependencies: {e}", None
    
    def _get_issue_data_for_parsing(self, issue_number: int) -> Optional[Dict[str, Any]]:
        """Get issue data specifically for dependency parsing"""
        try:
            cmd = f"gh issue view {issue_number} --json number,title,body,state,labels"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to fetch issue #{issue_number}: {result.stderr}")
                return None
            
            return json.loads(result.stdout)
            
        except Exception as e:
            self.logger.error(f"Error fetching issue #{issue_number} for parsing: {e}")
            return None
    
    def add_blocked_label(self, issue_number: int, reason: str) -> bool:
        """
        Add 'blocked' label to a GitHub issue.
        
        Args:
            issue_number: Issue to label
            reason: Reason for blocking
            
        Returns:
            True if successful
        """
        try:
            # Add blocked label
            cmd_label = f"gh issue edit {issue_number} --add-label 'blocked'"
            result = subprocess.run(cmd_label, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to add blocked label to issue #{issue_number}: {result.stderr}")
                return False
            
            # Add comment explaining the block
            comment = f"⚠️ **Issue Blocked**\n\n{reason}\n\nThis issue will be automatically unblocked when dependencies are satisfied."
            cmd_comment = f"gh issue comment {issue_number} --body {json.dumps(comment)}"
            subprocess.run(cmd_comment, shell=True, capture_output=True)
            
            self.logger.info(f"Added blocked label to issue #{issue_number}: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding blocked label to issue #{issue_number}: {e}")
            return False
    
    def remove_blocked_label(self, issue_number: int) -> bool:
        """
        Remove 'blocked' label from a GitHub issue.
        
        Args:
            issue_number: Issue to unblock
            
        Returns:
            True if successful
        """
        try:
            cmd = f"gh issue edit {issue_number} --remove-label 'blocked'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Label might not exist, which is okay
                if "not found" not in result.stderr.lower():
                    self.logger.warning(f"Could not remove blocked label from issue #{issue_number}: {result.stderr}")
                return True
            
            # Add comment that issue is unblocked
            comment = "✅ **Issue Unblocked**\n\nAll dependencies have been satisfied. Work can now proceed on this issue."
            cmd_comment = f"gh issue comment {issue_number} --body {json.dumps(comment)}"
            subprocess.run(cmd_comment, shell=True, capture_output=True)
            
            self.logger.info(f"Removed blocked label from issue #{issue_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing blocked label from issue #{issue_number}: {e}")
            return False
    
    def get_blocked_issues(self) -> List[int]:
        """
        Get all issues that are currently blocked.
        
        Returns:
            List of blocked issue numbers
        """
        try:
            cmd = "gh issue list --state open --label 'blocked' --json number --limit 100"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to fetch blocked issues: {result.stderr}")
                return []
            
            issues_data = json.loads(result.stdout)
            return [issue['number'] for issue in issues_data]
            
        except Exception as e:
            self.logger.error(f"Error fetching blocked issues: {e}")
            return []
    
    def check_blocked_issues_for_unblocking(self) -> List[int]:
        """
        Check all blocked issues to see if any can be unblocked.
        
        Returns:
            List of issue numbers that were unblocked
        """
        blocked_issues = self.get_blocked_issues()
        unblocked = []
        
        for issue_number in blocked_issues:
            can_proceed, reason, result = self.can_work_on_issue(issue_number)
            
            if can_proceed:
                if self.remove_blocked_label(issue_number):
                    unblocked.append(issue_number)
                    self.logger.info(f"Unblocked issue #{issue_number}")
        
        return unblocked
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            'parser_patterns_count': sum(len(patterns) for patterns in self.parser.patterns.values()),
            'checker_cache_stats': self.checker.get_cache_stats(),
            'patterns_valid': all(self.parser.validate_patterns().values())
        }


# Convenience function for orchestration utilities
def create_dependency_manager() -> DependencyManager:
    """Create a dependency manager with default configuration"""
    return DependencyManager()


# Example usage and testing functions
def main():
    """Example usage of the dependency management system"""
    print("RIF Dependency Management System")
    print("=" * 50)
    
    manager = create_dependency_manager()
    
    # Test with issue #143
    can_proceed, reason, result = manager.can_work_on_issue(143)
    print(f"\nIssue #143 dependency check:")
    print(f"Can proceed: {can_proceed}")
    if reason:
        print(f"Reason: {reason}")
    
    if result:
        print(f"Dependencies found: {len(result.satisfied_dependencies + result.blocking_dependencies)}")
        print(f"Blocking dependencies: {len(result.blocking_dependencies)}")
        print(f"Satisfied dependencies: {len(result.satisfied_dependencies)}")
    
    # Performance metrics
    metrics = manager.get_performance_metrics()
    print(f"\nPerformance metrics:")
    print(f"Pattern count: {metrics['parser_patterns_count']}")
    print(f"Cache size: {metrics['checker_cache_stats']['cache_size']}")
    print(f"Patterns valid: {metrics['patterns_valid']}")


if __name__ == "__main__":
    main()