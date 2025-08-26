#!/usr/bin/env python3
"""
Evidence Collection and Validation Framework for GitHub Issue #234
Implements comprehensive evidence gathering with validation gates
"""

import json
import logging
import hashlib
import subprocess
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import importlib.util
import ast
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Types of evidence collected"""
    IMPLEMENTATION = "implementation"
    TESTS = "tests"
    DOCUMENTATION = "documentation"
    COMMITS = "commits"
    PULL_REQUESTS = "pull_requests"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    INTEGRATION = "integration"

class EvidenceQuality(Enum):
    """Quality levels for evidence"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    MISSING = "missing"

@dataclass
class EvidenceItem:
    """Individual piece of evidence"""
    evidence_type: EvidenceType
    file_path: str
    content_hash: str
    size_bytes: int
    last_modified: str
    validation_status: str
    quality_score: float
    metadata: Dict[str, Any]
    collected_at: str

@dataclass
class ValidationResult:
    """Result of evidence validation"""
    is_valid: bool
    confidence: float
    quality: EvidenceQuality
    findings: List[str]
    issues: List[str]
    recommendations: List[str]
    validation_timestamp: str

@dataclass
class EvidencePackage:
    """Complete evidence package for an issue"""
    issue_number: int
    classification: str
    evidence_items: List[EvidenceItem]
    validation_results: Dict[str, ValidationResult]
    overall_quality: EvidenceQuality
    overall_confidence: float
    completeness_score: float
    collection_timestamp: str
    validation_timestamp: str

class EvidenceCollectionFramework:
    """
    Comprehensive evidence collection and validation system
    Implements adversarial verification with quality gates
    """
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = Path(repo_path)
        self.evidence_dir = self.repo_path / "knowledge" / "evidence_collection"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Evidence validation thresholds
        self.quality_thresholds = {
            "excellent": 90.0,
            "good": 75.0,
            "fair": 60.0,
            "poor": 40.0
        }
        
        # File type patterns
        self.code_extensions = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h'}
        self.test_patterns = ['test_', '_test.', 'tests/', '/test/', 'spec_', '_spec.']
        self.doc_extensions = {'.md', '.txt', '.rst', '.adoc'}
        
        logger.info(f"Evidence collection framework initialized for repo: {self.repo_path}")

    def collect_comprehensive_evidence(self, issue_number: int, classification: str) -> EvidencePackage:
        """
        Main method to collect comprehensive evidence for an issue
        Returns complete evidence package with validation results
        """
        logger.info(f"Starting comprehensive evidence collection for issue #{issue_number}")
        
        evidence_items = []
        collection_start = datetime.now()
        
        try:
            # Collect different types of evidence
            evidence_items.extend(self._collect_implementation_evidence(issue_number))
            evidence_items.extend(self._collect_test_evidence(issue_number))
            evidence_items.extend(self._collect_documentation_evidence(issue_number))
            evidence_items.extend(self._collect_commit_evidence(issue_number))
            evidence_items.extend(self._collect_pr_evidence(issue_number))
            evidence_items.extend(self._collect_performance_evidence(issue_number))
            evidence_items.extend(self._collect_quality_evidence(issue_number))
            
            logger.info(f"Collected {len(evidence_items)} evidence items for issue #{issue_number}")
            
            # Validate collected evidence
            validation_results = self._validate_evidence_package(evidence_items, classification)
            
            # Calculate overall quality and confidence
            overall_quality, overall_confidence, completeness_score = self._calculate_overall_metrics(
                validation_results, evidence_items
            )
            
            # Create evidence package
            evidence_package = EvidencePackage(
                issue_number=issue_number,
                classification=classification,
                evidence_items=evidence_items,
                validation_results=validation_results,
                overall_quality=overall_quality,
                overall_confidence=overall_confidence,
                completeness_score=completeness_score,
                collection_timestamp=collection_start.isoformat(),
                validation_timestamp=datetime.now().isoformat()
            )
            
            # Save evidence package
            self._save_evidence_package(evidence_package)
            
            logger.info(f"Evidence collection complete for issue #{issue_number}: "
                       f"quality={overall_quality.value}, confidence={overall_confidence:.1f}%")
            
            return evidence_package
            
        except Exception as e:
            logger.error(f"Evidence collection failed for issue #{issue_number}: {e}")
            # Return minimal package with error info
            return EvidencePackage(
                issue_number=issue_number,
                classification=classification,
                evidence_items=evidence_items,
                validation_results={"error": ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    quality=EvidenceQuality.MISSING,
                    findings=[],
                    issues=[f"Collection failed: {e}"],
                    recommendations=["Manual investigation required"],
                    validation_timestamp=datetime.now().isoformat()
                )},
                overall_quality=EvidenceQuality.MISSING,
                overall_confidence=0.0,
                completeness_score=0.0,
                collection_timestamp=collection_start.isoformat(),
                validation_timestamp=datetime.now().isoformat()
            )

    def _collect_implementation_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect implementation files related to issue"""
        evidence_items = []
        
        try:
            # Search patterns for issue references
            search_patterns = [
                f"issue-{issue_number}",
                f"issue_{issue_number}",
                f"#{issue_number}",
                f"issue{issue_number}",
                f"ISSUE_{issue_number}",
                f"Issue {issue_number}"
            ]
            
            found_files = set()
            
            for pattern in search_patterns:
                # Search in code files
                result = subprocess.run([
                    "grep", "-r", "-l", pattern, str(self.repo_path),
                    "--include=*.py", "--include=*.js", "--include=*.ts", 
                    "--include=*.java", "--include=*.go", "--exclude-dir=.git"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                    found_files.update(files)
            
            # Create evidence items for found files
            for file_path in found_files:
                try:
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        # Verify it's actually a code file
                        if any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.go']):
                            evidence_item = self._create_evidence_item(
                                file_path, EvidenceType.IMPLEMENTATION
                            )
                            if evidence_item:
                                evidence_items.append(evidence_item)
                                
                except Exception as e:
                    logger.warning(f"Failed to process implementation file {file_path}: {e}")
                    
            logger.debug(f"Found {len(evidence_items)} implementation evidence items")
            
        except Exception as e:
            logger.error(f"Implementation evidence collection failed: {e}")
            
        return evidence_items

    def _collect_test_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect test files related to issue"""
        evidence_items = []
        
        try:
            # Search in test directories
            test_dirs = ['tests', 'test', '__tests__', 'spec', 'testing']
            
            for test_dir in test_dirs:
                test_path = self.repo_path / test_dir
                if test_path.exists():
                    # Find test files mentioning the issue
                    result = subprocess.run([
                        "grep", "-r", "-l", f"#{issue_number}", str(test_path),
                        "--include=*.py", "--include=*.js", "--include=*.ts"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                        
                        for file_path in files:
                            evidence_item = self._create_evidence_item(
                                file_path, EvidenceType.TESTS
                            )
                            if evidence_item:
                                evidence_items.append(evidence_item)
            
            # Also search for test files with issue number in filename
            result = subprocess.run([
                "find", str(self.repo_path), "-name", f"*{issue_number}*",
                "-name", "*.py", "-o", "-name", "*.js", "-o", "-name", "*.ts"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                for file_path in files:
                    # Check if it looks like a test file
                    if any(pattern in file_path.lower() for pattern in self.test_patterns):
                        evidence_item = self._create_evidence_item(file_path, EvidenceType.TESTS)
                        if evidence_item:
                            evidence_items.append(evidence_item)
                            
            logger.debug(f"Found {len(evidence_items)} test evidence items")
            
        except Exception as e:
            logger.error(f"Test evidence collection failed: {e}")
            
        return evidence_items

    def _collect_documentation_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect documentation related to issue"""
        evidence_items = []
        
        try:
            # Search for documentation files mentioning the issue
            result = subprocess.run([
                "grep", "-r", "-l", f"#{issue_number}", str(self.repo_path),
                "--include=*.md", "--include=*.txt", "--include=*.rst", 
                "--exclude-dir=.git"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                
                for file_path in files:
                    evidence_item = self._create_evidence_item(
                        file_path, EvidenceType.DOCUMENTATION
                    )
                    if evidence_item:
                        evidence_items.append(evidence_item)
                        
            logger.debug(f"Found {len(evidence_items)} documentation evidence items")
            
        except Exception as e:
            logger.error(f"Documentation evidence collection failed: {e}")
            
        return evidence_items

    def _collect_commit_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect commit history evidence"""
        evidence_items = []
        
        try:
            # Search git history for commits mentioning the issue
            search_patterns = [f"#{issue_number}", f"issue {issue_number}", f"issue-{issue_number}"]
            
            for pattern in search_patterns:
                result = subprocess.run([
                    "git", "log", "--grep", pattern, "--oneline", "--all"
                ], capture_output=True, text=True, cwd=self.repo_path)
                
                if result.returncode == 0 and result.stdout.strip():
                    # Create virtual evidence item for commit history
                    commit_data = {
                        "commits": result.stdout.strip().split('\n'),
                        "search_pattern": pattern,
                        "commit_count": len(result.stdout.strip().split('\n'))
                    }
                    
                    evidence_item = EvidenceItem(
                        evidence_type=EvidenceType.COMMITS,
                        file_path=f"git_commits_{issue_number}_{pattern}",
                        content_hash=hashlib.md5(result.stdout.encode()).hexdigest(),
                        size_bytes=len(result.stdout),
                        last_modified=datetime.now().isoformat(),
                        validation_status="pending",
                        quality_score=0.0,
                        metadata=commit_data,
                        collected_at=datetime.now().isoformat()
                    )
                    
                    evidence_items.append(evidence_item)
                    
            logger.debug(f"Found {len(evidence_items)} commit evidence items")
            
        except Exception as e:
            logger.error(f"Commit evidence collection failed: {e}")
            
        return evidence_items

    def _collect_pr_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect pull request evidence"""
        evidence_items = []
        
        try:
            # Use gh CLI to find PRs mentioning the issue
            result = subprocess.run([
                "gh", "pr", "list", "--state", "all", "--search", f"#{issue_number}",
                "--json", "number,title,state,mergedAt,body"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                prs = json.loads(result.stdout)
                
                if prs:
                    pr_data = {
                        "pull_requests": prs,
                        "pr_count": len(prs),
                        "merged_count": len([pr for pr in prs if pr.get('state') == 'MERGED'])
                    }
                    
                    evidence_item = EvidenceItem(
                        evidence_type=EvidenceType.PULL_REQUESTS,
                        file_path=f"pull_requests_{issue_number}",
                        content_hash=hashlib.md5(json.dumps(prs).encode()).hexdigest(),
                        size_bytes=len(json.dumps(prs)),
                        last_modified=datetime.now().isoformat(),
                        validation_status="pending",
                        quality_score=0.0,
                        metadata=pr_data,
                        collected_at=datetime.now().isoformat()
                    )
                    
                    evidence_items.append(evidence_item)
                    
            logger.debug(f"Found {len(evidence_items)} PR evidence items")
            
        except Exception as e:
            logger.error(f"PR evidence collection failed: {e}")
            
        return evidence_items

    def _collect_performance_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect performance-related evidence"""
        evidence_items = []
        
        try:
            # Search for performance/benchmark files
            perf_patterns = ["benchmark", "performance", "speed", "timing", "metrics"]
            
            for pattern in perf_patterns:
                result = subprocess.run([
                    "find", str(self.repo_path), "-name", f"*{pattern}*",
                    "-name", "*.py", "-o", "-name", "*.js", "-o", "-name", "*.json"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                    
                    for file_path in files:
                        # Check if file mentions the issue
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if f"#{issue_number}" in content or f"issue-{issue_number}" in content:
                                    evidence_item = self._create_evidence_item(
                                        file_path, EvidenceType.PERFORMANCE
                                    )
                                    if evidence_item:
                                        evidence_items.append(evidence_item)
                        except Exception:
                            continue
                            
            logger.debug(f"Found {len(evidence_items)} performance evidence items")
            
        except Exception as e:
            logger.error(f"Performance evidence collection failed: {e}")
            
        return evidence_items

    def _collect_quality_evidence(self, issue_number: int) -> List[EvidenceItem]:
        """Collect quality-related evidence (tests, linting, etc.)"""
        evidence_items = []
        
        try:
            # Look for quality configuration files that might reference the issue
            quality_files = [
                ".github/workflows/*.yml", 
                "pytest.ini", "setup.cfg", "tox.ini",
                ".eslintrc*", ".prettierrc*",
                "quality/", "reports/"
            ]
            
            for pattern in quality_files:
                result = subprocess.run([
                    "find", str(self.repo_path), "-path", pattern, "-o", "-name", pattern
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                    
                    for file_path in files:
                        try:
                            if os.path.isfile(file_path):
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if f"#{issue_number}" in content:
                                        evidence_item = self._create_evidence_item(
                                            file_path, EvidenceType.QUALITY
                                        )
                                        if evidence_item:
                                            evidence_items.append(evidence_item)
                        except Exception:
                            continue
                            
            logger.debug(f"Found {len(evidence_items)} quality evidence items")
            
        except Exception as e:
            logger.error(f"Quality evidence collection failed: {e}")
            
        return evidence_items

    def _create_evidence_item(self, file_path: str, evidence_type: EvidenceType) -> Optional[EvidenceItem]:
        """Create evidence item from file path"""
        try:
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                return None
                
            stat = path.stat()
            
            # Calculate content hash
            with open(path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
                
            # Extract metadata based on file type
            metadata = self._extract_file_metadata(path, evidence_type)
            
            return EvidenceItem(
                evidence_type=evidence_type,
                file_path=str(path),
                content_hash=content_hash,
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                validation_status="pending",
                quality_score=0.0,
                metadata=metadata,
                collected_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"Failed to create evidence item for {file_path}: {e}")
            return None

    def _extract_file_metadata(self, path: Path, evidence_type: EvidenceType) -> Dict[str, Any]:
        """Extract metadata specific to file type and evidence type"""
        metadata = {
            "file_extension": path.suffix,
            "file_name": path.name,
            "relative_path": str(path.relative_to(self.repo_path)) if path.is_relative_to(self.repo_path) else str(path)
        }
        
        try:
            if evidence_type == EvidenceType.IMPLEMENTATION and path.suffix == '.py':
                # Python-specific metadata
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Count functions, classes, lines of code
                tree = ast.parse(content)
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                metadata.update({
                    "language": "python",
                    "function_count": len(functions),
                    "class_count": len(classes),
                    "lines_of_code": len(content.split('\n')),
                    "has_docstring": any(ast.get_docstring(node) for node in [tree] + classes + functions if ast.get_docstring(node))
                })
                
            elif evidence_type == EvidenceType.TESTS:
                # Test-specific metadata
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Count test functions/methods
                if path.suffix == '.py':
                    test_functions = len(re.findall(r'def test_\w+', content))
                    assert_statements = len(re.findall(r'assert\s+', content))
                    
                    metadata.update({
                        "test_function_count": test_functions,
                        "assert_count": assert_statements,
                        "test_framework": self._detect_test_framework(content)
                    })
                    
            elif evidence_type == EvidenceType.DOCUMENTATION:
                # Documentation-specific metadata
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                metadata.update({
                    "word_count": len(content.split()),
                    "line_count": len(content.split('\n')),
                    "has_code_blocks": '```' in content,
                    "heading_count": len(re.findall(r'^#+\s+', content, re.MULTILINE))
                })
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {path}: {e}")
            
        return metadata

    def _detect_test_framework(self, content: str) -> str:
        """Detect which testing framework is used"""
        if 'import pytest' in content or 'from pytest' in content:
            return 'pytest'
        elif 'import unittest' in content or 'from unittest' in content:
            return 'unittest'
        elif 'describe(' in content and 'it(' in content:
            return 'mocha/jest'
        else:
            return 'unknown'

    def _validate_evidence_package(self, evidence_items: List[EvidenceItem], classification: str) -> Dict[str, ValidationResult]:
        """Validate complete evidence package"""
        validation_results = {}
        
        # Group evidence by type
        evidence_by_type = {}
        for item in evidence_items:
            evidence_type = item.evidence_type.value
            if evidence_type not in evidence_by_type:
                evidence_by_type[evidence_type] = []
            evidence_by_type[evidence_type].append(item)
        
        # Validate each evidence type
        for evidence_type, items in evidence_by_type.items():
            validation_result = self._validate_evidence_type(evidence_type, items, classification)
            validation_results[evidence_type] = validation_result
            
        # Add overall validation
        validation_results['overall'] = self._validate_overall_package(evidence_by_type, classification)
        
        return validation_results

    def _validate_evidence_type(self, evidence_type: str, items: List[EvidenceItem], classification: str) -> ValidationResult:
        """Validate specific type of evidence"""
        findings = []
        issues = []
        recommendations = []
        confidence = 0.0
        
        try:
            if evidence_type == "implementation":
                confidence, findings, issues, recommendations = self._validate_implementation_evidence(items)
                
            elif evidence_type == "tests":
                confidence, findings, issues, recommendations = self._validate_test_evidence(items)
                
            elif evidence_type == "documentation":
                confidence, findings, issues, recommendations = self._validate_documentation_evidence(items)
                
            elif evidence_type == "commits":
                confidence, findings, issues, recommendations = self._validate_commit_evidence(items)
                
            elif evidence_type == "pull_requests":
                confidence, findings, issues, recommendations = self._validate_pr_evidence(items)
                
            else:
                # Generic validation
                confidence = min(75.0, len(items) * 25.0)  # Up to 75% for having evidence
                findings = [f"Found {len(items)} {evidence_type} evidence items"]
                
            # Determine quality based on confidence
            quality = self._confidence_to_quality(confidence)
            
            return ValidationResult(
                is_valid=confidence > 40.0,
                confidence=confidence,
                quality=quality,
                findings=findings,
                issues=issues,
                recommendations=recommendations,
                validation_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {evidence_type}: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                quality=EvidenceQuality.MISSING,
                findings=[],
                issues=[f"Validation error: {e}"],
                recommendations=["Manual validation required"],
                validation_timestamp=datetime.now().isoformat()
            )

    def _validate_implementation_evidence(self, items: List[EvidenceItem]) -> Tuple[float, List[str], List[str], List[str]]:
        """Validate implementation evidence specifically"""
        confidence = 0.0
        findings = []
        issues = []
        recommendations = []
        
        if not items:
            issues.append("No implementation files found")
            recommendations.append("Search for implementation using broader patterns")
            return 0.0, findings, issues, recommendations
            
        # Base confidence for having implementation files
        confidence = 40.0
        findings.append(f"Found {len(items)} implementation files")
        
        # Analyze file quality
        total_size = sum(item.size_bytes for item in items)
        python_files = [item for item in items if item.file_path.endswith('.py')]
        
        if total_size > 1000:  # Non-trivial implementation
            confidence += 20.0
            findings.append(f"Substantial implementation: {total_size} bytes")
        else:
            issues.append("Implementation appears minimal")
            
        # Check for Python-specific quality indicators
        if python_files:
            for item in python_files:
                metadata = item.metadata
                if metadata.get('function_count', 0) > 0:
                    confidence += 10.0
                    findings.append(f"Found {metadata['function_count']} functions in {item.file_path}")
                    
                if metadata.get('class_count', 0) > 0:
                    confidence += 10.0
                    findings.append(f"Found {metadata['class_count']} classes in {item.file_path}")
                    
                if metadata.get('has_docstring', False):
                    confidence += 5.0
                    findings.append("Code includes documentation")
        
        # Functional validation - try to import Python files
        functional_files = 0
        for item in python_files:
            try:
                # Basic syntax check
                with open(item.file_path, 'r') as f:
                    ast.parse(f.read())
                functional_files += 1
            except SyntaxError:
                issues.append(f"Syntax error in {item.file_path}")
            except Exception:
                issues.append(f"Cannot validate {item.file_path}")
        
        if functional_files == len(python_files) and python_files:
            confidence += 15.0
            findings.append("All Python files have valid syntax")
        elif functional_files > 0:
            confidence += 10.0
            findings.append(f"{functional_files}/{len(python_files)} Python files validated")
            
        # Recommendations based on findings
        if not python_files:
            recommendations.append("Verify non-Python implementations manually")
        if len(items) == 1:
            recommendations.append("Look for additional implementation files")
        if total_size < 500:
            recommendations.append("Investigate if implementation is complete")
            
        return min(100.0, confidence), findings, issues, recommendations

    def _validate_test_evidence(self, items: List[EvidenceItem]) -> Tuple[float, List[str], List[str], List[str]]:
        """Validate test evidence specifically"""
        confidence = 0.0
        findings = []
        issues = []
        recommendations = []
        
        if not items:
            issues.append("No test files found")
            recommendations.append("Look for integration tests or manual testing evidence")
            return 0.0, findings, issues, recommendations
            
        confidence = 30.0
        findings.append(f"Found {len(items)} test files")
        
        # Analyze test quality
        total_tests = 0
        total_assertions = 0
        
        for item in items:
            metadata = item.metadata
            test_count = metadata.get('test_function_count', 0)
            assert_count = metadata.get('assert_count', 0)
            
            total_tests += test_count
            total_assertions += assert_count
            
        if total_tests > 0:
            confidence += 30.0
            findings.append(f"Found {total_tests} test functions")
            
        if total_assertions > 0:
            confidence += 20.0
            findings.append(f"Found {total_assertions} assertions")
            
        # Framework detection
        frameworks = set()
        for item in items:
            framework = item.metadata.get('test_framework', 'unknown')
            if framework != 'unknown':
                frameworks.add(framework)
                
        if frameworks:
            confidence += 10.0
            findings.append(f"Test frameworks: {', '.join(frameworks)}")
            
        # Test execution validation would go here
        # For now, we'll assume tests are potentially runnable if they exist
        
        if total_tests == 0:
            issues.append("No test functions found in test files")
            recommendations.append("Verify test files contain actual tests")
        elif total_assertions == 0:
            issues.append("No assertions found in test functions")
            recommendations.append("Check if tests include proper assertions")
            
        return min(100.0, confidence), findings, issues, recommendations

    def _validate_documentation_evidence(self, items: List[EvidenceItem]) -> Tuple[float, List[str], List[str], List[str]]:
        """Validate documentation evidence"""
        confidence = 0.0
        findings = []
        issues = []
        recommendations = []
        
        if not items:
            issues.append("No documentation found")
            recommendations.append("Check for inline documentation or comments")
            return 0.0, findings, issues, recommendations
            
        confidence = 25.0
        findings.append(f"Found {len(items)} documentation files")
        
        total_words = sum(item.metadata.get('word_count', 0) for item in items)
        has_code_examples = any(item.metadata.get('has_code_blocks', False) for item in items)
        total_headings = sum(item.metadata.get('heading_count', 0) for item in items)
        
        if total_words > 100:
            confidence += 25.0
            findings.append(f"Substantial documentation: {total_words} words")
        elif total_words > 50:
            confidence += 15.0
            findings.append(f"Moderate documentation: {total_words} words")
        else:
            issues.append("Documentation appears minimal")
            
        if has_code_examples:
            confidence += 20.0
            findings.append("Documentation includes code examples")
            
        if total_headings > 0:
            confidence += 10.0
            findings.append(f"Well-structured with {total_headings} headings")
            
        if total_words < 50:
            recommendations.append("Expand documentation with more detail")
        if not has_code_examples:
            recommendations.append("Consider adding code examples")
            
        return min(100.0, confidence), findings, issues, recommendations

    def _validate_commit_evidence(self, items: List[EvidenceItem]) -> Tuple[float, List[str], List[str], List[str]]:
        """Validate commit history evidence"""
        confidence = 0.0
        findings = []
        issues = []
        recommendations = []
        
        if not items:
            return 0.0, findings, issues, recommendations
            
        total_commits = sum(item.metadata.get('commit_count', 0) for item in items)
        
        if total_commits > 0:
            confidence = min(80.0, total_commits * 20.0)  # Up to 80% for commit evidence
            findings.append(f"Found {total_commits} relevant commits")
            
            if total_commits >= 3:
                findings.append("Multiple commits suggest substantial work")
            elif total_commits == 1:
                issues.append("Only one commit found - work may be incomplete")
                recommendations.append("Look for additional commits or implementation")
        
        return confidence, findings, issues, recommendations

    def _validate_pr_evidence(self, items: List[EvidenceItem]) -> Tuple[float, List[str], List[str], List[str]]:
        """Validate pull request evidence"""
        confidence = 0.0
        findings = []
        issues = []
        recommendations = []
        
        if not items:
            return 0.0, findings, issues, recommendations
            
        for item in items:
            metadata = item.metadata
            pr_count = metadata.get('pr_count', 0)
            merged_count = metadata.get('merged_count', 0)
            
            if pr_count > 0:
                confidence += 40.0
                findings.append(f"Found {pr_count} related pull requests")
                
                if merged_count > 0:
                    confidence += 30.0
                    findings.append(f"{merged_count} PRs were merged")
                else:
                    issues.append("No merged PRs found")
                    recommendations.append("Verify implementation was actually merged")
                    
                if merged_count < pr_count:
                    findings.append(f"{pr_count - merged_count} PRs not merged (may be drafts or closed)")
        
        return min(100.0, confidence), findings, issues, recommendations

    def _validate_overall_package(self, evidence_by_type: Dict[str, List[EvidenceItem]], classification: str) -> ValidationResult:
        """Validate overall evidence package completeness"""
        findings = []
        issues = []
        recommendations = []
        
        # Required evidence types by classification
        required_evidence = {
            "current_feature": ["implementation", "tests"],
            "bug_fix": ["implementation", "tests"],
            "enhancement": ["implementation"],
            "test_issue": ["tests"],
            "documentation": ["documentation"],
            "obsolete_issue": []  # Flexible requirements
        }
        
        required = required_evidence.get(classification, ["implementation"])
        
        # Check for required evidence
        missing_required = []
        present_evidence = set(evidence_by_type.keys())
        
        for req in required:
            if req not in present_evidence:
                missing_required.append(req)
                
        # Calculate completeness confidence
        if not missing_required:
            confidence = 90.0
            findings.append("All required evidence types present")
        else:
            confidence = 60.0 - (len(missing_required) * 15.0)
            issues.extend([f"Missing {req} evidence" for req in missing_required])
            recommendations.extend([f"Collect {req} evidence" for req in missing_required])
            
        # Bonus for additional evidence types
        bonus_types = ["commits", "pull_requests", "performance", "quality"]
        present_bonus = [t for t in bonus_types if t in present_evidence]
        
        if present_bonus:
            confidence += len(present_bonus) * 5.0  # 5% per bonus type
            findings.append(f"Additional evidence: {', '.join(present_bonus)}")
            
        quality = self._confidence_to_quality(confidence)
        
        return ValidationResult(
            is_valid=confidence > 50.0,
            confidence=min(100.0, confidence),
            quality=quality,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            validation_timestamp=datetime.now().isoformat()
        )

    def _confidence_to_quality(self, confidence: float) -> EvidenceQuality:
        """Convert confidence score to quality level"""
        if confidence >= self.quality_thresholds["excellent"]:
            return EvidenceQuality.EXCELLENT
        elif confidence >= self.quality_thresholds["good"]:
            return EvidenceQuality.GOOD
        elif confidence >= self.quality_thresholds["fair"]:
            return EvidenceQuality.FAIR
        elif confidence >= self.quality_thresholds["poor"]:
            return EvidenceQuality.POOR
        else:
            return EvidenceQuality.MISSING

    def _calculate_overall_metrics(self, validation_results: Dict[str, ValidationResult], evidence_items: List[EvidenceItem]) -> Tuple[EvidenceQuality, float, float]:
        """Calculate overall quality, confidence, and completeness scores"""
        
        # Calculate weighted confidence
        evidence_weights = {
            "implementation": 0.30,
            "tests": 0.25,
            "documentation": 0.20,
            "commits": 0.10,
            "pull_requests": 0.10,
            "overall": 0.05
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for evidence_type, result in validation_results.items():
            weight = evidence_weights.get(evidence_type, 0.05)
            total_confidence += result.confidence * weight
            total_weight += weight
            
        overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        # Calculate completeness score (different from confidence)
        evidence_types_present = len(set(item.evidence_type.value for item in evidence_items))
        total_possible_types = len(EvidenceType)
        completeness_score = (evidence_types_present / total_possible_types) * 100
        
        # Overall quality based on confidence
        overall_quality = self._confidence_to_quality(overall_confidence)
        
        return overall_quality, overall_confidence, completeness_score

    def _save_evidence_package(self, package: EvidencePackage) -> None:
        """Save evidence package to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_package_issue_{package.issue_number}_{timestamp}.json"
        filepath = self.evidence_dir / filename
        
        # Convert to dict for JSON serialization
        package_dict = {
            "issue_number": package.issue_number,
            "classification": package.classification,
            "evidence_items": [
                {
                    "evidence_type": item.evidence_type.value,
                    "file_path": item.file_path,
                    "content_hash": item.content_hash,
                    "size_bytes": item.size_bytes,
                    "last_modified": item.last_modified,
                    "validation_status": item.validation_status,
                    "quality_score": item.quality_score,
                    "metadata": item.metadata,
                    "collected_at": item.collected_at
                } for item in package.evidence_items
            ],
            "validation_results": {
                k: {
                    "is_valid": v.is_valid,
                    "confidence": v.confidence,
                    "quality": v.quality.value,
                    "findings": v.findings,
                    "issues": v.issues,
                    "recommendations": v.recommendations,
                    "validation_timestamp": v.validation_timestamp
                } for k, v in package.validation_results.items()
            },
            "overall_quality": package.overall_quality.value,
            "overall_confidence": package.overall_confidence,
            "completeness_score": package.completeness_score,
            "collection_timestamp": package.collection_timestamp,
            "validation_timestamp": package.validation_timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(package_dict, f, indent=2)
            
        logger.info(f"Evidence package saved: {filepath}")

    def create_audit_evidence_report(self, package: EvidencePackage) -> str:
        """Generate human-readable audit evidence report"""
        
        report = f"""
# Evidence Collection Report - Issue #{package.issue_number}

**Classification**: {package.classification}
**Overall Quality**: {package.overall_quality.value.title()}
**Confidence**: {package.overall_confidence:.1f}%
**Completeness**: {package.completeness_score:.1f}%

## Evidence Summary

**Total Evidence Items**: {len(package.evidence_items)}

### Evidence by Type:
"""
        
        # Group evidence by type for reporting
        evidence_by_type = {}
        for item in package.evidence_items:
            evidence_type = item.evidence_type.value
            if evidence_type not in evidence_by_type:
                evidence_by_type[evidence_type] = []
            evidence_by_type[evidence_type].append(item)
            
        for evidence_type, items in evidence_by_type.items():
            report += f"\n**{evidence_type.title()}**: {len(items)} items\n"
            
            for item in items[:3]:  # Show first 3 items
                report += f"  - {item.file_path} ({item.size_bytes} bytes)\n"
                
            if len(items) > 3:
                report += f"  - ... and {len(items) - 3} more\n"
        
        report += "\n## Validation Results\n"
        
        for evidence_type, result in package.validation_results.items():
            report += f"\n### {evidence_type.title()}\n"
            report += f"- **Quality**: {result.quality.value.title()}\n"
            report += f"- **Confidence**: {result.confidence:.1f}%\n"
            report += f"- **Valid**: {'Yes' if result.is_valid else 'No'}\n"
            
            if result.findings:
                report += "- **Findings**:\n"
                for finding in result.findings:
                    report += f"  - {finding}\n"
                    
            if result.issues:
                report += "- **Issues**:\n"
                for issue in result.issues:
                    report += f"  - {issue}\n"
                    
            if result.recommendations:
                report += "- **Recommendations**:\n"
                for rec in result.recommendations:
                    report += f"  - {rec}\n"
        
        report += f"\n---\n*Report generated at {package.validation_timestamp}*\n"
        
        return report


# Main execution for testing
def main():
    """Test the evidence collection framework"""
    framework = EvidenceCollectionFramework()
    
    # Test with a known issue
    test_issue = 225  # Known issue with implementation
    
    package = framework.collect_comprehensive_evidence(test_issue, "current_feature")
    
    print("="*60)
    print(f"EVIDENCE COLLECTION RESULTS - ISSUE #{test_issue}")
    print("="*60)
    
    print(f"Overall Quality: {package.overall_quality.value.title()}")
    print(f"Overall Confidence: {package.overall_confidence:.1f}%")
    print(f"Completeness Score: {package.completeness_score:.1f}%")
    print(f"Evidence Items: {len(package.evidence_items)}")
    
    # Generate and print report
    report = framework.create_audit_evidence_report(package)
    print(report)


if __name__ == "__main__":
    main()