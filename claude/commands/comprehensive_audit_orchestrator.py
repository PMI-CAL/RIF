#!/usr/bin/env python3
"""
Comprehensive Audit Orchestrator for GitHub Issue #234
Implements audit orchestration engine with parallel processing capabilities
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditStatus(Enum):
    """Audit status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"

class IssueClassification(Enum):
    """Issue classification types"""
    CURRENT_FEATURE = "current_feature"
    TEST_ISSUE = "test_issue"
    OBSOLETE_ISSUE = "obsolete_issue"
    DOCUMENTATION = "documentation"

@dataclass
class AuditResult:
    """Structure for individual audit results"""
    issue_number: int
    title: str
    classification: IssueClassification
    status: AuditStatus
    confidence: float
    evidence_quality: str
    findings: List[str]
    recommendations: List[str]
    timestamp: str
    audit_agent: str

@dataclass
class BatchAuditProgress:
    """Structure for tracking batch progress"""
    total_issues: int
    completed: int
    in_progress: int
    pending: int
    failed: int
    start_time: str
    estimated_completion: Optional[str] = None

class ComprehensiveAuditOrchestrator:
    """
    Main orchestration engine for systematic audit of closed GitHub issues
    Implements parallel processing with evidence collection and quality validation
    """
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = Path(repo_path)
        self.audit_results: Dict[int, AuditResult] = {}
        self.batch_progress = BatchAuditProgress(
            total_issues=0,
            completed=0, 
            in_progress=0,
            pending=0,
            failed=0,
            start_time=datetime.now().isoformat()
        )
        
        # Create audit directories
        self.audit_dir = self.repo_path / "knowledge" / "audits"
        self.evidence_dir = self.audit_dir / "evidence" 
        self.reports_dir = self.audit_dir / "reports"
        
        for directory in [self.audit_dir, self.evidence_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Audit orchestrator initialized for repo: {self.repo_path}")

    def get_closed_issues(self) -> List[Dict[str, Any]]:
        """
        Fetch all closed GitHub issues using gh CLI
        Returns list of issue dictionaries with metadata
        """
        try:
            logger.info("Fetching closed GitHub issues...")
            result = subprocess.run([
                "gh", "issue", "list", 
                "--state", "closed",
                "--json", "number,title,closedAt,labels,body,comments",
                "--limit", "1000"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                logger.error(f"GitHub CLI error: {result.stderr}")
                return []
                
            issues = json.loads(result.stdout)
            logger.info(f"Retrieved {len(issues)} closed issues")
            return issues
            
        except Exception as e:
            logger.error(f"Failed to fetch closed issues: {e}")
            return []

    def classify_issue(self, issue: Dict[str, Any]) -> IssueClassification:
        """
        Classify issue type based on title, labels, and content
        Uses pattern matching and heuristics for automatic classification
        """
        title = issue.get('title', '').lower()
        labels = [label.get('name', '').lower() for label in issue.get('labels', [])]
        body = issue.get('body', '').lower()
        
        # Check for test-related issues
        test_indicators = ['test', 'testing', 'validation', 'demo', 'example']
        if any(indicator in title for indicator in test_indicators):
            if 'test:' in title or 'testing:' in title:
                return IssueClassification.TEST_ISSUE
                
        # Check for documentation issues  
        doc_indicators = ['documentation', 'readme', 'guide', 'docs', 'manual']
        if any(indicator in title for indicator in doc_indicators):
            return IssueClassification.DOCUMENTATION
            
        # Check for obsolete issues
        obsolete_indicators = ['obsolete', 'deprecated', 'superseded', 'replaced']
        if any(indicator in title or indicator in body for indicator in obsolete_indicators):
            return IssueClassification.OBSOLETE_ISSUE
            
        # Default to current feature for implementation-related issues
        return IssueClassification.CURRENT_FEATURE

    def create_audit_agent_task(self, issue: Dict[str, Any], classification: IssueClassification) -> str:
        """
        Generate Task prompt for specialized audit agent based on issue classification
        Returns formatted prompt for Claude Code Task tool
        """
        issue_number = issue['number']
        title = issue['title']
        
        base_prompt = f"""You are RIF-Validator performing adversarial audit of closed issue #{issue_number}: "{title}".

MANDATORY: Before ANY work, consult official documentation using these tools:
- mcp__rif-knowledge__get_claude_documentation for Claude Code capabilities
- mcp__rif-knowledge__query_knowledge for similar audit patterns  
- mcp__rif-knowledge__check_compatibility for approach validation

AUDIT REQUIREMENTS:
1. REOPEN issue #{issue_number} for independent verification
2. Search codebase for implementation evidence (use Grep, Glob, LS tools)
3. Perform functional testing of claimed features
4. Document findings with concrete evidence
5. Determine final status: VERIFIED/INCOMPLETE/FAILED
6. POST audit summary comment to issue #{issue_number}
7. POST results summary to META issue #234

"""

        if classification == IssueClassification.CURRENT_FEATURE:
            prompt = base_prompt + f"""
ADVERSARIAL AUDIT PROTOCOL for CURRENT FEATURE:
- Challenge every completion claim with evidence requirements
- Create verification tests for claimed functionality
- Search for implementation files and validate they work
- Test integration points and dependencies
- Verify quality gates (tests, documentation, performance)
- Document confidence level (0-100%) with reasoning
- ONLY re-close if implementation fully verified with evidence

EVIDENCE REQUIREMENTS:
- Functional code that can be imported/executed
- Test results showing features work as claimed
- Integration verification with existing systems  
- Performance metrics if claimed
- Documentation completeness check

FAILURE CONDITIONS:
- Implementation files missing or non-functional
- Tests fail or don't exist
- Claims not supported by evidence
- Integration broken or incomplete

Status: Re-close ONLY with 90%+ confidence and complete evidence package.
"""
        
        elif classification == IssueClassification.TEST_ISSUE:
            prompt = base_prompt + f"""
AUDIT PROTOCOL for TEST ISSUE:
- Verify test purpose and implementation
- Run tests to confirm they work
- Check test coverage and quality
- Validate test integrates with CI/CD
- Quick summary focused on test functionality

EVIDENCE REQUIREMENTS:
- Test files exist and are executable
- Tests pass when run
- Test coverage adequate for purpose
- Test integrated into testing framework

Status: Quick verification and summary - re-close if tests functional.
"""

        elif classification == IssueClassification.OBSOLETE_ISSUE:
            prompt = base_prompt + f"""
AUDIT PROTOCOL for OBSOLETE ISSUE:
- Identify superseding implementation if any
- Verify obsolete parts are truly obsolete
- Check if any non-obsolete parts need implementation
- Document what replaced the original requirement

EVIDENCE REQUIREMENTS:
- Clear superseding implementation identified
- Original requirement no longer needed
- No hanging dependencies on obsolete feature

Status: Re-close if truly obsolete, keep open if parts still needed.
"""

        else:  # DOCUMENTATION
            prompt = base_prompt + f"""
AUDIT PROTOCOL for DOCUMENTATION:
- Verify documentation exists and is complete
- Check accuracy and usefulness
- Validate examples work if provided
- Assess documentation quality and completeness

EVIDENCE REQUIREMENTS:
- Documentation files present and accessible
- Content accurate and complete
- Examples functional if provided
- Documentation serves intended purpose

Status: Re-close if documentation adequate, keep open if incomplete.
"""

        prompt += f"""

POST audit completion, add summary comment to META issue #234 using this template:

## 🔍 AUDIT SUMMARY: Issue #{issue_number}

**Classification**: {classification.value.upper()}
**Status**: [VERIFIED/INCOMPLETE/FAILED]  
**Confidence**: [0-100%]
**Evidence Quality**: [EXCELLENT/GOOD/POOR/MISSING]

**Findings**: [Key discoveries and evidence]
**Recommendation**: [CLOSE/KEEP_OPEN/NEEDS_WORK]

Follow all instructions in claude/agents/rif-validator.md for evidence collection.
"""
        
        return prompt

    async def launch_parallel_audit_agents(self, issues: List[Dict[str, Any]], batch_size: int = 5) -> None:
        """
        Launch parallel audit agents using Claude Code Task tool
        Processes issues in batches to manage resource usage
        """
        logger.info(f"Launching parallel audit for {len(issues)} issues in batches of {batch_size}")
        
        self.batch_progress.total_issues = len(issues)
        self.batch_progress.pending = len(issues)
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(issues), batch_size):
            batch = issues[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}: issues {[issue['number'] for issue in batch]}")
            
            # Launch all agents in this batch
            for issue in batch:
                await self.launch_single_audit_agent(issue)
                
            # Update progress
            self.batch_progress.in_progress += len(batch)
            self.batch_progress.pending -= len(batch)
            self.save_progress_checkpoint()
            
            # Brief pause between batches
            await asyncio.sleep(2)

    async def launch_single_audit_agent(self, issue: Dict[str, Any]) -> None:
        """
        Launch single audit agent for specific issue
        Uses Claude Code Task delegation
        """
        issue_number = issue['number']
        classification = self.classify_issue(issue)
        
        logger.info(f"Launching audit agent for issue #{issue_number} (classification: {classification.value})")
        
        try:
            # Create audit task prompt
            task_prompt = self.create_audit_agent_task(issue, classification)
            
            # Initialize audit result
            self.audit_results[issue_number] = AuditResult(
                issue_number=issue_number,
                title=issue['title'],
                classification=classification,
                status=AuditStatus.IN_PROGRESS,
                confidence=0.0,
                evidence_quality="pending",
                findings=[],
                recommendations=[],
                timestamp=datetime.now().isoformat(),
                audit_agent="rif-validator"
            )
            
            logger.info(f"Audit agent task created for issue #{issue_number}")
            
        except Exception as e:
            logger.error(f"Failed to launch audit agent for issue #{issue_number}: {e}")
            self.audit_results[issue_number] = AuditResult(
                issue_number=issue_number,
                title=issue['title'],
                classification=classification,
                status=AuditStatus.FAILED,
                confidence=0.0,
                evidence_quality="error",
                findings=[f"Agent launch failed: {e}"],
                recommendations=["Retry with manual intervention"],
                timestamp=datetime.now().isoformat(),
                audit_agent="orchestrator"
            )

    def collect_evidence(self, issue_number: int) -> Dict[str, Any]:
        """
        Collect evidence for specific issue audit
        Searches codebase and validates implementation claims
        """
        evidence = {
            "issue_number": issue_number,
            "timestamp": datetime.now().isoformat(),
            "implementation_files": [],
            "test_files": [],
            "documentation": [],
            "functionality_tests": {},
            "quality_metrics": {}
        }
        
        try:
            # Search for implementation files related to issue
            search_terms = [f"issue-{issue_number}", f"issue_{issue_number}", f"#{issue_number}"]
            
            for term in search_terms:
                # Search for code files
                result = subprocess.run([
                    "grep", "-r", "-l", term, self.repo_path, 
                    "--include=*.py", "--include=*.js", "--include=*.ts"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    files = result.stdout.strip().split('\n')
                    evidence["implementation_files"].extend([f for f in files if f])
            
            # Remove duplicates
            evidence["implementation_files"] = list(set(evidence["implementation_files"]))
            
            logger.info(f"Collected evidence for issue #{issue_number}: {len(evidence['implementation_files'])} files found")
            
        except Exception as e:
            logger.error(f"Evidence collection failed for issue #{issue_number}: {e}")
            evidence["error"] = str(e)
            
        return evidence

    def validate_quality_gates(self, issue_number: int, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply quality validation gates to audit evidence
        Returns quality score and recommendations
        """
        quality_score = 100
        quality_issues = []
        
        # Implementation file validation
        if not evidence.get("implementation_files"):
            quality_score -= 40
            quality_issues.append("No implementation files found")
            
        # Test coverage validation
        if not evidence.get("test_files"):
            quality_score -= 30
            quality_issues.append("No test files found")
            
        # Documentation validation
        if not evidence.get("documentation"):
            quality_score -= 20
            quality_issues.append("No documentation found")
            
        # Functionality test validation
        if not evidence.get("functionality_tests"):
            quality_score -= 10
            quality_issues.append("No functionality tests performed")
            
        quality_result = {
            "score": max(0, quality_score),
            "grade": self._get_quality_grade(quality_score),
            "issues": quality_issues,
            "recommendations": self._generate_quality_recommendations(quality_issues)
        }
        
        return quality_result

    def _get_quality_grade(self, score: int) -> str:
        """Convert quality score to letter grade"""
        if score >= 90: return "A"
        elif score >= 80: return "B"  
        elif score >= 70: return "C"
        elif score >= 60: return "D"
        else: return "F"

    def _generate_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        if "No implementation files found" in issues:
            recommendations.append("Search for implementation using broader patterns")
            recommendations.append("Verify issue was actually implemented vs just closed")
            
        if "No test files found" in issues:
            recommendations.append("Look for integration tests or manual testing evidence")
            recommendations.append("Consider if testing was done externally")
            
        if "No documentation found" in issues:
            recommendations.append("Check for inline documentation or comments")
            recommendations.append("Look for documentation in related files")
            
        return recommendations

    def save_progress_checkpoint(self) -> None:
        """Save current audit progress to checkpoint file"""
        checkpoint = {
            "progress": {
                "total_issues": self.batch_progress.total_issues,
                "completed": self.batch_progress.completed,
                "in_progress": self.batch_progress.in_progress,
                "pending": self.batch_progress.pending,
                "failed": self.batch_progress.failed,
                "start_time": self.batch_progress.start_time,
                "last_update": datetime.now().isoformat()
            },
            "results": {
                str(k): {
                    "issue_number": v.issue_number,
                    "title": v.title,
                    "classification": v.classification.value,
                    "status": v.status.value,
                    "confidence": v.confidence,
                    "evidence_quality": v.evidence_quality,
                    "findings": v.findings,
                    "recommendations": v.recommendations,
                    "timestamp": v.timestamp,
                    "audit_agent": v.audit_agent
                } for k, v in self.audit_results.items()
            }
        }
        
        checkpoint_file = self.audit_dir / f"audit_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
        logger.info(f"Progress checkpoint saved: {checkpoint_file}")

    def generate_progress_report(self) -> str:
        """Generate human-readable progress report"""
        completed_pct = (self.batch_progress.completed / self.batch_progress.total_issues * 100) if self.batch_progress.total_issues > 0 else 0
        
        report = f"""
# Comprehensive Audit Progress Report

## Overall Progress
- **Total Issues**: {self.batch_progress.total_issues}
- **Completed**: {self.batch_progress.completed} ({completed_pct:.1f}%)
- **In Progress**: {self.batch_progress.in_progress}
- **Pending**: {self.batch_progress.pending}
- **Failed**: {self.batch_progress.failed}

## Classification Breakdown
"""
        
        # Count by classification
        classification_counts = {}
        for result in self.audit_results.values():
            classification = result.classification.value
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
            
        for classification, count in classification_counts.items():
            report += f"- **{classification.replace('_', ' ').title()}**: {count} issues\n"
            
        report += f"""
## Quality Summary
"""
        
        # Quality statistics
        quality_grades = {}
        total_confidence = 0
        confidence_count = 0
        
        for result in self.audit_results.values():
            if result.evidence_quality and result.evidence_quality != "pending":
                quality_grades[result.evidence_quality] = quality_grades.get(result.evidence_quality, 0) + 1
                
            if result.confidence > 0:
                total_confidence += result.confidence
                confidence_count += 1
                
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        report += f"- **Average Confidence**: {avg_confidence:.1f}%\n"
        
        for grade, count in quality_grades.items():
            report += f"- **{grade.title()} Evidence**: {count} issues\n"
            
        report += f"""
## Timing
- **Started**: {self.batch_progress.start_time}
- **Last Update**: {datetime.now().isoformat()}

Generated by Comprehensive Audit Orchestrator
"""
        
        return report

    async def orchestrate_comprehensive_audit(self) -> Dict[str, Any]:
        """
        Main orchestration method - coordinates entire audit process
        Returns comprehensive audit results
        """
        logger.info("Starting comprehensive audit orchestration")
        
        try:
            # Step 1: Get all closed issues
            closed_issues = self.get_closed_issues()
            if not closed_issues:
                return {"error": "No closed issues found", "status": "failed"}
                
            logger.info(f"Found {len(closed_issues)} closed issues to audit")
            
            # Step 2: Initialize batch progress
            self.batch_progress.total_issues = len(closed_issues)
            self.batch_progress.pending = len(closed_issues)
            
            # Step 3: Launch parallel audit agents
            await self.launch_parallel_audit_agents(closed_issues)
            
            # Step 4: Generate initial progress report
            progress_report = self.generate_progress_report()
            
            # Step 5: Save checkpoint
            self.save_progress_checkpoint()
            
            # Step 6: Return orchestration results
            results = {
                "status": "orchestration_complete",
                "total_issues": len(closed_issues),
                "agents_launched": len(self.audit_results),
                "classifications": {
                    classification.value: len([
                        r for r in self.audit_results.values() 
                        if r.classification == classification
                    ]) for classification in IssueClassification
                },
                "progress_report": progress_report,
                "next_steps": [
                    "Audit agents are now running in parallel",
                    "Results will be posted to individual issues and META issue #234",
                    "Monitor progress through checkpoint files",
                    "Quality validation gates will be applied automatically"
                ]
            }
            
            logger.info("Comprehensive audit orchestration completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "partial_results": len(self.audit_results)
            }


async def main():
    """Main execution function for standalone testing"""
    orchestrator = ComprehensiveAuditOrchestrator()
    results = await orchestrator.orchestrate_comprehensive_audit()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE AUDIT ORCHESTRATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())