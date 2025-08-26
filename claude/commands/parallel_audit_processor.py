#!/usr/bin/env python3
"""
Parallel Audit Processing Pipeline for GitHub Issue #234
Implements high-performance batch processing of audit tasks with Task delegation
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import subprocess
import concurrent.futures
from queue import Queue, Empty
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingPhase(Enum):
    """Processing phases for audit pipeline"""
    CLASSIFICATION = "classification"
    EVIDENCE_COLLECTION = "evidence_collection"  
    AGENT_LAUNCH = "agent_launch"
    QUALITY_VALIDATION = "quality_validation"
    RESULTS_AGGREGATION = "results_aggregation"

@dataclass
class ProcessingTask:
    """Individual task for parallel processing"""
    task_id: str
    issue_number: int
    phase: ProcessingPhase
    priority: int
    data: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass 
class ProcessingMetrics:
    """Performance metrics for pipeline monitoring"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: float = 0.0
    throughput_per_minute: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    peak_concurrency: int = 0

class ParallelAuditProcessor:
    """
    High-performance parallel processing pipeline for audit tasks
    Implements worker pool pattern with Task delegation for Claude Code
    """
    
    def __init__(self, max_workers: int = 5, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.task_queue = Queue()
        self.results_queue = Queue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingTask] = {}
        self.metrics = ProcessingMetrics()
        self.shutdown_event = threading.Event()
        self.workers: List[threading.Thread] = []
        
        # Processing results storage
        self.repo_path = Path("/Users/cal/DEV/RIF")
        self.processing_dir = self.repo_path / "knowledge" / "audits" / "processing"
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Parallel audit processor initialized: {max_workers} workers, batch size {batch_size}")

    def create_classification_tasks(self, issues: List[Dict[str, Any]]) -> List[ProcessingTask]:
        """Create classification tasks for issue analysis"""
        tasks = []
        
        for i, issue in enumerate(issues):
            task = ProcessingTask(
                task_id=f"classify_{issue['number']}_{int(time.time())}",
                issue_number=issue['number'],
                phase=ProcessingPhase.CLASSIFICATION,
                priority=1,  # High priority for classification
                data={"issue": issue},
                created_at=datetime.now().isoformat()
            )
            tasks.append(task)
            
        logger.info(f"Created {len(tasks)} classification tasks")
        return tasks

    def create_evidence_collection_tasks(self, classified_issues: List[Dict[str, Any]]) -> List[ProcessingTask]:
        """Create evidence collection tasks for classified issues"""
        tasks = []
        
        for i, issue_data in enumerate(classified_issues):
            task = ProcessingTask(
                task_id=f"evidence_{issue_data['issue_number']}_{int(time.time())}",
                issue_number=issue_data['issue_number'],
                phase=ProcessingPhase.EVIDENCE_COLLECTION,
                priority=2,  # Medium priority
                data=issue_data,
                created_at=datetime.now().isoformat()
            )
            tasks.append(task)
            
        logger.info(f"Created {len(tasks)} evidence collection tasks")
        return tasks

    def create_agent_launch_tasks(self, evidence_data: List[Dict[str, Any]]) -> List[ProcessingTask]:
        """Create agent launch tasks for audit execution"""
        tasks = []
        
        for issue_data in evidence_data:
            task = ProcessingTask(
                task_id=f"agent_{issue_data['issue_number']}_{int(time.time())}",
                issue_number=issue_data['issue_number'],
                phase=ProcessingPhase.AGENT_LAUNCH,
                priority=3,  # Lower priority - resource intensive
                data=issue_data,
                created_at=datetime.now().isoformat()
            )
            tasks.append(task)
            
        logger.info(f"Created {len(tasks)} agent launch tasks")
        return tasks

    def process_classification_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process single issue classification task"""
        try:
            issue = task.data["issue"]
            title = issue.get('title', '').lower()
            labels = [label.get('name', '').lower() for label in issue.get('labels', [])]
            body = issue.get('body', '').lower()
            
            # Classification logic (from orchestrator)
            classification = self._classify_issue_type(title, labels, body)
            
            result = {
                "issue_number": issue['number'],
                "title": issue['title'],
                "classification": classification,
                "complexity": self._assess_complexity(issue),
                "priority": self._calculate_priority(issue, classification),
                "estimated_duration": self._estimate_duration(classification)
            }
            
            logger.debug(f"Classified issue #{issue['number']} as {classification}")
            return result
            
        except Exception as e:
            raise Exception(f"Classification failed: {e}")

    def process_evidence_collection_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process evidence collection for specific issue"""
        try:
            issue_number = task.issue_number
            classification = task.data.get('classification')
            
            evidence = {
                "issue_number": issue_number,
                "classification": classification,
                "timestamp": datetime.now().isoformat(),
                "implementation_files": self._search_implementation_files(issue_number),
                "test_files": self._search_test_files(issue_number),
                "documentation": self._search_documentation(issue_number),
                "related_prs": self._search_related_prs(issue_number),
                "commit_history": self._search_commit_history(issue_number)
            }
            
            # Quality assessment of evidence
            evidence["quality_score"] = self._assess_evidence_quality(evidence)
            evidence["completeness"] = self._assess_evidence_completeness(evidence)
            
            logger.debug(f"Collected evidence for issue #{issue_number}: quality {evidence['quality_score']}")
            return evidence
            
        except Exception as e:
            raise Exception(f"Evidence collection failed: {e}")

    def process_agent_launch_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process agent launch task - creates Task delegation"""
        try:
            issue_number = task.issue_number
            evidence = task.data
            classification = evidence.get('classification')
            
            # Generate Task prompt based on evidence and classification
            task_prompt = self._generate_audit_task_prompt(evidence)
            
            # Create Task delegation specification
            agent_task = {
                "task_type": "adversarial_audit",
                "issue_number": issue_number,
                "classification": classification,
                "prompt": task_prompt,
                "expected_duration": evidence.get('estimated_duration', '30 minutes'),
                "required_tools": ["gh", "grep", "find", "python3"],
                "evidence_requirements": evidence.get('completeness', {}),
                "created_at": datetime.now().isoformat()
            }
            
            # Save agent task for execution
            self._save_agent_task(agent_task)
            
            logger.info(f"Generated audit agent task for issue #{issue_number}")
            return agent_task
            
        except Exception as e:
            raise Exception(f"Agent launch task failed: {e}")

    def _classify_issue_type(self, title: str, labels: List[str], body: str) -> str:
        """Classify issue type using enhanced heuristics"""
        
        # Test issue patterns
        test_patterns = ['test', 'testing', 'validation', 'demo', 'example', 'spec']
        if any(pattern in title for pattern in test_patterns):
            return "test_issue"
            
        # Documentation patterns  
        doc_patterns = ['documentation', 'readme', 'guide', 'docs', 'manual', 'tutorial']
        if any(pattern in title for pattern in doc_patterns):
            return "documentation"
            
        # Obsolete patterns
        obsolete_patterns = ['obsolete', 'deprecated', 'superseded', 'replaced', 'duplicate']
        if any(pattern in title or pattern in body for pattern in obsolete_patterns):
            return "obsolete_issue"
            
        # Enhancement patterns
        enhancement_patterns = ['enhance', 'improve', 'optimize', 'upgrade', 'refactor']
        if any(pattern in title for pattern in enhancement_patterns):
            return "enhancement"
            
        # Bug fix patterns
        bug_patterns = ['fix', 'bug', 'error', 'issue', 'problem', 'crash']
        if any(pattern in title for pattern in bug_patterns):
            return "bug_fix"
            
        # Default to feature implementation
        return "current_feature"

    def _assess_complexity(self, issue: Dict[str, Any]) -> str:
        """Assess issue complexity for resource planning"""
        title = issue.get('title', '').lower()
        body = issue.get('body', '')
        
        # High complexity indicators
        high_complexity = ['architecture', 'system', 'framework', 'infrastructure', 'migration']
        if any(indicator in title for indicator in high_complexity):
            return "high"
            
        # Medium complexity indicators  
        medium_complexity = ['integration', 'api', 'database', 'performance', 'security']
        if any(indicator in title for indicator in medium_complexity):
            return "medium"
            
        # Body length as complexity indicator
        if len(body) > 1000:
            return "medium"
        elif len(body) > 500:
            return "low"
        else:
            return "trivial"

    def _calculate_priority(self, issue: Dict[str, Any], classification: str) -> int:
        """Calculate processing priority (1=highest, 5=lowest)"""
        
        # Priority by classification
        priority_map = {
            "current_feature": 1,  # Highest priority
            "bug_fix": 2,
            "enhancement": 3, 
            "test_issue": 4,
            "documentation": 4,
            "obsolete_issue": 5  # Lowest priority
        }
        
        base_priority = priority_map.get(classification, 3)
        
        # Adjust for recency (more recent = higher priority)
        try:
            closed_at = issue.get('closedAt', '')
            if closed_at:
                days_ago = (datetime.now() - datetime.fromisoformat(closed_at.replace('Z', '+00:00'))).days
                if days_ago < 7:
                    base_priority = max(1, base_priority - 1)  # Boost recent issues
                elif days_ago > 30:
                    base_priority = min(5, base_priority + 1)  # Lower priority for old issues
        except:
            pass  # Skip if date parsing fails
            
        return base_priority

    def _estimate_duration(self, classification: str) -> str:
        """Estimate audit duration by classification"""
        duration_map = {
            "current_feature": "45 minutes",
            "bug_fix": "30 minutes", 
            "enhancement": "35 minutes",
            "test_issue": "15 minutes",
            "documentation": "20 minutes",
            "obsolete_issue": "10 minutes"
        }
        return duration_map.get(classification, "30 minutes")

    def _search_implementation_files(self, issue_number: int) -> List[str]:
        """Search for implementation files related to issue"""
        files = []
        search_patterns = [
            f"issue-{issue_number}",
            f"issue_{issue_number}", 
            f"#{issue_number}",
            f"issue{issue_number}"
        ]
        
        try:
            for pattern in search_patterns:
                result = subprocess.run([
                    "grep", "-r", "-l", pattern, str(self.repo_path),
                    "--include=*.py", "--include=*.js", "--include=*.ts", 
                    "--include=*.md", "--exclude-dir=.git"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    found_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                    files.extend(found_files)
                    
        except Exception as e:
            logger.warning(f"File search failed for issue #{issue_number}: {e}")
            
        return list(set(files))  # Remove duplicates

    def _search_test_files(self, issue_number: int) -> List[str]:
        """Search for test files related to issue"""
        test_files = []
        
        try:
            # Search in test directories
            test_dirs = ['tests', 'test', '__tests__', 'spec']
            for test_dir in test_dirs:
                test_path = self.repo_path / test_dir
                if test_path.exists():
                    result = subprocess.run([
                        "find", str(test_path), "-name", f"*{issue_number}*",
                        "-o", "-name", f"*issue*{issue_number}*"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        found_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                        test_files.extend(found_files)
                        
        except Exception as e:
            logger.warning(f"Test file search failed for issue #{issue_number}: {e}")
            
        return test_files

    def _search_documentation(self, issue_number: int) -> List[str]:
        """Search for documentation related to issue"""
        docs = []
        
        try:
            # Search for markdown files mentioning the issue
            result = subprocess.run([
                "grep", "-r", "-l", f"#{issue_number}", str(self.repo_path),
                "--include=*.md", "--exclude-dir=.git"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                docs = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                
        except Exception as e:
            logger.warning(f"Documentation search failed for issue #{issue_number}: {e}")
            
        return docs

    def _search_related_prs(self, issue_number: int) -> List[Dict[str, Any]]:
        """Search for PRs related to issue"""
        prs = []
        
        try:
            # Use gh CLI to search for PRs mentioning the issue
            result = subprocess.run([
                "gh", "pr", "list", "--state", "all", "--search", f"#{issue_number}",
                "--json", "number,title,state,mergedAt"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                prs = json.loads(result.stdout)
                
        except Exception as e:
            logger.warning(f"PR search failed for issue #{issue_number}: {e}")
            
        return prs

    def _search_commit_history(self, issue_number: int) -> List[Dict[str, Any]]:
        """Search commit history for issue references"""
        commits = []
        
        try:
            # Search git history for commits mentioning the issue
            result = subprocess.run([
                "git", "log", "--grep", f"#{issue_number}", 
                "--pretty=format:%H|%s|%an|%ad", "--date=iso"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('|')
                        if len(parts) == 4:
                            commits.append({
                                "hash": parts[0],
                                "message": parts[1], 
                                "author": parts[2],
                                "date": parts[3]
                            })
                            
        except Exception as e:
            logger.warning(f"Commit search failed for issue #{issue_number}: {e}")
            
        return commits

    def _assess_evidence_quality(self, evidence: Dict[str, Any]) -> float:
        """Assess quality of collected evidence (0-100)"""
        score = 0.0
        
        # Implementation files (40 points)
        impl_files = evidence.get('implementation_files', [])
        if len(impl_files) > 0:
            score += min(40, len(impl_files) * 10)  # 10 points per file, max 40
            
        # Test files (25 points)  
        test_files = evidence.get('test_files', [])
        if len(test_files) > 0:
            score += min(25, len(test_files) * 8)  # 8 points per test file
            
        # Documentation (20 points)
        docs = evidence.get('documentation', [])
        if len(docs) > 0:
            score += min(20, len(docs) * 7)  # 7 points per doc file
            
        # Related PRs (10 points)
        prs = evidence.get('related_prs', [])
        if len(prs) > 0:
            score += min(10, len(prs) * 5)  # 5 points per PR
            
        # Commit history (5 points)
        commits = evidence.get('commit_history', [])
        if len(commits) > 0:
            score += min(5, len(commits) * 1)  # 1 point per commit
            
        return min(100.0, score)

    def _assess_evidence_completeness(self, evidence: Dict[str, Any]) -> Dict[str, bool]:
        """Assess completeness of evidence categories"""
        return {
            "has_implementation": len(evidence.get('implementation_files', [])) > 0,
            "has_tests": len(evidence.get('test_files', [])) > 0,
            "has_documentation": len(evidence.get('documentation', [])) > 0,
            "has_prs": len(evidence.get('related_prs', [])) > 0,
            "has_commits": len(evidence.get('commit_history', [])) > 0
        }

    def _generate_audit_task_prompt(self, evidence: Dict[str, Any]) -> str:
        """Generate comprehensive audit task prompt for Claude Code Task"""
        issue_number = evidence['issue_number']
        classification = evidence.get('classification', 'unknown')
        
        prompt = f"""You are RIF-Validator performing adversarial audit of closed issue #{issue_number}.

MANDATORY DOCUMENTATION CONSULTATION:
Before ANY audit work, you MUST:
1. Use mcp__rif-knowledge__get_claude_documentation for Claude Code capabilities
2. Use mcp__rif-knowledge__query_knowledge for similar audit patterns
3. Use mcp__rif-knowledge__check_compatibility for approach validation

EVIDENCE PRE-ANALYSIS:
Issue Classification: {classification}
Implementation Files Found: {len(evidence.get('implementation_files', []))}
Test Files Found: {len(evidence.get('test_files', []))}  
Documentation Found: {len(evidence.get('documentation', []))}
Related PRs: {len(evidence.get('related_prs', []))}
Quality Score: {evidence.get('quality_score', 0):.1f}/100

AUDIT PROTOCOL:
1. REOPEN issue #{issue_number} for independent verification
2. Validate each evidence file exists and is functional:
"""
        
        # Add specific files to validate
        impl_files = evidence.get('implementation_files', [])
        if impl_files:
            prompt += f"\n   Implementation Files to Validate:"
            for file_path in impl_files[:5]:  # Limit to top 5 files
                prompt += f"\n   - {file_path}"
                
        test_files = evidence.get('test_files', [])  
        if test_files:
            prompt += f"\n   Test Files to Validate:"
            for file_path in test_files[:3]:  # Limit to top 3 test files
                prompt += f"\n   - {file_path}"
                
        prompt += f"""

3. Perform functional testing of claimed features
4. Cross-reference with PRs and commits for implementation proof
5. Apply adversarial questioning to all claims
6. Generate confidence score (0-100%) based on evidence strength

VALIDATION REQUIREMENTS:
- Functional imports/execution of implementation code  
- Test execution and pass rates
- Integration verification with existing systems
- Documentation accuracy and completeness

POST AUDIT:
1. Comment detailed findings on issue #{issue_number}
2. Post summary to META issue #234 using template:

## 🔍 AUDIT COMPLETE: Issue #{issue_number}
**Classification**: {classification}
**Evidence Quality**: [EXCELLENT/GOOD/FAIR/POOR]
**Confidence**: [0-100%]
**Status**: [VERIFIED/INCOMPLETE/FAILED]
**Findings**: [Key evidence and testing results]

Follow all rif-validator.md instructions for evidence documentation.
"""
        
        return prompt

    def _save_agent_task(self, agent_task: Dict[str, Any]) -> None:
        """Save agent task for execution tracking"""
        task_file = self.processing_dir / f"agent_task_{agent_task['issue_number']}.json"
        with open(task_file, 'w') as f:
            json.dump(agent_task, f, indent=2)
            
        logger.debug(f"Agent task saved: {task_file}")

    def worker_thread(self, worker_id: int) -> None:
        """Worker thread for processing tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                
                # Update metrics
                self.metrics.peak_concurrency = max(
                    self.metrics.peak_concurrency,
                    len(self.active_tasks) + 1
                )
                
                # Process the task
                self._process_task(task, worker_id)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Empty:
                continue  # Timeout - check shutdown event
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} shut down")

    def _process_task(self, task: ProcessingTask, worker_id: int) -> None:
        """Process individual task with error handling"""
        task.started_at = datetime.now().isoformat()
        task.status = "processing"
        self.active_tasks[task.task_id] = task
        
        logger.info(f"Worker {worker_id} processing {task.phase.value} task for issue #{task.issue_number}")
        
        try:
            start_time = time.time()
            
            # Route to appropriate processor
            if task.phase == ProcessingPhase.CLASSIFICATION:
                result = self.process_classification_task(task)
            elif task.phase == ProcessingPhase.EVIDENCE_COLLECTION:
                result = self.process_evidence_collection_task(task)
            elif task.phase == ProcessingPhase.AGENT_LAUNCH:
                result = self.process_agent_launch_task(task)
            else:
                raise Exception(f"Unknown task phase: {task.phase}")
                
            # Update task with success
            task.completed_at = datetime.now().isoformat()
            task.status = "completed"
            task.result = result
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.completed_tasks += 1
            self._update_average_duration(duration)
            
            logger.info(f"Task completed: {task.task_id} in {duration:.2f}s")
            
        except Exception as e:
            # Handle task failure
            task.completed_at = datetime.now().isoformat()
            task.status = "failed"
            task.error = str(e)
            
            self.metrics.failed_tasks += 1
            logger.error(f"Task failed: {task.task_id} - {e}")
            
        finally:
            # Move to completed tasks and remove from active
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
                
            # Add to results queue for aggregation
            self.results_queue.put(task)

    def _update_average_duration(self, duration: float) -> None:
        """Update running average of task durations"""
        if self.metrics.completed_tasks == 1:
            self.metrics.average_task_duration = duration
        else:
            # Rolling average
            current_avg = self.metrics.average_task_duration
            new_avg = ((current_avg * (self.metrics.completed_tasks - 1)) + duration) / self.metrics.completed_tasks
            self.metrics.average_task_duration = new_avg

    def start_workers(self) -> None:
        """Start worker threads for parallel processing"""
        logger.info(f"Starting {self.max_workers} worker threads")
        
        self.metrics.start_time = datetime.now().isoformat()
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self.worker_thread,
                args=(i + 1,),
                name=f"AuditWorker-{i + 1}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        logger.info("All workers started")

    def stop_workers(self) -> None:
        """Stop all worker threads gracefully"""
        logger.info("Stopping workers...")
        
        self.shutdown_event.set()
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        self.metrics.end_time = datetime.now().isoformat()
        logger.info("All workers stopped")

    def add_tasks(self, tasks: List[ProcessingTask]) -> None:
        """Add tasks to processing queue"""
        for task in tasks:
            self.task_queue.put(task)
            self.metrics.total_tasks += 1
            
        logger.info(f"Added {len(tasks)} tasks to queue")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete"""
        logger.info("Waiting for task completion...")
        
        try:
            self.task_queue.join()  # Wait for all tasks to be processed
            logger.info("All tasks completed")
            return True
        except KeyboardInterrupt:
            logger.warning("Task completion interrupted by user")
            return False

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary and metrics"""
        
        # Calculate throughput
        if self.metrics.start_time and self.metrics.end_time:
            start_dt = datetime.fromisoformat(self.metrics.start_time)
            end_dt = datetime.fromisoformat(self.metrics.end_time)
            duration_minutes = (end_dt - start_dt).total_seconds() / 60
            
            if duration_minutes > 0:
                self.metrics.throughput_per_minute = self.metrics.completed_tasks / duration_minutes
        
        # Aggregate results by phase
        results_by_phase = {}
        for task in self.completed_tasks.values():
            phase = task.phase.value
            if phase not in results_by_phase:
                results_by_phase[phase] = {"completed": 0, "failed": 0}
                
            if task.status == "completed":
                results_by_phase[phase]["completed"] += 1
            else:
                results_by_phase[phase]["failed"] += 1
        
        return {
            "processing_metrics": asdict(self.metrics),
            "results_by_phase": results_by_phase,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "success_rate": (self.metrics.completed_tasks / self.metrics.total_tasks * 100) if self.metrics.total_tasks > 0 else 0,
            "estimated_remaining_time": self._estimate_remaining_time()
        }

    def _estimate_remaining_time(self) -> Optional[str]:
        """Estimate remaining processing time"""
        if self.metrics.average_task_duration <= 0 or self.task_queue.empty():
            return None
            
        remaining_tasks = self.task_queue.qsize() + len(self.active_tasks)
        estimated_seconds = (remaining_tasks * self.metrics.average_task_duration) / self.max_workers
        
        return str(timedelta(seconds=int(estimated_seconds)))

    async def process_pipeline_batch(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main pipeline method - processes full batch through all phases
        Returns comprehensive results and metrics
        """
        logger.info(f"Starting parallel processing pipeline for {len(issues)} issues")
        
        try:
            # Start worker threads
            self.start_workers()
            
            # Phase 1: Classification
            logger.info("Phase 1: Starting classification tasks")
            classification_tasks = self.create_classification_tasks(issues)
            self.add_tasks(classification_tasks)
            
            # Wait for classification to complete
            self.wait_for_completion()
            
            # Collect classification results  
            classified_issues = []
            for task in self.completed_tasks.values():
                if task.phase == ProcessingPhase.CLASSIFICATION and task.status == "completed":
                    classified_issues.append(task.result)
                    
            logger.info(f"Phase 1 complete: {len(classified_issues)} issues classified")
            
            # Phase 2: Evidence Collection
            logger.info("Phase 2: Starting evidence collection tasks")
            evidence_tasks = self.create_evidence_collection_tasks(classified_issues)
            self.add_tasks(evidence_tasks)
            
            # Wait for evidence collection to complete
            self.wait_for_completion()
            
            # Collect evidence results
            evidence_data = []
            for task in self.completed_tasks.values():
                if task.phase == ProcessingPhase.EVIDENCE_COLLECTION and task.status == "completed":
                    evidence_data.append(task.result)
                    
            logger.info(f"Phase 2 complete: Evidence collected for {len(evidence_data)} issues")
            
            # Phase 3: Agent Launch
            logger.info("Phase 3: Starting agent launch tasks") 
            agent_tasks = self.create_agent_launch_tasks(evidence_data)
            self.add_tasks(agent_tasks)
            
            # Wait for agent launches to complete
            self.wait_for_completion()
            
            # Collect agent task results
            launched_agents = []
            for task in self.completed_tasks.values():
                if task.phase == ProcessingPhase.AGENT_LAUNCH and task.status == "completed":
                    launched_agents.append(task.result)
                    
            logger.info(f"Phase 3 complete: {len(launched_agents)} audit agents launched")
            
            # Generate final summary
            summary = self.get_processing_summary()
            summary.update({
                "pipeline_status": "completed",
                "issues_processed": len(issues),
                "issues_classified": len(classified_issues),
                "evidence_collected": len(evidence_data),
                "agents_launched": len(launched_agents),
                "next_steps": [
                    "Audit agents are now executing adversarial validation",
                    "Results will be posted to individual issues and META issue #234",
                    "Quality validation gates will be applied automatically",
                    "Monitor agent progress through individual issue comments"
                ]
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                "pipeline_status": "failed",
                "error": str(e),
                "partial_results": self.get_processing_summary()
            }
            
        finally:
            # Clean shutdown
            self.stop_workers()


# Main execution for testing
async def main():
    """Test the parallel processing pipeline"""
    processor = ParallelAuditProcessor(max_workers=3, batch_size=5)
    
    # Mock issues for testing
    test_issues = [
        {"number": 1, "title": "Test Implementation", "labels": [], "body": "Test body"},
        {"number": 2, "title": "Bug Fix", "labels": [], "body": "Fix bug"},
        {"number": 3, "title": "Documentation Update", "labels": [], "body": "Update docs"}
    ]
    
    results = await processor.process_pipeline_batch(test_issues)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())