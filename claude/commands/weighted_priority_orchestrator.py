#!/usr/bin/env python3
"""
Weighted Priority Orchestrator for RIF GitHub-Native Automation

This module implements the weighted priority system that replaces the old
blocking PR approach with intelligent parallel processing capabilities.
"""

import json
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    PR = "pr"
    ISSUE = "issue"
    BLOCKING_ISSUE = "blocking_issue"

class AutomationLevel(Enum):
    GITHUB_NATIVE = "github_native"
    COPILOT_ASSISTED = "copilot_assisted" 
    RIF_MANAGED = "rif_managed"

class PRComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class PriorityTask:
    """Represents a prioritized task in the orchestration queue"""
    task_id: str
    task_type: TaskType
    priority_weight: float
    title: str
    complexity: Optional[str] = None
    automation_level: Optional[AutomationLevel] = None
    dependencies: List[str] = None
    blocked_by: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.blocked_by is None:
            self.blocked_by = []

class WeightedPriorityOrchestrator:
    """
    Main orchestrator implementing the weighted priority system for RIF.
    
    This replaces the old blocking PR approach with intelligent parallel 
    processing based on GitHub-native automation capabilities.
    """
    
    # Priority weights as defined in CLAUDE.md
    PRIORITY_WEIGHTS = {
        TaskType.BLOCKING_ISSUE: 3.0,
        TaskType.PR: 2.0,
        TaskType.ISSUE: 1.0
    }
    
    # Additional weight modifiers
    COMPLEXITY_MODIFIERS = {
        PRComplexity.SIMPLE: 0.8,
        PRComplexity.MEDIUM: 1.0,
        PRComplexity.COMPLEX: 1.2
    }
    
    # Progressive automation thresholds
    AUTOMATION_THRESHOLDS = {
        "max_lines_simple": 50,
        "max_files_simple": 5,
        "security_patterns": ["auth", "crypto", "token", "secret", "password"],
        "docs_only_patterns": [".md", ".txt", ".rst"],
        "test_only_patterns": ["test_", "_test.py", ".spec.", ".test."]
    }

    def __init__(self, max_parallel_capacity: int = 4):
        """Initialize the weighted priority orchestrator"""
        self.max_parallel_capacity = max_parallel_capacity
        
    def get_open_prs(self) -> List[Dict[str, Any]]:
        """Fetch open PRs using GitHub CLI"""
        try:
            cmd = [
                "gh", "pr", "list", 
                "--state", "open",
                "--json", "number,title,files,additions,deletions,labels,author,createdAt"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch PRs: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PR JSON: {e}")
            return []
    
    def get_open_issues(self) -> List[Dict[str, Any]]:
        """Fetch open issues using GitHub CLI"""
        try:
            cmd = [
                "gh", "issue", "list",
                "--state", "open", 
                "--json", "number,title,labels,body,comments,assignees"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch issues: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse issue JSON: {e}")
            return []
    
    def classify_pr_complexity(self, pr_data: Dict[str, Any]) -> PRComplexity:
        """Classify PR complexity for automation routing"""
        additions = pr_data.get("additions", 0)
        deletions = pr_data.get("deletions", 0)
        total_changes = additions + deletions
        files_changed = len(pr_data.get("files", []))
        
        # Check if it's a simple change
        if total_changes <= self.AUTOMATION_THRESHOLDS["max_lines_simple"]:
            if files_changed <= self.AUTOMATION_THRESHOLDS["max_files_simple"]:
                # Check for security-sensitive patterns
                title_lower = pr_data.get("title", "").lower()
                is_security_related = any(
                    pattern in title_lower 
                    for pattern in self.AUTOMATION_THRESHOLDS["security_patterns"]
                )
                
                if not is_security_related:
                    return PRComplexity.SIMPLE
        
        # Check if it's a complex architectural change
        if total_changes > 500 or files_changed > 20:
            return PRComplexity.COMPLEX
            
        return PRComplexity.MEDIUM
    
    def determine_automation_level(self, pr_data: Dict[str, Any]) -> AutomationLevel:
        """Determine appropriate automation level for PR"""
        complexity = self.classify_pr_complexity(pr_data)
        
        # Check for documentation-only changes
        files = pr_data.get("files", [])
        if files:
            file_extensions = [f.split('.')[-1] if '.' in f else '' for f in files]
            is_docs_only = all(
                any(pattern in f for pattern in self.AUTOMATION_THRESHOLDS["docs_only_patterns"])
                for f in files
            )
            
            # Check for test-only changes  
            is_test_only = all(
                any(pattern in f for pattern in self.AUTOMATION_THRESHOLDS["test_only_patterns"])
                for f in files
            )
            
            if is_docs_only or is_test_only:
                return AutomationLevel.GITHUB_NATIVE
        
        # Route based on complexity
        if complexity == PRComplexity.SIMPLE:
            return AutomationLevel.COPILOT_ASSISTED
        elif complexity == PRComplexity.COMPLEX:
            return AutomationLevel.RIF_MANAGED
        else:
            return AutomationLevel.COPILOT_ASSISTED
    
    def detect_blocking_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect issues that block all other orchestration"""
        blocking_phrases = [
            "THIS ISSUE BLOCKS ALL OTHERS",
            "THIS ISSUE BLOCKS ALL OTHER WORK", 
            "BLOCKS ALL OTHER WORK",
            "BLOCKS ALL OTHERS",
            "STOP ALL WORK",
            "MUST COMPLETE BEFORE ALL",
            "MUST COMPLETE BEFORE ALL OTHER WORK",
            "MUST COMPLETE BEFORE ALL OTHERS"
        ]
        
        blocking_issues = []
        for issue in issues:
            issue_body = issue.get("body", "").upper()
            issue_title = issue.get("title", "").upper()
            
            # Check comments as well
            comments = issue.get("comments", [])
            comment_text = " ".join([c.get("body", "") for c in comments]).upper()
            
            full_text = f"{issue_title} {issue_body} {comment_text}"
            
            if any(phrase in full_text for phrase in blocking_phrases):
                blocking_issues.append(issue)
                
        return blocking_issues
    
    def calculate_weighted_priorities(self, prs: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> List[PriorityTask]:
        """Calculate weighted priorities for all tasks"""
        priority_tasks = []
        
        # Process PRs
        for pr in prs:
            complexity = self.classify_pr_complexity(pr)
            automation_level = self.determine_automation_level(pr)
            
            base_weight = self.PRIORITY_WEIGHTS[TaskType.PR]
            complexity_modifier = self.COMPLEXITY_MODIFIERS[complexity]
            final_weight = base_weight * complexity_modifier
            
            task = PriorityTask(
                task_id=f"pr-{pr['number']}",
                task_type=TaskType.PR,
                priority_weight=final_weight,
                title=pr["title"],
                complexity=complexity.value,
                automation_level=automation_level
            )
            priority_tasks.append(task)
        
        # Process issues - check for blocking first
        blocking_issues = self.detect_blocking_issues(issues)
        
        for issue in issues:
            is_blocking = issue in blocking_issues
            task_type = TaskType.BLOCKING_ISSUE if is_blocking else TaskType.ISSUE
            
            base_weight = self.PRIORITY_WEIGHTS[task_type] 
            
            # Apply research weight reduction for analysis/research issues
            labels = [label.get("name", "") for label in issue.get("labels", [])]
            if any("research" in label.lower() or "analysis" in label.lower() for label in labels):
                base_weight *= 0.8  # Research weight from CLAUDE.md
            
            task = PriorityTask(
                task_id=f"issue-{issue['number']}", 
                task_type=task_type,
                priority_weight=base_weight,
                title=issue["title"]
            )
            priority_tasks.append(task)
        
        # Sort by priority weight (descending)
        priority_tasks.sort(key=lambda x: x.priority_weight, reverse=True)
        
        return priority_tasks
    
    def get_parallel_tasks(self, priority_tasks: List[PriorityTask]) -> List[PriorityTask]:
        """Get tasks that can run in parallel based on capacity and dependencies"""
        
        # If blocking issues exist, only return those
        blocking_tasks = [t for t in priority_tasks if t.task_type == TaskType.BLOCKING_ISSUE]
        if blocking_tasks:
            logger.info(f"Found {len(blocking_tasks)} blocking issues - halting other orchestration")
            return blocking_tasks[:self.max_parallel_capacity]
        
        # Otherwise, return mix of highest priority tasks up to capacity
        parallel_tasks = []
        task_counter = 0
        
        for task in priority_tasks:
            if task_counter >= self.max_parallel_capacity:
                break
                
            # Check dependencies (simplified - could be enhanced)
            if not task.blocked_by:
                parallel_tasks.append(task)
                task_counter += 1
                
        return parallel_tasks
    
    def trigger_github_automation(self, pr_number: int) -> bool:
        """Trigger GitHub automation for simple PRs"""
        try:
            # Enable auto-merge for the PR
            cmd = ["gh", "pr", "merge", str(pr_number), "--auto", "--squash"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Enabled auto-merge for PR #{pr_number}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to enable auto-merge for PR #{pr_number}: {e}")
            return False
    
    def generate_task_launch_code(self, task: PriorityTask) -> str:
        """Generate Task() launch code for RIF agents"""
        
        if task.task_type == TaskType.PR:
            if task.automation_level == AutomationLevel.GITHUB_NATIVE:
                return f"# PR #{task.task_id.split('-')[1]} handled by GitHub automation"
            elif task.automation_level == AutomationLevel.RIF_MANAGED:
                return f'''Task(
    description="RIF-Validator: Review complex PR #{task.task_id.split('-')[1]}",
    subagent_type="general-purpose",
    prompt="You are RIF-Validator. Review complex PR #{task.task_id.split('-')[1]} '{task.title}'. This has 2x priority weight. Run tests, check quality gates, validate implementation. Follow all instructions in claude/agents/rif-validator.md."
)'''
            else:
                return f"# PR #{task.task_id.split('-')[1]} handled by Copilot automation"
                
        elif task.task_type == TaskType.BLOCKING_ISSUE:
            return f'''Task(
    description="RIF-Implementer: URGENT - Resolve blocking issue #{task.task_id.split('-')[1]}",
    subagent_type="general-purpose", 
    prompt="You are RIF-Implementer. URGENT: Resolve BLOCKING issue #{task.task_id.split('-')[1]} '{task.title}'. This issue blocks all other work. Complete immediately. Follow all instructions in claude/agents/rif-implementer.md."
)'''
        else:
            return f'''Task(
    description="RIF-Implementer: Work on issue #{task.task_id.split('-')[1]}",
    subagent_type="general-purpose",
    prompt="You are RIF-Implementer. Work on issue #{task.task_id.split('-')[1]} '{task.title}'. Follow all instructions in claude/agents/rif-implementer.md."
)'''

def main():
    """CLI interface for the weighted priority orchestrator"""
    orchestrator = WeightedPriorityOrchestrator()
    
    print("🚀 RIF Weighted Priority Orchestrator")
    print("=====================================")
    
    # Fetch current state
    print("📋 Fetching open PRs and issues...")
    prs = orchestrator.get_open_prs()
    issues = orchestrator.get_open_issues()
    
    print(f"Found {len(prs)} open PRs and {len(issues)} open issues")
    
    # Calculate priorities
    print("\n⚖️  Calculating weighted priorities...")
    priority_tasks = orchestrator.calculate_weighted_priorities(prs, issues)
    
    # Show priority analysis
    print("\n📊 Priority Analysis:")
    for task in priority_tasks[:10]:  # Show top 10
        print(f"  {task.task_id}: {task.title[:50]}")
        print(f"    Priority Weight: {task.priority_weight:.1f}")
        print(f"    Type: {task.task_type.value}")
        if task.complexity:
            print(f"    Complexity: {task.complexity}")
        if task.automation_level:
            print(f"    Automation: {task.automation_level.value}")
        print()
    
    # Get parallel tasks
    parallel_tasks = orchestrator.get_parallel_tasks(priority_tasks)
    
    print(f"\n🔄 Recommended Parallel Execution ({len(parallel_tasks)} tasks):")
    print("=" * 50)
    
    for task in parallel_tasks:
        if task.task_type == TaskType.PR and task.automation_level == AutomationLevel.GITHUB_NATIVE:
            print(f"🤖 GitHub Auto: PR #{task.task_id.split('-')[1]} - {task.title}")
            # Actually trigger automation
            pr_number = int(task.task_id.split('-')[1])
            orchestrator.trigger_github_automation(pr_number)
        else:
            print("🧠 RIF Agent Task:")
            print(orchestrator.generate_task_launch_code(task))
            print()

if __name__ == "__main__":
    main()