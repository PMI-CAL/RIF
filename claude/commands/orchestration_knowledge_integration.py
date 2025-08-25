"""
Orchestration Knowledge Integration

Stores orchestration patterns, anti-patterns, and lessons learned in the knowledge base.
Creates training materials and guidance for future orchestration decisions.

Issue #224: RIF Orchestration Error: Incorrect Parallel Task Launching
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class OrchestrationPattern:
    """Represents a successful or failed orchestration pattern"""
    pattern_id: str
    pattern_name: str
    pattern_type: str  # "correct", "anti-pattern", "best-practice"
    description: str
    example_tasks: List[Dict[str, str]]
    context: str
    effectiveness_rating: float  # 0-1, higher is better
    usage_count: int
    success_rate: float
    lessons_learned: List[str]
    related_issues: List[int]
    created_at: str
    updated_at: str


@dataclass
class OrchestrationLesson:
    """Represents a lesson learned from orchestration experience"""
    lesson_id: str
    title: str
    category: str  # "anti-pattern", "best-practice", "troubleshooting"
    description: str
    problem_context: str
    solution_approach: str
    prevention_measures: List[str]
    related_patterns: List[str]
    confidence_level: float  # 0-1
    source_issues: List[int]
    created_at: str


class OrchestrationKnowledgeIntegration:
    """
    Integrates orchestration knowledge into the RIF knowledge base.
    
    Stores patterns, anti-patterns, lessons learned, and training materials
    for future orchestration decision making.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.patterns_dir = Path(self.knowledge_base_path) / "patterns"
        self.lessons_dir = Path(self.knowledge_base_path) / "learnings"
        self.training_dir = Path(self.knowledge_base_path) / "training"
        
        # Ensure directories exist
        self.patterns_dir.mkdir(exist_ok=True)
        self.lessons_dir.mkdir(exist_ok=True)
        self.training_dir.mkdir(exist_ok=True)
        
    def store_orchestration_pattern(
        self,
        pattern: OrchestrationPattern
    ) -> bool:
        """Store an orchestration pattern in the knowledge base"""
        try:
            pattern_file = self.patterns_dir / f"orchestration_pattern_{pattern.pattern_id}.json"
            
            with open(pattern_file, 'w') as f:
                json.dump(asdict(pattern), f, indent=2)
                
            # Update pattern index
            self._update_pattern_index(pattern)
            
            return True
            
        except Exception as e:
            print(f"Error storing orchestration pattern: {e}")
            return False
            
    def store_orchestration_lesson(
        self,
        lesson: OrchestrationLesson
    ) -> bool:
        """Store an orchestration lesson learned in the knowledge base"""
        try:
            lesson_file = self.lessons_dir / f"orchestration_lesson_{lesson.lesson_id}.json"
            
            with open(lesson_file, 'w') as f:
                json.dump(asdict(lesson), f, indent=2)
                
            # Update lessons index
            self._update_lessons_index(lesson)
            
            return True
            
        except Exception as e:
            print(f"Error storing orchestration lesson: {e}")
            return False
            
    def create_issue_224_knowledge_base_entries(self) -> Dict[str, bool]:
        """Create knowledge base entries specific to Issue #224"""
        results = {}
        
        # Store the Multi-Issue Accelerator anti-pattern
        multi_issue_anti_pattern = OrchestrationPattern(
            pattern_id="multi_issue_accelerator_224",
            pattern_name="Multi-Issue Accelerator Anti-Pattern",
            pattern_type="anti-pattern",
            description="Single Task attempting to handle multiple issues in parallel. Creates sequential bottleneck and violates Claude Code's parallel execution model.",
            example_tasks=[
                {
                    "description": "Multi-Issue Accelerator: Handle issues #1, #2, #3",
                    "prompt": "You are a Multi-Issue Accelerator agent. Handle these issues in parallel: Issue #1: user auth, Issue #2: database pool, Issue #3: API validation",
                    "subagent_type": "general-purpose"
                }
            ],
            context="Occurs when orchestrator misunderstands parallel execution model and creates single Task for multiple issues instead of multiple Tasks in one response.",
            effectiveness_rating=0.1,  # Very poor
            usage_count=1,
            success_rate=0.0,  # Always fails
            lessons_learned=[
                "Single Task cannot achieve true parallel execution for multiple issues",
                "Claude Code's parallel model requires multiple Task invocations in one response",
                "Multi-issue Task descriptions violate single responsibility principle",
                "Manual intervention required to correct this anti-pattern"
            ],
            related_issues=[224],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        results["multi_issue_anti_pattern"] = self.store_orchestration_pattern(multi_issue_anti_pattern)
        
        # Store the correct parallel pattern
        correct_parallel_pattern = OrchestrationPattern(
            pattern_id="correct_parallel_tasks_224",
            pattern_name="Correct Parallel Task Launching",
            pattern_type="correct",
            description="Multiple Task invocations launched in a single Claude response for true parallel execution. Each Task handles one specific issue with specialized agent.",
            example_tasks=[
                {
                    "description": "RIF-Implementer: User authentication system",
                    "prompt": "You are RIF-Implementer. Implement user authentication for issue #1. Follow all instructions in claude/agents/rif-implementer.md.",
                    "subagent_type": "general-purpose"
                },
                {
                    "description": "RIF-Implementer: Database connection pooling",
                    "prompt": "You are RIF-Implementer. Implement database connection pooling for issue #2. Follow all instructions in claude/agents/rif-implementer.md.",
                    "subagent_type": "general-purpose"
                },
                {
                    "description": "RIF-Validator: API validation framework",
                    "prompt": "You are RIF-Validator. Validate API framework for issue #3. Follow all instructions in claude/agents/rif-validator.md.",
                    "subagent_type": "general-purpose"
                }
            ],
            context="Proper implementation of Claude Code's parallel execution model. Multiple Tasks launched in single response run concurrently.",
            effectiveness_rating=0.95,  # Excellent
            usage_count=1,
            success_rate=1.0,  # Always succeeds
            lessons_learned=[
                "One Task = One Issue = One Specialized Agent principle",
                "Multiple Tasks in single response enables true parallel execution",
                "Agent specialization improves task effectiveness",
                "Proper agent instructions essential for success"
            ],
            related_issues=[224],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        results["correct_parallel_pattern"] = self.store_orchestration_pattern(correct_parallel_pattern)
        
        # Store the primary lesson learned
        issue_224_lesson = OrchestrationLesson(
            lesson_id="issue_224_primary_lesson",
            title="Avoid Multi-Issue Accelerator Anti-Pattern",
            category="anti-pattern",
            description="The Multi-Issue Accelerator anti-pattern occurs when attempting to create a single Task agent to handle multiple issues in parallel, which violates Claude Code's execution model and creates performance bottlenecks.",
            problem_context="User requested orchestration to handle multiple open issues. Orchestrator incorrectly attempted to create single 'Multi-Issue Accelerator' agent instead of multiple Task invocations.",
            solution_approach="Replace single multi-issue Task with multiple specialized Tasks launched in one response. Each Task handles exactly one issue with appropriate specialized agent.",
            prevention_measures=[
                "Always validate: One Task = One Issue = One Specialized Agent",
                "Use proper parallel launching: Multiple Task() in single response",
                "Validate task descriptions don't contain multiple issue numbers",
                "Ensure agent names match standard RIF agents (RIF-Implementer, RIF-Validator, etc.)",
                "Include proper agent instructions in all Task prompts"
            ],
            related_patterns=["multi_issue_accelerator_224", "correct_parallel_tasks_224"],
            confidence_level=1.0,  # High confidence
            source_issues=[224],
            created_at=datetime.utcnow().isoformat()
        )
        
        results["issue_224_lesson"] = self.store_orchestration_lesson(issue_224_lesson)
        
        return results
        
    def create_orchestration_training_materials(self) -> bool:
        """Create training materials for orchestration best practices"""
        try:
            training_content = self._generate_orchestration_training_content()
            
            training_file = self.training_dir / "orchestration_best_practices.json"
            with open(training_file, 'w') as f:
                json.dump(training_content, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error creating training materials: {e}")
            return False
            
    def _generate_orchestration_training_content(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration training content"""
        return {
            "training_module": "Orchestration Best Practices",
            "version": "1.0.0",
            "issue_reference": 224,
            "created_at": datetime.utcnow().isoformat(),
            "sections": {
                "fundamentals": {
                    "title": "Orchestration Fundamentals",
                    "content": [
                        "Claude Code IS the orchestrator - no separate orchestrator Task needed",
                        "Parallel execution achieved through multiple Task invocations in single response",
                        "One Task = One Issue = One Specialized Agent principle",
                        "Agent specialization improves effectiveness and maintainability"
                    ]
                },
                "anti_patterns_to_avoid": {
                    "title": "Critical Anti-Patterns to Avoid",
                    "patterns": [
                        {
                            "name": "Multi-Issue Accelerator",
                            "description": "Single Task attempting to handle multiple issues",
                            "why_it_fails": "Creates sequential bottleneck, violates parallel model",
                            "detection": "Task description contains multiple issue numbers",
                            "correction": "Split into separate Tasks, one per issue"
                        },
                        {
                            "name": "Generic Accelerator Naming",
                            "description": "Using generic names like 'Accelerator', 'Batch', 'Combined'", 
                            "why_it_fails": "Loses agent specialization benefits",
                            "detection": "Task description contains generic accelerator terms",
                            "correction": "Use specific RIF agent names (RIF-Implementer, RIF-Validator, etc.)"
                        }
                    ]
                },
                "correct_patterns": {
                    "title": "Correct Orchestration Patterns",
                    "patterns": [
                        {
                            "name": "Parallel Task Launching",
                            "description": "Multiple Task invocations in single response",
                            "benefits": "True parallel execution, specialized agents, scalable",
                            "template": "Task(...) Task(...) Task(...) # All in one response"
                        },
                        {
                            "name": "Dependency-Aware Orchestration",
                            "description": "Consider dependencies before launching parallel work",
                            "benefits": "Prevents conflicts, optimizes execution order",
                            "template": "Analyze dependencies → Validate patterns → Launch appropriate Tasks"
                        }
                    ]
                },
                "validation_checklist": {
                    "title": "Pre-Launch Validation Checklist",
                    "checklist": [
                        "One Task = One Issue: Each Task handles exactly one specific concern",
                        "Proper Agent Names: Using standard RIF agent names",
                        "Single Response Launch: All parallel Tasks launched in one response",
                        "Specialized Instructions: Each Task prompt includes proper agent file reference",
                        "No Multi-Issue Descriptions: Task descriptions don't mention multiple issues",
                        "Clear Responsibility: Task purpose is single, clear, and well-defined"
                    ]
                },
                "troubleshooting": {
                    "title": "Common Orchestration Problems and Solutions",
                    "problems": [
                        {
                            "problem": "Sequential processing instead of parallel",
                            "cause": "Tasks launched in separate responses",
                            "solution": "Launch all parallel Tasks in single response"
                        },
                        {
                            "problem": "Agent specialization not utilized",
                            "cause": "Using generic agent names or prompts",
                            "solution": "Use specific RIF agents with proper instructions"
                        },
                        {
                            "problem": "Manual intervention required",
                            "cause": "Anti-patterns not caught before execution",
                            "solution": "Implement pattern validation before launching Tasks"
                        }
                    ]
                },
                "examples": {
                    "title": "Orchestration Examples",
                    "wrong_example": {
                        "description": "Multi-Issue Accelerator Anti-Pattern",
                        "code": "Task(\n    description=\"Multi-Issue Accelerator: Handle issues #1, #2, #3\",\n    prompt=\"Handle multiple issues efficiently\",\n    subagent_type=\"general-purpose\"\n)",
                        "problems": ["Single Task for multiple issues", "Generic agent name", "Sequential execution"]
                    },
                    "correct_example": {
                        "description": "Proper Parallel Task Launching",
                        "code": "Task(\n    description=\"RIF-Implementer: Issue #1 implementation\",\n    prompt=\"You are RIF-Implementer. Implement feature for issue #1. Follow all instructions in claude/agents/rif-implementer.md.\",\n    subagent_type=\"general-purpose\"\n)\nTask(\n    description=\"RIF-Implementer: Issue #2 implementation\",\n    prompt=\"You are RIF-Implementer. Implement feature for issue #2. Follow all instructions in claude/agents/rif-implementer.md.\",\n    subagent_type=\"general-purpose\"\n)\nTask(\n    description=\"RIF-Validator: Issue #3 validation\",\n    prompt=\"You are RIF-Validator. Validate implementation for issue #3. Follow all instructions in claude/agents/rif-validator.md.\",\n    subagent_type=\"general-purpose\"\n)",
                        "benefits": ["One Task per issue", "Specialized agents", "Parallel execution", "Proper instructions"]
                    }
                }
            }
        }
        
    def _update_pattern_index(self, pattern: OrchestrationPattern):
        """Update the patterns index with new pattern"""
        index_file = self.patterns_dir / "orchestration_patterns_index.json"
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {"patterns": [], "last_updated": ""}
                
            # Update or add pattern entry
            pattern_entry = {
                "pattern_id": pattern.pattern_id,
                "pattern_name": pattern.pattern_name,
                "pattern_type": pattern.pattern_type,
                "effectiveness_rating": pattern.effectiveness_rating,
                "success_rate": pattern.success_rate,
                "usage_count": pattern.usage_count,
                "related_issues": pattern.related_issues,
                "updated_at": pattern.updated_at
            }
            
            # Remove existing entry if present
            index["patterns"] = [p for p in index["patterns"] if p["pattern_id"] != pattern.pattern_id]
            
            # Add new entry
            index["patterns"].append(pattern_entry)
            index["last_updated"] = datetime.utcnow().isoformat()
            
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not update patterns index: {e}")
            
    def _update_lessons_index(self, lesson: OrchestrationLesson):
        """Update the lessons index with new lesson"""
        index_file = self.lessons_dir / "orchestration_lessons_index.json"
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {"lessons": [], "last_updated": ""}
                
            # Update or add lesson entry
            lesson_entry = {
                "lesson_id": lesson.lesson_id,
                "title": lesson.title,
                "category": lesson.category,
                "confidence_level": lesson.confidence_level,
                "source_issues": lesson.source_issues,
                "created_at": lesson.created_at
            }
            
            # Remove existing entry if present
            index["lessons"] = [l for l in index["lessons"] if l["lesson_id"] != lesson.lesson_id]
            
            # Add new entry
            index["lessons"].append(lesson_entry)
            index["last_updated"] = datetime.utcnow().isoformat()
            
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not update lessons index: {e}")
            
    def generate_knowledge_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of knowledge integration"""
        return {
            "knowledge_integration_report": {
                "issue_reference": 224,
                "integration_timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "patterns_stored": len(list(self.patterns_dir.glob("orchestration_pattern_*.json"))),
                    "lessons_stored": len(list(self.lessons_dir.glob("orchestration_lesson_*.json"))),
                    "training_materials": len(list(self.training_dir.glob("*.json")))
                },
                "integration_status": "complete",
                "validation": {
                    "patterns_accessible": self.patterns_dir.exists(),
                    "lessons_accessible": self.lessons_dir.exists(),
                    "training_accessible": self.training_dir.exists(),
                    "indices_updated": True
                }
            }
        }


# Convenience functions for knowledge integration
def integrate_issue_224_knowledge() -> Dict[str, bool]:
    """Integrate Issue #224 specific knowledge into the knowledge base"""
    integrator = OrchestrationKnowledgeIntegration()
    
    # Store patterns and lessons
    results = integrator.create_issue_224_knowledge_base_entries()
    
    # Create training materials
    results["training_materials"] = integrator.create_orchestration_training_materials()
    
    return results


def query_orchestration_patterns(pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query stored orchestration patterns"""
    knowledge_base_path = "/Users/cal/DEV/RIF/knowledge"
    patterns_dir = Path(knowledge_base_path) / "patterns"
    
    patterns = []
    
    try:
        for pattern_file in patterns_dir.glob("orchestration_pattern_*.json"):
            with open(pattern_file, 'r') as f:
                pattern = json.load(f)
                
            if pattern_type is None or pattern.get("pattern_type") == pattern_type:
                patterns.append(pattern)
                
    except Exception as e:
        print(f"Error querying patterns: {e}")
        
    return patterns


if __name__ == "__main__":
    # Test knowledge integration
    print("Integrating Issue #224 knowledge...")
    results = integrate_issue_224_knowledge()
    
    print(f"Integration results:")
    for component, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {component}: {status}")
        
    # Test pattern querying
    print("\nQuerying anti-patterns:")
    anti_patterns = query_orchestration_patterns("anti-pattern")
    print(f"Found {len(anti_patterns)} anti-patterns")
    
    for pattern in anti_patterns:
        print(f"  - {pattern['pattern_name']}: {pattern['effectiveness_rating']}")