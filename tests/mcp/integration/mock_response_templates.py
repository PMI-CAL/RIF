"""
Mock Response Templates for MCP Integration Testing

Provides realistic response templates and scenarios for different MCP server types
to ensure comprehensive testing coverage.

Issue: #86 - Build MCP integration tests  
Component: Response Templates
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import random
import time


class ResponseScenario(Enum):
    """Response scenario types"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_ERROR = "auth_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ResponseTemplate:
    """Template for mock responses"""
    scenario: ResponseScenario
    data: Dict[str, Any]
    delay_ms: int = 100
    should_raise_exception: bool = False
    exception_type: str = ""


class GitHubResponseTemplates:
    """Response templates for GitHub MCP server"""
    
    @staticmethod
    def get_repository_info_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "repository": {
                    "id": 123456789,
                    "node_id": "MDEwOlJlcG9zaXRvcnkxMjM0NTY3ODk=",
                    "name": "RIF",
                    "full_name": "PMI-CAL/RIF",
                    "description": "Reactive Intelligence Framework - Automatic intelligent development system",
                    "private": False,
                    "html_url": "https://github.com/PMI-CAL/RIF",
                    "clone_url": "https://github.com/PMI-CAL/RIF.git",
                    "default_branch": "main",
                    "language": "Python",
                    "size": 5420,
                    "stargazers_count": 15,
                    "watchers_count": 15,
                    "forks_count": 3,
                    "open_issues_count": 12,
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-12-23T14:30:00Z",
                    "pushed_at": "2024-12-23T14:25:00Z",
                    "topics": ["ai", "automation", "development", "framework"],
                    "visibility": "public"
                }
            },
            delay_ms=150
        )
    
    @staticmethod
    def get_repository_info_not_found() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.FAILURE,
            data={
                "message": "Not Found",
                "documentation_url": "https://docs.github.com/rest/repos/repos#get-a-repository"
            },
            delay_ms=100,
            should_raise_exception=True,
            exception_type="NotFoundError"
        )
    
    @staticmethod
    def list_issues_success() -> ResponseTemplate:
        issues = []
        for i in range(random.randint(3, 15)):
            issue_num = i + 80  # Start from issue #80
            issues.append({
                "id": 2000000000 + i,
                "number": issue_num,
                "title": f"Implement feature {chr(65 + i % 26)}",
                "body": f"Description for issue #{issue_num}",
                "state": random.choice(["open", "closed"]),
                "locked": False,
                "assignee": {
                    "login": "rif-implementer",
                    "id": 100000001,
                    "avatar_url": "https://avatars.githubusercontent.com/u/100000001?v=4"
                } if random.random() > 0.3 else None,
                "labels": [
                    {
                        "id": 4000000001,
                        "name": f"state:{random.choice(['new', 'implementing', 'validating', 'complete'])}",
                        "color": random.choice(["0052CC", "5319E7", "D4C5F9", "28A745"])
                    },
                    {
                        "id": 4000000002,
                        "name": f"complexity:{random.choice(['low', 'medium', 'high'])}",
                        "color": random.choice(["FFA500", "FF6B35", "DC3545"])
                    }
                ],
                "created_at": "2024-12-20T10:00:00Z",
                "updated_at": "2024-12-23T14:00:00Z",
                "html_url": f"https://github.com/PMI-CAL/RIF/issues/{issue_num}",
                "comments": random.randint(0, 8),
                "user": {
                    "login": "cal",
                    "id": 100000000,
                    "avatar_url": "https://avatars.githubusercontent.com/u/100000000?v=4"
                }
            })
        
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={"issues": issues, "total_count": len(issues)},
            delay_ms=random.randint(200, 400)
        )
    
    @staticmethod
    def get_issue_success(issue_number: int = 86) -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "issue": {
                    "id": 2000000086,
                    "number": issue_number,
                    "title": "Build MCP integration tests",
                    "body": "## Issue #86: MCP Integration Enhancement\n\n### Objective\nCreate mock framework for testing, design integration scenarios, and establish performance benchmarks.\n\n### Scope\n- Mock server framework\n- Integration test scenarios\n- Performance benchmarks\n- Test automation",
                    "state": "open",
                    "locked": False,
                    "assignee": {
                        "login": "rif-implementer",
                        "id": 100000001
                    },
                    "labels": [
                        {
                            "id": 4000000001,
                            "name": "state:implementing",
                            "color": "0e8a16"
                        },
                        {
                            "id": 4000000002,
                            "name": "complexity:medium",
                            "color": "FFA500"
                        }
                    ],
                    "created_at": "2024-12-22T09:00:00Z",
                    "updated_at": "2024-12-23T15:00:00Z",
                    "html_url": f"https://github.com/PMI-CAL/RIF/issues/{issue_number}",
                    "comments": 3,
                    "user": {
                        "login": "cal",
                        "id": 100000000
                    }
                }
            },
            delay_ms=120
        )
    
    @staticmethod
    def create_issue_comment_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "comment": {
                    "id": 3000000001,
                    "html_url": "https://github.com/PMI-CAL/RIF/issues/86#issuecomment-3000000001",
                    "issue_url": "https://api.github.com/repos/PMI-CAL/RIF/issues/86",
                    "body": "Integration test implementation completed successfully.",
                    "user": {
                        "login": "rif-implementer",
                        "id": 100000001
                    },
                    "created_at": "2024-12-23T15:30:00Z",
                    "updated_at": "2024-12-23T15:30:00Z"
                }
            },
            delay_ms=250
        )
    
    @staticmethod
    def rate_limited_error() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.RATE_LIMITED,
            data={
                "message": "API rate limit exceeded for user.",
                "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"
            },
            delay_ms=100,
            should_raise_exception=True,
            exception_type="RateLimitError"
        )


class MemoryResponseTemplates:
    """Response templates for Memory MCP server"""
    
    @staticmethod
    def store_memory_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "stored": True,
                "memory_id": f"mem_{int(time.time())}_{random.randint(1000, 9999)}",
                "key": "integration_test_memory",
                "content_hash": "sha256:abcd1234...",
                "timestamp": time.time(),
                "size_bytes": random.randint(500, 5000),
                "indexed": True
            },
            delay_ms=80
        )
    
    @staticmethod
    def retrieve_memory_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "found": True,
                "memory": {
                    "id": "mem_1703344800_1234",
                    "key": "integration_test_memory",
                    "content": {
                        "type": "implementation_pattern",
                        "description": "MCP integration test pattern",
                        "code_examples": [
                            "async def test_parallel_execution():",
                            "    results = await aggregator.get_context(...)"
                        ],
                        "metadata": {
                            "created_by": "rif-implementer",
                            "complexity": "medium",
                            "success_rate": 0.95
                        }
                    },
                    "relevance_score": 0.87,
                    "last_accessed": time.time() - 3600,
                    "access_count": 15
                }
            },
            delay_ms=90
        )
    
    @staticmethod
    def search_memories_success() -> ResponseTemplate:
        results = []
        for i in range(random.randint(2, 8)):
            results.append({
                "memory_id": f"mem_{int(time.time()) - random.randint(0, 86400)}_{random.randint(1000, 9999)}",
                "key": f"memory_key_{i}",
                "snippet": f"Memory content snippet {i + 1} with relevant information...",
                "relevance_score": random.uniform(0.6, 0.95),
                "metadata": {
                    "type": random.choice(["pattern", "decision", "implementation", "issue_resolution"]),
                    "created": time.time() - random.randint(0, 2592000)  # Within last 30 days
                }
            })
        
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "results": results,
                "total_count": len(results),
                "search_query": "integration test patterns",
                "execution_time_ms": random.randint(50, 200)
            },
            delay_ms=150
        )
    
    @staticmethod
    def get_context_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "context": {
                    "summary": "Integration testing context for MCP services",
                    "key_points": [
                        "Mock servers provide realistic response simulation",
                        "Performance benchmarks establish baseline expectations",
                        "Failure scenarios ensure robust error handling"
                    ],
                    "related_memories": 12,
                    "confidence_score": 0.89,
                    "context_relevance": {
                        "patterns": 0.92,
                        "implementations": 0.85,
                        "decisions": 0.78
                    }
                }
            },
            delay_ms=200
        )


class SequentialThinkingResponseTemplates:
    """Response templates for Sequential Thinking MCP server"""
    
    @staticmethod
    def start_reasoning_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "reasoning_session": {
                    "session_id": f"think_{int(time.time())}_{random.randint(100, 999)}",
                    "problem": "Design integration test strategy for MCP services",
                    "status": "initialized",
                    "initial_analysis": {
                        "problem_type": "system_integration_testing",
                        "complexity_assessment": "medium",
                        "approach_options": [
                            "Mock-based integration testing",
                            "End-to-end testing with real services",
                            "Hybrid approach with selective mocking"
                        ]
                    },
                    "reasoning_framework": "systematic_analysis",
                    "estimated_steps": random.randint(4, 8)
                }
            },
            delay_ms=300
        )
    
    @staticmethod
    def continue_reasoning_success(step_number: int = 1) -> ResponseTemplate:
        reasoning_steps = {
            1: {
                "analysis": "Analyzing requirements for MCP integration testing",
                "considerations": [
                    "Need for realistic response simulation",
                    "Performance benchmarking requirements",
                    "Error scenario coverage"
                ],
                "conclusions": ["Mock framework approach is most suitable"]
            },
            2: {
                "analysis": "Designing mock server architecture",
                "considerations": [
                    "Configurability for different scenarios",
                    "Metrics collection capabilities",
                    "Health state simulation"
                ],
                "conclusions": ["Enhanced mock servers with state management needed"]
            },
            3: {
                "analysis": "Planning performance benchmark approach",
                "considerations": [
                    "Concurrent request handling",
                    "Response time measurement",
                    "Resource utilization tracking"
                ],
                "conclusions": ["Multi-dimensional benchmarking framework required"]
            }
        }
        
        step_data = reasoning_steps.get(step_number, reasoning_steps[1])
        
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "reasoning_step": {
                    "step_number": step_number,
                    "step_type": "analysis_and_evaluation",
                    "analysis": step_data["analysis"],
                    "considerations": step_data["considerations"],
                    "conclusions": step_data["conclusions"],
                    "confidence_level": random.uniform(0.75, 0.92),
                    "next_step_preview": "Continue with implementation planning"
                }
            },
            delay_ms=random.randint(200, 500)
        )
    
    @staticmethod
    def get_conclusion_success() -> ResponseTemplate:
        return ResponseTemplate(
            scenario=ResponseScenario.SUCCESS,
            data={
                "conclusion": {
                    "final_recommendation": "Implement enhanced mock server framework with configurable scenarios",
                    "reasoning_chain": [
                        "Integration testing requires realistic service simulation",
                        "Mock servers provide controlled, repeatable test environments",
                        "Enhanced configuration enables comprehensive scenario coverage",
                        "Performance benchmarking establishes baseline expectations",
                        "Health monitoring ensures system resilience"
                    ],
                    "implementation_approach": {
                        "primary_components": [
                            "EnhancedMockMCPServer with state management",
                            "Configurable response templates",
                            "Performance benchmarking framework",
                            "Integration test suite with parallel execution"
                        ],
                        "success_criteria": [
                            "Mock servers accurately simulate real MCP behavior",
                            "Performance benchmarks provide actionable insights",
                            "Test coverage includes failure and recovery scenarios"
                        ]
                    },
                    "confidence_level": 0.88,
                    "alternative_approaches": [
                        "Docker-based real service testing",
                        "Stub-based minimal mocking"
                    ],
                    "risk_assessment": {
                        "low": ["Mock fidelity variations"],
                        "medium": ["Performance benchmark accuracy"],
                        "high": []
                    }
                }
            },
            delay_ms=400
        )


class ResponseTemplateManager:
    """Manages response templates for different scenarios"""
    
    def __init__(self):
        self.github_templates = GitHubResponseTemplates()
        self.memory_templates = MemoryResponseTemplates()
        self.thinking_templates = SequentialThinkingResponseTemplates()
    
    def get_template(self, server_type: str, operation: str, scenario: ResponseScenario = ResponseScenario.SUCCESS) -> ResponseTemplate:
        """Get response template for specific server type and operation"""
        
        if server_type == "github":
            return self._get_github_template(operation, scenario)
        elif server_type == "memory":
            return self._get_memory_template(operation, scenario)
        elif server_type == "sequential_thinking":
            return self._get_thinking_template(operation, scenario)
        else:
            return ResponseTemplate(
                scenario=ResponseScenario.SUCCESS,
                data={"message": f"Unknown server type: {server_type}"},
                delay_ms=100
            )
    
    def _get_github_template(self, operation: str, scenario: ResponseScenario) -> ResponseTemplate:
        """Get GitHub-specific template"""
        templates = {
            ("get_repository_info", ResponseScenario.SUCCESS): self.github_templates.get_repository_info_success,
            ("get_repository_info", ResponseScenario.FAILURE): self.github_templates.get_repository_info_not_found,
            ("list_issues", ResponseScenario.SUCCESS): self.github_templates.list_issues_success,
            ("get_issue", ResponseScenario.SUCCESS): self.github_templates.get_issue_success,
            ("create_issue_comment", ResponseScenario.SUCCESS): self.github_templates.create_issue_comment_success,
            ("rate_limit", ResponseScenario.RATE_LIMITED): self.github_templates.rate_limited_error,
        }
        
        template_func = templates.get((operation, scenario))
        if template_func:
            return template_func()
        else:
            return ResponseTemplate(
                scenario=scenario,
                data={"message": f"Template not found for {operation} with {scenario}"},
                delay_ms=100
            )
    
    def _get_memory_template(self, operation: str, scenario: ResponseScenario) -> ResponseTemplate:
        """Get Memory-specific template"""
        templates = {
            ("store_memory", ResponseScenario.SUCCESS): self.memory_templates.store_memory_success,
            ("retrieve_memory", ResponseScenario.SUCCESS): self.memory_templates.retrieve_memory_success,
            ("search_memories", ResponseScenario.SUCCESS): self.memory_templates.search_memories_success,
            ("get_context", ResponseScenario.SUCCESS): self.memory_templates.get_context_success,
        }
        
        template_func = templates.get((operation, scenario))
        if template_func:
            return template_func()
        else:
            return ResponseTemplate(
                scenario=scenario,
                data={"message": f"Template not found for {operation} with {scenario}"},
                delay_ms=100
            )
    
    def _get_thinking_template(self, operation: str, scenario: ResponseScenario) -> ResponseTemplate:
        """Get Sequential Thinking-specific template"""
        templates = {
            ("start_reasoning", ResponseScenario.SUCCESS): self.thinking_templates.start_reasoning_success,
            ("continue_reasoning", ResponseScenario.SUCCESS): lambda: self.thinking_templates.continue_reasoning_success(random.randint(1, 3)),
            ("get_conclusion", ResponseScenario.SUCCESS): self.thinking_templates.get_conclusion_success,
        }
        
        template_func = templates.get((operation, scenario))
        if template_func:
            return template_func()
        else:
            return ResponseTemplate(
                scenario=scenario,
                data={"message": f"Template not found for {operation} with {scenario}"},
                delay_ms=100
            )


# Predefined scenario sets for comprehensive testing
class TestScenarios:
    """Predefined test scenarios for different testing needs"""
    
    HAPPY_PATH = [
        ("github", "get_repository_info", ResponseScenario.SUCCESS),
        ("github", "list_issues", ResponseScenario.SUCCESS),
        ("memory", "retrieve_memory", ResponseScenario.SUCCESS),
        ("sequential_thinking", "start_reasoning", ResponseScenario.SUCCESS)
    ]
    
    ERROR_HANDLING = [
        ("github", "get_repository_info", ResponseScenario.FAILURE),
        ("github", "rate_limit", ResponseScenario.RATE_LIMITED),
        ("memory", "retrieve_memory", ResponseScenario.FAILURE),
        ("sequential_thinking", "start_reasoning", ResponseScenario.FAILURE)
    ]
    
    PERFORMANCE_STRESS = [
        ("github", "list_issues", ResponseScenario.SUCCESS),
        ("memory", "search_memories", ResponseScenario.SUCCESS),
        ("sequential_thinking", "continue_reasoning", ResponseScenario.SUCCESS)
    ] * 5  # Repeat for stress testing
    
    MIXED_SCENARIOS = [
        ("github", "get_repository_info", ResponseScenario.SUCCESS),
        ("memory", "retrieve_memory", ResponseScenario.FAILURE),
        ("sequential_thinking", "start_reasoning", ResponseScenario.SUCCESS),
        ("github", "create_issue_comment", ResponseScenario.SUCCESS),
        ("memory", "store_memory", ResponseScenario.SUCCESS)
    ]