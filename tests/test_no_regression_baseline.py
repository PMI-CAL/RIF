#!/usr/bin/env python3
"""
RIF No-Regression Baseline Test Suite

This test suite validates that ALL currently working RIF functionality
remains intact after any modifications. These tests establish the 
baseline truth that the system IS working and must continue working.

Run before and after ANY changes to ensure no regression.
"""

import os
import json
import subprocess
import pytest
from pathlib import Path
import re

# Test configuration
RIF_ROOT = Path("/Users/cal/DEV/RIF")
CLAUDE_AGENTS_DIR = RIF_ROOT / "claude" / "agents"
CLAUDE_MD_PATH = RIF_ROOT / "CLAUDE.md"
BASELINE_DATA_DIR = RIF_ROOT / "knowledge" / "validation"

class TestTaskOrchestrationBaseline:
    """Test that Task() orchestration patterns are preserved and functional"""
    
    def test_task_patterns_in_claude_md(self):
        """Verify CLAUDE.md contains working Task() orchestration examples"""
        claude_md_content = CLAUDE_MD_PATH.read_text()
        
        # Verify key Task() patterns exist
        assert "Task(" in claude_md_content, "Task() pattern missing from CLAUDE.md"
        assert "subagent_type=\"general-purpose\"" in claude_md_content, "subagent_type pattern missing"
        assert "RIF-Analyst" in claude_md_content, "RIF-Analyst Task pattern missing"
        assert "RIF-Implementer" in claude_md_content, "RIF-Implementer Task pattern missing"
        
        # Verify parallel execution documentation
        assert "These Tasks run IN PARALLEL" in claude_md_content, "Parallel execution documentation missing"
        assert "multiple Task tool invocations in a single Claude response" in claude_md_content, "Parallel pattern explanation missing"
        
    def test_task_patterns_in_agents(self):
        """Verify agent files contain Task() usage patterns"""
        agent_files = list(CLAUDE_AGENTS_DIR.glob("*.md"))
        assert len(agent_files) > 0, "No agent files found"
        
        task_pattern_found = False
        for agent_file in agent_files:
            content = agent_file.read_text()
            if "Task(" in content:
                task_pattern_found = True
                # Verify Task patterns include proper structure
                assert "subagent_type" in content or "general-purpose" in content, f"Invalid Task pattern in {agent_file}"
                
        assert task_pattern_found, "No Task() patterns found in agent files"
        
    def test_orchestration_examples_valid(self):
        """Verify orchestration examples in CLAUDE.md are syntactically valid"""
        claude_md_content = CLAUDE_MD_PATH.read_text()
        
        # Extract Task() examples
        task_examples = re.findall(r'Task\([^)]+\)', claude_md_content, re.DOTALL)
        assert len(task_examples) > 0, "No Task examples found"
        
        for example in task_examples:
            # Verify required parameters present
            assert "description=" in example, f"Missing description in Task example: {example}"
            assert "subagent_type=" in example, f"Missing subagent_type in Task example: {example}"
            assert "prompt=" in example, f"Missing prompt in Task example: {example}"


class TestAgentSystemBaseline:
    """Test that all RIF agents are properly defined and functional"""
    
    def test_all_rif_agents_exist(self):
        """Verify all core RIF agent files exist"""
        required_agents = [
            "rif-analyst.md",
            "rif-planner.md", 
            "rif-architect.md",
            "rif-implementer.md",
            "rif-validator.md",
            "rif-learner.md"
        ]
        
        for agent_file in required_agents:
            agent_path = CLAUDE_AGENTS_DIR / agent_file
            assert agent_path.exists(), f"Required agent file missing: {agent_file}"
            assert agent_path.stat().st_size > 0, f"Agent file is empty: {agent_file}"
            
    def test_agent_structure_integrity(self):
        """Verify agent files have proper structure"""
        agent_files = [f for f in CLAUDE_AGENTS_DIR.glob("rif-*.md")]
        
        for agent_file in agent_files:
            content = agent_file.read_text()
            
            # Verify essential sections exist
            assert "## Role" in content or "Role" in content, f"Missing Role section in {agent_file}"
            assert "## Responsibilities" in content or "Responsibilities" in content, f"Missing Responsibilities in {agent_file}"
            assert "## Workflow" in content or "Workflow" in content, f"Missing Workflow in {agent_file}"
            
    def test_github_integration_patterns(self):
        """Verify agents contain GitHub integration patterns"""
        agent_files = [f for f in CLAUDE_AGENTS_DIR.glob("rif-*.md")]
        
        github_patterns_found = 0
        for agent_file in agent_files:
            content = agent_file.read_text()
            if any(pattern in content for pattern in ["state:", "GitHub", "issue", "label"]):
                github_patterns_found += 1
                
        assert github_patterns_found > 0, "No GitHub integration patterns found in agents"


class TestParallelExecutionBaseline:
    """Test that parallel execution patterns are documented and preserved"""
    
    def test_parallel_execution_documentation(self):
        """Verify parallel execution is properly documented"""
        claude_md_content = CLAUDE_MD_PATH.read_text()
        
        # Verify parallel execution concepts
        assert "parallel" in claude_md_content.lower(), "Parallel execution not documented"
        assert "multiple Task" in claude_md_content, "Multiple Task pattern not documented"
        assert "single response" in claude_md_content, "Single response pattern not documented"
        
    def test_orchestration_examples_show_parallel(self):
        """Verify orchestration examples demonstrate parallel patterns"""
        claude_md_content = CLAUDE_MD_PATH.read_text()
        
        # Find sections with multiple Task() calls
        sections = claude_md_content.split("```python")
        parallel_examples = 0
        
        for section in sections:
            if section.count("Task(") > 1:
                parallel_examples += 1
                
        assert parallel_examples > 0, "No parallel Task execution examples found"


class TestGitHubIntegrationBaseline:
    """Test that GitHub integration patterns are preserved"""
    
    def test_github_state_management_documented(self):
        """Verify GitHub state management is documented"""
        claude_md_content = CLAUDE_MD_PATH.read_text()
        
        # Verify state management concepts
        assert "state:" in claude_md_content, "GitHub state labels not documented"
        assert "GitHub issues" in claude_md_content, "GitHub issues not mentioned"
        
    def test_current_github_connectivity(self):
        """Verify gh CLI is available and configured (basic test)"""
        try:
            result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
            assert result.returncode == 0, "GitHub CLI not authenticated"
        except FileNotFoundError:
            pytest.skip("GitHub CLI not available")
            
    def test_workflow_labels_documented(self):
        """Verify workflow labels are documented"""
        claude_md_content = CLAUDE_MD_PATH.read_text()
        
        expected_labels = ["state:new", "state:analyzing", "state:planning", "state:implementing"]
        for label in expected_labels:
            assert label in claude_md_content, f"Workflow label {label} not documented"


class TestBaselinePerformance:
    """Establish performance baselines that must be maintained"""
    
    def test_agent_file_loading_performance(self):
        """Measure agent file loading performance (baseline)"""
        import time
        
        start_time = time.time()
        agent_files = list(CLAUDE_AGENTS_DIR.glob("*.md"))
        for agent_file in agent_files:
            content = agent_file.read_text()
            assert len(content) > 0
        end_time = time.time()
        
        load_time = end_time - start_time
        # Store baseline (should be very fast)
        assert load_time < 1.0, f"Agent file loading too slow: {load_time}s"
        
        # Store baseline for future comparison
        baseline_file = BASELINE_DATA_DIR / "performance_baseline.json"
        baseline_file.parent.mkdir(exist_ok=True)
        
        if baseline_file.exists():
            baseline_data = json.loads(baseline_file.read_text())
        else:
            baseline_data = {}
            
        baseline_data["agent_file_load_time"] = load_time
        baseline_file.write_text(json.dumps(baseline_data, indent=2))
        
    def test_claude_md_parsing_performance(self):
        """Measure CLAUDE.md parsing performance (baseline)"""
        import time
        
        start_time = time.time()
        claude_md_content = CLAUDE_MD_PATH.read_text()
        task_patterns = re.findall(r'Task\([^)]+\)', claude_md_content, re.DOTALL)
        end_time = time.time()
        
        parse_time = end_time - start_time
        assert parse_time < 0.5, f"CLAUDE.md parsing too slow: {parse_time}s"
        
        # Store baseline
        baseline_file = BASELINE_DATA_DIR / "performance_baseline.json"
        baseline_file.parent.mkdir(exist_ok=True)
        
        if baseline_file.exists():
            baseline_data = json.loads(baseline_file.read_text())
        else:
            baseline_data = {}
            
        baseline_data["claude_md_parse_time"] = parse_time
        baseline_data["task_patterns_found"] = len(task_patterns)
        baseline_file.write_text(json.dumps(baseline_data, indent=2))


class TestBaselinePreservation:
    """Test that baseline functionality is preserved exactly"""
    
    def test_create_baseline_snapshot(self):
        """Create snapshot of current working state"""
        baseline_file = BASELINE_DATA_DIR / "baseline_snapshot.json"
        baseline_file.parent.mkdir(exist_ok=True)
        
        snapshot = {
            "timestamp": subprocess.run(["date", "-u"], capture_output=True, text=True).stdout.strip(),
            "claude_md_size": CLAUDE_MD_PATH.stat().st_size,
            "agent_count": len(list(CLAUDE_AGENTS_DIR.glob("*.md"))),
            "task_patterns": len(re.findall(r'Task\(', CLAUDE_MD_PATH.read_text())),
            "rif_agents": [f.name for f in CLAUDE_AGENTS_DIR.glob("rif-*.md")],
        }
        
        baseline_file.write_text(json.dumps(snapshot, indent=2))
        assert baseline_file.exists(), "Failed to create baseline snapshot"
        
    def test_validate_against_baseline(self):
        """Validate current state matches baseline (if baseline exists)"""
        baseline_file = BASELINE_DATA_DIR / "baseline_snapshot.json"
        
        if not baseline_file.exists():
            pytest.skip("No baseline exists yet - this test creates the baseline")
            
        baseline = json.loads(baseline_file.read_text())
        
        # Verify no regression in key metrics
        current_agent_count = len(list(CLAUDE_AGENTS_DIR.glob("*.md")))
        assert current_agent_count >= baseline["agent_count"], "Agent files missing - regression detected"
        
        current_task_patterns = len(re.findall(r'Task\(', CLAUDE_MD_PATH.read_text()))
        assert current_task_patterns >= baseline["task_patterns"], "Task patterns missing - regression detected"
        
        current_rif_agents = [f.name for f in CLAUDE_AGENTS_DIR.glob("rif-*.md")]
        for required_agent in baseline["rif_agents"]:
            assert required_agent in current_rif_agents, f"RIF agent {required_agent} missing - regression detected"


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])